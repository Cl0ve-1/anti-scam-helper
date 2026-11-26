import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1 import (
    OcrClient,
    RecognizeWebImageRequest,
    WebImageRequestBody,
)
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdksis.v1 import (
    Config,
    PostShortAudioReq,
    RecognizeShortAudioRequest,
    SisClient,
)
from huaweicloudsdksis.v1.region.sis_region import SisRegion
from openai import AsyncOpenAI
from pydantic import BaseModel

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

app = FastAPI(title="安心防詐助手 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

HUAWEI_API_KEY = os.getenv("HUAWEI_API_KEY", "")
HUAWEI_BASE_URL = os.getenv("HUAWEI_BASE_URL", "https://api.modelarts-maas.com/v2")
HUAWEI_MODEL_NAME = os.getenv("HUAWEI_MODEL_NAME", "deepseek-v4.1-flash-1017")

llm_client = AsyncOpenAI(
    api_key=HUAWEI_API_KEY,
    base_url=HUAWEI_BASE_URL,
)

# 澳門常用官方熱線白名單（可隨時自行補充常用快遞、診所、管理處號碼）
OFFICIAL_PHONES = {
    "28727688": "澳門通 / MPay 24小時官方客戶服務熱線",
    "28558033": "澳門通 / MPay 24小時報失熱線",
    "88007777": "澳門司法警察局防詐騙查詢熱線",
    "993": "澳門司法警察局報案熱線",
    "999": "澳門緊急求助熱線",
    "88895566": "中國銀行澳門分行客戶服務熱線",
    "28335533": "大西洋銀行 (BNU) 客戶服務熱線",
    "88995588": "工銀澳門客戶服務熱線",
    "1000": "澳門電訊 (CTM) 客戶服務熱線",
    "28339911": "澳門電力 (CEM) 客戶服務熱線",
    "28220088": "澳門自來水客戶服務熱線",
}

OFFICIAL_DOMAINS = (
    "gov.mo",
    "macaupass.com",
    "bocmacau.com",
    "bnu.com.mo",
    "icbc.com.mo",
    "ctm.net",
    "macau.ctm.net",
    "cem-macau.com",
    "macaowater.com",
)

HIGH_RISK_RULES = [
    ("假冒公檢法詐騙", ["安全帳戶", "安全账户", "洗黑錢", "洗黑钱", "通緝令", "通缉令", "拘捕令", "清查資金", "清查资金", "資產公證", "资产公证", "凍結資產", "冻结资产"]),
    ("假冒客服／遠程控屏詐騙", ["螢幕共享", "屏幕共享", "遠程控制", "远程控制", "百萬保障", "百万保障", "自動扣費", "自动扣费"]),
    ("猜猜我是誰（假冒親友）", ["住院按金", "保釋金", "保释金", "俾人拉咗", "換咗新號碼", "换了新号码"]),
    ("保密／轉帳高危話術", ["唔好同屋企人講", "不要告訴家人", "不要告诉家人", "保密協議", "保密协议"]),
]

URL_EXTRACT_PATTERN = re.compile(
    r"(https?://[^\s\u4e00-\u9fff]+|www\.[^\s\u4e00-\u9fff]+|\b[a-zA-Z0-9-]+\.(?:com|net|org|mo|hk|cn|xyz|top|vip|club|cc|info|icu|tk|app|io)(?:/[^\s\u4e00-\u9fff]*)?)",
    re.IGNORECASE,
)
PHONE_EXTRACT_PATTERN = re.compile(r"(?:\+?853[\s\-]?|\+?852[\s\-]?|00\d{2,3}[\s\-]?)?(?<!\d)(?:\d{8}|\d{3,4}-\d{4}|99[39]|1000)(?!\d)")

SYSTEM_PROMPT = """你是一位專為港澳長輩服務的「安心防詐助手」。
輸入可能包含粵語口語（例如：過數＝轉帳、撳＝點擊、俾＝給、呃＝騙、差佬＝警察、戶口＝帳戶），並會附上系統透過【官方白名單與即時聯網搜尋】查到的電話與網址核實資料。

請結合訊息內容與核實資料，嚴格只輸出以下 JSON 格式（不要輸出任何多餘文字或 Markdown 標記）：
{
  "risk_level": "high" 或 "warn" 或 "safe",
  "scam_type": "簡短類型名稱（若安全則寫『官方／正規商戶通知』或『一般正常訊息』）",
  "reply_message": "用溫暖、簡單、長輩易懂的繁體中文回覆（60至110字內）。明確說明是否安全、查核到的電話或網址歸屬，以及具體建議。"
}

判斷原則：
1. 若訊息只是單純的業務結果、取件或預約通知，沒有要求點擊陌生連結、沒有要求轉帳或交驗證碼，且電話／網址經白名單或聯網核實屬於官方、郵政、快遞或本地正規公司，應判為 "safe"。
2. 若聯網查無該小公司或私人號碼的紀錄，但內容沒有明顯詐騙誘導（如單純通知取件、聯絡送貨），應判為 "warn"，溫和提醒長輩核實即可，不要過度驚嚇。
3. 凡是要求轉帳、借錢、交按金、點擊非官方陌生網址、下載軟件、開螢幕共享、透露驗證碼，或查核到與宣稱機構不符，一律判為 "high"。
"""

PHONE_SYSTEM_PROMPT = """你是一位專為港澳長輩服務的「安心防詐助手」，負責根據【號碼格式】與【聯網搜尋結果】判斷一個陌生來電號碼的風險。

請嚴格只輸出以下 JSON 格式（不要輸出任何多餘文字或 Markdown 標記）：
{
  "risk_level": "high" 或 "warn" 或 "safe",
  "scam_type": "簡短歸屬或風險分類（例如：澳門快遞公司熱線、本地商戶號碼、疑似詐騙號碼、未登錄本地號碼）",
  "reply_message": "用溫暖、長輩易懂的繁體中文回覆（55至100字內），說明這支電話查到屬於哪間公司或機構；若查無公開資料，溫和提醒對方若談到轉帳或驗證碼才要立刻掛斷。"
}

判斷原則：
1. 若聯網搜尋結果明確顯示該號碼屬於正規郵政、快遞公司、銀行、政府或澳門本地商戶／診所／物業管理，且無詐騙舉報，判為 "safe" 並寫出機構名稱。
2. 若搜尋結果出現詐騙舉報、推銷騷擾紀錄，或屬於境外不明來電／高危虛擬號段，判為 "high"。
3. 若屬於澳門正常 8 位數本地號碼（28/6/8 開頭）但網上查無公開資料（常見於快遞送貨員私人手機或小型商戶），判為 "warn"，說明「暫無詐騙紀錄但查不到公開公司名稱」，提醒長輩正常接聽即可，只要不轉帳、不報驗證碼就安全。
"""


class SosRequest(BaseModel):
    user_id: str = "elder_01"
    message: str


class PhoneRequest(BaseModel):
    phone: str


def _is_official_domain(host: str) -> bool:
    host = host.lower().split(":")[0].strip(".")
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def _extract_entities(text: str):
    urls = list(dict.fromkeys(URL_EXTRACT_PATTERN.findall(text)))
    raw_phones = PHONE_EXTRACT_PATTERN.findall(text)
    phones = []
    seen = set()
    for p in raw_phones:
        cleaned = re.sub(r"[\s\-]", "", p)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            phones.append(cleaned)
    return phones[:3], urls[:3]


def _web_search_sync(query: str, max_results: int = 4) -> list[str]:
    if DDGS is None:
        return []
    try:
        results = DDGS().text(query, max_results=max_results)
        snippets = []
        for r in results or []:
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()[:150]
            snippets.append(f"- {title} ({href}): {body}")
        return snippets
    except Exception:
        return []


def _verify_entities_sync(text: str) -> str:
    phones, urls = _extract_entities(text)
    if not phones and not urls:
        return "訊息中未偵測到電話號碼或網址。"

    reports = []

    for phone in phones:
        local_num = re.sub(r"^(\+?853|00853)", "", phone)
        if local_num in OFFICIAL_PHONES:
            reports.append(f"【電話 {phone}】已確認為官方白名單號碼：{OFFICIAL_PHONES[local_num]}。")
        else:
            hits = _web_search_sync(f'"{phone}" 澳門 OR 快遞 OR 郵政 OR 公司 OR 詐騙', max_results=4)
            if hits:
                reports.append(f"【電話 {phone} 聯網搜尋結果】\n" + "\n".join(hits))
            else:
                reports.append(f"【電話 {phone}】不在官方白名單內，且聯網暫無公開公司或詐騙紀錄。")

    for raw_url in urls:
        parsed = urlparse(raw_url if raw_url.startswith("http") else "https://" + raw_url)
        host = parsed.netloc or raw_url
        if _is_official_domain(host):
            reports.append(f"【網址 {raw_url}】網域 ({host}) 屬於已驗證的官方網域。")
        else:
            hits = _web_search_sync(f'"{host}" 官網 OR 詐騙 OR phishing', max_results=3)
            if hits:
                reports.append(f"【網址 {raw_url} ({host}) 聯網搜尋結果】\n" + "\n".join(hits))
            else:
                reports.append(f"【網址 {raw_url} ({host})】非已知官方網域，且未查到可靠官方紀錄，疑似釣魚連結。")

    return "\n\n".join(reports)


def rule_based_check(text: str):
    for scam_type, keywords in HIGH_RISK_RULES:
        for kw in keywords:
            if kw in text:
                return {
                    "risk_level": "high",
                    "scam_type": scam_type,
                    "reply_message": f"高度危險！內容提到「{kw}」，這是非常典型的{scam_type}手法。請千萬不要轉帳、不要開啟螢幕共享，也不要透露驗證碼，立刻掛斷電話或刪除訊息，並打給家人確認！",
                }
    return None


def _parse_llm_json(raw_text: str, default_risk: str = "high") -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    raw_risk = str(data.get("risk_level", "")).lower()
    risk = raw_risk if raw_risk in ("high", "warn", "safe") else default_risk
    return {
        "risk_level": risk,
        "scam_type": str(data.get("scam_type") or ("疑似詐騙" if risk == "high" else "分析完成")),
        "reply_message": str(data.get("reply_message") or "請保持冷靜，凡涉及轉帳、借錢或點擊陌生連結，請先向家人確認。"),
    }


def _call_huawei_sis(audio_bytes: bytes, lang: str) -> str:
    ak = os.getenv("HUAWEI_AK", "")
    sk = os.getenv("HUAWEI_SK", "")
    region = os.getenv("HUAWEI_SIS_REGION", "cn-north-4")

    if not ak or not sk:
        raise RuntimeError("尚未設定 HUAWEI_AK 或 HUAWEI_SK 環境變數")

    with tempfile.NamedTemporaryFile(suffix=".input", delete=True) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f_out:
        f_in.write(audio_bytes)
        f_in.flush()
        subprocess.run(
            ["ffmpeg", "-y", "-i", f_in.name, "-ar", "16000", "-ac", "1", f_out.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        wav_bytes = f_out.read()

    credentials = BasicCredentials(ak, sk)
    client = (
        SisClient.new_builder()
        .with_credentials(credentials)
        .with_region(SisRegion.value_of(region))
        .build()
    )

    model_prop = "cantonese_16k_common" if lang == "zh-HK" else "chinese_16k_general"
    config = Config(audio_format="wav", _property=model_prop, add_punc="yes")
    req = RecognizeShortAudioRequest(
        body=PostShortAudioReq(
            config=config,
            data=base64.b64encode(wav_bytes).decode("utf-8"),
        )
    )
    resp = client.recognize_short_audio(req)
    return resp.result.text if resp.result and resp.result.text else ""


def _call_huawei_ocr(image_bytes: bytes) -> str:
    ak = os.getenv("HUAWEI_AK", "")
    sk = os.getenv("HUAWEI_SK", "")
    region = os.getenv("HUAWEI_OCR_REGION", os.getenv("HUAWEI_SIS_REGION", "cn-north-4"))

    if not ak or not sk:
        raise RuntimeError("尚未設定 HUAWEI_AK 或 HUAWEI_SK 環境變數")

    credentials = BasicCredentials(ak, sk)
    client = (
        OcrClient.new_builder()
        .with_credentials(credentials)
        .with_region(OcrRegion.value_of(region))
        .build()
    )

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    req = RecognizeWebImageRequest(
        body=WebImageRequestBody(image=img_b64)
    )
    resp = client.recognize_web_image(req)
    if not resp.result or not resp.result.words_block_list:
        return ""

    lines = [block.words.strip() for block in resp.result.words_block_list if block.words and block.words.strip()]
    return "\n".join(lines)


@app.get("/")
async def serve_home():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="找不到 index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/sos")
async def check_scam(req: SosRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="訊息內容不可為空")

    rule_hit = rule_based_check(msg)
    if rule_hit:
        return rule_hit

    try:
        verify_report = await asyncio.wait_for(
            asyncio.to_thread(_verify_entities_sync, msg),
            timeout=5.0,
        )
    except Exception:
        verify_report = "聯網核實逾時，請僅依據訊息內容謹慎判斷。"

    user_payload = f"【用戶收到的訊息】\n{msg}\n\n【電話與網址核實資料】\n{verify_report}"

    try:
        completion = await llm_client.chat.completions.create(
            model=HUAWEI_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        content = completion.choices[0].message.content or ""
        return _parse_llm_json(content)
    except Exception:
        return {
            "risk_level": "high",
            "scam_type": "系統連線繁忙（建議提高警覺）",
            "reply_message": "剛才雲端分析稍微有點慢。記住，只要對方提到借錢、轉帳、點連結或要驗證碼，一律先停下來，打原本電話向家人確認！",
        }


@app.post("/api/check_phone")
async def check_phone(req: PhoneRequest):
    raw = req.phone.strip()
    num = re.sub(r"[\s\-()]", "", raw)
    if not num or len(num) < 3:
        raise HTTPException(status_code=400, detail="請輸入完整的來電號碼")

    # 1. 境外偽裝本地改號直接攔截（即使後面 8 位數像本地號碼也屬高危）
    if re.match(r"^(\+|00)(853|852)", num):
        return {
            "risk_level": "high",
            "scam_type": "境外偽裝本地來電（改號詐騙）",
            "reply_message": "請特別小心！真正澳門或香港本地打來的電話，前面不會帶有 +853、+852 或 00。這類號碼極大機會是境外詐騙集團偽裝的，建議直接掛斷或拒接！",
        }

    # 2. 官方白名單秒查
    if num in OFFICIAL_PHONES:
        return {
            "risk_level": "safe",
            "scam_type": f"官方白名單：{OFFICIAL_PHONES[num]}",
            "reply_message": f"這個號碼已確認是「{OFFICIAL_PHONES[num]}」，屬於正規官方熱線，可以安心接聽或撥打查詢。",
        }

    # 3. 高危虛擬／境外號段特徵標記
    format_hint = "一般陌生號碼"
    if re.match(r"^(\+|00)", num):
        format_hint = "境外國際長途號碼（高風險，正規本地銀行、郵政或政府部門不會用境外號碼聯絡）"
    elif re.match(r"^(170|171|165|400|95\d{6,})", num):
        format_hint = "內地網路虛擬號段或高頻推銷客服號段（常見於假冒客服或推銷詐騙）"
    elif re.match(r"^28\d{6}$", num):
        format_hint = "澳門本地 8 位數固網電話（常見於本地公司、商戶、快遞門市或診所）"
    elif re.match(r"^6\d{7}$", num):
        format_hint = "澳門本地 8 位數手提電話（可能是親友、快遞送貨員私人手機，亦可能被騙徒用作本地太空卡）"
    elif re.match(r"^8\d{7}$", num):
        format_hint = "澳門本地 8 位數商業／熱線號碼"

    # 4. 聯網搜尋（針對快遞、郵政、小公司與詐騙舉報）
    try:
        hits = await asyncio.wait_for(
            asyncio.to_thread(_web_search_sync, f'"{num}" 澳門 OR 快遞 OR 郵政 OR 公司 OR 詐騙', 5),
            timeout=5.0,
        )
    except Exception:
        hits = []

    search_summary = "\n".join(hits) if hits else "網上暫時查無此號碼的公開公司資料或詐騙舉報紀錄。"
    payload = f"【查詢號碼】{raw}（清理後：{num}）\n【號碼格式判斷】{format_hint}\n\n【聯網搜尋結果】\n{search_summary}"

    # 5. 交由 DeepSeek 綜合判斷
    try:
        completion = await llm_client.chat.completions.create(
            model=HUAWEI_MODEL_NAME,
            messages=[
                {"role": "system", "content": PHONE_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        content = completion.choices[0].message.content or ""
        return _parse_llm_json(content, default_risk="warn")
    except Exception:
        return {
            "risk_level": "warn",
            "scam_type": format_hint,
            "reply_message": "剛才聯網查核稍微有點慢。如果是澳門本地 8 位數號碼，可能是快遞員或本地商戶；正常接聽沒關係，只要對方提到轉帳、借錢或驗證碼就立刻掛線！",
        }


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("zh-HK"),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"text": ""}

    try:
        text = await asyncio.to_thread(_call_huawei_sis, audio_bytes, lang)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"語音辨識失敗: {str(e)}")


@app.post("/api/ocr")
async def extract_image_text(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="圖片內容不可為空")

    try:
        text = await asyncio.to_thread(_call_huawei_ocr, image_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片文字辨識失敗: {str(e)}")
