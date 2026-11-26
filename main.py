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
HUAWEI_MODEL_NAME = os.getenv("HUAWEI_MODEL_NAME", "openpangu-2.0-flash")

llm_client = AsyncOpenAI(
    api_key=HUAWEI_API_KEY,
    base_url=HUAWEI_BASE_URL,
)

# 澳門常用官方熱線白名單（8 位數字）
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
}

# 澳門及常見官方域名白名單（支援子域名匹配）
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

# 第一層：極高危關鍵詞快速規則
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
輸入可能包含粵語口語（例如：過數＝轉帳、撳＝點擊、俾＝給、呃＝騙、差佬＝警察、戶口＝帳戶），並會附上系統剛才透過【官方白名單與即時聯網搜尋】查到的電話與網址核實資料。

請結合訊息內容與核實資料，嚴格只輸出以下 JSON 格式（不要輸出任何多餘文字或 Markdown 標記）：
{
  "risk_level": "high" 或 "safe",
  "scam_type": "簡短類型名稱（若安全則寫『官方正常通知』或『一般正常訊息』）",
  "reply_message": "用溫暖、簡單、長輩易懂的繁體中文回覆（60至110字內）。明確說明是否安全、剛才查核到的電話或網址真偽，以及具體建議。"
}

判斷原則：
1. 若訊息只是單純的業務結果通知（例如「申請審核成功」、「帳單已出」、「交易提醒」），**沒有**要求點擊陌生連結、**沒有**要求轉帳或交出驗證碼，且附帶的聯絡電話／網址經核實屬於該機構官方，應判為 "safe"，並在回覆中讓長輩放心（同時可順口提醒：若有疑問可直接撥打該官方熱線查詢）。
2. 凡是要求轉帳、借錢、交按金、點擊非官方陌生網址、下載軟件、開螢幕共享、透露短訊驗證碼，或附帶的網址／電話經查證與該官方不符，一律判為 "high"。
3. 注意：有時騙徒會在詐騙短訊結尾故意附上真實官方電話來混淆視聽，因此若訊息同時包含「可疑非官方連結」或「催促轉帳／解封」，即使電話是真的，仍須判為 "high"。
"""


class SosRequest(BaseModel):
    user_id: str = "elder_01"
    message: str


def _is_official_domain(host: str) -> bool:
    host = host.lower().split(":")[0].strip(".")
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def _extract_entities(text: str):
    """從文字中提取電話號碼與網址"""
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


def _web_search_sync(query: str, max_results: int = 3) -> list[str]:
    if DDGS is None:
        return []
    try:
        results = DDGS().text(query, max_results=max_results)
        snippets = []
        for r in results or []:
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()[:140]
            snippets.append(f"- {title} ({href}): {body}")
        return snippets
    except Exception:
        return []


def _verify_entities_sync(text: str) -> str:
    """檢查訊息中的電話與網址（先查本地白名單，未知項目再聯網搜尋）"""
    phones, urls = _extract_entities(text)
    if not phones and not urls:
        return "訊息中未偵測到電話號碼或網址。"

    reports = []

    for phone in phones:
        # 去除 +853 / 853 前綴比對澳門本地號碼
        local_num = re.sub(r"^(\+?853|00853)", "", phone)
        if local_num in OFFICIAL_PHONES:
            reports.append(f"【電話 {phone}】已確認為官方白名單號碼：{OFFICIAL_PHONES[local_num]}。")
        else:
            hits = _web_search_sync(f'"{phone}" 澳門 OR 香港 OR 客服 OR 詐騙', max_results=3)
            if hits:
                reports.append(f"【電話 {phone} 聯網搜尋結果】\n" + "\n".join(hits))
            else:
                reports.append(f"【電話 {phone}】不在官方白名單內，且聯網未查到明確官方歸屬，請謹慎核對。")

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


def _parse_llm_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    risk = "high" if str(data.get("risk_level", "")).lower() == "high" else "safe"
    return {
        "risk_level": risk,
        "scam_type": str(data.get("scam_type") or ("疑似詐騙" if risk == "high" else "一般正常訊息")),
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
    """呼叫華為雲 OCR 網絡圖片識別提取截圖文字"""
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

    # 1. 本地極高危規則秒判
    rule_hit = rule_based_check(msg)
    if rule_hit:
        return rule_hit

    # 2. 先核實訊息中的電話與網址（白名單 + 聯網搜尋，最多等 5 秒）
    try:
        verify_report = await asyncio.wait_for(
            asyncio.to_thread(_verify_entities_sync, msg),
            timeout=5.0,
        )
    except Exception:
        verify_report = "聯網核實逾時，請僅依據訊息內容謹慎判斷。"

    # 3. 將原始訊息與核實結果一併送給華為雲大模型判斷
    user_payload = f"【用戶收到的訊息】\n{msg}\n\n【電話與網址核實資料】\n{verify_report}"

    try:
        completion = await llm_client.chat.completions.create(
            model=HUAWEI_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.1,
            max_tokens=320,
        )
        content = completion.choices[0].message.content or ""
        return _parse_llm_json(content)
    except Exception:
        return {
            "risk_level": "high",
            "scam_type": "系統連線繁忙（建議提高警覺）",
            "reply_message": "剛才雲端分析稍微有點慢。記住，只要對方提到借錢、轉帳、點連結或要驗證碼，一律先停下來，打原本電話向家人確認！",
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
