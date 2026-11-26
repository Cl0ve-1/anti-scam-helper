import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

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

# 第一層：高危關鍵詞與釣魚特徵快速規則
HIGH_RISK_RULES = [
    ("假冒公檢法詐騙", ["安全帳戶", "安全账户", "洗黑錢", "洗黑钱", "通緝令", "通缉令", "拘捕令", "清查資金", "清查资金", "資產公證", "资产公证", "凍結資產", "冻结资产"]),
    ("假冒客服／遠程控屏詐騙", ["螢幕共享", "屏幕共享", "遠程控制", "远程控制", "百萬保障", "百万保障", "自動扣費", "自动扣费"]),
    ("猜猜我是誰（假冒親友）", ["住院按金", "保釋金", "保释金", "俾人拉咗", "換咗新號碼", "换了新号码"]),
    ("驗證碼／轉帳高危話術", ["驗證碼", "验证码", "唔好同屋企人講", "不要告訴家人", "不要告诉家人", "保密協議", "保密协议"]),
]

PHISHING_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(xyz|top|vip|club|cc|info|icu|tk)\b)", re.IGNORECASE)
PHISHING_HINTS = ["封禁", "凍結", "冻结", "異常", "异常", "解除", "積分", "积分", "過期", "过期", "包裹", "領取", "领取", "驗證", "验证", "撳", "點擊", "点击"]

SYSTEM_PROMPT = """你是一位專為港澳長輩服務的「安心防詐助手」。
輸入的內容可能來自語音轉錄或短訊／聊天截圖，包含粵語口語（例如：過數＝轉帳、撳＝點擊、俾＝給、呃＝騙、差佬＝警察、戶口＝帳戶、急住要錢＝急需借錢）。

請判斷該內容是否涉及詐騙，並嚴格只輸出以下 JSON 格式（不要輸出任何多餘文字或 Markdown 標記）：
{
  "risk_level": "high" 或 "safe",
  "scam_type": "簡短詐騙類型名稱（若安全則寫『一般正常訊息』）",
  "reply_message": "用溫暖、簡單、長輩易懂的繁體中文回覆（60至100字內），明確講出是否危險，並給出具體做法（例如：立刻掛線、不要點擊連結、先打原本電話找家人確認）。"
}

判斷原則：
1. 凡是涉及要求轉帳、借錢、交按金、點陌生網址、開螢幕共享、交出銀行密碼或短訊驗證碼、自稱公檢法或客服威脅扣費／封禁帳號，一律判為 "high"。
2. 寧可從嚴保護長輩，只要有明顯可疑跡象就判為 "high"。
"""


class SosRequest(BaseModel):
    user_id: str = "elder_01"
    message: str


def rule_based_check(text: str):
    for scam_type, keywords in HIGH_RISK_RULES:
        for kw in keywords:
            if kw in text:
                return {
                    "risk_level": "high",
                    "scam_type": scam_type,
                    "reply_message": f"高度危險！內容提到「{kw}」，這是非常典型的{scam_type}手法。請千萬不要轉帳、不要開啟螢幕共享，也不要透露驗證碼，立刻掛斷電話或刪除訊息，並打給家人確認！",
                }

    if PHISHING_URL_PATTERN.search(text) and any(h in text for h in PHISHING_HINTS):
        return {
            "risk_level": "high",
            "scam_type": "釣魚連結短訊詐騙",
            "reply_message": "高度危險！這條訊息帶有可疑網址並催促你處理帳戶或包裹問題，極大機會是釣魚詐騙短訊。千萬不要點擊裡面的連結，直接刪除即可！",
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

    rule_hit = rule_based_check(msg)
    if rule_hit:
        return rule_hit

    try:
        completion = await llm_client.chat.completions.create(
            model=HUAWEI_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            temperature=0.2,
            max_tokens=300,
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
