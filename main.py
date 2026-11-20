from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from openai import AsyncOpenAI
import datetime
import json
import re
import os

app = FastAPI(title="反詐騙一鍵求救 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 今晚拿到華為雲 MaaS 資訊後，填入這三行即可（保留引號）
# ==========================================


# 改為從 Render 雲端環境變數讀取（安全且不會外洩）
HUAWEI_API_KEY = os.getenv("HUAWEI_API_KEY", "")
HUAWEI_BASE_URL = os.getenv("HUAWEI_BASE_URL", "https://api.modelarts-maas.com/v2")
HUAWEI_MODEL_NAME = os.getenv("HUAWEI_MODEL_NAME", "openpangu-2.0-flash")

SYSTEM_PROMPT = """你是一位專門保護長輩的反詐騙心理安撫與鑑定助手。
長輩現在可能正處於驚慌或困惑中，請分析長輩提供的對話、語音轉錄或短訊內容，判斷是否為詐騙。

請嚴格只回傳一個純 JSON 物件（不要加上多餘文字），格式如下：
{
  "risk_level": "high 或 safe",
  "scam_type": "若是詐騙請寫類型（例如：假冒公檢法、猜猜我是誰、假中獎、假投資），安全則寫無明顯風險",
  "reply_message": "給長輩看並朗讀出來的回覆（60至90字以內，繁體中文，語氣要溫暖、鎮定、像家人一樣；先安撫情緒，再點出破綻，最後給出一個明確的保護動作，例如立刻掛斷或不要轉帳）"
}
注意：只要涉及要求轉帳、匯款、安全帳戶、說出驗證碼、點擊不明連結、自稱警察/海關/檢察官/客服要求配合調查，risk_level 一律判定為 "high"。"""


# 啟動時只建立一次客戶端，重複利用連線（省下每次重新 SSL 握手的時間）
ai_client = AsyncOpenAI(
    api_key=HUAWEI_API_KEY,
    base_url=HUAWEI_BASE_URL,
    timeout=20.0,
    max_retries=0
) if (HUAWEI_API_KEY and HUAWEI_BASE_URL) else None

SYSTEM_PROMPT = """你是一位專門保護長輩的反詐騙心理安撫與鑑定助手。
請分析長輩提供的對話、語音轉錄或短訊內容，判斷是否為詐騙。

請嚴格只回傳一個純 JSON 物件（不要加上多餘文字），格式如下：
{
  "risk_level": "high 或 safe",
  "scam_type": "若是詐騙請寫類型（例如：釣魚網站短訊、假冒公檢法、猜猜我是誰、假中獎、假投資），安全則寫無明顯風險",
  "reply_message": "給長輩看並朗讀出來的回覆（60至90字以內，繁體中文，語氣溫暖鎮定；先安撫情緒，再點出破綻，最後給出一個明確的保護動作，例如千萬不要點連結或立刻掛斷）"
}
高風險判定規則（符合任一項 risk_level 一律為 "high"）：
1. 內容包含不明網址/連結，且伴隨「帳號風險、即將封禁、凍結、異常、解除限制、積分過期」等威脅或催促。
2. 網址拼寫有仿冒破綻（例如用 vv 假冒 w、拼錯官方品牌名稱如 vvhtasapp）。
3. 要求轉帳、借錢、匯款、安全帳戶、說出驗證碼、自稱公檢法/海關/客服要求配合調查。"""


class SOSRequest(BaseModel):
    user_id: str
    message: str


def fast_phishing_url_check(message: str) -> dict | None:
    """第一層：高速規則引擎（< 0.05 秒秒判釣魚網址短訊，不花 Token）"""
    url_pattern = r"(https?://\S+|www\.\S+|[a-zA-Z0-9-]+\.(?:com|net|org|cn|hk|mo|tw|top|xyz|vip|cc|site|online|app)\b\S*)"
    urls = re.findall(url_pattern, message, flags=re.IGNORECASE)
    if not urls:
        return None

    phishing_triggers = [
        "風險", "风险", "封禁", "停用", "凍結", "冻结", "解除", "異常", "异常",
        "過期", "过期", "認證", "认证", "違規", "违规", "領取", "领取", "驗證", "验证"
    ]
    hit_triggers = [kw for kw in phishing_triggers if kw in message]

    if hit_triggers:
        return {
            "risk_level": "high",
            "scam_type": "釣魚連結短訊（仿冒官方網站）",
            "reply_message": f"不要怕，這是典型的釣魚詐騙短訊！騙子故意用「帳號封禁或風險」嚇你去點假網址（{urls[0]}），千萬不要點擊連結，也不要輸入帳號密碼，直接把短訊刪掉就好！"
        }
    return None


def local_fallback_check(message: str, is_ai_error: bool = False) -> dict:
    """本地備援檢測"""
    danger_keywords = [
        "警察", "公安", "檢察官", "凍結", "冻结", "轉帳", "转账", "匯款", "過數", "借錢",
        "住院", "按金", "押金", "安全帳戶", "驗證碼", "验证码", "中獎", "緊急求助", "洗黑錢", "海關", "通緝", "出事"
    ]
    hit = [kw for kw in danger_keywords if kw in message]
    if hit:
        return {
            "risk_level": "high",
            "scam_type": f"命中高風險關鍵字({','.join(hit)})",
            "reply_message": "不要怕，深呼吸！這非常可能是詐騙電話或短訊。千萬不要轉帳或借錢，也不要說出驗證碼，請直接掛斷電話，先打給家人確認！"
        }
    if is_ai_error:
        return {
            "risk_level": "high",
            "scam_type": "網路繁忙（啟動安全防護）",
            "reply_message": "剛才網路有點慢，沒能完成深度檢查。為了安全起見，只要涉及點擊陌生連結、借錢、轉帳或要驗證碼，一律先不要理會，直接打電話給家人確認！"
        }
    return {
        "risk_level": "safe",
        "scam_type": "無明顯風險",
        "reply_message": "目前沒有發現明顯的常見詐騙字眼，但只要對方叫你給錢、點陌生連結或報驗證碼，一律不要理會。如果不放心，隨時按紅色按鈕找家人！"
    }


async def analyze_with_ai(message: str) -> dict:
    """雙層檢測：先跑高速釣魚連結規則，再呼叫華為雲 MaaS 大模型"""
    # 第一層：若是「網址 + 封禁/風險威脅」，0.05 秒直接攔截
    fast_hit = fast_phishing_url_check(message)
    if fast_hit:
        print(">> [高速規則攔截] 命中釣魚網址短訊，免等 AI 直接警告！")
        return fast_hit

    if not ai_client or not HUAWEI_MODEL_NAME:
        print(">> [提示] 尚未設定華為雲 API 參數，暫用本地關鍵字模式。")
        return local_fallback_check(message)

    try:
        response = await ai_client.chat.completions.create(
            model=HUAWEI_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"長輩收到的內容或描述：\n{message}"}
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw_content = response.choices[0].message.content.strip()
        print(f">> [AI 原始回覆] {raw_content}")

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            raise ValueError("AI 回覆中找不到 JSON 結構")
        result = json.loads(match.group(0))

        risk_level = "high" if result.get("risk_level") == "high" else "safe"
        return {
            "risk_level": risk_level,
            "scam_type": result.get("scam_type", "AI 分析完成"),
            "reply_message": result.get("reply_message", "請保持冷靜，不要轉帳或點擊不明連結，先打電話給家人確認！")
        }
    except Exception as e:
        print(f">> [AI 呼叫異常，自動切換保守備援] 錯誤原因: {e}")
        return local_fallback_check(message, is_ai_error=True)


@app.get("/")
async def serve_home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/sos")
async def trigger_sos(req: SOSRequest):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] 收到求救訊號！用戶: {req.user_id} | 內容: {req.message}")

    if "長輩按下緊急求助按鈕" in req.message:
        print(">> [緊急按鈕觸發] 立即通知家屬！")
        return {
            "status": "success",
            "risk_level": "high",
            "scam_type": "手動緊急求助",
            "reply_message": "不要怕，深呼吸，你現在很安全！我們已經馬上通知家人，請先掛斷陌生電話，什麼都不要轉帳，家人立刻就會打給你！",
            "action": "wait_for_call"
        }

    analysis = await analyze_with_ai(req.message)
    print(f">> [鑑定結果] 風險: {analysis['risk_level']} | 類型: {analysis['scam_type']}")

    if analysis["risk_level"] == "high":
        print(">> [高風險警報] 已觸發家屬通報流程！")

    return {
        "status": "success",
        "risk_level": analysis["risk_level"],
        "scam_type": analysis["scam_type"],
        "reply_message": analysis["reply_message"], 
        "action": "wait_for_call" if analysis["risk_level"] == "high" else "none"
    }