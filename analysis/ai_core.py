import re
import logging
import requests

from utils.helpers import robust_json_extract

logger = logging.getLogger(__name__)


def _sanitize_prompt_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input before embedding into LLM prompts.

    - Strips instruction-like patterns (system/assistant role injections)
    - Removes backtick fences and curly-brace instruction blocks
    - Truncates to max_length
    """
    original = text
    text = str(text).strip()
    # Remove instruction injection attempts
    text = re.sub(r'(?i)(system\s*:|assistant\s*:|<<\s*SYS\s*>>|<\|im_start\|>|<\|im_end\|>)', '', text)
    # Remove backtick code fences that could wrap injected instructions
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove curly-brace blocks that look like template injections
    text = re.sub(r'\{[^}]{20,}\}', '', text)
    text = text[:max_length]
    if text != original:
        logger.warning(f"Prompt input sanitized: original length={len(original)}, sanitized length={len(text)}")
    return text


def resolve_ticker_and_market(query, client=None, gemini_model=None):
    query = _sanitize_prompt_input(str(query).strip(), max_length=100)
    if re.match(r'^\d{4,6}$', query):
        return query, "🇹🇼 台股 (TW)", query
    if re.match(r'^[A-Z]{1,5}$', query.upper()):
        return query.upper(), "🗽 美股 (US)", query.upper()

    # 先嘗試使用開源台股 API 搜尋中文名稱 (TWSE / TPEx)
    try:
        res_twse = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=3)
        if res_twse.status_code == 200:
            data = res_twse.json()
            for item in data:
                if query in item.get('Name', '') or query == item.get('Code', ''):
                    return item['Code'], "🇹🇼 台股 (TW)", item['Name']
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.debug(f"TWSE API 查詢失敗: {e}")

    try:
        res_tpex = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", timeout=3)
        if res_tpex.status_code == 200:
            data = res_tpex.json()
            for item in data:
                if query in item.get('CompanyName', '') or query == item.get('SecuritiesCompanyCode', ''):
                    return item['SecuritiesCompanyCode'], "🇹🇼 台股 (TW)", item['CompanyName']
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.debug(f"TPEx API 查詢失敗: {e}")

    if not client:
        return None, None, None
    prompt = f"將'{query}'轉為股票代碼。回傳JSON:{{'market':'TW'或'US', 'ticker':'代碼', 'name':'中文名'}}。台股代碼僅數字。"
    try:
        res = client.models.generate_content(model=gemini_model, contents=prompt)
        data = robust_json_extract(res.text)
        if data and 'market' in data:
            return data['ticker'], "🇹🇼 台股 (TW)" if data['market'] == "TW" else "🗽 美股 (US)", data.get('name', query)
        return None, None, None
    except Exception as e:
        logger.error(f"AI 翻譯失敗: {e}")
        return None, None, None


def analyze_signals(df):
    if df.empty or len(df) < 30:
        return "資料不足"
    signals = []

    price_slope = (df['Close'].iloc[-1] - df['Close'].iloc[-10])
    rsi_slope = (df['RSI'].iloc[-1] - df['RSI'].iloc[-10])
    if price_slope > 0 and rsi_slope < 0:
        signals.append("⚠️ 頂背離")
    elif price_slope < 0 and rsi_slope > 0:
        signals.append("✨ 底背離")

    k, d = df['K'].iloc[-1], df['D'].iloc[-1]
    prev_k = df['K'].iloc[-2]
    if prev_k < df['D'].iloc[-2] and k > d and k < 80:
        signals.append("⚡ KD 金叉")
    elif prev_k > df['D'].iloc[-2] and k < d and k > 20:
        signals.append("💀 KD 死叉")

    macd = df['MACD'].iloc[-1]
    if df['MACD'].iloc[-2] < 0 and macd > 0:
        signals.append("🔥 MACD 翻紅")
    elif df['MACD'].iloc[-2] > 0 and macd < 0:
        signals.append("❄️ MACD 翻綠")

    return " | ".join(signals) if signals else "無明顯訊號"


def detect_hot_themes(market, client=None, gemini_model=None):
    if not client:
        return []
    market = _sanitize_prompt_input(str(market), max_length=100)
    q = "今日台股熱門族群" if "台股" in market else "Top US sectors today"
    prompt = f"搜'{q}'，歸納3~5個主題，回傳List JSON (純文字列表)。"
    try:
        res = client.models.generate_content(model=gemini_model, contents=prompt)
        return robust_json_extract(res.text) or []
    except Exception as e:
        logger.warning(f"AI 熱門族群偵測失敗: {e}")
        return []


def generate_supply_chain_structure(market, keyword, client=None, gemini_model=None):
    if not client:
        return None
    keyword = _sanitize_prompt_input(str(keyword), max_length=100)
    prompt = f"拆解'{keyword}'產業鏈，回傳JSON: {{'部位': {{'代碼': '中文名'}}}}"
    try:
        res = client.models.generate_content(model=gemini_model, contents=prompt)
        return robust_json_extract(res.text)
    except Exception as e:
        logger.warning(f"AI 產業鏈生成失敗: {e}")
        return None


def generate_ai_analysis(market, ticker, name, price, change, sector, technicals, strategy,
                         extra_data="", timeframe="1d", signal_context="",
                         client=None, gemini_model=None):
    if not client:
        return "請先輸入 API Key。"
    # Sanitize user-controllable inputs
    ticker = _sanitize_prompt_input(str(ticker), max_length=100)
    name = _sanitize_prompt_input(str(name), max_length=100)
    signal_context = _sanitize_prompt_input(str(signal_context), max_length=500)
    extra_data = _sanitize_prompt_input(str(extra_data), max_length=500)
    desc = "週線(Weekly)" if timeframe == "1wk" else "日線(Daily)"
    prompt = f"""
    角色：全方位技術分析大師。標的：{market} {ticker} {name}。
    分析週期：{desc}。數據：{price} ({change}%) | {technicals}
    **重點訊號：{signal_context}**
    {extra_data}

    請進行分析 (Markdown)：
    1. 🔍 訊號判讀
    2. 📐 形態與趨勢
    3. 🛡️ 實戰指令 ({strategy})
    """
    try:
        res = client.models.generate_content(model=gemini_model, contents=prompt)
        return res.text
    except Exception as e:
        return f"分析失敗: {e}"
