"""
AI Portfolio Review (P5-2)
Uses Gemini AI to provide intelligent portfolio analysis and rebalancing suggestions.
"""
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_portfolio_review(
    holdings: list,
    price_data: dict,
    total_value: float,
    gemini_client=None,
    model_name: str = 'gemini-3.1-pro-preview',
) -> dict:
    """Generate AI-powered portfolio review.

    Args:
        holdings: List of dicts with {ticker, name, shares, entry_price, current_price, sector}
        price_data: Dict of {ticker: DataFrame} with recent price history
        total_value: Total portfolio value
        gemini_client: Google Gemini client instance
        model_name: Gemini model to use

    Returns:
        Dict with structured review: {summary, sector_analysis, risk_assessment,
        rebalancing_suggestions, market_regime, confidence}
    """
    if not holdings:
        return _empty_review("投資組合為空")

    # --- Compute portfolio metrics ---
    metrics = _compute_portfolio_metrics(holdings, price_data, total_value)

    # --- Build AI prompt ---
    prompt = _build_review_prompt(holdings, metrics, total_value)

    # --- Call Gemini ---
    if gemini_client is None:
        logger.warning("[AI Portfolio] No Gemini client, returning rule-based review")
        return _rule_based_review(holdings, metrics, total_value)

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return _parse_ai_response(response.text, metrics)
    except Exception as e:
        logger.error(f"[AI Portfolio] Gemini API error: {e}")
        return _rule_based_review(holdings, metrics, total_value)


def _compute_portfolio_metrics(holdings, price_data, total_value):
    """Compute quantitative portfolio metrics."""
    metrics = {
        'n_positions': len(holdings),
        'total_value': total_value,
        'sector_weights': {},
        'top_holdings': [],
        'total_pnl_pct': 0,
        'avg_correlation': None,
        'portfolio_beta': None,
    }

    # Sector concentration
    sector_values = {}
    for h in holdings:
        sector = h.get('sector', '未知')
        pos_value = h.get('shares', 0) * h.get('current_price', 0)
        sector_values[sector] = sector_values.get(sector, 0) + pos_value

    if total_value > 0:
        metrics['sector_weights'] = {
            s: v / total_value * 100 for s, v in sector_values.items()
        }

    # Top holdings by weight
    holding_weights = []
    total_pnl = 0
    for h in holdings:
        pos_value = h.get('shares', 0) * h.get('current_price', 0)
        entry_value = h.get('shares', 0) * h.get('entry_price', 0)
        pnl = pos_value - entry_value
        total_pnl += pnl
        weight = pos_value / total_value * 100 if total_value > 0 else 0
        holding_weights.append({
            'ticker': h['ticker'],
            'name': h.get('name', h['ticker']),
            'weight': weight,
            'pnl_pct': (pnl / entry_value * 100) if entry_value > 0 else 0,
        })

    holding_weights.sort(key=lambda x: x['weight'], reverse=True)
    metrics['top_holdings'] = holding_weights[:5]
    metrics['total_pnl_pct'] = (total_pnl / (total_value - total_pnl) * 100) if total_value > total_pnl else 0

    # Correlation (if enough price data)
    if price_data and len(price_data) >= 2:
        try:
            returns_dict = {}
            for ticker, df in price_data.items():
                if df is not None and len(df) >= 30:
                    returns_dict[ticker] = df['Close'].pct_change().dropna()

            if len(returns_dict) >= 2:
                returns_df = pd.DataFrame(returns_dict).dropna()
                if len(returns_df) >= 20:
                    corr_matrix = returns_df.corr()
                    # Average off-diagonal correlation
                    n = len(corr_matrix)
                    if n > 1:
                        mask = ~np.eye(n, dtype=bool)
                        metrics['avg_correlation'] = float(corr_matrix.values[mask].mean())
        except Exception as e:
            logger.warning(f"[AI Portfolio] Correlation calc failed: {e}")

    return metrics


def _build_review_prompt(holdings, metrics, total_value):
    """Build structured prompt for Gemini."""
    holdings_text = "\n".join(
        f"- {h['ticker']} ({h.get('name', '?')}): "
        f"持股 {h.get('shares', 0)} 股, "
        f"進場 {h.get('entry_price', 0):.2f}, "
        f"現價 {h.get('current_price', 0):.2f}, "
        f"產業: {h.get('sector', '未知')}"
        for h in holdings
    )

    sector_text = "\n".join(
        f"- {s}: {w:.1f}%" for s, w in metrics['sector_weights'].items()
    )

    corr_text = f"平均相關性: {metrics['avg_correlation']:.2f}" if metrics['avg_correlation'] is not None else "相關性資料不足"

    return f"""你是一位資深的投資組合經理，擁有 20 年台灣和美國市場經驗。
請用繁體中文分析以下投資組合，並提供具體的建議。

## 投資組合概況
- 總市值: {total_value:,.0f} TWD
- 持股數: {metrics['n_positions']}
- 整體損益: {metrics['total_pnl_pct']:+.1f}%
- {corr_text}

## 持股明細
{holdings_text}

## 產業分布
{sector_text}

## 請以 JSON 格式回覆，包含以下欄位:
{{
    "summary": "一句話整體評價（不超過 50 字）",
    "sector_analysis": "產業集中度分析（是否過於集中？建議如何分散？）",
    "risk_assessment": "風險評估（主要風險來源、系統性風險、個股風險）",
    "rebalancing_suggestions": ["具體的調整建議 1", "建議 2", "建議 3"],
    "market_regime": "當前市場體制判斷（多頭/空頭/盤整）及其對組合的影響",
    "confidence": 75
}}

注意：
- confidence 範圍 0-100，反映你對分析的信心程度
- 建議要具體可執行，不要泛泛而談
- 如果有明顯問題（如過度集中），要直接指出"""


def _parse_ai_response(text, metrics):
    """Parse AI response into structured format."""
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            result['_source'] = 'ai'
            result['_metrics'] = metrics
            return result
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"[AI Portfolio] Failed to parse JSON response: {e}")

    # Fallback: return raw text
    return {
        'summary': text[:200] if text else '分析失敗',
        'sector_analysis': '',
        'risk_assessment': '',
        'rebalancing_suggestions': [],
        'market_regime': '',
        'confidence': 0,
        '_source': 'ai_raw',
        '_metrics': metrics,
    }


def _rule_based_review(holdings, metrics, total_value):
    """Generate rule-based review when AI is unavailable."""
    warnings = []
    suggestions = []

    # Check sector concentration
    for sector, weight in metrics['sector_weights'].items():
        if weight > 40:
            warnings.append(f"⚠️ {sector} 佔比 {weight:.0f}%，建議不超過 40%")
            suggestions.append(f"減少 {sector} 持股至 40% 以下，分散到其他產業")

    # Check single stock concentration
    for h in metrics['top_holdings']:
        if h['weight'] > 20:
            warnings.append(f"⚠️ {h['ticker']} 佔比 {h['weight']:.0f}%，單一持股過高")
            suggestions.append(f"考慮減少 {h['ticker']} 至 15% 以下")

    # Check correlation
    if metrics['avg_correlation'] is not None and metrics['avg_correlation'] > 0.7:
        warnings.append(f"⚠️ 持股平均相關性 {metrics['avg_correlation']:.2f}，分散效果有限")
        suggestions.append("加入低相關性資產（如債券 ETF、不同產業股票）")

    # Check position count
    if metrics['n_positions'] < 5:
        suggestions.append("建議持有 5-15 檔股票以獲得適當分散")
    elif metrics['n_positions'] > 20:
        suggestions.append("持股超過 20 檔，可能難以有效管理，考慮精簡")

    if not suggestions:
        suggestions.append("投資組合分散度良好，持續監控個股表現即可")

    return {
        'summary': f"持有 {metrics['n_positions']} 檔股票，整體損益 {metrics['total_pnl_pct']:+.1f}%",
        'sector_analysis': '\n'.join(warnings) if warnings else '產業分布尚可',
        'risk_assessment': f"{'、'.join(warnings)}" if warnings else '未發現明顯風險集中問題',
        'rebalancing_suggestions': suggestions,
        'market_regime': '需要更多市場數據才能判斷',
        'confidence': 50,
        '_source': 'rule_based',
        '_metrics': metrics,
    }


def _empty_review(reason):
    return {
        'summary': reason,
        'sector_analysis': '',
        'risk_assessment': '',
        'rebalancing_suggestions': [],
        'market_regime': '',
        'confidence': 0,
        '_source': 'empty',
        '_metrics': {},
    }
