"""
AI Strategy Description (P5-3)
Generate natural language descriptions of backtested strategies using Gemini.
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


def generate_strategy_description(
    strategy_name: str,
    stats: dict,
    trades_summary: dict,
    yearly_breakdown: list = None,
    gemini_client=None,
    model_name: str = 'gemini-3.1-pro-preview',
) -> dict:
    """Generate a plain-language description of a backtested strategy.

    Args:
        strategy_name: Name of the strategy (e.g., "Isaac V3.7")
        stats: Backtest statistics dict (cagr, max_drawdown, sharpe, win_ratio, etc.)
        trades_summary: Summary of trades {total, avg_return, avg_period, best, worst}
        yearly_breakdown: List of {year, return_pct, trades, max_dd}
        gemini_client: Google Gemini client
        model_name: Model to use

    Returns:
        Dict with {overview, strengths, weaknesses, regime_analysis, recommendation}
    """
    prompt = _build_prompt(strategy_name, stats, trades_summary, yearly_breakdown)

    if gemini_client is None:
        logger.warning("[AI Strategy] No Gemini client, returning rule-based description")
        return _rule_based_description(strategy_name, stats, trades_summary)

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return _parse_response(response.text)
    except Exception as e:
        logger.error(f"[AI Strategy] Gemini error: {e}")
        return _rule_based_description(strategy_name, stats, trades_summary)


def _build_prompt(strategy_name, stats, trades_summary, yearly_breakdown):
    """Build Gemini prompt."""
    yearly_text = ""
    if yearly_breakdown:
        yearly_text = "\n".join(
            f"  - {y.get('year', '?')}: 報酬 {y.get('return_pct', 0):+.1f}%, "
            f"交易 {y.get('trades', 0)} 筆, MDD {y.get('max_dd', 0):.1f}%"
            for y in yearly_breakdown
        )
    else:
        yearly_text = "  (年度資料不可用)"

    return f"""你是一位資深量化策略分析師。請用繁體中文，為以下回測結果撰寫一份白話文的策略說明。
目標讀者是「剛接觸量化交易的新手投資人」，所以請用淺顯易懂的語言。

## 策略名稱: {strategy_name}

## 回測統計
- 年化報酬率 (CAGR): {stats.get('cagr', 0)*100:.2f}%
- 最大回撤 (MDD): {stats.get('max_drawdown', 0)*100:.2f}%
- 夏普比率: {stats.get('daily_sharpe', 0):.2f}
- 勝率: {stats.get('win_ratio', 0)*100:.1f}%
- 總交易筆數: {trades_summary.get('total', 0)}
- 平均持有天數: {trades_summary.get('avg_period', 0):.1f}
- 最佳單筆: {trades_summary.get('best', 0)*100:+.1f}%
- 最差單筆: {trades_summary.get('worst', 0)*100:+.1f}%

## 年度表現
{yearly_text}

## 請以 JSON 格式回覆:
{{
    "overview": "策略整體描述，用 2-3 句話說明這個策略在做什麼、適合什麼樣的市場（100-150字）",
    "strengths": ["優勢 1", "優勢 2", "優勢 3"],
    "weaknesses": ["弱點或風險 1", "弱點或風險 2"],
    "regime_analysis": "分析策略在不同市場環境（多頭/空頭/盤整）下的表現差異",
    "recommendation": "一句話結論：這個策略適合誰？應該如何使用？"
}}"""


def _parse_response(text):
    """Parse AI response."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            result['_source'] = 'ai'
            return result
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"[AI Strategy] JSON parse failed: {e}")

    return {
        'overview': text[:300] if text else '描述生成失敗',
        'strengths': [],
        'weaknesses': [],
        'regime_analysis': '',
        'recommendation': '',
        '_source': 'ai_raw',
    }


def _rule_based_description(strategy_name, stats, trades_summary):
    """Generate description based on rules when AI is unavailable."""
    cagr = stats.get('cagr', 0) * 100
    mdd = abs(stats.get('max_drawdown', 0) * 100)
    sharpe = stats.get('daily_sharpe', 0)
    win_rate = stats.get('win_ratio', 0) * 100
    avg_period = trades_summary.get('avg_period', 0)

    # Overview
    if cagr > 20:
        perf_desc = "高報酬"
    elif cagr > 10:
        perf_desc = "中等報酬"
    else:
        perf_desc = "穩健型"

    if avg_period < 10:
        style = "短線交易"
    elif avg_period < 30:
        style = "波段交易"
    else:
        style = "中長期持有"

    overview = (
        f"{strategy_name} 是一個{perf_desc}的{style}策略，"
        f"年化報酬率 {cagr:.1f}%，最大回撤 {mdd:.1f}%。"
        f"平均每筆交易持有 {avg_period:.0f} 天，勝率 {win_rate:.0f}%。"
    )

    # Strengths
    strengths = []
    if cagr > 15:
        strengths.append(f"年化報酬率 {cagr:.1f}% 優於大盤平均")
    if sharpe > 1.0:
        strengths.append(f"夏普比率 {sharpe:.2f}，風險調整後報酬良好")
    if win_rate > 50:
        strengths.append(f"勝率 {win_rate:.0f}%，超過半數交易獲利")
    if mdd < 25:
        strengths.append(f"最大回撤 {mdd:.1f}%，風控表現良好")
    if not strengths:
        strengths.append("策略具有系統性規則，避免情緒化交易")

    # Weaknesses
    weaknesses = []
    if mdd > 30:
        weaknesses.append(f"最大回撤 {mdd:.1f}% 偏高，需要較強的心理承受能力")
    if win_rate < 45:
        weaknesses.append(f"勝率僅 {win_rate:.0f}%，需要良好的風報比才能獲利")
    if trades_summary.get('total', 0) < 100:
        weaknesses.append("交易樣本數較少，統計可靠度有待提升")
    if not weaknesses:
        weaknesses.append("回測結果不代表未來表現，實際交易可能因滑價和手續費而打折")

    # Recommendation
    if cagr > 15 and mdd < 30:
        rec = f"適合有一定經驗的投資人，建議搭配嚴格的部位管理使用。"
    elif cagr > 10:
        rec = f"適合穩健型投資人，可作為核心策略的一部分。"
    else:
        rec = f"建議進一步優化參數或搭配其他策略使用。"

    return {
        'overview': overview,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'regime_analysis': '需要 AI 分析功能（Gemini API）來提供不同市場環境下的表現分析。',
        'recommendation': rec,
        '_source': 'rule_based',
    }
