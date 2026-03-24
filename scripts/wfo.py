"""
Walk-Forward Optimization (WFO) 框架
避免回測 Overfitting 的核心工具

===========================================================
WFO 核心概念
===========================================================

傳統回測的 Overfitting 問題:
  - 在同一段歷史資料上反覆調參 → 參數「記住」了歷史的雜訊
  - 回測績效漂亮，但實盤必然衰退

WFO 的解法:
  - 把歷史切成多個「訓練 (IS) + 驗證 (OOS)」窗口對
  - 每個窗口只在 IS 期搜尋最佳參數
  - 將最佳參數「盲目」套用到它從未見過的 OOS 期
  - 拼接所有 OOS 績效 → 真正的 Out-of-Sample 曲線

===========================================================
窗口示意圖 (IS=24月, OOS=12月, 共3個窗口)
===========================================================

  |<--- IS1 24m --->|<OOS1 12m>|
                    |<--- IS2 24m --->|<OOS2 12m>|
                                      |<--- IS3 24m --->|<OOS3 12m>|

  → OOS1 + OOS2 + OOS3 拼接 = 純 OOS 績效曲線

===========================================================
Overfitting 診斷指標
===========================================================

1. OOS/IS CAGR Ratio  (目標 > 0.5，越接近 1 越好)
2. OOS/IS Sharpe Ratio (目標 > 0.5)
3. 參數穩定性 σ       (各窗口最佳參數的標準差，越小越穩健)
4. Efficiency Ratio    = 平均 OOS Score / 平均 IS Score

使用方式:
    cd "C:/Users/Dodo/Documents/AI Invest HQ"
    ./python_embed/python.exe scripts/wfo.py
    ./python_embed/python.exe scripts/wfo.py --is-months 36 --oos-months 12 --notify
    ./python_embed/python.exe scripts/wfo.py --quick   # 快速模式 (小參數網格)
"""

import sys
import os
import json
import argparse
import itertools
import logging
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# 確保可以 import 專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================
# 1. 日誌設定
# ===========================================================

def setup_logging(log_file="wfo_debug.log"):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logging.root.addHandler(fh)
    logging.root.addHandler(ch)
    logging.root.setLevel(logging.INFO)


# ===========================================================
# 2. 窗口生成
# ===========================================================

def generate_wfo_windows(
    start_date: str,
    end_date: str,
    is_months: int = 24,
    oos_months: int = 12,
) -> list[dict]:
    """
    產生 WFO 滾動窗口列表。

    每個窗口包含:
      is_start, is_end   — In-Sample 期 (訓練 / 優化期)
      oos_start, oos_end — Out-of-Sample 期 (驗證期，不接觸)

    窗口以 oos_months 為步長向前滾動，
    直到 oos_end 超過 end_date 為止。
    """
    windows = []
    cursor = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    while True:
        is_start = cursor
        is_end   = cursor + relativedelta(months=is_months) - relativedelta(days=1)
        oos_start = is_end + relativedelta(days=1)
        oos_end   = oos_start + relativedelta(months=oos_months) - relativedelta(days=1)

        if oos_end > end:
            break  # OOS 窗口超出資料範圍，停止

        windows.append({
            "is_start":  is_start.strftime("%Y-%m-%d"),
            "is_end":    is_end.strftime("%Y-%m-%d"),
            "oos_start": oos_start.strftime("%Y-%m-%d"),
            "oos_end":   oos_end.strftime("%Y-%m-%d"),
        })

        cursor += relativedelta(months=oos_months)  # 每次向前推 OOS 長度

    return windows


# ===========================================================
# 3. 評分函數 (IS 優化目標)
# ===========================================================

def score_stats(stats: dict, trades) -> float:
    """
    將回測結果轉換為單一評分，用於 IS 期的參數比較。

    設計原則 (與 CLAUDE.md 評估指標優先順序一致):
      - 風控優先: Max Drawdown 是否 < 30%
      - CAGR 越高越好
      - Sharpe 與 Win Ratio 作為輔助

    若 Max Drawdown < -30% 或交易次數太少，給嚴重懲罰。
    """
    try:
        cagr     = float(stats.get("cagr", 0) or 0)
        max_dd   = float(stats.get("max_drawdown", -1) or -1)
        sharpe   = float(stats.get("daily_sharpe", 0) or 0)
        win_rate = float(stats.get("win_ratio", 0) or 0)
        n_trades = len(trades) if trades is not None else 0

        # 硬性門檻懲罰
        if max_dd < -0.40:          # 超過 40% 大回撤 → 嚴重懲罰
            return -999.0
        if n_trades < 30:           # 交易次數太少 → 統計無意義
            return -999.0

        # 加權評分
        # 優先序: Max DD > CAGR > Sharpe > Win Rate
        score = (
            cagr   * 0.40 +        # 年化報酬 (最重要)
            max_dd * 0.30 +        # 最大回撤 (負值，乘正係數 → 越小越好)
            sharpe * 0.20 +        # 夏普比率
            win_rate * 0.10        # 勝率
        )
        return round(score, 6)

    except Exception as e:
        logging.warning(f"score_stats 發生錯誤: {e}")
        return -999.0


# ===========================================================
# 4. 單窗口網格搜索
# ===========================================================

def grid_search_window(
    api_token: str,
    is_start: str,
    is_end: str,
    param_grid: dict,
) -> tuple[dict, dict, float]:
    """
    在 IS 期對所有參數組合進行網格搜索。

    回傳: (best_params, best_stats, best_score)
    """
    from strategies.isaac import run_isaac_strategy

    # 產生所有參數組合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    logging.info(f"  IS [{is_start} ~ {is_end}]: 共 {len(combinations)} 組參數組合")

    best_score  = -float("inf")
    best_params = combinations[0] if combinations else {}
    best_stats  = {}
    best_trades = None

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        param_str = " | ".join(f"{k}={v}" for k, v in params.items())

        try:
            report = run_isaac_strategy(
                api_token,
                params=params,
                sim_start=is_start,
                sim_end=is_end,
            )
            stats  = report.get_stats()
            trades = report.get_trades()
            sc     = score_stats(stats, trades)

            logging.info(
                f"  [{i:3d}/{len(combinations)}] {param_str} "
                f"→ CAGR={stats.get('cagr', 0):.1%} "
                f"DD={stats.get('max_drawdown', 0):.1%} "
                f"Sharpe={stats.get('daily_sharpe', 0):.2f} "
                f"Score={sc:.4f}"
            )

            if sc > best_score:
                best_score  = sc
                best_params = params
                best_stats  = {k: v for k, v in stats.items()}
                best_trades = trades

        except Exception as e:
            logging.warning(f"  [{i:3d}/{len(combinations)}] {param_str} → 失敗: {e}")

    logging.info(
        f"  IS 最佳參數: {best_params} "
        f"(Score={best_score:.4f}, "
        f"CAGR={best_stats.get('cagr', 0):.1%}, "
        f"DD={best_stats.get('max_drawdown', 0):.1%})"
    )
    return best_params, best_stats, best_score


# ===========================================================
# 5. OOS 評估
# ===========================================================

def evaluate_oos(
    api_token: str,
    oos_start: str,
    oos_end: str,
    params: dict,
) -> tuple[dict, object]:
    """
    用 IS 期找到的最佳參數，在 OOS 期評估。
    這是「盲測」— 策略從未見過這段資料。

    回傳: (oos_stats, oos_trades)
    """
    from strategies.isaac import run_isaac_strategy

    logging.info(f"  OOS [{oos_start} ~ {oos_end}]: 使用參數 {params}")

    report = run_isaac_strategy(
        api_token,
        params=params,
        sim_start=oos_start,
        sim_end=oos_end,
    )
    stats  = report.get_stats()
    trades = report.get_trades()

    logging.info(
        f"  OOS 結果: CAGR={stats.get('cagr', 0):.1%} "
        f"DD={stats.get('max_drawdown', 0):.1%} "
        f"Sharpe={stats.get('daily_sharpe', 0):.2f} "
        f"Trades={len(trades)}"
    )
    return stats, trades


# ===========================================================
# 6. WFO 主函數
# ===========================================================

def walk_forward_optimize(
    api_token: str,
    start_date: str = "2015-01-01",
    end_date: str   = None,
    is_months: int  = 24,
    oos_months: int = 12,
    param_grid: dict = None,
    notify: bool = False,
) -> dict:
    """
    完整 Walk-Forward Optimization 流程。

    步驟:
      1. 產生滾動窗口
      2. 對每個窗口: IS 期網格搜索 → OOS 期盲測
      3. 拼接所有 OOS 結果
      4. 計算 IS/OOS Efficiency Ratio (Overfitting 指標)
      5. 輸出最終報告
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    # 預設參數網格 (依 CLAUDE.md 可調參數表)
    if param_grid is None:
        param_grid = {
            "trail_stop":        [0.12, 0.15, 0.18, 0.20],
            "rsi_threshold":     [25, 28, 32],
            "volume_mult":       [1.3, 1.5, 2.0],
            "supply_danger_pct": [0.95, 0.97],
        }

    logging.info("=" * 70)
    logging.info("Walk-Forward Optimization 啟動")
    logging.info(f"  資料範圍: {start_date} ~ {end_date}")
    logging.info(f"  IS 期長度: {is_months} 個月")
    logging.info(f"  OOS 期長度: {oos_months} 個月")
    logging.info(f"  參數網格: {param_grid}")
    param_count = 1
    for v in param_grid.values():
        param_count *= len(v)
    logging.info(f"  總組合數 (每窗口): {param_count}")
    logging.info("=" * 70)

    # 產生窗口
    windows = generate_wfo_windows(start_date, end_date, is_months, oos_months)
    if not windows:
        logging.error("無法產生任何 WFO 窗口，請確認日期範圍與 IS/OOS 長度設定")
        return {}

    logging.info(f"共產生 {len(windows)} 個 WFO 窗口")
    for i, w in enumerate(windows, 1):
        logging.info(f"  窗口 {i}: IS[{w['is_start']} ~ {w['is_end']}] OOS[{w['oos_start']} ~ {w['oos_end']}]")

    # 逐窗口執行
    window_results = []

    for idx, window in enumerate(windows, 1):
        logging.info("")
        logging.info(f"{'='*60}")
        logging.info(f"窗口 {idx}/{len(windows)}: IS={window['is_start']}~{window['is_end']} | OOS={window['oos_start']}~{window['oos_end']}")
        logging.info(f"{'='*60}")

        # === Step A: IS 期網格搜索 ===
        logging.info(f"[Step A] IS 期網格搜索...")
        best_params, is_stats, is_score = grid_search_window(
            api_token,
            window["is_start"],
            window["is_end"],
            param_grid,
        )

        # === Step B: OOS 期盲測 ===
        logging.info(f"[Step B] OOS 期盲測...")
        try:
            oos_stats, oos_trades = evaluate_oos(
                api_token,
                window["oos_start"],
                window["oos_end"],
                best_params,
            )
            oos_score = score_stats(oos_stats, oos_trades)
        except Exception as e:
            logging.error(f"OOS 期評估失敗: {e}")
            oos_stats  = {}
            oos_trades = None
            oos_score  = -999.0

        result = {
            "window_idx":  idx,
            "is_start":    window["is_start"],
            "is_end":      window["is_end"],
            "oos_start":   window["oos_start"],
            "oos_end":     window["oos_end"],
            "best_params": best_params,
            "is_score":    is_score,
            "oos_score":   oos_score,
            "is_stats":    _extract_key_stats(is_stats),
            "oos_stats":   _extract_key_stats(oos_stats),
            "oos_trades":  len(oos_trades) if oos_trades is not None else 0,
        }
        window_results.append(result)

        # 即時顯示窗口摘要
        _print_window_summary(result)

    # === Step C: 彙整分析 ===
    final_report = _compile_report(window_results, param_grid)

    # 儲存 JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"wfo_result_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    logging.info(f"\nWFO 結果已儲存至: {output_file}")

    # 打印最終報告
    _print_final_report(final_report)

    # Telegram 通知
    if notify:
        _send_wfo_telegram(final_report)

    return final_report


# ===========================================================
# 7. 報告生成輔助函數
# ===========================================================

def _extract_key_stats(stats: dict) -> dict:
    """從回測 stats 中提取關鍵指標"""
    if not stats:
        return {}
    keys = ["cagr", "max_drawdown", "daily_sharpe", "daily_sortino", "win_ratio",
            "total_return", "calmar", "ytd"]
    return {k: round(float(stats[k]), 4) for k in keys if k in stats and stats[k] is not None}


def _print_window_summary(result: dict):
    is_s = result["is_stats"]
    oos_s = result["oos_stats"]

    # 計算 Efficiency Ratio (OOS/IS)
    is_cagr  = is_s.get("cagr", 0) or 0
    oos_cagr = oos_s.get("cagr", 0) or 0
    eff = (oos_cagr / is_cagr) if is_cagr != 0 else float("nan")

    is_sharpe  = is_s.get("daily_sharpe", 0) or 0
    oos_sharpe = oos_s.get("daily_sharpe", 0) or 0
    sharpe_eff = (oos_sharpe / is_sharpe) if is_sharpe != 0 else float("nan")

    logging.info(f"\n  [窗口 {result['window_idx']} 摘要]")
    logging.info(f"  最佳參數: {result['best_params']}")
    logging.info(f"  IS  → CAGR={is_s.get('cagr',0):.1%}  DD={is_s.get('max_drawdown',0):.1%}  Sharpe={is_s.get('daily_sharpe',0):.2f}")
    logging.info(f"  OOS → CAGR={oos_s.get('cagr',0):.1%}  DD={oos_s.get('max_drawdown',0):.1%}  Sharpe={oos_s.get('daily_sharpe',0):.2f}  Trades={result['oos_trades']}")
    logging.info(f"  Efficiency Ratio → CAGR: {eff:.2f}  Sharpe: {sharpe_eff:.2f}  "
                 f"({'良好 ✓' if eff > 0.5 else '可能 Overfit ✗'})")


def _compile_report(window_results: list[dict], param_grid: dict) -> dict:
    """彙整所有窗口結果，計算 Overfitting 診斷指標"""
    import numpy as np

    valid = [r for r in window_results if r["oos_stats"]]

    # OOS 整體統計
    oos_cagrs   = [r["oos_stats"].get("cagr", 0) for r in valid]
    oos_dds     = [r["oos_stats"].get("max_drawdown", 0) for r in valid]
    oos_sharpes = [r["oos_stats"].get("daily_sharpe", 0) for r in valid]
    is_cagrs    = [r["is_stats"].get("cagr", 0) for r in valid]
    is_sharpes  = [r["is_stats"].get("daily_sharpe", 0) for r in valid]

    avg_oos_cagr   = float(np.mean(oos_cagrs))   if oos_cagrs   else 0
    avg_oos_dd     = float(np.mean(oos_dds))     if oos_dds     else 0
    avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
    avg_is_cagr    = float(np.mean(is_cagrs))    if is_cagrs    else 0
    avg_is_sharpe  = float(np.mean(is_sharpes))  if is_sharpes  else 0

    cagr_eff_ratio   = avg_oos_cagr / avg_is_cagr     if avg_is_cagr   != 0 else float("nan")
    sharpe_eff_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else float("nan")

    # 參數穩定性 (各窗口最佳參數的標準差)
    param_stability = {}
    for key in param_grid.keys():
        vals = [r["best_params"].get(key) for r in window_results if r["best_params"].get(key) is not None]
        if vals:
            param_stability[key] = {
                "values": vals,
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "mode": max(set(vals), key=vals.count),  # 最常出現的參數值 (建議部署值)
            }

    # 綜合 Overfitting 診斷
    overfit_risk = _assess_overfit_risk(cagr_eff_ratio, sharpe_eff_ratio, param_stability)

    # 建議部署參數 (各窗口最常選中的值 → 最穩健)
    recommended_params = {k: v["mode"] for k, v in param_stability.items()}

    return {
        "generated_at":       datetime.now().isoformat(),
        "window_count":       len(window_results),
        "valid_windows":      len(valid),
        "param_grid":         param_grid,
        "window_results":     window_results,
        "summary": {
            "avg_oos_cagr":        round(avg_oos_cagr,   4),
            "avg_oos_max_dd":      round(avg_oos_dd,     4),
            "avg_oos_sharpe":      round(avg_oos_sharpe, 4),
            "avg_is_cagr":         round(avg_is_cagr,    4),
            "avg_is_sharpe":       round(avg_is_sharpe,  4),
            "cagr_efficiency_ratio":   round(cagr_eff_ratio,   3),
            "sharpe_efficiency_ratio": round(sharpe_eff_ratio, 3),
        },
        "param_stability":        param_stability,
        "recommended_params":     recommended_params,
        "overfit_assessment":     overfit_risk,
    }


def _assess_overfit_risk(cagr_eff: float, sharpe_eff: float, param_stability: dict) -> dict:
    """
    依據 Efficiency Ratio 和參數穩定性評估 Overfitting 風險。

    判斷標準:
      - Efficiency Ratio > 0.7 → 低風險
      - Efficiency Ratio 0.4~0.7 → 中等風險
      - Efficiency Ratio < 0.4 → 高風險 (明顯 Overfit)

    參數穩定性:
      - std/mean < 0.2 → 穩定 (robust)
      - std/mean > 0.5 → 不穩定 (可能過度擬合特定噪音)
    """
    import math

    risk_factors = []
    risk_score = 0  # 0=無風險, 1=低, 2=中, 3=高

    # CAGR Efficiency Ratio
    if math.isnan(cagr_eff):
        risk_factors.append("CAGR Efficiency Ratio 無法計算")
        risk_score = max(risk_score, 2)
    elif cagr_eff < 0:
        risk_factors.append(f"CAGR Efficiency Ratio 為負 ({cagr_eff:.2f}) — OOS 表現比 IS 差很多")
        risk_score = max(risk_score, 3)
    elif cagr_eff < 0.4:
        risk_factors.append(f"CAGR Efficiency Ratio 過低 ({cagr_eff:.2f} < 0.4) — 明顯 Overfitting")
        risk_score = max(risk_score, 3)
    elif cagr_eff < 0.7:
        risk_factors.append(f"CAGR Efficiency Ratio 中等 ({cagr_eff:.2f}) — 輕度 Overfitting")
        risk_score = max(risk_score, 2)
    else:
        risk_factors.append(f"CAGR Efficiency Ratio 良好 ({cagr_eff:.2f} ≥ 0.7) ✓")

    # Sharpe Efficiency Ratio
    if not math.isnan(sharpe_eff) and sharpe_eff < 0.4:
        risk_factors.append(f"Sharpe Efficiency Ratio 過低 ({sharpe_eff:.2f} < 0.4)")
        risk_score = max(risk_score, 3)
    elif not math.isnan(sharpe_eff) and sharpe_eff >= 0.7:
        risk_factors.append(f"Sharpe Efficiency Ratio 良好 ({sharpe_eff:.2f} ≥ 0.7) ✓")

    # 參數穩定性
    for key, info in param_stability.items():
        mean_val = info["mean"]
        std_val  = info["std"]
        if mean_val != 0:
            cv = std_val / abs(mean_val)  # Coefficient of Variation
            if cv > 0.5:
                risk_factors.append(f"參數 '{key}' 不穩定 (CV={cv:.2f} > 0.5) — 各窗口最佳值差異大")
                risk_score = max(risk_score, 2)
            elif cv < 0.2:
                risk_factors.append(f"參數 '{key}' 穩定 (CV={cv:.2f} < 0.2) ✓")

    risk_labels = {0: "極低", 1: "低", 2: "中等", 3: "高"}
    return {
        "risk_level":  risk_labels.get(risk_score, "未知"),
        "risk_score":  risk_score,
        "risk_factors": risk_factors,
        "interpretation": (
            "策略參數穩健，OOS 績效接近 IS 期，可信度高。" if risk_score <= 1 else
            "策略存在輕度 Overfitting，建議以 WFO 推薦參數部署，並持續監控。" if risk_score == 2 else
            "策略存在明顯 Overfitting！OOS 績效大幅遜於 IS 期，不建議以當前參數實盤。"
        ),
    }


def _print_final_report(report: dict):
    s = report.get("summary", {})
    overfit = report.get("overfit_assessment", {})
    rec = report.get("recommended_params", {})
    stability = report.get("param_stability", {})

    logging.info("\n" + "=" * 70)
    logging.info("Walk-Forward Optimization 最終報告")
    logging.info("=" * 70)

    logging.info(f"\n[OOS 綜合績效 (Out-of-Sample，未見過的資料)]")
    logging.info(f"  平均 CAGR:      {s.get('avg_oos_cagr', 0):.1%}")
    logging.info(f"  平均 Max DD:    {s.get('avg_oos_max_dd', 0):.1%}")
    logging.info(f"  平均 Sharpe:    {s.get('avg_oos_sharpe', 0):.2f}")

    logging.info(f"\n[IS 綜合績效 (In-Sample，訓練期參考)]")
    logging.info(f"  平均 CAGR:      {s.get('avg_is_cagr', 0):.1%}")
    logging.info(f"  平均 Sharpe:    {s.get('avg_is_sharpe', 0):.2f}")

    logging.info(f"\n[Overfitting 診斷]")
    logging.info(f"  CAGR Efficiency Ratio:   {s.get('cagr_efficiency_ratio', 0):.2f}  (OOS/IS, 目標>0.7)")
    logging.info(f"  Sharpe Efficiency Ratio: {s.get('sharpe_efficiency_ratio', 0):.2f}  (OOS/IS, 目標>0.7)")
    logging.info(f"  Overfitting 風險:         {overfit.get('risk_level', '?')}")
    for factor in overfit.get("risk_factors", []):
        logging.info(f"    → {factor}")
    logging.info(f"  結論: {overfit.get('interpretation', '')}")

    logging.info(f"\n[參數穩定性分析]")
    for key, info in stability.items():
        logging.info(
            f"  {key}: mean={info['mean']}  std={info['std']}  "
            f"各窗口值={info['values']}  建議值={info['mode']}"
        )

    logging.info(f"\n[建議部署參數 (各窗口最常被選中的值)]")
    for k, v in rec.items():
        logging.info(f"  {k}: {v}")

    logging.info("\n" + "=" * 70)


def _send_wfo_telegram(report: dict):
    """透過 Telegram 發送 WFO 摘要報告"""
    try:
        import toml
        from utils.notify import send_telegram

        secrets = toml.load(".streamlit/secrets.toml")
        s = report.get("summary", {})
        overfit = report.get("overfit_assessment", {})
        rec = report.get("recommended_params", {})

        msg = (
            f"📊 *Walk-Forward Optimization 完成*\n\n"
            f"*OOS 平均績效 (未見過的資料):*\n"
            f"  CAGR: {s.get('avg_oos_cagr', 0):.1%}\n"
            f"  Max DD: {s.get('avg_oos_max_dd', 0):.1%}\n"
            f"  Sharpe: {s.get('avg_oos_sharpe', 0):.2f}\n\n"
            f"*Overfitting 診斷:*\n"
            f"  CAGR Efficiency: {s.get('cagr_efficiency_ratio', 0):.2f}\n"
            f"  Sharpe Efficiency: {s.get('sharpe_efficiency_ratio', 0):.2f}\n"
            f"  風險等級: {overfit.get('risk_level', '?')}\n\n"
            f"*建議部署參數:*\n"
            + "\n".join(f"  {k}={v}" for k, v in rec.items())
        )

        send_telegram(msg)
        logging.info("[WFO] Telegram 通知已發送")
    except Exception as e:
        logging.warning(f"[WFO] Telegram 通知失敗: {e}")


# ===========================================================
# 8. 快速模式 (測試用，小參數網格)
# ===========================================================

QUICK_PARAM_GRID = {
    "trail_stop":    [0.15, 0.18],
    "rsi_threshold": [28, 32],
    "volume_mult":   [1.5, 2.0],
}

FULL_PARAM_GRID = {
    "trail_stop":        [0.12, 0.15, 0.18, 0.20],
    "rsi_threshold":     [25, 28, 32],
    "volume_mult":       [1.3, 1.5, 2.0],
    "supply_danger_pct": [0.95, 0.97],
}


# ===========================================================
# 9. CLI 入口
# ===========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization — 避免回測 Overfitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  ./python_embed/python.exe scripts/wfo.py
  ./python_embed/python.exe scripts/wfo.py --is-months 36 --oos-months 12
  ./python_embed/python.exe scripts/wfo.py --quick   # 小參數網格，快速驗證
  ./python_embed/python.exe scripts/wfo.py --notify  # 完成後 Telegram 通知
        """,
    )
    parser.add_argument("--start",      default="2015-01-01", help="回測起始日 (YYYY-MM-DD)")
    parser.add_argument("--end",        default=None,         help="回測結束日 (YYYY-MM-DD，預設今天)")
    parser.add_argument("--is-months",  type=int, default=24, help="IS 期長度 (月，預設 24)")
    parser.add_argument("--oos-months", type=int, default=12, help="OOS 期長度 (月，預設 12)")
    parser.add_argument("--quick",      action="store_true",  help="快速模式 (小參數網格)")
    parser.add_argument("--notify",     action="store_true",  help="完成後 Telegram 通知")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    setup_logging()

    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    api_token = secrets.get("FINLAB_API_KEY", "")
    if not api_token:
        print("ERROR: FINLAB_API_KEY not found in .streamlit/secrets.toml")
        sys.exit(1)

    param_grid = QUICK_PARAM_GRID if args.quick else FULL_PARAM_GRID
    n_combo = 1
    for v in param_grid.values():
        n_combo *= len(v)

    windows = generate_wfo_windows(args.start, args.end or date.today().strftime("%Y-%m-%d"),
                                   args.is_months, args.oos_months)
    total_runs = n_combo * len(windows) + len(windows)  # IS grid + OOS evals

    print(f"\n{'='*60}")
    print(f"WFO 設定摘要")
    print(f"{'='*60}")
    print(f"  IS 期: {args.is_months} 個月 | OOS 期: {args.oos_months} 個月")
    print(f"  窗口數: {len(windows)}")
    print(f"  參數組合數: {n_combo} ({'快速模式' if args.quick else '完整模式'})")
    print(f"  預計回測次數: {total_runs} 次")
    print(f"  (每次回測約 1-3 分鐘，預計總耗時 {total_runs * 2 // 60}-{total_runs * 3 // 60} 分鐘)")
    print(f"{'='*60}\n")

    walk_forward_optimize(
        api_token=api_token,
        start_date=args.start,
        end_date=args.end,
        is_months=args.is_months,
        oos_months=args.oos_months,
        param_grid=param_grid,
        notify=args.notify,
    )
