"""
Walk-Forward Optimization (WFO) 框架
=====================================
用途: 將 Isaac 策略從靜態回測升級為滾動式 out-of-sample 驗證，
      防止 overfitting，產出真實可交易的績效估計。

設計:
  - Training window: 3 年 (需經歷完整牛熊循環)
  - Test window: 6 個月
  - Roll step: 6 個月
  - 最終績效 = 拼接所有 out-of-sample 區間
"""

import pandas as pd
import numpy as np
import itertools
import logging
import json
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 確保能 import strategies/ 和專案根目錄的模組
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


# ==========================================
# 1. 參數空間定義
# ==========================================

# 固定參數 (業界標準，不優化)
FIXED_PARAMS = {
    'ma_periods': [20, 50, 60, 120, 150, 200],  # MA 週期
}

# 粗調參數 (WFO 搜索，步長大)
COARSE_GRID = {
    'trail_stop':   [0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
}

# 微調參數 (網格搜索 + 穩定性檢驗)
FINE_GRID = {
    'rsi_threshold':     [25, 28, 30],
    'volume_mult':       [1.2, 1.5, 1.8],
    'supply_danger_pct': [0.95, 0.97, 0.99],
    'liq_min':           [300000, 500000],
}

# 預設參數 (V3.4 目前最佳)
DEFAULT_PARAMS = {
    'trail_stop':        0.18,
    'rsi_threshold':     28,
    'volume_mult':       1.5,
    'supply_danger_pct': 0.97,
    'liq_min':           500000,
}


def generate_param_grid(grid_dict):
    """從參數字典生成所有參數組合"""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, c)) for c in combos]


# ==========================================
# 2. 單窗口回測
# ==========================================

def run_single_window(api_token, params, train_start, train_end, test_start, test_end,
                      train_only=False):
    """
    執行單個 WFO 窗口: 在 train 區間用 params 回測，在 test 區間驗證。

    train_only: 若 True，只跑 train（用於 optimize 模式的候選篩選，節省一半時間）
    """
    from isaac import run_isaac_strategy

    result = {
        'params': params,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
    }

    # --- In-Sample (Training) ---
    try:
        train_report = run_isaac_strategy(
            api_token, params=params,
            sim_start=train_start, sim_end=train_end
        )
        train_stats = train_report.get_stats()
        result['train_stats'] = {
            'cagr': train_stats.get('cagr', 0),
            'max_drawdown': train_stats.get('max_drawdown', 0),
            'daily_sharpe': train_stats.get('daily_sharpe', 0),
            'win_ratio': train_stats.get('win_ratio', 0),
        }
    except Exception as e:
        log.error(f"Train window failed ({train_start}~{train_end}): {e}")
        result['train_stats'] = {'cagr': 0, 'max_drawdown': -1, 'daily_sharpe': 0, 'win_ratio': 0}

    if train_only:
        result['test_stats'] = {'cagr': 0, 'max_drawdown': 0, 'daily_sharpe': 0, 'win_ratio': 0}
        result['test_equity'] = None
        result['test_trades'] = []
        return result

    # --- Out-of-Sample (Test) ---
    try:
        test_report = run_isaac_strategy(
            api_token, params=params,
            sim_start=test_start, sim_end=test_end
        )
        test_stats = test_report.get_stats()
        result['test_stats'] = {
            'cagr': test_stats.get('cagr', 0),
            'max_drawdown': test_stats.get('max_drawdown', 0),
            'daily_sharpe': test_stats.get('daily_sharpe', 0),
            'win_ratio': test_stats.get('win_ratio', 0),
        }
        try:
            equity = test_report.portfolio_value
            result['test_equity'] = equity
        except Exception:
            result['test_equity'] = None

        try:
            result['test_trades'] = test_report.get_trades().to_dict('records')
        except Exception:
            result['test_trades'] = []
    except Exception as e:
        log.error(f"Test window failed ({test_start}~{test_end}): {e}")
        result['test_stats'] = {'cagr': 0, 'max_drawdown': -1, 'daily_sharpe': 0, 'win_ratio': 0}
        result['test_equity'] = None
        result['test_trades'] = []

    return result


# ==========================================
# 3. WFO 目標函數
# ==========================================

def objective_score(stats, weights=None):
    """
    計算單一回測結果的綜合評分 (越高越好)。

    公式: w1*CAGR + w2*Sharpe + w3*(1+MaxDD) + w4*WinRatio
    MaxDD 為負數，所以 (1+MaxDD) 越接近 1 越好。

    預設權重偏重風險調整後報酬 (Sharpe) 和回撤控制。
    """
    if weights is None:
        weights = {
            'cagr': 0.25,
            'sharpe': 0.35,
            'maxdd': 0.25,
            'win': 0.15,
        }

    cagr = stats.get('cagr', 0)
    sharpe = stats.get('daily_sharpe', 0)
    maxdd = stats.get('max_drawdown', 0)  # 負數
    win = stats.get('win_ratio', 0)

    score = (
        weights['cagr'] * cagr +
        weights['sharpe'] * sharpe +
        weights['maxdd'] * (1 + maxdd) +  # maxdd=-0.3 → 0.7
        weights['win'] * win
    )
    return score


# ==========================================
# 4. Walk-Forward 主引擎
# ==========================================

def generate_wfo_windows(data_start='2014-01-01', data_end=None,
                         train_years=3, test_months=6, step_months=6):
    """
    生成 WFO 滾動窗口列表。

    Returns:
        list of dict: [{'train_start', 'train_end', 'test_start', 'test_end'}, ...]
    """
    if data_end is None:
        data_end = datetime.now().strftime('%Y-%m-%d')

    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)
    windows = []

    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years) - relativedelta(days=1)
        test_start = train_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end:
            # 最後一個窗口: test 截至資料結束
            test_end = end
            if test_start >= test_end:
                break
            windows.append({
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
            })
            break

        windows.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })

        train_start += relativedelta(months=step_months)

    return windows


def run_wfo(api_token, param_grid=None, mode='fixed',
            train_years=3, test_months=6, step_months=6,
            data_start='2014-01-01', data_end=None,
            save_path=None):
    """
    執行完整的 Walk-Forward Optimization。

    Args:
        api_token:   FinLab API token
        param_grid:  參數組合列表 (list of dict)，None 則用預設
        mode:        'fixed' (固定參數跑所有窗口) / 'optimize' (每窗口選最佳參數)
        train_years: 訓練窗口年數
        test_months: 測試窗口月數
        step_months: 滾動步長月數
        data_start:  資料起始日
        data_end:    資料結束日
        save_path:   結果儲存路徑 (JSON)

    Returns:
        dict: {
            'mode': str,
            'windows': list,         # 每個窗口的完整結果
            'oos_summary': dict,     # 拼接後的 OOS 績效摘要
            'is_vs_oos': dict,       # IS vs OOS 比較 (衡量 overfitting)
            'best_params_per_window': list,  # 每窗口最佳參數 (optimize 模式)
        }
    """
    windows = generate_wfo_windows(
        data_start=data_start, data_end=data_end,
        train_years=train_years, test_months=test_months, step_months=step_months
    )

    log.info(f"WFO 啟動: {len(windows)} 個滾動窗口, mode={mode}")
    for i, w in enumerate(windows):
        log.info(f"  Window {i+1}: Train {w['train_start']}~{w['train_end']} | "
                 f"Test {w['test_start']}~{w['test_end']}")

    if param_grid is None:
        param_grid = [DEFAULT_PARAMS]

    all_results = []
    best_params_per_window = []
    oos_equities = []
    is_stats_list = []
    oos_stats_list = []

    for wi, w in enumerate(windows):
        log.info(f"\n{'='*60}")
        log.info(f"Window {wi+1}/{len(windows)}: "
                 f"Train {w['train_start']}~{w['train_end']} | "
                 f"Test {w['test_start']}~{w['test_end']}")
        log.info(f"{'='*60}")

        if mode == 'optimize':
            # --- 優化模式: 只在 training 上測試所有參數，選最佳 ---
            best_score = -np.inf
            best_params = param_grid[0]

            for pi, params in enumerate(param_grid):
                log.info(f"  Param {pi+1}/{len(param_grid)}: {params}")
                result = run_single_window(
                    api_token, params,
                    w['train_start'], w['train_end'],
                    w['test_start'], w['test_end'],
                    train_only=True  # 只跑 train，節省一半時間
                )
                score = objective_score(result['train_stats'])
                if score > best_score:
                    best_score = score
                    best_params = params

            log.info(f"  >> Best params: {best_params} (IS score: {best_score:.4f})")
            best_params_per_window.append(best_params)

            # 用最佳參數跑完整 train+test
            oos_result = run_single_window(
                api_token, best_params,
                w['train_start'], w['train_end'],
                w['test_start'], w['test_end']
            )
            all_results.append(oos_result)

        else:
            # --- 固定模式: 用預設參數跑所有窗口 ---
            result = run_single_window(
                api_token, param_grid[0],
                w['train_start'], w['train_end'],
                w['test_start'], w['test_end']
            )
            all_results.append(result)
            best_params_per_window.append(param_grid[0])

        # 收集 IS/OOS 統計
        latest = all_results[-1]
        is_stats_list.append(latest['train_stats'])
        oos_stats_list.append(latest['test_stats'])

        if latest.get('test_equity') is not None:
            oos_equities.append(latest['test_equity'])

        # 即時印出結果
        is_s = latest['train_stats']
        oos_s = latest['test_stats']
        log.info(f"  IS:  CAGR={is_s['cagr']*100:+.1f}% | DD={is_s['max_drawdown']*100:.1f}% | "
                 f"Sharpe={is_s['daily_sharpe']:.2f} | Win={is_s['win_ratio']*100:.1f}%")
        log.info(f"  OOS: CAGR={oos_s['cagr']*100:+.1f}% | DD={oos_s['max_drawdown']*100:.1f}% | "
                 f"Sharpe={oos_s['daily_sharpe']:.2f} | Win={oos_s['win_ratio']*100:.1f}%")

    # ==========================================
    # 5. 彙整 OOS 績效
    # ==========================================
    oos_summary = _summarize_oos(oos_stats_list, windows)
    is_summary = _summarize_oos(is_stats_list, windows)
    is_vs_oos = _compare_is_oos(is_summary, oos_summary)

    # 印出最終報告
    log.info(f"\n{'='*60}")
    log.info("WFO 最終報告")
    log.info(f"{'='*60}")
    log.info(f"窗口數: {len(windows)}")
    log.info(f"模式: {mode}")
    log.info(f"\nIn-Sample 平均績效:")
    _print_summary(is_summary)
    log.info(f"\nOut-of-Sample 平均績效:")
    _print_summary(oos_summary)
    log.info(f"\n績效衰退 (IS→OOS):")
    log.info(f"  CAGR 衰退: {is_vs_oos['cagr_decay']*100:.1f}%")
    log.info(f"  Sharpe 衰退: {is_vs_oos['sharpe_decay']:.2f}")
    log.info(f"  Overfitting 指數: {is_vs_oos['overfit_index']:.2f} "
             f"({'⚠️ 可能過擬合' if is_vs_oos['overfit_index'] > 0.5 else '✅ 穩健'})")

    final_result = {
        'mode': mode,
        'windows': _serialize_results(all_results),
        'oos_summary': oos_summary,
        'is_summary': is_summary,
        'is_vs_oos': is_vs_oos,
        'best_params_per_window': best_params_per_window,
    }

    # 存檔
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"\n結果已儲存: {save_path}")

    return final_result


# ==========================================
# 6. 輔助函數
# ==========================================

def _summarize_oos(stats_list, windows):
    """計算 OOS 區間的加權平均績效"""
    if not stats_list:
        return {'avg_cagr': 0, 'avg_sharpe': 0, 'worst_dd': 0, 'avg_win': 0}

    cagrs = [s.get('cagr', 0) for s in stats_list]
    sharpes = [s.get('daily_sharpe', 0) for s in stats_list]
    dds = [s.get('max_drawdown', 0) for s in stats_list]
    wins = [s.get('win_ratio', 0) for s in stats_list]

    return {
        'avg_cagr': np.mean(cagrs),
        'median_cagr': np.median(cagrs),
        'std_cagr': np.std(cagrs),
        'avg_sharpe': np.mean(sharpes),
        'worst_dd': np.min(dds),
        'avg_dd': np.mean(dds),
        'avg_win': np.mean(wins),
        'window_count': len(stats_list),
        'positive_windows': sum(1 for c in cagrs if c > 0),
        'per_window_cagr': cagrs,
    }


def _compare_is_oos(is_summary, oos_summary):
    """比較 IS 和 OOS 績效，計算 overfitting 指數"""
    is_cagr = is_summary.get('avg_cagr', 0)
    oos_cagr = oos_summary.get('avg_cagr', 0)
    is_sharpe = is_summary.get('avg_sharpe', 0)
    oos_sharpe = oos_summary.get('avg_sharpe', 0)

    cagr_decay = is_cagr - oos_cagr
    sharpe_decay = is_sharpe - oos_sharpe

    # Overfitting 指數: (IS-OOS)/IS, 0=完美, 1=完全過擬合
    if is_cagr > 0:
        overfit_index = cagr_decay / is_cagr
    else:
        overfit_index = 0

    return {
        'cagr_decay': cagr_decay,
        'sharpe_decay': sharpe_decay,
        'overfit_index': max(0, min(1, overfit_index)),  # clamp to [0,1]
    }


def _print_summary(summary):
    """印出績效摘要"""
    log.info(f"  平均 CAGR: {summary['avg_cagr']*100:+.1f}% "
             f"(中位數: {summary['median_cagr']*100:+.1f}%, σ={summary['std_cagr']*100:.1f}%)")
    log.info(f"  平均 Sharpe: {summary['avg_sharpe']:.2f}")
    log.info(f"  最差 MaxDD: {summary['worst_dd']*100:.1f}%")
    log.info(f"  平均 Win: {summary['avg_win']*100:.1f}%")
    log.info(f"  正報酬窗口: {summary['positive_windows']}/{summary['window_count']}")


def _serialize_results(results):
    """將結果序列化為可 JSON 儲存的格式"""
    serialized = []
    for r in results:
        s = {k: v for k, v in r.items() if k != 'test_equity'}
        if r.get('test_equity') is not None:
            try:
                eq = r['test_equity']
                s['test_equity_start'] = float(eq.iloc[0]) if len(eq) > 0 else 0
                s['test_equity_end'] = float(eq.iloc[-1]) if len(eq) > 0 else 0
            except Exception:
                pass
        serialized.append(s)
    return serialized


# ==========================================
# 7. 快捷入口
# ==========================================

def run_wfo_fixed(api_token, save_path=None):
    """
    用現有 V3.4 預設參數跑 WFO (最快，用來看 overfitting 程度)。
    這是 Step 1 的入口。
    """
    return run_wfo(
        api_token, mode='fixed',
        param_grid=[DEFAULT_PARAMS],
        save_path=save_path or 'wfo_fixed_result.json'
    )


def run_wfo_coarse(api_token, save_path=None):
    """
    粗調參數 WFO (trail_stop 掃描)。
    這是 Step 2 的入口。
    """
    grid = []
    for ts in COARSE_GRID['trail_stop']:
        p = DEFAULT_PARAMS.copy()
        p['trail_stop'] = ts
        grid.append(p)

    return run_wfo(
        api_token, mode='optimize',
        param_grid=grid,
        save_path=save_path or 'wfo_coarse_result.json'
    )


def run_plateau_test(api_token, save_path=None):
    """
    Step 3: Plateau Test — 穩定性檢驗
    用全期回測跑 trail_stop 細粒度掃描 (0.08~0.26, 步長 0.02)
    驗證最佳參數附近是「高原」而非「尖峰」
    """
    from isaac import run_isaac_strategy

    test_values = [round(x, 2) for x in np.arange(0.08, 0.27, 0.02)]
    log.info(f"\n{'='*60}")
    log.info(f"Plateau Test: trail_stop 細粒度掃描")
    log.info(f"測試值: {test_values}")
    log.info(f"{'='*60}")

    results = []
    for i, ts in enumerate(test_values):
        params = DEFAULT_PARAMS.copy()
        params['trail_stop'] = ts
        log.info(f"\n[{i+1}/{len(test_values)}] trail_stop = {ts}")

        try:
            report = run_isaac_strategy(api_token, params=params)
            stats = report.get_stats()
            r = {
                'trail_stop': ts,
                'cagr': stats.get('cagr', 0),
                'max_drawdown': stats.get('max_drawdown', 0),
                'daily_sharpe': stats.get('daily_sharpe', 0),
                'win_ratio': stats.get('win_ratio', 0),
            }
            r['score'] = objective_score(r)
            log.info(f"  CAGR={r['cagr']*100:+.1f}% | DD={r['max_drawdown']*100:.1f}% | "
                     f"Sharpe={r['daily_sharpe']:.2f} | Win={r['win_ratio']*100:.1f}% | "
                     f"Score={r['score']:.4f}")
        except Exception as e:
            log.error(f"  失敗: {e}")
            r = {'trail_stop': ts, 'cagr': 0, 'max_drawdown': -1,
                 'daily_sharpe': 0, 'win_ratio': 0, 'score': 0}

        results.append(r)

    # --- 分析高原穩定性 ---
    scores = [r['score'] for r in results]
    ts_values = [r['trail_stop'] for r in results]
    best_idx = np.argmax(scores)
    best_ts = ts_values[best_idx]
    best_score = scores[best_idx]

    log.info(f"\n{'='*60}")
    log.info(f"Plateau Test 結果")
    log.info(f"{'='*60}")
    log.info(f"最佳 trail_stop: {best_ts} (score: {best_score:.4f})")

    # 計算高原寬度: score > best_score * 0.9 的連續區間
    threshold = best_score * 0.9
    plateau_values = [ts for ts, s in zip(ts_values, scores) if s >= threshold]
    if plateau_values:
        plateau_width = max(plateau_values) - min(plateau_values)
        log.info(f"高原範圍 (>90% best): {min(plateau_values):.2f} ~ {max(plateau_values):.2f} "
                 f"(寬度: {plateau_width:.2f})")
    else:
        plateau_width = 0
        log.info(f"高原範圍: 無 (尖峰!)")

    # 計算鄰居穩定性
    if best_idx > 0 and best_idx < len(scores) - 1:
        left = scores[best_idx - 1]
        right = scores[best_idx + 1]
        neighbor_avg = (left + right) / 2
        neighbor_ratio = neighbor_avg / best_score if best_score > 0 else 0
        log.info(f"鄰居穩定性: 左={left:.4f} 右={right:.4f} "
                 f"鄰居/最佳={neighbor_ratio:.2f}")
    else:
        neighbor_ratio = 0

    # 判定
    is_plateau = plateau_width >= 0.06 and neighbor_ratio >= 0.85
    verdict = "✅ 高原 (穩健)" if is_plateau else "⚠️ 尖峰 (可能 overfitting)"
    log.info(f"判定: {verdict}")

    # 績效表格
    log.info(f"\n全部結果:")
    log.info(f"{'trail_stop':>12} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'Win':>6} | {'Score':>8}")
    log.info("-" * 65)
    for r in results:
        marker = " ← BEST" if r['trail_stop'] == best_ts else ""
        log.info(f"{r['trail_stop']:>12.2f} | {r['cagr']*100:>+7.1f}% | "
                 f"{r['max_drawdown']*100:>7.1f}% | {r['daily_sharpe']:>7.2f} | "
                 f"{r['win_ratio']*100:>5.1f}% | {r['score']:>8.4f}{marker}")

    # 存檔
    final = {
        'test_type': 'plateau_test',
        'param': 'trail_stop',
        'results': results,
        'best_value': best_ts,
        'best_score': best_score,
        'plateau_width': plateau_width,
        'plateau_range': [min(plateau_values), max(plateau_values)] if plateau_values else [],
        'neighbor_ratio': neighbor_ratio,
        'is_plateau': is_plateau,
        'verdict': verdict,
    }

    path = save_path or 'wfo_plateau_result.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\n結果已儲存: {path}")

    return final


def run_wfo_full(api_token, save_path=None):
    """
    完整參數網格 WFO (粗調 + 微調)。
    注意: 組合數 = 6 * 3 * 3 * 3 * 2 = 324，每窗口跑 324 次。
    """
    all_grid = {**COARSE_GRID, **FINE_GRID}
    grid = generate_param_grid(all_grid)
    log.info(f"完整網格: {len(grid)} 種參數組合")

    return run_wfo(
        api_token, mode='optimize',
        param_grid=grid,
        save_path=save_path or 'wfo_full_result.json'
    )


# ==========================================
# 8. CLI 入口
# ==========================================

if __name__ == '__main__':
    import toml
    import sys

    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                '.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    api_token = secrets.get('FINLAB_API_KEY', '')

    if not api_token:
        print("缺少 FINLAB_API_KEY")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'fixed'

    if mode == 'fixed':
        print("執行 WFO (固定參數模式)...")
        run_wfo_fixed(api_token)
    elif mode == 'coarse':
        print("執行 WFO (粗調參數模式)...")
        run_wfo_coarse(api_token)
    elif mode == 'plateau':
        print("執行 Plateau Test (穩定性檢驗)...")
        run_plateau_test(api_token)
    elif mode == 'full':
        print("執行 WFO (完整參數網格)...")
        run_wfo_full(api_token)
    else:
        print(f"未知模式: {mode}")
        print("用法: python strategies/wfo.py [fixed|coarse|plateau|full]")
