"""
回測比較腳本 — ATR 部位規模 vs 等權重 / 滑價+流動性影響
執行: python scripts/backtest_comparison.py

比較三種配置:
  A) 原始 Isaac V3.7 (等權重, position_limit=1/10)
  B) ATR 正規化部位 (波動率越高，分配越少)
  C) 原始 + 滑價 0.5% + 流動性門檻 100萬股
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import toml
import numpy as np
import pandas as pd

secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_API_KEY', secrets.get('FINLAB_API_TOKEN', ''))

if not API_TOKEN:
    print("ERROR: 找不到 FinLab API Token")
    sys.exit(1)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'backtest_comparison.json')


def run_baseline():
    """A) 原始 Isaac V3.7"""
    print("\n{'='*60}")
    print("A) 原始 Isaac V3.7 (等權重)")
    print("{'='*60}")
    from strategies.isaac import run_isaac_strategy
    import importlib
    import strategies.isaac as mod
    importlib.reload(mod)
    report = mod.run_isaac_strategy(API_TOKEN)
    return report


def run_atr_sizing():
    """B) ATR 正規化部位規模"""
    print("\n{'='*60}")
    print("B) ATR 正規化部位規模")
    print("{'='*60}")
    import importlib
    import strategies.isaac as mod
    importlib.reload(mod)

    # 用 raw_mode 取得原始部位和價格數據
    raw = mod.run_isaac_strategy(API_TOKEN, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']

    # 計算 ATR (14日)
    from finlab import data
    high = data.get('price:最高價')
    low = data.get('price:最低價')

    # True Range
    prev_close = close.shift(1)
    tr1 = high.reindex_like(close) - low.reindex_like(close)
    tr2 = (high.reindex_like(close) - prev_close).abs()
    tr3 = (low.reindex_like(close) - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 因為 true_range 變成 1D series，需要用不同方式
    # 改為逐股計算 ATR
    print("  計算 ATR-14...")
    atr_14 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        h = high[col] if col in high.columns else close[col]
        l = low[col] if col in low.columns else close[col]
        c = close[col]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr_14[col] = tr.rolling(14).mean()

    # ATR 正規化: position_weight = target_risk / (ATR / price)
    # target_risk = 等權重時的平均波動 (用中位數 ATR% 作為基準)
    atr_pct = (atr_14 / close).replace([np.inf, -np.inf], np.nan)

    # 中位數 ATR% 作為基準
    median_atr_pct = atr_pct.median(axis=1)  # 每天的跨股票中位數

    # 正規化權重: median_atr / stock_atr (波動大的分配少)
    atr_weight = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        stock_atr_pct = atr_pct[col]
        # 避免除零
        safe_atr = stock_atr_pct.replace(0, np.nan)
        atr_weight[col] = (median_atr_pct / safe_atr).clip(0.3, 2.0)  # 限制 0.3x ~ 2x

    atr_weight = atr_weight.fillna(1.0)

    # 套用 ATR 權重到部位
    atr_pos = final_pos.copy()
    active_mask = atr_pos > 0
    atr_pos[active_mask] = atr_pos[active_mask] * atr_weight.reindex_like(atr_pos).fillna(1.0)[active_mask]

    # 正規化使每天總部位權重 ≈ 原始總權重
    daily_orig_sum = final_pos[final_pos > 0].sum(axis=1)
    daily_atr_sum = atr_pos[atr_pos > 0].sum(axis=1)
    scale = (daily_orig_sum / daily_atr_sum.replace(0, 1)).clip(0.5, 2.0)
    atr_pos[active_mask] = atr_pos[active_mask].mul(scale, axis=0)[active_mask]

    atr_pos = atr_pos.replace([np.inf, -np.inf], 0).fillna(0)

    print(f"  ATR 部位調整完成: {(atr_pos > 0).any(axis=1).sum()} 天有持倉")

    # 執行回測
    from data.provider import safe_finlab_sim
    report = safe_finlab_sim(
        atr_pos,
        name='Isaac V3.7 + ATR Sizing',
        upload=False,
        trail_stop=raw['trail_stop'],
        position_limit=1.0 / raw['max_concurrent'],
        touched_exit=False,
    )
    return report


def run_with_slippage_and_liquidity():
    """C) 滑價 + 流動性過濾"""
    print("\n{'='*60}")
    print("C) 原始 + 滑價 0.5% + 流動性 100萬股")
    print("{'='*60}")
    import importlib
    import strategies.isaac as mod
    importlib.reload(mod)

    # 提高流動性門檻
    report = mod.run_isaac_strategy(
        API_TOKEN,
        params={'liq_min': 1_000_000},  # 100萬股 (原始 50萬)
    )

    # 滑價影響用 stats 調整估算 (FinLab 不直接支援滑價參數)
    # 用交易次數 × 0.5% 估算
    return report


def extract_stats(report, label):
    """從回測報告提取關鍵指標"""
    try:
        stats = report.get_stats()
        trades = report.get_trades()

        # 基礎指標
        cagr = stats.get('cagr', 0)
        mdd = stats.get('max_drawdown', 0)
        sharpe = stats.get('sharpe', 0)
        win_ratio = stats.get('win_ratio', 0)
        n_trades = len(trades)

        # 風報比
        if not trades.empty:
            avg_win = trades[trades['return'] > 0]['return'].mean() if len(trades[trades['return'] > 0]) > 0 else 0
            avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if len(trades[trades['return'] <= 0]) > 0 else 0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
            avg_hold = trades['period'].mean()
            max_win = trades['return'].max()
            max_loss = trades['return'].min()
        else:
            avg_win = avg_loss = risk_reward = avg_hold = max_win = max_loss = 0

        # 交易成本估算
        slippage_per_trade = 0.005  # 0.5% round-trip
        commission_per_trade = 0.001425 * 2  # 買賣各 0.1425%
        tax_per_trade = 0.003  # 證交稅
        total_cost_per_trade = slippage_per_trade + commission_per_trade + tax_per_trade
        total_cost_drag = n_trades * total_cost_per_trade

        result = {
            'label': label,
            'cagr': round(cagr * 100, 2),
            'mdd': round(mdd * 100, 2),
            'sharpe': round(sharpe, 3),
            'win_ratio': round(win_ratio * 100, 1),
            'risk_reward': round(risk_reward, 2),
            'n_trades': n_trades,
            'avg_hold_days': round(avg_hold, 1),
            'max_win': round(max_win * 100, 2),
            'max_loss': round(max_loss * 100, 2),
            'estimated_cost_drag_pct': round(total_cost_drag * 100, 2),
            'net_cagr_est': round((cagr - total_cost_drag / 5) * 100, 2),  # 假設 5 年
        }

        print(f"\n  [{label}] 回測結果:")
        print(f"  CAGR:      {result['cagr']:>8.2f}%")
        print(f"  MDD:       {result['mdd']:>8.2f}%")
        print(f"  Sharpe:    {result['sharpe']:>8.3f}")
        print(f"  勝率:      {result['win_ratio']:>8.1f}%")
        print(f"  風報比:    {result['risk_reward']:>8.2f}")
        print(f"  交易次數:  {result['n_trades']:>8d}")
        print(f"  平均持有:  {result['avg_hold_days']:>8.1f} 天")
        print(f"  最大獲利:  {result['max_win']:>8.2f}%")
        print(f"  最大虧損:  {result['max_loss']:>8.2f}%")
        print(f"  成本拖累:  {result['estimated_cost_drag_pct']:>8.2f}%")

        return result
    except Exception as e:
        print(f"  ERROR extracting stats: {e}")
        import traceback
        traceback.print_exc()
        return {'label': label, 'error': str(e)}


def main():
    print("=" * 60)
    print("  Isaac V3.7 回測比較")
    print("  A) 原始等權重  B) ATR正規化  C) 滑價+流動性")
    print("=" * 60)

    results = []

    # A) 原始
    t0 = time.time()
    try:
        report_a = run_baseline()
        stats_a = extract_stats(report_a, "A) 原始 Isaac V3.7")
        results.append(stats_a)
    except Exception as e:
        print(f"  A) 回測失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append({'label': 'A) 原始 Isaac V3.7', 'error': str(e)})
    print(f"  耗時: {time.time() - t0:.1f}s")

    # B) ATR 正規化
    t0 = time.time()
    try:
        report_b = run_atr_sizing()
        stats_b = extract_stats(report_b, "B) ATR 正規化部位")
        results.append(stats_b)
    except Exception as e:
        print(f"  B) 回測失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append({'label': 'B) ATR 正規化部位', 'error': str(e)})
    print(f"  耗時: {time.time() - t0:.1f}s")

    # C) 滑價 + 流動性
    t0 = time.time()
    try:
        report_c = run_with_slippage_and_liquidity()
        stats_c = extract_stats(report_c, "C) 流動性100萬股")
        results.append(stats_c)
    except Exception as e:
        print(f"  C) 回測失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append({'label': 'C) 流動性100萬股', 'error': str(e)})
    print(f"  耗時: {time.time() - t0:.1f}s")

    # 輸出比較表
    print("\n" + "=" * 60)
    print("  比較結果")
    print("=" * 60)

    valid = [r for r in results if 'error' not in r]
    if valid:
        header = f"{'指標':<16}" + "".join(f"{r['label']:<24}" for r in valid)
        print(header)
        print("-" * len(header))

        for key in ['cagr', 'mdd', 'sharpe', 'win_ratio', 'risk_reward', 'n_trades',
                     'avg_hold_days', 'max_win', 'max_loss', 'estimated_cost_drag_pct']:
            label_map = {
                'cagr': 'CAGR (%)', 'mdd': 'MDD (%)', 'sharpe': 'Sharpe',
                'win_ratio': '勝率 (%)', 'risk_reward': '風報比', 'n_trades': '交易次數',
                'avg_hold_days': '平均持有天', 'max_win': '最大獲利%', 'max_loss': '最大虧損%',
                'estimated_cost_drag_pct': '成本拖累%',
            }
            row = f"{label_map.get(key, key):<16}"
            for r in valid:
                val = r.get(key, 'N/A')
                row += f"{val:<24}"
            print(row)

    # 儲存結果
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果已儲存: {OUTPUT_PATH}")

    # 建議
    print("\n" + "=" * 60)
    print("  建議")
    print("=" * 60)
    if len(valid) >= 2:
        a = next((r for r in valid if 'A)' in r['label']), None)
        b = next((r for r in valid if 'B)' in r['label']), None)
        c = next((r for r in valid if 'C)' in r['label']), None)

        if a and b:
            cagr_diff = b.get('cagr', 0) - a.get('cagr', 0)
            mdd_diff = b.get('mdd', 0) - a.get('mdd', 0)
            sharpe_diff = b.get('sharpe', 0) - a.get('sharpe', 0)
            print(f"\n  ATR 正規化 vs 等權重:")
            print(f"    CAGR 變化:   {cagr_diff:+.2f}%")
            print(f"    MDD 變化:    {mdd_diff:+.2f}%")
            print(f"    Sharpe 變化: {sharpe_diff:+.3f}")
            if sharpe_diff > 0.05:
                print(f"    → 建議: ATR 正規化改善了風險調整後報酬，建議整合")
            elif sharpe_diff > -0.05:
                print(f"    → 建議: ATR 正規化影響不大，可選擇性整合")
            else:
                print(f"    → 建議: ATR 正規化反而劣化績效，不建議整合")

        if a and c:
            cagr_diff = c.get('cagr', 0) - a.get('cagr', 0)
            print(f"\n  流動性過濾 (100萬 vs 50萬股):")
            print(f"    CAGR 變化:   {cagr_diff:+.2f}%")
            print(f"    交易次數變化: {c.get('n_trades', 0) - a.get('n_trades', 0):+d}")
            cost_a = a.get('estimated_cost_drag_pct', 0)
            cost_c = c.get('estimated_cost_drag_pct', 0)
            print(f"    成本拖累變化: {cost_c - cost_a:+.2f}%")


if __name__ == '__main__':
    main()
