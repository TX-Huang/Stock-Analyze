"""
P0 回測比較：產業集中限制 + 滑價模擬
比較 4 組設定的績效差異，決定是否整合進主策略。

Variants:
  A. Baseline  — 原始 Isaac V3.7（無改動）
  B. Industry  — max_per_industry=3
  C. Slippage  — fee_ratio=0.002（模擬 0.2% 滑價+手續費）
  D. Combined  — B + C 同時啟用
"""
import sys, os, logging, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_comparison():
    import toml
    secrets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                '.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    api_token = secrets.get('FINLAB_API_KEY', '')
    if not api_token:
        logger.error("FINLAB_API_KEY not found in secrets.toml")
        return None

    from strategies.isaac import run_isaac_strategy

    variants = {
        'A_Baseline': {
            'params': {},
            'sim_override': {},
            'desc': '原始 Isaac V3.7（無改動）',
        },
        'B_Industry3': {
            'params': {'max_per_industry': 3},
            'sim_override': {},
            'desc': '每產業最多 3 檔',
        },
        'C_Slippage': {
            'params': {},
            'sim_override': {'fee_ratio': 0.002},  # 0.2% 來回滑價+手續費
            'desc': '模擬 0.2% 滑價+手續費',
        },
        'D_Combined': {
            'params': {'max_per_industry': 3},
            'sim_override': {'fee_ratio': 0.002},
            'desc': '產業限制 + 滑價模擬',
        },
    }

    results = {}

    for name, config in variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name} — {config['desc']}")
        logger.info(f"{'='*60}")

        try:
            t0 = time.time()

            # 如果有 sim_override（fee_ratio 等），需要用 raw_mode 取得 position 後自己跑 sim
            if config['sim_override']:
                raw_result = run_isaac_strategy(
                    api_token,
                    params=config['params'],
                    raw_mode=True,  # 回傳 dict: {final_pos, close, ...}
                )
                final_pos = raw_result['final_pos']
                trail_stop = raw_result['trail_stop']
                max_concurrent = raw_result['max_concurrent']

                from data.provider import safe_finlab_sim
                sim_kwargs = {
                    'name': f'Isaac V3.7 [{name}]',
                    'upload': False,
                    'trail_stop': trail_stop,
                    'position_limit': 1.0 / max_concurrent,
                    'touched_exit': False,
                }
                sim_kwargs.update(config['sim_override'])
                report = safe_finlab_sim(final_pos, **sim_kwargs)
            else:
                report = run_isaac_strategy(
                    api_token,
                    params=config['params'],
                )

            elapsed = time.time() - t0
            stats = report.get_stats()
            trades = report.get_trades()

            # 計算額外指標
            n_trades = len(trades)
            win_trades = trades[trades['return'] > 0] if not trades.empty else trades
            loss_trades = trades[trades['return'] <= 0] if not trades.empty else trades
            avg_win = float(win_trades['return'].mean()) if len(win_trades) > 0 else 0
            avg_loss = float(abs(loss_trades['return'].mean())) if len(loss_trades) > 0 else 0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
            avg_hold = float(trades['period'].mean()) if not trades.empty else 0

            result = {
                'desc': config['desc'],
                'cagr': float(stats.get('cagr', 0)),
                'max_drawdown': float(stats.get('max_drawdown', 0)),
                'sharpe': float(stats.get('sharpe', 0)),
                'win_ratio': float(stats.get('win_ratio', 0)),
                'n_trades': n_trades,
                'risk_reward': round(risk_reward, 3),
                'avg_hold_days': round(avg_hold, 1),
                'avg_win': round(avg_win * 100, 2),
                'avg_loss': round(avg_loss * 100, 2),
                'elapsed_sec': round(elapsed, 1),
            }
            results[name] = result

            logger.info(f"  CAGR:     {result['cagr']*100:.2f}%")
            logger.info(f"  MDD:      {result['max_drawdown']*100:.2f}%")
            logger.info(f"  Sharpe:   {result['sharpe']:.3f}")
            logger.info(f"  WinRate:  {result['win_ratio']*100:.1f}%")
            logger.info(f"  Trades:   {result['n_trades']}")
            logger.info(f"  R/R:      {result['risk_reward']:.3f}")
            logger.info(f"  Time:     {result['elapsed_sec']:.1f}s")

        except Exception as e:
            logger.error(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'desc': config['desc'], 'error': str(e)}

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'p0_backtest_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("P0 回測比較結果")
    print("=" * 90)
    print(f"{'Variant':<16} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'WinRate':>8} {'Trades':>7} {'R/R':>6} {'AvgHold':>8}")
    print("-" * 90)
    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<16} ERROR: {r['error'][:50]}")
            continue
        print(f"{name:<16} {r['cagr']*100:>7.2f}% {r['max_drawdown']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['win_ratio']*100:>7.1f}% {r['n_trades']:>7} "
              f"{r['risk_reward']:>6.3f} {r['avg_hold_days']:>7.1f}d")
    print("=" * 90)

    # Delta analysis
    if 'A_Baseline' in results and 'error' not in results['A_Baseline']:
        base = results['A_Baseline']
        print("\n差異分析 (vs Baseline):")
        print(f"{'Variant':<16} {'ΔCAGR':>10} {'ΔMDD':>10} {'ΔSharpe':>10} {'ΔWinRate':>10}")
        print("-" * 60)
        for name, r in results.items():
            if name == 'A_Baseline' or 'error' in r:
                continue
            d_cagr = (r['cagr'] - base['cagr']) * 100
            d_mdd = (r['max_drawdown'] - base['max_drawdown']) * 100
            d_sharpe = r['sharpe'] - base['sharpe']
            d_wr = (r['win_ratio'] - base['win_ratio']) * 100
            print(f"{name:<16} {d_cagr:>+9.2f}% {d_mdd:>+9.2f}% {d_sharpe:>+10.3f} {d_wr:>+9.1f}%")

    return results


if __name__ == '__main__':
    run_comparison()
