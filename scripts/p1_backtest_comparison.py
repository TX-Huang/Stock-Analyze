"""
P1 回測比較：Volume Mult + ATR Trailing
Variants:
  A. Baseline     — 原始 V3.7
  B. VolMult 2.0  — volume_mult=2.0（減少假突破）
  C. ATR Exit     — atr_exit=True（動態停損替代固定 18%）
  D. ATR + VolMult — 同時啟用 B + C
  E. Trail 15%    — trail_stop=0.15（縮緊固定停損）
  F. Trail 12%    — trail_stop=0.12（更緊固定停損）
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
        logger.error("FINLAB_API_KEY not found")
        return None

    from strategies.isaac import run_isaac_strategy

    variants = {
        'A_Baseline': {
            'params': {},
            'desc': '原始 V3.7 (vol=1.5, trail=18%, no ATR)',
        },
        'B_Vol2.0': {
            'params': {'volume_mult': 2.0},
            'desc': 'Volume multiplier 提高至 2.0',
        },
        'C_ATR_Exit': {
            'params': {'atr_exit': True},
            'desc': 'ATR 動態出場 (60日高-3×ATR14)',
        },
        'D_ATR+Vol': {
            'params': {'atr_exit': True, 'volume_mult': 2.0},
            'desc': 'ATR 出場 + Volume 2.0',
        },
        'E_Trail15': {
            'params': {'trail_stop': 0.15},
            'desc': 'Trailing stop 縮緊至 15%',
        },
        'F_Trail12': {
            'params': {'trail_stop': 0.12},
            'desc': 'Trailing stop 縮緊至 12%',
        },
    }

    results = {}

    for name, config in variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name} — {config['desc']}")
        logger.info(f"{'='*60}")

        try:
            t0 = time.time()
            report = run_isaac_strategy(api_token, params=config['params'])
            elapsed = time.time() - t0

            stats = report.get_stats()
            trades = report.get_trades()

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

            logger.info(f"  CAGR: {result['cagr']*100:.2f}% | MDD: {result['max_drawdown']*100:.2f}% | "
                        f"WR: {result['win_ratio']*100:.1f}% | R/R: {result['risk_reward']:.3f} | "
                        f"Trades: {result['n_trades']} | Hold: {result['avg_hold_days']}d | {result['elapsed_sec']:.0f}s")

        except Exception as e:
            logger.error(f"  FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'desc': config['desc'], 'error': str(e)}

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'p1_backtest_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print table
    print("\n" + "=" * 100)
    print("P1 回測比較結果")
    print("=" * 100)
    print(f"{'Variant':<14} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'WinRate':>8} {'Trades':>7} {'R/R':>6} {'AvgWin':>7} {'AvgLoss':>8} {'Hold':>6}")
    print("-" * 100)
    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<14} ERROR: {r['error'][:50]}")
            continue
        print(f"{name:<14} {r['cagr']*100:>7.2f}% {r['max_drawdown']*100:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['win_ratio']*100:>7.1f}% {r['n_trades']:>7} "
              f"{r['risk_reward']:>6.3f} {r['avg_win']:>6.2f}% {r['avg_loss']:>7.2f}% {r['avg_hold_days']:>5.1f}d")
    print("=" * 100)

    # Delta
    if 'A_Baseline' in results and 'error' not in results['A_Baseline']:
        base = results['A_Baseline']
        print(f"\n{'Variant':<14} {'ΔCAGR':>10} {'ΔMDD':>10} {'ΔWinRate':>10} {'ΔR/R':>8}")
        print("-" * 55)
        for name, r in results.items():
            if name == 'A_Baseline' or 'error' in r:
                continue
            print(f"{name:<14} {(r['cagr']-base['cagr'])*100:>+9.2f}% "
                  f"{(r['max_drawdown']-base['max_drawdown'])*100:>+9.2f}% "
                  f"{(r['win_ratio']-base['win_ratio'])*100:>+9.1f}% "
                  f"{r['risk_reward']-base['risk_reward']:>+8.3f}")

    return results


if __name__ == '__main__':
    run_comparison()
