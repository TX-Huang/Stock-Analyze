"""
自動回測執行器
供 Claude 自主執行策略回測並產出結構化報告。

使用方式:
    python scripts/auto_backtest.py [--notify] [--version V3.3]

輸出:
    - 結構化回測報告 (stdout)
    - finlab_debug.log (詳細日誌)
    - Telegram 通知 (需設定 --notify)
"""

import sys
import os
import json
import argparse
from datetime import datetime

# 確保可以 import 專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_backtest(notify=False, version="V3"):
    """執行 Isaac 策略回測並產出報告"""
    import toml

    secrets = toml.load('.streamlit/secrets.toml')
    api_token = secrets.get('FINLAB_API_KEY', '')

    if not api_token:
        print("ERROR: FINLAB_API_KEY not found in .streamlit/secrets.toml")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Isaac {version} backtest...")

    from strategies.isaac import run_isaac_strategy
    report = run_isaac_strategy(api_token)

    trades = report.get_trades()
    stats = report.get_stats()

    # 結構化輸出
    result = {
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "trade_count": len(trades),
        "cagr": round(stats["cagr"] * 100, 2),
        "max_drawdown": round(stats["max_drawdown"] * 100, 2),
        "sharpe": round(stats["daily_sharpe"], 3),
        "sortino": round(stats["daily_sortino"], 3),
        "win_ratio": round(stats["win_ratio"] * 100, 1),
        "total_return": round(stats["total_return"] * 100, 1),
        "calmar": round(stats["calmar"], 3),
        "ytd": round(stats["ytd"] * 100, 1),
    }

    if not trades.empty and 'period' in trades.columns:
        result["period_mean"] = round(trades['period'].mean(), 1)
        result["period_median"] = round(trades['period'].median(), 1)
        result["period_max"] = int(trades['period'].max())
        result["period_1_ratio"] = round((trades['period'] == 1).sum() / len(trades) * 100, 1)

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULT ({version})")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # JSON 輸出（供程式讀取）
    print(f"\nJSON_OUTPUT: {json.dumps(result)}")

    # Telegram 通知
    if notify:
        try:
            from utils.notify import send_telegram, format_backtest_report
            msg = format_backtest_report(stats, trades, version=version)
            send_telegram(msg)
            print("\n[NOTIFY] Telegram notification sent.")
        except Exception as e:
            print(f"\n[NOTIFY] Telegram failed: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Backtest Runner")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification")
    parser.add_argument("--version", default="V3", help="Strategy version label")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_backtest(notify=args.notify, version=args.version)
