"""
突破偵測排程器 — 獨立於 Streamlit 運行

啟動方式:
  python scheduler.py                    # 前景執行 (Ctrl+C 停止)
  python scheduler.py --once             # 只掃描一次
  python scheduler.py --interval 15      # 每 15 分鐘掃描一次 (預設 30)
  python scheduler.py --notify           # 有信號時發 Telegram 通知

也可透過 Windows 工作排程器設定自動執行。
"""
import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta

# 確保 project root 在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.provider import get_data_provider
from data.watchlist import WatchlistManager
from analysis.breakout import scan_breakouts, format_scan_results_for_telegram

from config.paths import SCAN_RESULTS_PATH, PAPER_TRADE_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
TW_TZ = timezone(timedelta(hours=8))

logger = logging.getLogger(__name__)


def _get_portfolio_tickers():
    """從 paper_trade.json 取得目前持倉的股票清單。"""
    if not os.path.exists(PAPER_TRADE_PATH):
        return []
    try:
        with open(PAPER_TRADE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        positions = data.get('positions', [])
        return [
            {
                'ticker': p['ticker'],
                'name': p.get('name', p['ticker']),
                'source': 'portfolio'
            }
            for p in positions if p.get('ticker')
        ]
    except Exception as e:
        logger.warning(f"讀取持倉失敗: {e}")
        return []


def _get_watchlist_tickers():
    """從 watchlist.json 取得自選股清單。"""
    try:
        wm = WatchlistManager()
        stocks = wm.get_all()
        return [
            {
                'ticker': s['ticker'],
                'name': s.get('name', s['ticker']),
                'source': 'watchlist'
            }
            for s in stocks if s.get('ticker')
        ]
    except Exception as e:
        logger.warning(f"讀取自選股失敗: {e}")
        return []


def _merge_tickers(watchlist, portfolio):
    """合併並去重。"""
    seen = set()
    merged = []
    for item in watchlist + portfolio:
        t = item['ticker']
        if t not in seen:
            seen.add(t)
            merged.append(item)
        else:
            # 如果兩邊都有，標記 source
            for m in merged:
                if m['ticker'] == t and m['source'] != item['source']:
                    m['source'] = 'both'
    return merged


def _is_market_hours():
    """判斷是否在台股交易時段 (09:00 - 13:30 週一~五)。"""
    now = datetime.now(TW_TZ)
    if now.weekday() >= 5:  # 週六日
        return False
    hour_min = now.hour * 100 + now.minute
    return 850 <= hour_min <= 1335  # 8:50 ~ 13:35 (含盤前盤後緩衝)


def run_scan(notify=False, force=False):
    """
    執行一次完整掃描。

    Args:
        notify: 是否發送 Telegram 通知
        force: 是否強制執行（忽略交易時段限制）

    Returns:
        list: scan results
    """
    if not force and not _is_market_hours():
        logger.info("非交易時段，跳過掃描。使用 --force 可強制執行。")
        return []

    logger.info("=" * 50)
    logger.info("開始突破偵測掃描...")

    # 1. 收集股票清單
    watchlist = _get_watchlist_tickers()
    portfolio = _get_portfolio_tickers()
    tickers = _merge_tickers(watchlist, portfolio)

    if not tickers:
        logger.warning("自選股和持倉皆為空，無標的可掃描。")
        return []

    logger.info(f"掃描標的: {len(tickers)} 檔 "
                f"(自選股 {len(watchlist)}, 持倉 {len(portfolio)})")

    # 2. 建立 provider
    provider = get_data_provider("yfinance", market_type="TW")

    # 3. 執行掃描
    results = scan_breakouts(tickers, provider, period="6mo")

    signals_count = len([r for r in results if r['signal'] is not None])
    critical_count = len([r for r in results if r.get('signal_info', {}).get('level') == 'critical'])

    logger.info(f"掃描完成: {len(results)} 檔完成, "
                f"{signals_count} 檔有信號, "
                f"{critical_count} 檔重大信號")

    # 4. 儲存結果
    scan_data = {
        'scan_time': datetime.now(TW_TZ).isoformat(),
        'total_scanned': len(tickers),
        'signals_count': signals_count,
        'critical_count': critical_count,
        'results': results,
    }

    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(SCAN_RESULTS_PATH), suffix='.tmp')
    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
        json.dump(scan_data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, SCAN_RESULTS_PATH)

    logger.info(f"結果已儲存: {SCAN_RESULTS_PATH}")

    # 5. Telegram 通知 (只推送有信號的)
    if notify and signals_count > 0:
        try:
            from utils.notify import send_telegram
            msg = format_scan_results_for_telegram(results)
            send_telegram(msg, parse_mode=None)  # 用純文字，避免 markdown 問題
            logger.info("Telegram 通知已發送")
        except Exception as e:
            logger.error(f"Telegram 發送失敗: {e}")

    # 6. 打印摘要
    for r in results:
        if r['signal']:
            info = r['signal_info']
            logger.info(
                f"  {info['icon']} {r['name']} ({r['ticker']}): "
                f"{info['label']} | 價: {r['price']:,.1f} | "
                f"量比: {r['volume_ratio']:.1f}x"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description='突破偵測排程器')
    parser.add_argument('--once', action='store_true', help='只掃描一次')
    parser.add_argument('--interval', type=int, default=30, help='掃描間隔(分鐘)')
    parser.add_argument('--notify', action='store_true', help='發送 Telegram 通知')
    parser.add_argument('--force', action='store_true', help='強制執行(忽略交易時段)')
    args = parser.parse_args()

    logger.info(f"突破偵測排程器啟動")
    logger.info(f"  模式: {'單次' if args.once else f'循環 (每 {args.interval} 分鐘)'}")
    logger.info(f"  通知: {'開啟' if args.notify else '關閉'}")
    logger.info(f"  時段: {'強制執行' if args.force else '僅交易時段'}")

    if args.once:
        run_scan(notify=args.notify, force=args.force)
        return

    # 循環模式
    try:
        while True:
            try:
                run_scan(notify=args.notify, force=args.force)
            except Exception as e:
                logger.error(f"Scan failed: {e}", exc_info=True)
            logger.info(f"下次掃描: {args.interval} 分鐘後")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        logger.info("排程器已停止 (Ctrl+C)")


def run_daily_market_scan():
    """APScheduler 用的每日市場掃描 job — 使用 data/scanner.py 掃描邏輯。"""
    from data.scanner import scan_single_stock_deep
    from data.watchlist import WatchlistManager

    logger.info("=== 每日市場掃描 (APScheduler) ===")

    # 收集自選股 + 持倉
    watchlist = _get_watchlist_tickers()
    portfolio = _get_portfolio_tickers()
    tickers = _merge_tickers(watchlist, portfolio)

    if not tickers:
        logger.warning("無標的可掃描 — 自選股和持倉皆為空")
        return

    logger.info(f"掃描標的: {len(tickers)} 檔")

    results = []
    for item in tickers:
        try:
            result = scan_single_stock_deep(
                market="🇹🇼 台股 (TW)",
                ticker=item['ticker'],
                strategy="Isaac V3",
                timeframe="1d",
                data_source="yfinance",
            )
            if result is not None:
                results.append({
                    'ticker': item['ticker'],
                    'name': item['name'],
                    'source': item['source'],
                    'scan': result,
                })
        except Exception as e:
            logger.warning(f"掃描 {item['ticker']} 失敗: {e}")

    # 寫入 scan_results.json
    scan_data = {
        'scan_time': datetime.now(TW_TZ).isoformat(),
        'scheduler': 'apscheduler',
        'total_scanned': len(tickers),
        'results_count': len(results),
        'results': results,
    }

    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(SCAN_RESULTS_PATH), suffix='.tmp'
    )
    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
        json.dump(scan_data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, SCAN_RESULTS_PATH)

    logger.info(f"每日掃描完成 — {len(results)}/{len(tickers)} 檔成功，已寫入 {SCAN_RESULTS_PATH}")

    # 掃描後推送每日摘要
    try:
        send_daily_digest()
    except Exception as e:
        logger.error(f"每日摘要推送失敗: {e}")


def send_daily_digest():
    """
    讀取最新掃描結果，格式化 Top 5 機會摘要，群發給所有訂閱者。

    讀取 data/scan_results.json，使用 format_report_summary() 格式化，
    透過 send_to_all_subscribers() 推送。無訂閱者或無 bot token 時靜默退出。
    """
    from utils.helpers import safe_json_read
    from utils.notify import send_to_all_subscribers, get_subscribers, _get_bot_token

    # Pre-flight checks
    if not _get_bot_token():
        logger.info("每日摘要: 無 bot token，跳過推送")
        return

    if not get_subscribers():
        logger.info("每日摘要: 無訂閱者，跳過推送")
        return

    scan_data = safe_json_read(SCAN_RESULTS_PATH, default=None)
    if not scan_data:
        logger.warning("每日摘要: 無掃描結果可推送")
        return

    results = scan_data.get('results', [])
    if not results:
        logger.info("每日摘要: 掃描結果為空，跳過推送")
        return

    scan_time = scan_data.get('scan_time', '---')

    # Build digest using format_report_summary from stock_report
    from analysis.stock_report import format_report_summary

    # Each result from run_daily_market_scan has a 'scan' key holding the report dict
    # Sort by composite score (descending) to pick top opportunities
    scored = []
    for r in results:
        report = r.get('scan') or {}
        thesis = report.get('thesis') or {}
        score = thesis.get('composite_score')
        scored.append((score if score is not None else -1, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:5]

    # Format message
    lines = [f"📋 AI Invest HQ 每日戰情摘要", f"🕐 {scan_time}", ""]

    for rank, (score, r) in enumerate(top5, 1):
        report = r.get('scan') or {}
        # Inject ticker/name into report for format_report_summary
        report_with_meta = {**report}
        if 'ticker' not in report_with_meta:
            report_with_meta['ticker'] = r.get('ticker', '?')
        if 'name' not in report_with_meta:
            report_with_meta['name'] = r.get('name', '')

        summary = format_report_summary(report_with_meta)
        lines.append(f"{rank}. {summary}")

    total = scan_data.get('total_scanned', len(results))
    lines.append("")
    lines.append(f"共掃描 {total} 檔，顯示前 {len(top5)} 名")

    message = "\n".join(lines)

    result = send_to_all_subscribers(message, parse_mode=None)
    logger.info(
        f"每日摘要推送完成: {result['sent']}/{result['total']} 成功, "
        f"{result['failed']} 失敗"
    )


def run_apscheduler():
    """以 APScheduler BackgroundScheduler 獨立運行，每天 08:20 掃描市場。"""
    import signal
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(timezone='Asia/Taipei')
    scheduler.add_job(
        run_daily_market_scan,
        trigger='cron',
        hour=8,
        minute=20,
        id='daily_market_scan',
        name='每日 08:20 市場掃描',
        misfire_grace_time=3600,
    )
    scheduler.start()

    logger.info("APScheduler 已啟動 — 每日 08:20 執行市場掃描")
    logger.info("按 Ctrl+C 停止")

    # Graceful shutdown
    shutdown_event = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_event
        logger.info(f"收到信號 {signum}，正在關閉排程器...")
        shutdown_event = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not shutdown_event:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler 已停止")


if __name__ == '__main__':
    # 支援兩種模式:
    #   python scheduler.py              → 原有的突破偵測循環
    #   python scheduler.py --apscheduler → APScheduler 獨立排程
    if '--apscheduler' in sys.argv:
        sys.argv.remove('--apscheduler')
        run_apscheduler()
    else:
        main()
