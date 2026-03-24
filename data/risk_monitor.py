"""
即時風控模組 — Risk Monitor
監控持倉停損、部位限制、大盤警報、組合回撤

使用方式:
    python data/risk_monitor.py              # 執行風控檢查
    from data.risk_monitor import RiskMonitor
    monitor = RiskMonitor()
    alerts = monitor.check_all()
"""
import sys
import os
import json
import logging
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import PAPER_TRADE_PATH, RISK_CONFIG_PATH
from utils.helpers import safe_json_read, safe_json_write

TW_TZ = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _default_risk_config():
    return {
        'trail_stop_pct': 18.0,         # 個股追蹤停損 (%)
        'max_loss_per_stock': 15.0,     # 個股最大虧損 (%)
        'max_portfolio_dd': 20.0,       # 組合最大回撤 (%)
        'max_single_weight': 15.0,      # 單檔最大權重 (%)
        'max_positions': 10,            # 最大持倉數
        'etf_ma60_alert': True,         # 大盤跌破 MA60 警報
        'etf_ma120_alert': True,        # 大盤跌破 MA120 警報
        'daily_loss_alert': 3.0,        # 單日虧損超過 N% 警報
    }


class RiskMonitor:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        saved = safe_json_read(RISK_CONFIG_PATH)
        if saved is not None:
            return saved
        config = _default_risk_config()
        safe_json_write(RISK_CONFIG_PATH, config)
        return config

    def _load_paper_trade(self):
        return safe_json_read(PAPER_TRADE_PATH)

    def _get_quotes(self, tickers):
        """取得即時報價"""
        quote_map = {}
        try:
            import toml
            secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
            from data.provider import SinoPacProvider
            SinoPacProvider._api_instance = None
            SinoPacProvider._logged_in = False
            provider = SinoPacProvider(
                api_key=secrets.get('SINOPAC_API_KEY', ''),
                secret_key=secrets.get('SINOPAC_SECRET_KEY', ''),
                simulation=False,
            )
            if SinoPacProvider._logged_in:
                snaps = provider.get_snapshots(list(tickers))
                for s in snaps:
                    quote_map[s['code']] = s
                SinoPacProvider.logout()
        except Exception as e:
            logging.warning(f"報價取得失敗: {e}")
        return quote_map

    def check_all(self):
        """
        執行完整風控檢查

        Returns:
            dict: {
                'timestamp': ...,
                'alerts': [{'level': 'WARNING/DANGER', 'type': ..., 'message': ...}],
                'positions_risk': [...],
                'portfolio_risk': {...},
                'market_risk': {...},
            }
        """
        account = self._load_paper_trade()
        if not account:
            return {'timestamp': datetime.now(TW_TZ).isoformat(), 'alerts': [], 'message': '無模擬帳戶資料'}

        positions = account.get('positions', [])
        all_tickers = [p['ticker'] for p in positions] + ['0050']
        quote_map = self._get_quotes(all_tickers)

        alerts = []
        positions_risk = []
        total_value = account.get('cash', 0)

        # ==========================================
        # 1. 個股風險檢查
        # ==========================================
        _positions_updated = False
        for pos in positions:
            ticker = pos['ticker']
            entry = pos['entry_price']
            current = quote_map.get(ticker, {}).get('close', pos.get('current_price', entry))
            if current <= 0:
                current = entry
            shares = pos['shares']
            value = current * shares
            total_value += value

            pnl_pct = (current - entry) / entry * 100
            change_today = quote_map.get(ticker, {}).get('change_rate', 0)

            risk_info = {
                'ticker': ticker,
                'name': pos.get('name', ''),
                'entry_price': entry,
                'current_price': current,
                'pnl_pct': round(pnl_pct, 2),
                'change_today': round(change_today, 2),
                'value': round(value, 0),
                'alerts': [],
            }

            # 真正的追蹤停損: 從進場以來最高價回撤
            high_since_entry = pos.get('high_since_entry', entry)
            if current > high_since_entry:
                high_since_entry = current
                pos['high_since_entry'] = high_since_entry
                _positions_updated = True
            trail_pct = self.config['trail_stop_pct'] / 100.0
            trail_stop_price = high_since_entry * (1 - trail_pct)
            trail_drop_pct = (current - high_since_entry) / high_since_entry * 100 if high_since_entry > 0 else 0

            risk_info['high_since_entry'] = round(high_since_entry, 2)
            risk_info['trail_stop_price'] = round(trail_stop_price, 2)

            if current < trail_stop_price:
                alert = {
                    'level': 'DANGER',
                    'type': 'trail_stop',
                    'ticker': ticker,
                    'message': (
                        f"{ticker} {pos.get('name', '')} 觸及追蹤停損! "
                        f"(現價 {current:.2f} < 停損 {trail_stop_price:.2f}, "
                        f"高點 {high_since_entry:.2f} 回撤 {trail_drop_pct:+.1f}%)"
                    ),
                }
                alerts.append(alert)
                risk_info['alerts'].append('TRAIL_STOP')

            # 最大虧損檢查 (從進場價)
            elif pnl_pct <= -self.config['max_loss_per_stock']:
                alert = {
                    'level': 'WARNING',
                    'type': 'max_loss',
                    'ticker': ticker,
                    'message': f"{ticker} {pos.get('name', '')} 接近停損 ({pnl_pct:+.1f}%)",
                }
                alerts.append(alert)
                risk_info['alerts'].append('NEAR_STOP')

            # 單日大跌檢查
            if change_today <= -self.config['daily_loss_alert']:
                alert = {
                    'level': 'WARNING',
                    'type': 'daily_drop',
                    'ticker': ticker,
                    'message': f"{ticker} {pos.get('name', '')} 今日大跌 {change_today:+.1f}%",
                }
                alerts.append(alert)
                risk_info['alerts'].append('DAILY_DROP')

            positions_risk.append(risk_info)

        # 持久化 high_since_entry 回 paper_trade.json
        if _positions_updated and account:
            safe_json_write(PAPER_TRADE_PATH, account, default=str)

        # ==========================================
        # 2. 組合風險檢查
        # ==========================================
        initial = account.get('initial_capital', 1_000_000)
        portfolio_return = (total_value / initial - 1) * 100

        # 最高權益 (從 equity history)
        equity_hist = account.get('daily_equity', [])
        # Use recent 120-day peak instead of all-time peak
        recent_equities = [e.get('equity', initial) for e in equity_hist[-120:]] + [initial]
        peak_equity = max(recent_equities)
        current_dd = (total_value / peak_equity - 1) * 100 if peak_equity > 0 else 0

        portfolio_risk = {
            'total_equity': round(total_value, 0),
            'portfolio_return': round(portfolio_return, 2),
            'current_drawdown': round(current_dd, 2),
            'peak_equity': round(peak_equity, 0),
            'n_positions': len(positions),
            'cash_ratio': round(account.get('cash', 0) / total_value * 100, 1) if total_value > 0 else 0,
        }

        # 最大回撤警報
        if current_dd <= -self.config['max_portfolio_dd']:
            alerts.append({
                'level': 'DANGER',
                'type': 'portfolio_dd',
                'message': f"組合回撤 {current_dd:.1f}% 超過上限 -{self.config['max_portfolio_dd']}%!",
            })

        # 集中度檢查
        if total_value > 0:
            for pr in positions_risk:
                weight = pr['value'] / total_value * 100
                pr['weight'] = round(weight, 1)
                if weight > self.config['max_single_weight']:
                    alerts.append({
                        'level': 'WARNING',
                        'type': 'concentration',
                        'ticker': pr['ticker'],
                        'message': f"{pr['ticker']} 權重 {weight:.1f}% 超過上限 {self.config['max_single_weight']}%",
                    })

        # 持倉數檢查
        if len(positions) > self.config['max_positions']:
            alerts.append({
                'level': 'WARNING',
                'type': 'position_count',
                'message': f"持倉數 {len(positions)} 超過上限 {self.config['max_positions']}",
            })

        # ==========================================
        # 3. 大盤風險檢查
        # ==========================================
        etf_quote = quote_map.get('0050', {})
        market_risk = {'etf_0050': {}}

        if etf_quote and etf_quote.get('close', 0) > 0:
            # 用推薦檔案中的 MA 數據
            from config.paths import RECOMMENDATION_PATH
            rec_path = RECOMMENDATION_PATH
            ma60 = 0
            ma120 = 0
            rec = safe_json_read(rec_path)
            if rec:
                etf_data = rec.get('etf_0050', {})
                ma60 = etf_data.get('ma60', 0)
                ma120 = etf_data.get('ma120', 0)

            etf_close = etf_quote['close']
            market_risk['etf_0050'] = {
                'close': etf_close,
                'change_rate': etf_quote.get('change_rate', 0),
                'ma60': ma60,
                'ma120': ma120,
                'above_ma60': etf_close > ma60 if ma60 > 0 else None,
                'above_ma120': etf_close > ma120 if ma120 > 0 else None,
            }

            if ma60 > 0 and etf_close < ma60 and self.config['etf_ma60_alert']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'market_ma60',
                    'message': f"0050 跌破 MA60! ({etf_close:.2f} < {ma60:.2f}) — 建議減倉至 60%",
                })
            if ma120 > 0 and etf_close < ma120 and self.config['etf_ma120_alert']:
                alerts.append({
                    'level': 'DANGER',
                    'type': 'market_ma120',
                    'message': f"0050 跌破 MA120! ({etf_close:.2f} < {ma120:.2f}) — 建議減倉至 30%",
                })

        # ==========================================
        # 排序 alerts: DANGER 優先
        # ==========================================
        alerts.sort(key=lambda x: 0 if x['level'] == 'DANGER' else 1)

        # Overall level
        n_danger = sum(1 for a in alerts if a['level'] == 'DANGER')
        n_warning = sum(1 for a in alerts if a['level'] == 'WARNING')
        if n_danger > 0:
            level = 'DANGER'
        elif n_warning > 0:
            level = 'WARNING'
        else:
            level = 'OK'

        return {
            'timestamp': datetime.now(TW_TZ).isoformat(),
            'level': level,
            'alerts': alerts,
            'n_alerts': len(alerts),
            'n_danger': n_danger,
            'n_warning': n_warning,
            'positions_risk': positions_risk,
            'portfolio_risk': portfolio_risk,
            'market_risk': market_risk,
        }

    def format_risk_text(self, result=None):
        """格式化風控報告為文字"""
        if result is None:
            result = self.check_all()

        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"  Risk Monitor Report")
        lines.append(f"  {result['timestamp'][:19]}")
        lines.append(f"{'='*50}")

        # 警報
        alerts = result.get('alerts', [])
        if alerts:
            lines.append(f"\n  [!! 警報 !!] ({result['n_danger']} DANGER, {result['n_warning']} WARNING)")
            for a in alerts:
                icon = '!!' if a['level'] == 'DANGER' else '!'
                lines.append(f"  {icon} [{a['level']}] {a['message']}")
        else:
            lines.append(f"\n  [OK] 無警報，所有風控指標正常")

        # 組合風險
        pr = result.get('portfolio_risk', {})
        lines.append(f"\n  [組合風險]")
        lines.append(f"  權益: {pr.get('total_equity', 0):>12,.0f} ({pr.get('portfolio_return', 0):+.2f}%)")
        lines.append(f"  回撤: {pr.get('current_drawdown', 0):+.2f}% (峰值: {pr.get('peak_equity', 0):,.0f})")
        lines.append(f"  持倉: {pr.get('n_positions', 0)}檔 | 現金比: {pr.get('cash_ratio', 0):.1f}%")

        # 大盤
        mr = result.get('market_risk', {}).get('etf_0050', {})
        if mr:
            ma60_status = 'Above' if mr.get('above_ma60') else 'Below'
            ma120_status = 'Above' if mr.get('above_ma120') else 'Below'
            lines.append(f"\n  [大盤 0050]")
            lines.append(f"  收盤: {mr.get('close', 0):.2f} ({mr.get('change_rate', 0):+.2f}%)")
            lines.append(f"  MA60: {mr.get('ma60', 0):.2f} ({ma60_status}) | MA120: {mr.get('ma120', 0):.2f} ({ma120_status})")

        # 個股風險
        positions_risk = result.get('positions_risk', [])
        if positions_risk:
            lines.append(f"\n  [個股風險]")
            lines.append(f"  {'代碼':<6} {'名稱':<8} {'損益%':>7} {'今日%':>7} {'權重':>5} {'狀態':>8}")
            lines.append(f"  {'-'*48}")
            for pr_item in sorted(positions_risk, key=lambda x: x.get('pnl_pct', 0)):
                name = pr_item.get('name', '')[:6]
                status = ','.join(pr_item.get('alerts', [])) or 'OK'
                lines.append(
                    f"  {pr_item['ticker']:<6} {name:<8} {pr_item['pnl_pct']:>+6.1f}% "
                    f"{pr_item.get('change_today', 0):>+6.1f}% {pr_item.get('weight', 0):>4.1f}% {status:>8}"
                )

        lines.append(f"{'='*50}")
        return '\n'.join(lines)


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == '__main__':
    monitor = RiskMonitor()
    result = monitor.check_all()
    print(monitor.format_risk_text(result))
