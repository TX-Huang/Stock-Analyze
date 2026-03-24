"""
自動化交易系統 — Auto Trading Engine
整合 Isaac V3.7 策略 + 永豐金 Shioaji 下單 API

功能:
  - 策略信號 → 自動產生委託單
  - 支援市價/限價單
  - 風控預檢 (pre-trade risk check)
  - 完整交易日誌
  - 支援模擬/實盤模式切換

使用方式:
    from data.auto_trader import AutoTrader
    trader = AutoTrader(mode='simulation')
    trader.execute_daily()
"""
import sys
import os
import json
import logging
from datetime import datetime, date, timedelta, timezone
from copy import deepcopy

TW_TZ = timezone(timedelta(hours=8))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import ORDER_LOG_PATH, AUTO_TRADE_CONFIG_PATH
from utils.helpers import safe_json_read, safe_json_write

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _default_config():
    return {
        'mode': 'simulation',       # simulation / live
        'order_type': 'market',     # market / limit
        'limit_offset_pct': 0.5,    # 限價偏移 (%)
        'max_order_value': 500_000, # 單筆最大金額
        'daily_loss_limit': 3.0,    # 當日虧損上限 (%)
        'enabled': False,           # 自動交易開關
        'initial_capital': 1_000_000,
        'schedule_time': '09:05',   # 每日執行時間
        'pre_trade_checks': True,   # 下單前風控檢查
        'telegram_notify': True,    # Telegram 通知
    }


class AutoTrader:
    """自動化交易引擎"""

    def __init__(self, config_path=AUTO_TRADE_CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.order_log = self._load_order_log()
        self._api = None

    # --------------------------------------------------
    # Config Management
    # --------------------------------------------------
    def _load_config(self):
        saved = safe_json_read(self.config_path)
        if saved is not None:
            cfg = _default_config()
            cfg.update(saved)
            return cfg
        cfg = _default_config()
        self._save_config(cfg)
        return cfg

    def _save_config(self, cfg=None):
        if cfg is None:
            cfg = self.config
        safe_json_write(self.config_path, cfg)

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.config:
                self.config[k] = v
        self._save_config()

    # --------------------------------------------------
    # Order Log
    # --------------------------------------------------
    def _load_order_log(self):
        data = safe_json_read(ORDER_LOG_PATH)
        return data if data is not None else {'orders': [], 'daily_summary': []}

    def _save_order_log(self):
        safe_json_write(ORDER_LOG_PATH, self.order_log, default=str)

    def _log_order(self, order):
        order['timestamp'] = datetime.now(TW_TZ).isoformat()
        # Ensure slippage fields are present in logged order
        order.setdefault('slippage_pct', None)
        order.setdefault('slippage_amount', None)
        self.order_log['orders'].append(order)
        self._rotate_order_log_if_needed()
        self._save_order_log()

    def _rotate_order_log_if_needed(self):
        """當 orders 超過 1000 筆，將舊紀錄歸檔，僅保留最近 500 筆"""
        orders = self.order_log.get('orders', [])
        if len(orders) <= 1000:
            return

        # 分割: 歸檔舊的，保留最近 500 筆
        to_archive = orders[:-500]
        self.order_log['orders'] = orders[-500:]

        # 歸檔檔名: order_log_archive_YYYYMM.json (使用 DATA_DIR)
        from config.paths import DATA_DIR
        archive_name = f"order_log_archive_{datetime.now(TW_TZ).strftime('%Y%m')}.json"
        archive_path = os.path.join(DATA_DIR, archive_name)

        # 如果歸檔檔案已存在，合併
        existing_archive = []
        if os.path.exists(archive_path):
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    existing_archive = json.load(f)
                if not isinstance(existing_archive, list):
                    existing_archive = existing_archive.get('orders', [])
            except Exception:
                existing_archive = []

        merged = existing_archive + to_archive

        safe_json_write(archive_path, merged, default=str)
        logging.info(f"[AutoTrader] 訂單日誌已歸檔 {len(to_archive)} 筆至 {archive_name}，主檔保留 {len(self.order_log['orders'])} 筆")

    # --------------------------------------------------
    # Shioaji API Connection
    # --------------------------------------------------
    def _connect_api(self):
        """連接永豐金 API"""
        if self._api is not None:
            return self._api
        try:
            import toml
            secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
            import shioaji as sj
            api = sj.Shioaji(simulation=(self.config['mode'] == 'simulation'))
            api.login(
                api_key=secrets.get('SINOPAC_API_KEY', ''),
                secret_key=secrets.get('SINOPAC_SECRET_KEY', ''),
            )
            self._api = api
            logging.info(f"[AutoTrader] API 連線成功 (mode={self.config['mode']})")
            return api
        except Exception as e:
            logging.error(f"[AutoTrader] API 連線失敗: {e}")
            return None

    def _disconnect_api(self):
        if self._api:
            try:
                self._api.logout()
            except Exception:
                pass
            self._api = None

    # --------------------------------------------------
    # Pre-Trade Risk Check
    # --------------------------------------------------
    def pre_trade_check(self, recommendation):
        """下單前風控檢查"""
        issues = []

        # 1. 檢查是否超過當日虧損上限
        today_orders = [o for o in self.order_log['orders']
                        if o.get('date') == str(date.today()) and o.get('status') == 'filled']
        today_pnl = sum(o.get('realized_pnl', 0) for o in today_orders)
        capital = self.config['initial_capital']
        if capital > 0 and (today_pnl / capital * 100) <= -self.config['daily_loss_limit']:
            issues.append(f"當日已虧損 {today_pnl/capital*100:.1f}%，超過上限 {self.config['daily_loss_limit']}%")

        # 2. 檢查市場環境
        market_regime = recommendation.get('market_regime', '')
        if market_regime == '空頭':
            issues.append(f"市場處於空頭環境，建議減少交易")

        # 3. 檢查單筆金額（用實際配置金額，不是整張價格）
        max_val = self.config['max_order_value']
        try:
            from data.paper_trader import PaperTrader
            pt = PaperTrader()
            equity = pt._calc_equity()
            max_pos = recommendation.get('summary', {}).get('max_concurrent', 8)
            exposure = recommendation.get('exposure', 1.0)
            per_stock = (equity * exposure) / max(max_pos, 1)
        except Exception:
            per_stock = max_val

        for rec in recommendation.get('recommendations', []):
            price = rec.get('close', 0)
            actual_budget = min(per_stock, max_val)
            if price > 0 and actual_budget < price:
                issues.append(f"{rec['ticker']} 股價 {price:,.0f} 超過每檔預算 {actual_budget:,.0f}（連一股都買不起）")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checked_at': datetime.now(TW_TZ).isoformat(),
        }

    # --------------------------------------------------
    # Order Generation
    # --------------------------------------------------
    def generate_orders(self, recommendation):
        """從推薦清單產生委託單"""
        orders = []
        recs = recommendation.get('recommendations', [])
        exits = recommendation.get('exits', [])
        exposure = recommendation.get('exposure', 1.0)
        max_positions = recommendation.get('summary', {}).get('max_concurrent', 8)

        # 出場單 — 從 PaperTrader 查持倉股數
        try:
            from data.paper_trader import PaperTrader
            pt_positions = {p['ticker']: p for p in PaperTrader().account['positions']}
        except Exception:
            pt_positions = {}

        for ex in exits:
            sell_shares = pt_positions.get(ex['ticker'], {}).get('shares', 0)
            orders.append({
                'action': 'SELL',
                'ticker': ex['ticker'],
                'name': ex.get('name', ''),
                'price': ex.get('close', 0),
                'shares': sell_shares,
                'reason': 'signal_exit',
                'order_type': self.config['order_type'],
            })

        # 進場單 (只有 new entries)
        # 用 PaperTrader 的實際權益和現金計算可用資金
        try:
            from data.paper_trader import PaperTrader
            pt = PaperTrader()
            available_cash = pt.account['cash']
            current_equity = pt._calc_equity()
            current_positions = len(pt.account['positions'])
        except Exception:
            available_cash = self.config['initial_capital']
            current_equity = self.config['initial_capital']
            current_positions = 0

        new_entries = recommendation.get('new_entries', [])
        # 按 score 排序，優先配置高分的
        entry_recs = sorted(
            [r for r in recs if r['ticker'] in new_entries],
            key=lambda x: x.get('score', 0), reverse=True,
        )
        available_slots = max(max_positions - current_positions, 0)
        entry_recs = entry_recs[:available_slots]

        if entry_recs:
            # 每檔目標配置 = 總權益 × 曝險 / 最大持倉數
            per_stock_target = (current_equity * exposure) / max_positions
            # 受限於 max_order_value
            per_stock_budget = min(per_stock_target, self.config['max_order_value'])

            for rec in entry_recs:
                price = rec.get('close', 0)
                if price <= 0:
                    continue

                # 實際可用 = min(每檔預算, 剩餘現金)
                budget = min(per_stock_budget, available_cash)
                if budget < price:
                    continue  # 連一股都買不起

                # 計算股數: 優先整張，剩餘用零股補滿
                full_lots = int(budget / (price * 1000))  # 可買幾張
                remaining = budget - (full_lots * 1000 * price)
                odd_shares = int(remaining / price)        # 零股 (1股為單位)
                shares = full_lots * 1000 + odd_shares

                if shares <= 0:
                    continue

                cost = round(price * shares, 0)
                buy_commission = round(cost * 0.001425, 0)  # 台股買入手續費 0.1425%
                total_cost = cost + buy_commission
                available_cash -= total_cost  # 扣除已配置的現金（含手續費）

                # 標記整股/零股下單方式
                order_lot = 'common' if odd_shares == 0 and full_lots > 0 else 'odd' if full_lots == 0 else 'mixed'

                # 限價計算
                if self.config['order_type'] == 'limit':
                    limit_price = round(price * (1 - self.config['limit_offset_pct'] / 100), 2)
                else:
                    limit_price = 0  # 市價

                orders.append({
                    'action': 'BUY',
                    'ticker': rec['ticker'],
                    'name': rec.get('name', ''),
                    'price': price,
                    'limit_price': limit_price,
                    'shares': shares,
                    'score': rec.get('score', 0),
                    'cost': cost,
                    'order_lot': order_lot,
                    'reason': 'signal_entry',
                    'order_type': self.config['order_type'],
                })

        return orders

    # --------------------------------------------------
    # Order Execution
    # --------------------------------------------------
    def execute_order(self, order):
        """執行單筆委託"""
        result = deepcopy(order)
        result['date'] = str(date.today())
        result['mode'] = self.config['mode']

        if self.config['mode'] == 'simulation':
            # 模擬模式: 直接以市價成交
            fill_price = order['price']
            result['status'] = 'filled'
            result['filled_price'] = fill_price
            result['filled_shares'] = order.get('shares', 0)
            result['message'] = '模擬成交'

            # Slippage tracking
            expected_price = order.get('price', 0)  # Signal price
            actual_price = fill_price  # Actual fill price
            if expected_price > 0:
                slippage = (actual_price - expected_price) / expected_price * 100
                result['slippage_pct'] = slippage
                result['slippage_amount'] = (actual_price - expected_price) * order.get('shares', 0)
        else:
            # 實盤模式: 透過 Shioaji API 下單
            api = self._connect_api()
            if not api:
                result['status'] = 'failed'
                result['message'] = 'API 連線失敗'
                self._log_order(result)
                return result

            try:
                import shioaji as sj
                contract = api.Contracts.Stocks.get(str(order['ticker']))
                if not contract:
                    result['status'] = 'failed'
                    result['message'] = f"找不到合約 {order['ticker']}"
                    self._log_order(result)
                    return result

                # 建立委託
                action = sj.Action.Buy if order['action'] == 'BUY' else sj.Action.Sell

                if order['order_type'] == 'market':
                    price_type = sj.StockPriceType.LMT  # 台股沒有真正的市價單，用漲/跌停價
                    price = order['price']
                else:
                    price_type = sj.StockPriceType.LMT
                    price = order.get('limit_price', order['price'])

                # 拆分整股 + 零股 (Shioaji 不支援混合下單)
                shares = order.get('shares', 0)
                full_lot_shares = (shares // 1000) * 1000
                odd_lot_shares = shares % 1000
                trade_ids = []

                # 整股委託
                if full_lot_shares > 0:
                    sj_order = api.Order(
                        price=price,
                        quantity=full_lot_shares // 1000,
                        action=action,
                        price_type=price_type,
                        order_type=sj.OrderType.ROD,
                        order_lot=sj.StockOrderLot.Common,
                    )
                    trade = api.place_order(contract, sj_order)
                    trade_ids.append(str(getattr(trade, 'order_id', '')))
                    logging.info(f"[AutoTrader] 整股委託: {order['action']} {order['ticker']} x{full_lot_shares} @ {price}")

                # 零股委託
                if odd_lot_shares > 0:
                    sj_order = api.Order(
                        price=price,
                        quantity=odd_lot_shares,
                        action=action,
                        price_type=price_type,
                        order_type=sj.OrderType.ROD,
                        order_lot=sj.StockOrderLot.IntradayOdd,
                    )
                    trade = api.place_order(contract, sj_order)
                    trade_ids.append(str(getattr(trade, 'order_id', '')))
                    logging.info(f"[AutoTrader] 零股委託: {order['action']} {order['ticker']} x{odd_lot_shares} @ {price}")

                result['status'] = 'submitted'
                result['trade_id'] = ','.join(trade_ids)
                result['filled_price'] = price
                result['filled_shares'] = shares
                result['message'] = f"委託已送出 (整股:{full_lot_shares} 零股:{odd_lot_shares})"

                # Slippage tracking
                expected_price = order.get('price', 0)  # Signal price
                actual_price = price  # Actual fill price
                if expected_price > 0:
                    slippage = (actual_price - expected_price) / expected_price * 100
                    result['slippage_pct'] = slippage
                    result['slippage_amount'] = (actual_price - expected_price) * order.get('shares', 0)

            except Exception as e:
                result['status'] = 'failed'
                result['message'] = f"下單失敗: {e}"
                logging.error(f"[AutoTrader] 下單失敗: {e}")

        self._log_order(result)
        self._sync_paper_trader(result)
        return result

    # --------------------------------------------------
    # Sync to PaperTrader
    # --------------------------------------------------
    def _sync_paper_trader(self, result):
        """委託成交後同步更新 PaperTrader 持倉，確保兩個模組資料一致"""
        if result.get('status') not in ('filled', 'submitted'):
            return  # 只同步成交的委託

        try:
            from data.paper_trader import PaperTrader
            pt = PaperTrader()
            ticker = result.get('ticker', '')
            price = result.get('filled_price', result.get('price', 0))
            shares = result.get('filled_shares', result.get('shares', 0))

            if result['action'] == 'BUY' and price > 0 and shares > 0:
                # 檢查是否已持有（避免重複進場）
                held = {p['ticker'] for p in pt.account['positions']}
                if ticker not in held:
                    cost = price * shares
                    buy_commission = round(cost * 0.001425, 0)
                    pt.account['cash'] -= (cost + buy_commission)
                    pt.account['positions'].append({
                        'ticker': ticker,
                        'name': result.get('name', ''),
                        'score': result.get('score', 0),
                        'entry_price': price,
                        'current_price': price,
                        'high_since_entry': price,
                        'shares': shares,
                        'entry_date': result.get('date', str(date.today())),
                        'change_rate': 0,
                    })
                    pt._save()
                    logging.info(f"[Sync] PaperTrader 同步進場: {ticker} {price} x {shares}")

            elif result['action'] == 'SELL':
                # 找到對應持倉並平倉
                new_positions = []
                for pos in pt.account['positions']:
                    if pos['ticker'] == ticker:
                        # 台股手續費: 買賣各 0.1425%, 賣出證交稅 0.3%
                        buy_fee = pos['entry_price'] * pos['shares'] * 0.001425
                        sell_fee = price * pos['shares'] * 0.001425
                        sell_tax = price * pos['shares'] * 0.003
                        total_fees = buy_fee + sell_fee + sell_tax
                        pnl_abs = (price - pos['entry_price']) * pos['shares'] - total_fees
                        pnl_pct = pnl_abs / (pos['entry_price'] * pos['shares']) * 100 if pos['entry_price'] > 0 else 0
                        pt.account['closed_trades'].append({
                            'ticker': ticker,
                            'name': pos.get('name', ''),
                            'entry_price': pos['entry_price'],
                            'exit_price': price,
                            'shares': pos['shares'],
                            'entry_date': pos.get('entry_date', ''),
                            'exit_date': result.get('date', str(date.today())),
                            'pnl_pct': round(pnl_pct, 2),
                            'pnl_abs': round(pnl_abs, 0),
                            'fees': round(total_fees, 0),
                        })
                        pt.account['cash'] += price * pos['shares'] - sell_fee - sell_tax
                        logging.info(f"[Sync] PaperTrader 同步出場: {ticker} {pnl_pct:+.1f}%")
                    else:
                        new_positions.append(pos)
                pt.account['positions'] = new_positions
                pt._save()

        except Exception as e:
            logging.warning(f"[Sync] PaperTrader 同步失敗 (不影響委託): {e}")

    # --------------------------------------------------
    # Daily Execution Pipeline
    # --------------------------------------------------
    def execute_daily(self, recommendation=None):
        """完整每日執行流程"""
        if not self.config.get('enabled', False):
            return {'status': 'disabled', 'message': 'Auto trading is disabled', 'orders': []}

        logging.info("[AutoTrader] 開始每日執行流程")

        # 1. 取得推薦
        if recommendation is None:
            from config.paths import RECOMMENDATION_PATH
            recommendation = safe_json_read(RECOMMENDATION_PATH)
            if recommendation is None:
                return {'status': 'error', 'message': '找不到推薦檔案'}

        # 2. 風控預檢
        if self.config['pre_trade_checks']:
            check = self.pre_trade_check(recommendation)
            if not check['passed']:
                logging.warning(f"[AutoTrader] 風控預檢未通過: {check['issues']}")
                return {
                    'status': 'blocked',
                    'message': '風控預檢未通過',
                    'risk_check': check,
                    'orders': [],
                }

        # 3. 產生委託單
        orders = self.generate_orders(recommendation)
        if not orders:
            return {
                'status': 'no_action',
                'message': '今日無新交易信號',
                'orders': [],
            }

        # 4. 執行委託
        results = []
        for order in orders:
            result = self.execute_order(order)
            results.append(result)

        # 5. 日結摘要
        filled = [r for r in results if r.get('status') == 'filled']
        failed = [r for r in results if r.get('status') == 'failed']
        summary = {
            'date': str(date.today()),
            'total_orders': len(results),
            'filled': len(filled),
            'failed': len(failed),
            'buy_orders': len([r for r in results if r.get('action') == 'BUY']),
            'sell_orders': len([r for r in results if r.get('action') == 'SELL']),
            'total_cost': sum(r.get('cost', 0) for r in results if r.get('action') == 'BUY' and r.get('status') == 'filled'),
            'mode': self.config['mode'],
            'timestamp': datetime.now(TW_TZ).isoformat(),
        }
        self.order_log['daily_summary'].append(summary)
        self._save_order_log()

        logging.info(f"[AutoTrader] 執行完成: {summary['filled']}/{summary['total_orders']} 成交")

        return {
            'status': 'completed',
            'summary': summary,
            'orders': results,
            'risk_check': None,
        }

    # --------------------------------------------------
    # Status & History
    # --------------------------------------------------
    def get_status(self):
        """取得交易系統狀態"""
        orders = self.order_log.get('orders', [])
        today_str = str(date.today())
        today_orders = [o for o in orders if o.get('date') == today_str]

        # 最近 30 天統計
        recent_orders = [o for o in orders if o.get('date', '') >= str(date.today() - timedelta(days=30))]
        total_buy = sum(o.get('cost', 0) for o in recent_orders if o.get('action') == 'BUY' and o.get('status') == 'filled')
        total_sell = sum(o.get('cost', 0) for o in recent_orders if o.get('action') == 'SELL' and o.get('status') == 'filled')

        return {
            'config': self.config,
            'today_orders': today_orders,
            'today_count': len(today_orders),
            'total_orders': len(orders),
            'recent_30d': {
                'buy_amount': total_buy,
                'sell_amount': total_sell,
                'order_count': len(recent_orders),
            },
            'daily_summaries': self.order_log.get('daily_summary', [])[-30:],
            'last_5_orders': orders[-5:] if orders else [],
        }

    def get_order_history(self, days=30):
        """取得歷史委託記錄"""
        cutoff = str(date.today() - timedelta(days=days))
        orders = [o for o in self.order_log.get('orders', []) if o.get('date', '') >= cutoff]
        return sorted(orders, key=lambda x: x.get('timestamp', ''), reverse=True)


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == '__main__':
    trader = AutoTrader()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'status':
            status = trader.get_status()
            print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        elif cmd == 'run':
            result = trader.execute_daily()
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        elif cmd == 'enable':
            trader.update_config(enabled=True)
            print("Auto trading enabled")
        elif cmd == 'disable':
            trader.update_config(enabled=False)
            print("Auto trading disabled")
    else:
        print("Usage: python auto_trader.py [status|run|enable|disable]")
