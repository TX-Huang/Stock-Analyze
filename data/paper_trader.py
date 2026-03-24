"""
模擬交易追蹤器 — Paper Trading Tracker
自動追蹤 Isaac V3.7 推薦的持股，模擬進出場、計算損益

使用方式:
    python data/paper_trader.py                  # 執行每日更新
    python data/paper_trader.py status           # 查看持倉狀態
    python data/paper_trader.py history          # 查看歷史交易
    python data/paper_trader.py reset            # 重置模擬帳戶

模組呼叫:
    from data.paper_trader import PaperTrader
    trader = PaperTrader()
    trader.update()           # 根據最新推薦更新持倉
    trader.get_status()       # 取得持倉狀態
"""
import sys
import os
import json
import logging
from datetime import datetime, date
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import PAPER_TRADE_PATH, RECOMMENDATION_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _default_account():
    """空白模擬帳戶"""
    return {
        'initial_capital': 1_000_000,
        'cash': 1_000_000,
        'positions': [],        # 目前持倉
        'closed_trades': [],    # 已平倉交易
        'daily_equity': [],     # 每日權益記錄
        'created_at': datetime.now().isoformat(),
        'last_updated': None,
    }


class PaperTrader:
    def __init__(self, path=PAPER_TRADE_PATH):
        self.path = path
        self.account = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return _default_account()

    def _save(self):
        import tempfile
        self.account['last_updated'] = datetime.now().isoformat()
        dir_name = os.path.dirname(self.path)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=dir_name,
                                          suffix='.tmp', delete=False) as tmp:
            json.dump(self.account, tmp, ensure_ascii=False, indent=2, default=str)
            tmp_name = tmp.name
        os.replace(tmp_name, self.path)

    def reset(self, initial_capital=1_000_000):
        """重置模擬帳戶"""
        self.account = _default_account()
        self.account['initial_capital'] = initial_capital
        self.account['cash'] = initial_capital
        self._save()
        logging.info(f"模擬帳戶已重置，初始資金: {initial_capital:,.0f}")

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

    def catch_up(self, generate_func=None):
        """
        補跟遺漏天數 — 如果上次更新是 N 天前，自動跑 N 次策略+更新。

        Args:
            generate_func: 產生每日推薦的函式 (無參數)，
                          預設為 data.daily_recommender.get_daily_recommendation
        Returns:
            int: 補跟了幾天
        """
        last_updated = self.account.get('last_updated', '')
        if not last_updated:
            return 0

        try:
            last_dt = datetime.strptime(last_updated[:10], '%Y-%m-%d').date()
        except Exception:
            return 0

        today = date.today()
        gap_days = (today - last_dt).days

        if gap_days <= 1:
            return 0  # 不需要補跟

        # 只補跟工作日 (跳過週末)
        import numpy as np
        trading_days = np.busday_count(last_dt, today) - 1  # 不含今天
        if trading_days <= 0:
            return 0

        logging.info(f"偵測到 {trading_days} 個遺漏交易日 ({last_dt} → {today})")

        if generate_func is None:
            try:
                from data.daily_recommender import get_daily_recommendation
                generate_func = get_daily_recommendation
            except ImportError:
                logging.error("無法載入 daily_recommender")
                return 0

        caught_up = 0
        for i in range(min(trading_days, 10)):  # 最多補 10 天
            try:
                generate_func()
                self.update()
                caught_up += 1
            except Exception as e:
                logging.warning(f"補跟第 {i+1} 天失敗: {e}")
                break

        logging.info(f"補跟完成: {caught_up}/{trading_days} 天")
        return caught_up

    def update(self, recommendation=None):
        """
        根據推薦更新持倉:
        1. 讀取最新推薦
        2. 新推薦股 → 進場
        3. 不在推薦的持倉 → 出場
        4. 更新即時價格和損益
        """
        # 讀取推薦
        if recommendation is None:
            if not os.path.exists(RECOMMENDATION_PATH):
                logging.error("找不到推薦檔案，請先執行 daily_recommender.py")
                return
            with open(RECOMMENDATION_PATH, 'r', encoding='utf-8') as f:
                recommendation = json.load(f)

        rec_date = recommendation.get('date', str(date.today()))
        recs = recommendation.get('recommendations', [])
        rec_tickers = {r['ticker'] for r in recs}
        exits = {ex['ticker'] for ex in recommendation.get('exits', [])}
        exposure = recommendation.get('exposure', 1.0)
        strategy_name = recommendation.get('strategy', 'isaac')

        # 分離手動持倉和策略持倉 — update() 只處理策略持倉
        manual_positions = [p for p in self.account['positions'] if p.get('source') == 'manual']
        strategy_positions = [p for p in self.account['positions'] if p.get('source', 'strategy:isaac') != 'manual']

        # 目前策略持倉的 ticker（不包含手動）
        held_tickers = {p['ticker'] for p in strategy_positions}

        # 需要出場的 (不在推薦清單中 或 在 exits 中) — 只對策略持倉
        to_exit = (held_tickers - rec_tickers) | (held_tickers & exits)
        # 需要進場的 (在推薦清單但不在策略持倉中)
        to_enter = rec_tickers - held_tickers

        # 取得所有相關股票報價（包含手動持倉，只更新價格不出場）
        manual_tickers = {p['ticker'] for p in manual_positions}
        all_tickers = held_tickers | rec_tickers | manual_tickers
        quote_map = self._get_quotes(all_tickers)

        # 更新手動持倉價格（只更新，不進出場）
        for pos in manual_positions:
            ticker = pos['ticker']
            if ticker in quote_map:
                pos['current_price'] = quote_map[ticker]['close']
                pos['change_rate'] = quote_map[ticker].get('change_rate', 0)

        # === 出場（只處理策略持倉）===
        new_strategy_positions = []
        for pos in strategy_positions:
            ticker = pos['ticker']
            if ticker in to_exit:
                # 平倉
                current_price = quote_map.get(ticker, {}).get('close', pos.get('current_price', pos['entry_price']))
                if current_price <= 0:
                    current_price = pos['entry_price']

                # 台股手續費: 買賣各 0.1425%, 賣出證交稅 0.3%
                buy_fee = pos['entry_price'] * pos['shares'] * 0.001425
                sell_fee = current_price * pos['shares'] * 0.001425
                sell_tax = current_price * pos['shares'] * 0.003
                total_fees = buy_fee + sell_fee + sell_tax

                pnl_abs = (current_price - pos['entry_price']) * pos['shares'] - total_fees
                pnl_pct = pnl_abs / (pos['entry_price'] * pos['shares']) * 100

                closed = {
                    'ticker': ticker,
                    'name': pos.get('name', ''),
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'shares': pos['shares'],
                    'entry_date': pos['entry_date'],
                    'exit_date': rec_date,
                    'pnl_pct': round(pnl_pct, 2),
                    'pnl_abs': round(pnl_abs, 0),
                    'fees': round(total_fees, 0),
                    'hold_days': (datetime.strptime(rec_date, '%Y-%m-%d') - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days if pos.get('entry_date') else 0,
                }
                self.account['closed_trades'].append(closed)
                self.account['cash'] += current_price * pos['shares'] - sell_fee - sell_tax
                logging.info(f"  出場: {ticker} {pos.get('name', '')} | {pnl_pct:+.1f}% | ${pnl_abs:+,.0f}")
            else:
                # 更新報價
                if ticker in quote_map:
                    pos['current_price'] = quote_map[ticker]['close']
                    pos['change_rate'] = quote_map[ticker].get('change_rate', 0)
                new_strategy_positions.append(pos)

        # 合併: 手動持倉（不動） + 策略持倉（更新後）
        self.account['positions'] = manual_positions + new_strategy_positions

        # === 進場 ===
        if to_enter and self.account['cash'] > 0:
            # 每檔分配的資金
            max_positions = recommendation.get('summary', {}).get('max_concurrent', 10)
            current_count = len(self.account['positions'])
            available_slots = max(max_positions - current_count, 0)

            if available_slots > 0:
                # 按 score 排序要進場的
                enter_recs = sorted(
                    [r for r in recs if r['ticker'] in to_enter],
                    key=lambda x: x['score'],
                    reverse=True,
                )[:available_slots]

                # 計算每檔分配金額 (考慮曝險比例)
                total_equity = self._calc_equity(quote_map)
                per_stock = (total_equity * exposure) / max_positions

                for rec in enter_recs:
                    ticker = rec['ticker']
                    price = quote_map.get(ticker, {}).get('close', rec.get('close', 0))
                    if price <= 0:
                        continue

                    # 實際可用 = min(每檔預算, 剩餘現金)
                    budget = min(per_stock, self.account['cash'])
                    if budget < price:
                        continue  # 連一股都買不起

                    # 計算股數: 優先整張，剩餘用零股補滿
                    full_lots = int(budget / (price * 1000))  # 可買幾張
                    remaining = budget - (full_lots * 1000 * price)
                    odd_shares = int(remaining / price)        # 零股 (1股為單位)
                    shares = full_lots * 1000 + odd_shares

                    if shares <= 0:
                        continue

                    # 台股手續費: 買入 0.1425%
                    cost = price * shares
                    buy_commission = cost * 0.001425
                    total_cost = cost + buy_commission

                    self.account['cash'] -= total_cost
                    self.account['positions'].append({
                        'ticker': ticker,
                        'name': rec.get('name', ''),
                        'score': rec['score'],
                        'entry_price': price,
                        'current_price': price,
                        'shares': shares,
                        'entry_date': rec_date,
                        'change_rate': rec.get('change_rate', 0),
                        'source': f"strategy:{recommendation.get('strategy', 'isaac').lower().replace(' ', '_')}",
                    })
                    logging.info(f"  進場: {ticker} {rec.get('name', '')} | {price} x {shares}股 | 成本 {cost:,.0f}")

        # === 更新權益記錄 ===
        equity = self._calc_equity(quote_map)
        # 去重: 同一天只保留最新一筆
        daily_rec = {
            'date': rec_date,
            'equity': round(equity, 0),
            'cash': round(self.account['cash'], 0),
            'positions_value': round(equity - self.account['cash'], 0),
            'n_positions': len(self.account['positions']),
            'return_pct': round((equity / self.account['initial_capital'] - 1) * 100, 2),
        }
        existing = [i for i, r in enumerate(self.account['daily_equity']) if r.get('date') == rec_date]
        if existing:
            self.account['daily_equity'][existing[-1]] = daily_rec
        else:
            self.account['daily_equity'].append(daily_rec)

        self._save()
        logging.info(f"模擬交易更新完成 | 權益: {equity:,.0f} | 持倉: {len(self.account['positions'])}檔 | 現金: {self.account['cash']:,.0f}")
        return self.get_status()

    def _calc_equity(self, quote_map=None):
        """計算目前總權益"""
        equity = self.account['cash']
        for pos in self.account['positions']:
            price = pos.get('current_price', pos['entry_price'])
            if quote_map and pos['ticker'] in quote_map:
                price = quote_map[pos['ticker']].get('close', price)
            equity += price * pos['shares']
        return equity

    def get_status(self):
        """取得持倉狀態"""
        positions = self.account['positions']
        closed = self.account['closed_trades']
        equity_hist = self.account['daily_equity']

        # 持倉損益
        total_unrealized = 0
        pos_details = []
        for p in positions:
            cur = p.get('current_price', p['entry_price'])
            # 含手續費的未實現損益 (買賣各 0.1425% + 賣出證交稅 0.3%)
            gross = (cur - p['entry_price']) * p['shares']
            est_fees = p['entry_price'] * p['shares'] * 0.001425 + cur * p['shares'] * (0.001425 + 0.003)
            pnl_abs = gross - est_fees
            pnl_pct = pnl_abs / (p['entry_price'] * p['shares']) * 100 if p['entry_price'] > 0 else 0
            total_unrealized += pnl_abs
            pos_details.append({
                **p,
                'pnl_pct': round(pnl_pct, 2),
                'pnl_abs': round(pnl_abs, 0),
            })

        # 已平倉統計
        total_realized = sum(t.get('pnl_abs', 0) for t in closed)
        n_wins = sum(1 for t in closed if t.get('pnl_pct', 0) > 0)
        win_rate = n_wins / len(closed) * 100 if closed else 0

        # 目前權益
        equity = self._calc_equity()
        return_pct = (equity / self.account['initial_capital'] - 1) * 100

        return {
            'equity': round(equity, 0),
            'cash': round(self.account['cash'], 0),
            'initial_capital': self.account['initial_capital'],
            'return_pct': round(return_pct, 2),
            'total_unrealized': round(total_unrealized, 0),
            'total_realized': round(total_realized, 0),
            'n_positions': len(positions),
            'n_closed_trades': len(closed),
            'win_rate': round(win_rate, 1),
            'positions': pos_details,
            'last_updated': self.account.get('last_updated', ''),
        }

    def _lookup_name(self, ticker):
        """Auto-lookup stock name. Priority: FinLab → SinoPac → yfinance."""
        # Method 1: FinLab security_categories (最完整的台股中文名)
        try:
            import re
            if re.match(r'^\d{4,6}[A-Za-z]?$', str(ticker)):
                from finlab import data
                cats = data.get('security_categories')
                if cats is not None and 'stock_id' in cats.columns:
                    cats_idx = cats.set_index('stock_id')
                    if ticker in cats_idx.index:
                        name = str(cats_idx.loc[ticker, 'name']) if 'name' in cats_idx.columns else ''
                        if name and name != 'nan':
                            return name
        except Exception:
            pass
        # Method 2: yfinance fallback
        try:
            from data.provider import get_data_provider
            provider = get_data_provider("auto", market_type="TW")
            info = provider.get_stock_info(ticker)
            name = info.get('name', '')
            if name and name != ticker:
                return name
        except Exception:
            pass
        return ticker

    # ------------------------------------------------------------------
    # Source-aware position management
    # ------------------------------------------------------------------
    def add_manual_position(self, ticker, name, entry_price, shares,
                            entry_date=None, note=""):
        """Add a manually-tracked position (source='manual')."""
        if entry_date is None:
            entry_date = str(date.today())
        elif hasattr(entry_date, 'strftime'):
            entry_date = entry_date.strftime('%Y-%m-%d')

        # Auto-lookup name if empty
        if not name:
            name = self._lookup_name(ticker)

        cost = entry_price * shares
        buy_commission = cost * 0.001425
        total_cost = cost + buy_commission

        self.account['positions'].append({
            'ticker': ticker,
            'name': name,
            'score': 0,
            'entry_price': entry_price,
            'current_price': entry_price,
            'shares': shares,
            'entry_date': entry_date,
            'change_rate': 0,
            'source': 'manual',
            'note': note,
        })
        self.account['cash'] -= total_cost
        self._save()
        logging.info(f"手動新增: {ticker} {name} | {entry_price} x {shares}股")

    def remove_position(self, ticker, source=None):
        """Remove a position, optionally filtering by source tag."""
        new_positions = []
        removed = False
        for p in self.account['positions']:
            pos_source = p.get('source', 'strategy:isaac')
            if p['ticker'] == ticker and not removed:
                if source is None or pos_source == source:
                    removed = True
                    # Return cash (simulate selling at current price)
                    cur = p.get('current_price', p['entry_price'])
                    sell_fee = cur * p['shares'] * 0.001425
                    sell_tax = cur * p['shares'] * 0.003
                    proceeds = cur * p['shares'] - sell_fee - sell_tax
                    self.account['cash'] += proceeds

                    # Record as closed trade
                    pnl_abs = (cur - p['entry_price']) * p['shares'] - (
                        p['entry_price'] * p['shares'] * 0.001425 + sell_fee + sell_tax
                    )
                    pnl_pct = pnl_abs / (p['entry_price'] * p['shares']) * 100 if p['entry_price'] > 0 else 0
                    self.account['closed_trades'].append({
                        'ticker': ticker,
                        'name': p.get('name', ''),
                        'entry_price': p['entry_price'],
                        'exit_price': cur,
                        'shares': p['shares'],
                        'entry_date': p.get('entry_date', ''),
                        'exit_date': str(date.today()),
                        'pnl_pct': round(pnl_pct, 2),
                        'pnl_abs': round(pnl_abs, 0),
                        'source': pos_source,
                    })
                    logging.info(f"平倉: {ticker} {p.get('name','')} | {pnl_pct:+.1f}%")
                    continue
            new_positions.append(p)
        self.account['positions'] = new_positions
        self._save()
        return removed

    def get_positions(self, source=None):
        """Get positions, optionally filtered by source tag.

        source can be:
          - None  -> all positions
          - 'manual' -> manual only
          - 'strategy:isaac' -> specific strategy
          - 'strategy:' -> prefix match (any strategy)
        """
        positions = self.account.get('positions', [])
        if source is None:
            return positions
        result = []
        for p in positions:
            pos_source = p.get('source', 'strategy:isaac')
            if source.endswith(':') and pos_source.startswith(source):
                result.append(p)
            elif pos_source == source:
                result.append(p)
        return result

    def format_status_text(self):
        """格式化持倉狀態為文字"""
        status = self.get_status()
        lines = []
        lines.append(f"{'='*45}")
        lines.append(f"  Paper Trading Status")
        lines.append(f"{'='*45}")

        # 帳戶總覽
        lines.append(f"\n  [帳戶總覽]")
        lines.append(f"  初始資金: {status['initial_capital']:>12,.0f}")
        lines.append(f"  目前權益: {status['equity']:>12,.0f}  ({status['return_pct']:+.2f}%)")
        lines.append(f"  可用現金: {status['cash']:>12,.0f}")
        lines.append(f"  未實現損益: {status['total_unrealized']:>+10,.0f}")
        lines.append(f"  已實現損益: {status['total_realized']:>+10,.0f}")
        lines.append(f"  已平倉: {status['n_closed_trades']} 筆 | 勝率: {status['win_rate']:.1f}%")

        # 持倉明細
        positions = status['positions']
        if positions:
            lines.append(f"\n  [持倉明細] ({len(positions)} 檔)")
            lines.append(f"  {'代碼':<6} {'名稱':<8} {'進場':>7} {'現價':>7} {'損益%':>7} {'損益$':>9}")
            lines.append(f"  {'-'*52}")
            for p in sorted(positions, key=lambda x: x.get('pnl_pct', 0), reverse=True):
                name = p.get('name', '')[:6]
                lines.append(
                    f"  {p['ticker']:<6} {name:<8} {p['entry_price']:>7.1f} "
                    f"{p.get('current_price', p['entry_price']):>7.1f} "
                    f"{p.get('pnl_pct', 0):>+6.1f}% {p.get('pnl_abs', 0):>+9,.0f}"
                )
        else:
            lines.append(f"\n  (目前無持倉)")

        lines.append(f"\n  更新: {status['last_updated'][:19] if status['last_updated'] else 'N/A'}")
        lines.append(f"{'='*45}")
        return '\n'.join(lines)

    def format_history_text(self, n=20):
        """格式化歷史交易"""
        closed = self.account['closed_trades'][-n:]
        if not closed:
            return "尚無已平倉交易記錄"

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  交易歷史 (最近 {len(closed)} 筆)")
        lines.append(f"{'='*60}")
        lines.append(f"  {'代碼':<6} {'名稱':<8} {'進場':>7} {'出場':>7} {'損益%':>7} {'天數':>4}")
        lines.append(f"  {'-'*48}")

        for t in reversed(closed):
            name = t.get('name', '')[:6]
            lines.append(
                f"  {t['ticker']:<6} {name:<8} {t['entry_price']:>7.1f} "
                f"{t['exit_price']:>7.1f} {t['pnl_pct']:>+6.1f}% {t.get('hold_days', 0):>4}d"
            )

        # 統計
        avg_pnl = sum(t['pnl_pct'] for t in closed) / len(closed) if closed else 0
        wins = sum(1 for t in closed if t['pnl_pct'] > 0)
        lines.append(f"  {'-'*48}")
        lines.append(f"  平均報酬: {avg_pnl:+.2f}% | 勝率: {wins}/{len(closed)} ({wins/len(closed)*100:.0f}%)")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == '__main__':
    trader = PaperTrader()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'status':
            print(trader.format_status_text())
        elif cmd == 'history':
            print(trader.format_history_text())
        elif cmd == 'reset':
            capital = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
            trader.reset(capital)
            print(f"帳戶已重置，初始資金: {capital:,.0f}")
        else:
            print(f"未知指令: {cmd}")
            print("用法: python paper_trader.py [status|history|reset]")
    else:
        # 預設: 執行更新
        status = trader.update()
        if status:
            print(trader.format_status_text())
