import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
import datetime
import time as _time
import re
import logging
import threading
from collections import OrderedDict

def sanitize_dataframe(df: pd.DataFrame, source_name: str = "Unknown") -> pd.DataFrame:
    """
    強制淨化 DataFrame：源頭防呆機制
    針對 FinLab 和其他 API 來源，強制將欄位型態和 Index 統一，
    避免 Pandas 1.5 到 2.0+ 的 CategoricalDtype 或是 float/int 混合報錯。
    """
    if df is None or df.empty:
        return df


    # 1. 強制對齊欄位型態 (Columns)
    if isinstance(df.columns, pd.CategoricalIndex):
        logging.info(f"[{source_name}] 偵測到 CategoricalIndex，強制洗為 string")
        df.columns = df.columns.astype(str)
    elif df.columns.dtype != 'object':
        try:
            df.columns = df.columns.astype(str)
        except (TypeError, ValueError) as e:
            logging.warning(f"[{source_name}] 欄位型態轉換失敗: {e}")

    # 2. 強制對齊索引型態 (Index) - 確保是時間格式
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logging.warning(f"[{source_name}] Index 無法轉換為 DatetimeIndex: {e}")

    return df

class BaseDataProvider(ABC):
    """
    虛擬中介層 (Data Adapter) 基礎類別
    所有實作的 API Provider 都必須確保回傳統一格式的資料。
    """

    @abstractmethod
    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        取得歷史 K 線資料
        :param ticker: 股票代碼 (純代碼，不含 .TW 結尾)
        :param period: 回溯時間 (ex: '1mo', '1y', '5y')
        :param interval: K 線週期 (ex: '1d', '1wk')
        :return: pd.DataFrame 必須包含 ['Open', 'High', 'Low', 'Close', 'Volume'] 且 Index 為 DatetimeIndex
        """
        pass

    @abstractmethod
    def get_stock_info(self, ticker: str) -> dict:
        """
        取得基本面/基本資料
        :param ticker: 股票代碼
        :return: dict 包含 {'name', 'pe', 'eps', 'yield'} 等統一鍵值
        """
        pass

class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance API 實作
    自動處理美股、台股後綴 (.TW, .TWO) 邏輯。
    """
    def __init__(self, market_type="TW"):
        self.market_type = market_type

    def _format_ticker(self, ticker: str) -> list:
        # 如果是美股，直接回傳
        if "US" in self.market_type:
            return [str(ticker)]

        # 台股代碼: 純數字(0050, 2330) 或 數字+字母(00631L, 00632R)
        if re.match(r'^\d{4,6}[A-Za-z]?$', str(ticker)):
            suffixes = [".TW", ".TWO"]
            return [f"{ticker}{suf}" for suf in suffixes]

        # 其他格式直接回傳
        return [str(ticker)]

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    
        tickers_to_try = self._format_ticker(ticker)
        df = pd.DataFrame()

        for t in tickers_to_try:
            try:
                # 使用預設 auto_adjust=True，相容新版 yfinance
                d = yf.download(t, period=period, interval=interval, progress=False)
                logging.info(f"[YFinance] {t}: shape={d.shape}, columns={list(d.columns)}, empty={d.empty}")

                if not d.empty:
                    # 處理 yfinance 可能回傳 MultiIndex columns 的問題
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                    # 去除可能的重複欄位名 (新版 yfinance MultiIndex 攤平後可能產生)
                    d = d.loc[:, ~d.columns.duplicated()]

                    # 確保有必要的欄位
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing = [c for c in required_cols if c not in d.columns]
                    if missing:
                        logging.warning(f"[YFinance] {t}: 缺少欄位 {missing}, 實際欄位: {list(d.columns)}")
                        continue

                    if len(d) > 30:
                        df = d[required_cols] # 提取並統一欄位順序
                        df.index.name = 'Date' # 統一 Index 名稱
                        # 儲存成功抓到的 raw_ticker 供 info 抓取使用
                        self._last_successful_ticker = t
                        logging.info(f"[YFinance] {t}: 成功取得 {len(df)} 筆資料")
                        break
                    else:
                        logging.warning(f"[YFinance] {t}: 資料筆數不足 ({len(d)} <= 30)")
                else:
                    logging.warning(f"[YFinance] {t}: 下載結果為空")
            except Exception as e:
                logging.error(f"[YFinance] {t}: 下載失敗 - {type(e).__name__}: {e}")
                continue

        if df.empty:
            logging.error(f"[YFinance] 所有嘗試皆失敗: {tickers_to_try}")
        return df

    def get_stock_info(self, ticker: str) -> dict:
        info_dict = {'name': str(ticker), 'pe': 'N/A', 'eps': 'N/A', 'yield': 'N/A', 'raw_ticker': ticker}

        # 嘗試使用剛剛成功抓到歷史資料的 ticker (確保後綴正確)
        target_ticker = getattr(self, '_last_successful_ticker', self._format_ticker(ticker)[0])
        info_dict['raw_ticker'] = target_ticker

        try:
            stock = yf.Ticker(target_ticker)
            info = stock.info

            # 名稱處理
            info_dict['name'] = info.get('longName', str(ticker))
            info_dict['pe'] = info.get('trailingPE', 'N/A')
            info_dict['eps'] = info.get('trailingEps', 'N/A')

            dy = info.get('dividendYield', None)
            if dy is not None:
                info_dict['yield'] = f"{dy*100:.2f}%"
        except (ConnectionError, TimeoutError, AttributeError, KeyError) as e:
            logging.warning(f"[YFinance] {target_ticker}: 取得基本面資訊失敗 - {type(e).__name__}: {e}")

        return info_dict

class FinlabProvider(BaseDataProvider):
    """
    Finlab API 實作
    從橫截面資料庫中抽取特定股票的 OHLCV 時間序列。
    """
    def __init__(self, api_token=""):
        self.api_token = api_token
        if api_token:
            try:
                import finlab
                finlab.login(api_token)
                logging.info("[FinLab] 登入成功")
            except Exception as e:
                logging.warning(f"[FinLab] 登入失敗: {e}")

    def _period_to_days(self, period: str) -> int:
        """將 period 字串轉換為天數"""
        period_map = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
        }
        return period_map.get(period, 730)

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        FinLab 實作: 從橫截面資料取出單一股票的 OHLCV 時間序列。
        """
        try:
            from finlab import data

            ticker_str = str(ticker).strip()

            close = data.get('price:收盤價')
            open_ = data.get('price:開盤價')
            high = data.get('price:最高價')
            low = data.get('price:最低價')
            vol = data.get('price:成交股數')

            # 確保 CategoricalIndex 轉為 string 以便比對
            for df_tmp in [close, open_, high, low, vol]:
                if isinstance(df_tmp.columns, pd.CategoricalIndex):
                    df_tmp.columns = df_tmp.columns.astype(str)

            if ticker_str not in close.columns:
                logging.warning(f"[FinLab] 股票 {ticker_str} 不存在於資料中")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Open': open_[ticker_str],
                'High': high[ticker_str],
                'Low': low[ticker_str],
                'Close': close[ticker_str],
                'Volume': vol[ticker_str],
            })
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            df = df.dropna(subset=['Close'])

            # 依 period 參數截取時間範圍
            days = self._period_to_days(period)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]

            # 週線重新取樣（不使用 resample='D'）
            if interval == '1wk':
                df = df.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                }).dropna(subset=['Close'])

            return df

        except Exception as e:
            logging.warning(f"[FinLab] 取得 {ticker} 歷史資料失敗: {e}")
            return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> dict:
        """從 FinLab 取得基本面資訊"""
        info_dict = {
            'name': str(ticker),
            'pe': 'N/A',
            'eps': 'N/A',
            'yield': 'N/A',
            'raw_ticker': ticker,
        }
        try:
            from finlab import data

            ticker_str = str(ticker).strip()

            # 公司基本資訊（名稱）
            try:
                company_info = data.get('company_basic_info')
                if isinstance(company_info.columns, pd.CategoricalIndex):
                    company_info.columns = company_info.columns.astype(str)
                if '公司簡稱' in company_info.columns:
                    row = company_info[company_info.index.astype(str) == ticker_str]
                    if not row.empty:
                        info_dict['name'] = str(row['公司簡稱'].iloc[0])
                elif ticker_str in company_info.index.astype(str).tolist():
                    # company_basic_info 格式可能是 stock_id 為 index
                    row = company_info.loc[company_info.index.astype(str) == ticker_str]
                    if not row.empty and '公司簡稱' in row.columns:
                        info_dict['name'] = str(row['公司簡稱'].iloc[0])
            except Exception as e:
                logging.debug(f"[FinLab] 取得 {ticker_str} 公司名稱失敗: {e}")

            # 本益比
            try:
                pe_df = data.get('price_earning_ratio:本益比')
                if isinstance(pe_df.columns, pd.CategoricalIndex):
                    pe_df.columns = pe_df.columns.astype(str)
                if ticker_str in pe_df.columns:
                    pe_val = pe_df[ticker_str].dropna()
                    if not pe_val.empty:
                        info_dict['pe'] = round(float(pe_val.iloc[-1]), 2)
            except Exception as e:
                logging.debug(f"[FinLab] 取得 {ticker_str} 本益比失敗: {e}")

            # EPS（每股盈餘）
            try:
                eps_df = data.get('fundamental_features:每股稅後淨利')
                if isinstance(eps_df.columns, pd.CategoricalIndex):
                    eps_df.columns = eps_df.columns.astype(str)
                if ticker_str in eps_df.columns:
                    eps_val = eps_df[ticker_str].dropna()
                    if not eps_val.empty:
                        info_dict['eps'] = round(float(eps_val.iloc[-1]), 2)
            except Exception as e:
                logging.debug(f"[FinLab] 取得 {ticker_str} EPS 失敗: {e}")

        except Exception as e:
            logging.warning(f"[FinLab] 取得 {ticker} 基本面資訊失敗: {e}")

        return info_dict

class SinoPacProvider(BaseDataProvider):
    """
    永豐金 Shioaji API 實作
    使用 Shioaji SDK 取得台股歷史 K 線資料。
    """
    _api_instance = None  # 類別層級快取，避免重複登入
    _logged_in = False
    _last_connected = None  # 上次連線時間戳
    _RECONNECT_INTERVAL = 4 * 3600  # 4 小時強制重連
    _lock = threading.Lock()  # 類別層級鎖，保護 _api_instance 等共享狀態

    def __init__(self, api_key="", secret_key="", **kwargs):

        self.api_key = api_key
        self.secret_key = secret_key
        self._cache = OrderedDict()  # {(ticker, period, interval): (df, timestamp)}
        self._cache_lock = threading.Lock()  # 實例層級鎖，保護 _cache 存取
        self._cache_ttl = 3600  # 1 小時快取
        self._cache_max = 200  # 最大快取數量

    def _get_api(self):
        """取得或建立 Shioaji API 連線（全域快取 + 健康檢查）"""
        with SinoPacProvider._lock:
            if SinoPacProvider._api_instance is not None:
                # 健康檢查 1: 超過 4 小時強制重連
                if (SinoPacProvider._last_connected is not None
                        and _time.time() - SinoPacProvider._last_connected > SinoPacProvider._RECONNECT_INTERVAL):
                    logging.info("[SinoPac] 連線超過 4 小時，強制重連")
                    self._reset_api()
                else:
                    # 健康檢查 2: 嘗試簡單操作確認連線仍有效
                    try:
                        _ = SinoPacProvider._api_instance.Contracts.Stocks
                        return SinoPacProvider._api_instance
                    except Exception as e:
                        logging.warning(f"[SinoPac] 連線健康檢查失敗 ({e})，重新連線")
                        self._reset_api()

            if not self.api_key or not self.secret_key:
                logging.warning("[SinoPac] API 金鑰未設定")
                return None

            try:
                import shioaji as sj
                api = sj.Shioaji()
                api.login(self.api_key, self.secret_key)
                SinoPacProvider._api_instance = api
                SinoPacProvider._logged_in = True
                SinoPacProvider._last_connected = _time.time()
                logging.info("[SinoPac] 登入成功")
                return api
            except Exception as e:
                logging.error(f"[SinoPac] 登入失敗: {e}")
                return None

    @classmethod
    def _reset_api(cls):
        """重設 API 連線（不 logout，因為可能已經斷線）
        Note: Caller is expected to already hold cls._lock when called from _get_api().
              Direct callers (e.g. logout) should acquire the lock themselves.
        """
        try:
            if cls._api_instance is not None:
                cls._api_instance.logout()
        except Exception as e:
            logging.debug(f"[SinoPac] 重設 API 時登出失敗 (可能已斷線): {e}")
        cls._api_instance = None
        cls._logged_in = False
        cls._last_connected = None

    def _evict_cache(self):
        """快取超過上限時，先清除過期項目，再淘汰最舊的。
        Note: Caller must hold self._cache_lock."""
        if len(self._cache) <= self._cache_max:
            return
        # Phase 1: 清除已過 TTL 的項目
        now = _time.time()
        expired_keys = [k for k, (_, ts) in self._cache.items() if now - ts >= self._cache_ttl]
        for k in expired_keys:
            del self._cache[k]
        # Phase 2: 仍超過上限，淘汰最舊的
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _period_to_dates(self, period: str):
        """將 period 字串轉換為 (start_date, end_date)"""
        end = datetime.date.today()
        period_map = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
        }
        days = period_map.get(period, 730)
        start = end - datetime.timedelta(days=days)
        return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

    def _find_contract(self, api, ticker: str):
        """在 TSE / OTC 中查找股票合約"""
        ticker_str = str(ticker)
        # 先嘗試上市 (TSE)
        try:
            contract = api.Contracts.Stocks["TSE"][ticker_str]
            if contract is not None:
                return contract
        except (KeyError, AttributeError):
            pass
        # 再嘗試上櫃 (OTC)
        try:
            contract = api.Contracts.Stocks["OTC"][ticker_str]
            if contract is not None:
                return contract
        except (KeyError, AttributeError):
            pass
        return None

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:

        # 快取檢查
        cache_key = (ticker, period, interval)
        with self._cache_lock:
            if cache_key in self._cache:
                df, ts = self._cache[cache_key]
                if _time.time() - ts < self._cache_ttl:
                    self._cache.move_to_end(cache_key)  # LRU: 標記為最近使用
                    logging.info(f"[SinoPac] {ticker}: 使用快取資料")
                    return df
                else:
                    del self._cache[cache_key]  # 過期，移除

        api = self._get_api()
        if api is None:
            logging.warning(f"[SinoPac] {ticker}: API 未連線，跳過")
            return pd.DataFrame()

        try:
            contract = self._find_contract(api, ticker)
            if contract is None:
                logging.warning(f"[SinoPac] {ticker}: 找不到合約")
                return pd.DataFrame()

            start_date, end_date = self._period_to_dates(period)
            kbars = api.kbars(contract, start=start_date, end=end_date)

            if not kbars or not kbars.ts:
                logging.warning(f"[SinoPac] {ticker}: 無 K 線資料")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Open': kbars.Open,
                'High': kbars.High,
                'Low': kbars.Low,
                'Close': kbars.Close,
                'Volume': kbars.Volume,
            }, index=pd.to_datetime(kbars.ts))
            df.index.name = 'Date'

            # Shioaji 回傳分鐘級資料，需 resample 為日 K
            if interval == '1d' and len(df) > 0:
                df = df.resample('D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum',
                }).dropna()
            elif interval == '1wk' and len(df) > 0:
                df = df.resample('W').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum',
                }).dropna()

            if len(df) > 30:
                with self._cache_lock:
                    self._cache[cache_key] = (df, _time.time())
                    self._evict_cache()
                logging.info(f"[SinoPac] {ticker}: 成功取得 {len(df)} 筆資料")
                return df
            else:
                logging.warning(f"[SinoPac] {ticker}: 資料筆數不足 ({len(df)})")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"[SinoPac] {ticker}: 取得歷史資料失敗 - {type(e).__name__}: {e}")
            return pd.DataFrame()

    def get_snapshots(self, tickers):
        """取得即時快照資料"""
        api = self._get_api()
        if api is None:
            return []
        try:
            contracts = []
            for t in tickers:
                c = self._find_contract(api, t)
                if c is not None:
                    contracts.append(c)
            if not contracts:
                return []
            snapshots = api.snapshots(contracts)
            return snapshots
        except Exception as e:
            logging.error(f"[SinoPac] get_snapshots 失敗: {e}")
            return []

    @classmethod
    def logout(cls):
        """登出並重設 API 連線"""
        if cls._api_instance is not None:
            try:
                cls._api_instance.logout()
            except Exception as e:
                logging.warning(f"[SinoPac] 登出時發生錯誤: {e}")
            cls._api_instance = None
            cls._logged_in = False
            logging.info("[SinoPac] 已登出")

    def get_stock_info(self, ticker: str) -> dict:
        """從合約取得基本資訊"""
        info_dict = {'name': str(ticker), 'pe': 'N/A', 'eps': 'N/A', 'yield': 'N/A', 'raw_ticker': ticker}
        api = self._get_api()
        if api is None:
            return info_dict
        try:
            contract = self._find_contract(api, ticker)
            if contract:
                info_dict['name'] = getattr(contract, 'name', str(ticker))
                info_dict['raw_ticker'] = ticker
        except (ConnectionError, AttributeError, KeyError) as e:
            logging.warning(f"[SinoPac] {ticker}: 取得基本面資訊失敗 - {type(e).__name__}: {e}")
        return info_dict


class SinoPacWithFallback(BaseDataProvider):
    """
    永豐金優先 + YFinance 備援
    台股預設使用 Shioaji，失敗時自動 fallback 到 YFinance。
    """
    def __init__(self, market_type="TW", api_key="", secret_key=""):
        self.primary = SinoPacProvider(api_key=api_key, secret_key=secret_key)
        self.fallback = YFinanceProvider(market_type=market_type)
        self.market_type = market_type

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    
        # 嘗試永豐金
        df = self.primary.get_historical_data(ticker, period, interval)
        if not df.empty:
            return df
        # Fallback 到 YFinance
        logging.info(f"[Fallback] {ticker}: 永豐金無資料，改用 YFinance")
        return self.fallback.get_historical_data(ticker, period, interval)

    def get_stock_info(self, ticker: str) -> dict:
        info = self.primary.get_stock_info(ticker)
        if info.get('name', '') == str(ticker):
            # 永豐金沒拿到名稱，試 YFinance
            return self.fallback.get_stock_info(ticker)
        return info


# ==========================================
# Data Factory
# ==========================================
_provider_cache = OrderedDict()  # {(source, market_type): instance}, LRU max 5
_PROVIDER_CACHE_MAX = 5

def get_data_provider(source_name: str = "auto", market_type: str = "TW", **kwargs) -> BaseDataProvider:
    """
    取得對應的資料中介層實體（含 singleton 快取）。
    預設行為 (auto):
      - 台股 → 永豐金 Shioaji 優先，YFinance 備援
      - 美股 → YFinance
    :param source_name: 'auto', 'yfinance', 'finlab', 'sinopac'
    :param market_type: 'TW' 或 'US'
    """
    cache_key = (source_name.lower(), market_type)

    # Finlab provider 不快取（每次可能帶不同 token）
    if source_name.lower() == 'finlab':
        return FinlabProvider(api_token=kwargs.get('api_token', ''))

    if cache_key in _provider_cache:
        _provider_cache.move_to_end(cache_key)  # LRU: mark as recently used
        return _provider_cache[cache_key]

    if source_name.lower() == 'sinopac':
        provider = SinoPacProvider(api_key=kwargs.get('api_key', ''), secret_key=kwargs.get('secret_key', ''))
    elif source_name.lower() == 'yfinance':
        provider = YFinanceProvider(market_type=market_type)
    else:
        # auto: 台股用永豐金+備援，美股用 YFinance
        if "US" in market_type:
            provider = YFinanceProvider(market_type=market_type)
        else:
            provider = SinoPacWithFallback(
                market_type=market_type,
                api_key=kwargs.get('api_key', ''),
                secret_key=kwargs.get('secret_key', ''),
            )

    # Evict oldest if over limit
    while len(_provider_cache) >= _PROVIDER_CACHE_MAX:
        evicted_key, _ = _provider_cache.popitem(last=False)
        logging.info(f"[ProviderCache] 快取已滿，淘汰最舊的: {evicted_key}")

    _provider_cache[cache_key] = provider
    return provider

def safe_finlab_sim(position, **kwargs):
    """
    防禦性封裝 finlab.backtest.sim。
    """

    import pandas as pd
    from finlab import backtest, data
    import shutil
    import os

    # [FIX] Pandas 2.3 + zoneinfo 在 Streamlit threading 環境下會觸發
    # SystemError: null argument to internal routine (in pd.Timestamp.now(tz=str))
    # Monkey-patch Finlab 的 _normalize_market_timestamp 使用 pytz 作為 fallback
    try:
        from finlab.market import Market
        _original_normalize = Market._normalize_market_timestamp

        @staticmethod
        def _safe_normalize(timestamp, timezone_name):
            if timestamp is None:
                try:
                    return pd.Timestamp.now(tz=timezone_name)
                except (SystemError, TypeError):
                    import pytz
                    return pd.Timestamp.now(tz=pytz.timezone(timezone_name))
            timestamp = pd.Timestamp(timestamp)
            try:
                if timestamp.tz is None:
                    return timestamp.tz_localize(timezone_name)
                return timestamp.tz_convert(timezone_name)
            except (SystemError, TypeError):
                import pytz
                tz = pytz.timezone(timezone_name)
                if timestamp.tz is None:
                    return timestamp.tz_localize(tz)
                return timestamp.tz_convert(tz)

        Market._normalize_market_timestamp = _safe_normalize
        logging.info("[safe_finlab_sim] Timezone patch applied")
    except Exception as patch_e:
        logging.warning(f"[safe_finlab_sim] Timezone patch skipped: {patch_e}")

    # 源頭防護：確保 position 本身沒問題
    # [FIX for 1-day hold bug] Do NOT convert position.columns to string if we can avoid it.
    # If we convert it to string, FinLab's internal `backtest.sim` alignment with `data.get('price:收盤價')` (which is Categorical)
    # will fail completely, resulting in immediate liquidation of all positions on the next day!
    # Instead, we force `position.columns` to be the EXACT same CategoricalIndex object as `price:收盤價`.
    try:
        raw_close = data.get('price:收盤價')
        if len(position.columns) == len(raw_close.columns):
            # Apply exactly identical CategoricalIndex to prevent Pandas `NotImplementedError`
            position.columns = raw_close.columns
            logging.info("成功對齊 position.columns 與 raw_close.columns (精確 CategoricalIndex 對接)。")
        else:
            # Fallback if somehow shapes differ
            logging.warning("position 與 raw_close 欄位數量不一致，嘗試轉換為 string Index 作為最後手段。")
            if isinstance(position.columns, pd.CategoricalIndex):
                position.columns = position.columns.astype(str)
    except Exception as e:
        logging.warning(f"對齊 CategoricalIndex 時發生錯誤: {e}")
        if isinstance(position.columns, pd.CategoricalIndex):
            position.columns = position.columns.astype(str)

    # Pre-emptive Cache Healing (毒樹果實預防)
    # The crash happens because FinLab internally tries to load 'security_categories' during sim()
    import pickle
    try:
        import requests as _requests
    except ImportError:
        _requests = None

    # 定義可恢復的錯誤類型 (網路問題，不需刪快取)
    _network_errors = (ConnectionError, TimeoutError)
    if _requests is not None:
        _network_errors = (ConnectionError, TimeoutError, _requests.exceptions.RequestException)

    # 定義快取損壞的錯誤類型 (需要刪快取)
    _corruption_errors = (pickle.UnpicklingError, EOFError, ValueError)

    home_finlab = os.path.expanduser("~/.finlab")
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            logging.info(f"執行前置快取健康檢查 (Pre-flight Cache Check, attempt {attempt + 1}/{max_retries + 1})...")
            _ = data.get('security_categories')
            break  # 檢查通過
        except _corruption_errors as check_e:
            logging.warning(f"快取損壞 ({type(check_e).__name__}): {check_e}")
            logging.info("啟動快取淨化流程 (刪除 ~/.finlab)...")
            if os.path.exists(home_finlab):
                try:
                    shutil.rmtree(home_finlab)
                    logging.info("物理刪除 ~/.finlab 成功！")
                except Exception as del_e:
                    logging.error(f"物理刪除 ~/.finlab 失敗: {del_e}")
            # 強制重新下載
            try:
                data.get('security_categories', force_download=True)
                logging.info("強制重新下載 security_categories 成功！")
                break
            except Exception as dl_e:
                logging.error(f"強制重新下載失敗: {dl_e}")
                if attempt >= max_retries:
                    logging.error("已達最大重試次數，放棄快取修復")
        except _network_errors as check_e:
            logging.warning(f"網路錯誤 ({type(check_e).__name__}): {check_e}，不刪除快取，重試中...")
            if attempt >= max_retries:
                logging.error("網路重試次數已達上限，繼續嘗試 sim()")
        except Exception as check_e:
            logging.warning(f"快取健康檢查失敗 (未知錯誤 {type(check_e).__name__}): {check_e}")
            # 未知錯誤：不刪快取，僅記錄
            break

    try:
        logging.info("開始執行 backtest.sim")
        report = backtest.sim(position, **kwargs)
        return report

    except Exception as e:
        logging.error(f"backtest.sim 發生未預期錯誤: {e}", exc_info=True)
        raise e
