import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
import datetime
import re
import streamlit as st

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
        if "US" in self.market_type or not re.match(r'^\d{4,6}$', str(ticker)):
            return [str(ticker)]

        # 如果是台股數字代碼，嘗試 TW 和 TWO
        suffixes = [".TW", ".TWO"]
        return [f"{ticker}{suf}" for suf in suffixes]

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        tickers_to_try = self._format_ticker(ticker)
        df = pd.DataFrame()

        for t in tickers_to_try:
            try:
                # auto_adjust=False 確保 Open/High/Low/Close 不被調整，且 Adj Close 會獨立出來
                d = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
                if not d.empty:
                    # 處理 yfinance 可能回傳 MultiIndex columns 的問題
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)

                    # 確保有必要的欄位
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in d.columns for col in required_cols) and len(d) > 30:
                        df = d[required_cols] # 提取並統一欄位順序
                        df.index.name = 'Date' # 統一 Index 名稱
                        # 儲存成功抓到的 raw_ticker 供 info 抓取使用
                        self._last_successful_ticker = t
                        break
            except Exception as e:
                continue

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
        except:
            pass

        return info_dict

class FinlabProvider(BaseDataProvider):
    """
    Finlab API 實作 (概念驗證)
    從橫截面資料庫中抽取特定股票的 OHLCV 時間序列。
    """
    def __init__(self, api_token=""):
        self.api_token = api_token
        # 在實際環境中需要 finlab.login(api_token)

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        FinLab 實作:
        from finlab import data
        close = data.get('price:收盤價')[ticker]
        ... 組合出 OHLCV 格式
        """
        # TODO: 實作真實的 Finlab 資料串接
        st.warning(f"FinlabProvider 尚未完全實作歷史資料擷取: {ticker}")
        return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> dict:
        # TODO: 實作真實的 Finlab 基本面擷取
        return {'name': str(ticker), 'pe': 'N/A', 'eps': 'N/A', 'yield': 'N/A', 'raw_ticker': ticker}

class SinoPacProvider(BaseDataProvider):
    """
    永豐金 Shioaji API 實作 (預留)
    """
    def __init__(self, api_key="", secret_key=""):
        pass

    def get_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        st.warning(f"SinoPacProvider 尚未實作歷史資料擷取: {ticker}")
        return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> dict:
        return {'name': str(ticker), 'pe': 'N/A', 'eps': 'N/A', 'yield': 'N/A', 'raw_ticker': ticker}

# ==========================================
# Data Factory
# ==========================================
def get_data_provider(source_name: str, market_type: str = "TW", **kwargs) -> BaseDataProvider:
    """
    取得對應的資料中介層實體
    :param source_name: 'yfinance', 'finlab', 'sinopac'
    :param market_type: 'TW' 或 'US'
    """
    if source_name.lower() == 'finlab':
        return FinlabProvider(api_token=kwargs.get('api_token', ''))
    elif source_name.lower() == 'sinopac':
        return SinoPacProvider(api_key=kwargs.get('api_key', ''), secret_key=kwargs.get('secret_key', ''))
    else:
        # Default fallback is YFinance
        return YFinanceProvider(market_type=market_type)
