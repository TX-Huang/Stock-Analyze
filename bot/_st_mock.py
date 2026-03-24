"""
Streamlit Mock — 讓 scanner/ai_core 在 Bot 環境中正常運作
"""
import sys


class _MockSessionState:
    def __init__(self):
        self.dynamic_name_map = {}

    def __contains__(self, key):
        return hasattr(self, key)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return None


class _MockProgress:
    def progress(self, x):
        return self

    def empty(self):
        return self

    def text(self, x):
        return self


class _MockSt:
    session_state = _MockSessionState()

    @staticmethod
    def warning(msg): pass

    @staticmethod
    def info(msg): pass

    @staticmethod
    def error(msg): pass

    @staticmethod
    def progress(n):
        return _MockProgress()

    @staticmethod
    def empty():
        return _MockProgress()

    @staticmethod
    def set_page_config(**kwargs): pass

    @staticmethod
    def sidebar():
        return _MockSt()

    @staticmethod
    def columns(n):
        return [_MockSt()] * n

    @staticmethod
    def metric(*args, **kwargs): pass

    @staticmethod
    def write(*args, **kwargs): pass

    @staticmethod
    def markdown(*args, **kwargs): pass


def install_mock():
    """在 import streamlit 之前呼叫，注入 mock"""
    if 'streamlit' not in sys.modules:
        sys.modules['streamlit'] = _MockSt()
