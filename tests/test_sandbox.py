"""Tests for utils/sandbox.py -- strategy safety validation."""
import pytest

from utils.sandbox import validate_strategy_safety


# ===================================================================
# Safe code
# ===================================================================

class TestSafeCode:
    def test_simple_arithmetic(self):
        code = "x = 1 + 2\nprint(x)"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is True
        assert len(warnings) == 0

    def test_pandas_numpy_allowed(self):
        code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3]})
result = np.mean(df['a'])
"""
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is True

    def test_finlab_strategy_code(self):
        code = """
import finlab
from finlab import data
from finlab.backtest import sim

close = data.get('price:收盤價')
ma20 = close.average(20)
position = (close > ma20).hold_until(close < ma20)
report = sim(position, upload=False)
"""
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is True

    def test_function_definition(self):
        code = """
def calculate_signal(df):
    return df['Close'].pct_change()
"""
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is True


# ===================================================================
# Blocked imports
# ===================================================================

class TestBlockedImports:
    @pytest.mark.parametrize("module", ["os", "sys", "subprocess", "shutil", "socket"])
    def test_blocked_stdlib_import(self, module):
        code = f"import {module}"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
        assert any(module in w for w in warnings)

    def test_blocked_from_import(self):
        code = "from os import path"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked_http(self):
        code = "import http.client"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked_urllib(self):
        code = "from urllib.request import urlopen"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked_ctypes(self):
        code = "import ctypes"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked_multiprocessing(self):
        code = "import multiprocessing"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False


# ===================================================================
# Blocked function calls
# ===================================================================

class TestBlockedFunctions:
    @pytest.mark.parametrize("func", ["eval", "exec", "compile"])
    def test_blocked_builtins(self, func):
        code = f'{func}("print(1)")'
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
        assert any(func in w for w in warnings)

    def test_blocked_open(self):
        code = 'f = open("/etc/passwd", "r")'
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked___import__(self):
        code = '__import__("os")'
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False

    def test_blocked_getattr(self):
        code = 'getattr(obj, "dangerous")'
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_syntax_error(self):
        code = "def broken(:"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
        assert any("Syntax error" in w for w in warnings)

    def test_empty_string(self):
        is_safe, warnings = validate_strategy_safety("")
        assert is_safe is True

    def test_multiple_violations(self):
        code = """
import os
import subprocess
eval("1+1")
exec("print('hi')")
"""
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
        assert len(warnings) >= 4

    def test_method_call_on_object(self):
        """Calling .eval() on an object should also be caught."""
        code = "df.eval('a + b')"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
        assert any("eval" in w for w in warnings)

    def test_safe_method_same_name_different_context(self):
        """Attribute access to 'eval' as a method is blocked regardless of object."""
        code = "result = my_obj.eval(expression)"
        is_safe, warnings = validate_strategy_safety(code)
        assert is_safe is False
