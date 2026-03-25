"""Strategy upload sandbox for safe code execution."""
import ast
import logging
import signal
import threading

logger = logging.getLogger(__name__)

# Modules that are NEVER allowed in uploaded strategies
BLOCKED_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'ftplib', 'smtplib', 'ctypes', 'importlib', 'code', 'codeop',
    'compile', 'compileall', 'py_compile', 'zipimport',
    'multiprocessing', 'signal', 'resource', 'pathlib',
    '__import__', 'eval', 'exec', 'compile', 'globals', 'locals',
    'getattr', 'setattr', 'delattr', '__builtins__',
}

# Functions that are dangerous
BLOCKED_FUNCTIONS = {
    'eval', 'exec', 'compile', 'open', '__import__',
    'globals', 'locals', 'getattr', 'setattr', 'delattr',
    'exit', 'quit',
}

def validate_strategy_safety(source_code: str) -> tuple[bool, list[str]]:
    """Validate uploaded strategy code for safety.

    WARNING: AST validation alone is insufficient for fully sandboxing untrusted
    code. This provides a first layer of defense but should be combined with
    process-level isolation (e.g., Docker, seccomp) for production use.

    Returns:
        (is_safe, list_of_warnings)
    """
    warnings = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    # Pre-scan: detect string concatenation that could build dangerous names
    # e.g. "ev" + "al", "ex" + "ec", "im" + "port", "op" + "en"
    _DANGEROUS_CONCAT_TARGETS = {'eval', 'exec', 'import', 'open', '__import__', 'compile'}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Check if both sides are string constants
            left = getattr(node.left, 'value', None) if isinstance(node.left, ast.Constant) else None
            right = getattr(node.right, 'value', None) if isinstance(node.right, ast.Constant) else None
            if isinstance(left, str) and isinstance(right, str):
                combined = left + right
                if combined.lower() in _DANGEROUS_CONCAT_TARGETS:
                    warnings.append(f"❌ 禁止的字串拼接 (可能規避檢查): '{left}' + '{right}' = '{combined}'")

        # Block getattr() on __builtins__
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Name) and first_arg.id == '__builtins__':
                        warnings.append("❌ 禁止的呼叫: getattr(__builtins__, ...)")

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [alias.name for alias in node.names]

            if module and module.split('.')[0] in BLOCKED_IMPORTS:
                warnings.append(f"❌ 禁止的 import: {module}")
            for name in names:
                if name.split('.')[0] in BLOCKED_IMPORTS:
                    warnings.append(f"❌ 禁止的 import: {name}")

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_FUNCTIONS:
                warnings.append(f"❌ 禁止的函數呼叫: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_FUNCTIONS:
                warnings.append(f"❌ 禁止的方法呼叫: .{node.func.attr}()")

    is_safe = not any("❌" in w for w in warnings)
    return is_safe, warnings


def run_with_timeout(func, timeout_seconds=120, *args, **kwargs):
    """Run a function with a timeout (Windows-compatible using threads)."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Strategy execution timed out after {timeout_seconds} seconds")

    if exception[0] is not None:
        raise exception[0]

    return result[0]
