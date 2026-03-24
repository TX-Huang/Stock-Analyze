"""
Standard signal format + auto-discovery of strategies.
Strategies are discovered from:
  1. strategies/*.py (built-in)
  2. strategies/custom/*.py (user-uploaded)

Each strategy file can optionally define:
  STRATEGY_NAME = "我的策略"
  STRATEGY_DESCRIPTION = "基於均線的多空策略"

If not defined, the filename is used as the display name.
The file must contain a `run_strategy(api_token)` function to be recognized.
"""
import os
import ast
import logging

logger = logging.getLogger(__name__)

STANDARD_SIGNAL = {
    "strategy": "strategy_name",
    "date": "YYYY-MM-DD",
    "recommendations": [
        {
            "ticker": "2330",
            "name": "台積電",
            "action": "BUY",  # BUY / SELL / HOLD
            "score": 8.0,
            "reason": "VCP成形+外資買超",
            "close": 892.0,
            "volume": 25000,
        }
    ]
}

# ─── Built-in strategies with known metadata ───
_BUILTIN_STRATEGIES = [
    {"key": "isaac", "label": "Isaac V3.7", "source_tag": "strategy:isaac",
     "module": "strategies.isaac", "func": "run_isaac_strategy", "builtin": True},
]

# ─── Known filenames → display names (for built-in .py that don't have metadata) ───
_KNOWN_NAMES = {
    "minervini": "Minervini VCP",
    "edwards_magee": "Edwards & Magee",
    "elder": "Elder Triple Screen",
    "momentum": "Momentum 動能策略",
    "vcp": "VCP 波動收縮",
    "long_only": "純多策略",
    "long_short": "多空策略",
    "candlestick": "K線型態策略",
    "wfo": "WFO 自適應",
    "adam": "Adam 策略",
}

# Files to skip (not strategies, just utilities/tests)
_SKIP_FILES = {
    "__init__", "template", "advanced_test", "allocation_test",
    "decay_test", "monte_carlo", "opt_test", "rotation_test",
    "round3_test", "v37_validation",
}


def _extract_metadata_from_file(filepath):
    """Parse a .py file with AST to extract STRATEGY_NAME, STRATEGY_DESCRIPTION, and check for run_strategy()."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return None

    name = None
    description = None
    has_run_strategy = False
    has_run_isaac = False
    supports_stress = False
    func_name = None

    for node in ast.walk(tree):
        # Check for STRATEGY_NAME = "xxx"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == 'STRATEGY_NAME' and isinstance(node.value, (ast.Constant, ast.Str)):
                        name = node.value.value if isinstance(node.value, ast.Constant) else node.value.s
                    elif target.id == 'STRATEGY_DESCRIPTION' and isinstance(node.value, (ast.Constant, ast.Str)):
                        description = node.value.value if isinstance(node.value, ast.Constant) else node.value.s

        # Check for def run_*strategy*(...) functions
        if isinstance(node, ast.FunctionDef):
            fn = node.name
            if fn == 'run_strategy':
                has_run_strategy = True
                func_name = fn
                args = [a.arg for a in node.args.args]
                supports_stress = {'stop_loss', 'take_profit'}.issubset(set(args)) or node.args.kwarg is not None
            elif fn == 'run_isaac_strategy':
                has_run_isaac = True
                func_name = fn
            elif fn.startswith('run_') and 'strategy' in fn:
                # Matches run_minervini_strategy, run_elder_strategy, etc.
                has_run_strategy = True
                func_name = fn
                args = [a.arg for a in node.args.args]
                supports_stress = {'stop_loss', 'take_profit'}.issubset(set(args)) or node.args.kwarg is not None

    if not has_run_strategy and not has_run_isaac:
        return None

    return {
        'name': name,
        'description': description,
        'has_run_strategy': has_run_strategy,
        'has_run_isaac': has_run_isaac,
        'func_name': func_name if (has_run_strategy or has_run_isaac) else None,
        'supports_stress': supports_stress,
    }


def discover_strategies():
    """
    Auto-discover all available strategies.
    Returns list of strategy dicts:
      [{"key": "isaac", "label": "Isaac V3.7", "source_tag": "strategy:isaac",
        "module": "strategies.isaac", "func": "run_isaac_strategy",
        "builtin": True, "custom": False, "description": "...", "supports_stress": False}]
    """
    strategies = []

    # 1. Built-in strategies (always first)
    for s in _BUILTIN_STRATEGIES:
        strategies.append({**s, "custom": False, "description": "", "supports_stress": True})

    # 2. Scan strategies/*.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    strategies_dir = os.path.join(project_root, 'strategies')

    if os.path.isdir(strategies_dir):
        for fname in sorted(os.listdir(strategies_dir)):
            if not fname.endswith('.py'):
                continue
            key = fname[:-3]  # remove .py
            if key in _SKIP_FILES or key.startswith('_'):
                continue
            # Skip if already in built-in list
            if any(s['key'] == key for s in strategies):
                continue

            filepath = os.path.join(strategies_dir, fname)
            meta = _extract_metadata_from_file(filepath)
            if meta is None:
                continue

            label = meta['name'] or _KNOWN_NAMES.get(key, key.replace('_', ' ').title())
            func = meta.get('func_name') or ('run_isaac_strategy' if meta['has_run_isaac'] else 'run_strategy')

            strategies.append({
                "key": key,
                "label": label,
                "source_tag": f"strategy:{key}",
                "module": f"strategies.{key}",
                "func": func,
                "builtin": True,
                "custom": False,
                "description": meta.get('description', ''),
                "supports_stress": meta.get('supports_stress', False),
            })

    # 3. Scan strategies/custom/*.py (user-uploaded)
    custom_dir = os.path.join(strategies_dir, 'custom')
    if os.path.isdir(custom_dir):
        for fname in sorted(os.listdir(custom_dir)):
            if not fname.endswith('.py') or fname.startswith('_'):
                continue
            key = f"custom_{fname[:-3]}"
            filepath = os.path.join(custom_dir, fname)
            meta = _extract_metadata_from_file(filepath)
            if meta is None:
                continue

            label = meta['name'] or f"📂 {fname[:-3].replace('_', ' ').title()}"
            func = 'run_strategy'

            strategies.append({
                "key": key,
                "label": label,
                "source_tag": f"strategy:{key}",
                "module": None,  # custom modules are loaded dynamically
                "filepath": filepath,
                "func": func,
                "builtin": False,
                "custom": True,
                "description": meta.get('description', ''),
                "supports_stress": meta.get('supports_stress', False),
            })

    return strategies


# Auto-discover on import (cached at module level)
AVAILABLE_STRATEGIES = discover_strategies()
