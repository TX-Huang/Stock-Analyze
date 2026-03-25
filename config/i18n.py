"""
Internationalization (i18n) Module
===================================
Supports: zh-TW (繁體中文, default), en (English), zh-CN (简体中文)

Usage::

    from config.i18n import t
    t('war_room.title')           # returns localized string
    t('war_room.analyzing', ticker='2330')  # variable substitution
"""

from config.locales import LOCALES, LOCALE_LABELS

DEFAULT_LOCALE = 'zh-TW'
SUPPORTED_LOCALES = list(LOCALES.keys())


def _get_session_state():
    """Return st.session_state if Streamlit is running, else a module-level dict."""
    try:
        import streamlit as st
        # Access .session_state to verify runtime is active
        _ = st.session_state
        return st.session_state
    except Exception:
        return _FALLBACK_STATE


# Module-level fallback when Streamlit is not running (e.g. tests, CLI).
_FALLBACK_STATE: dict = {}


def get_locale() -> str:
    """Return the current locale code (e.g. 'zh-TW', 'en', 'zh-CN')."""
    state = _get_session_state()
    return state.get('locale', DEFAULT_LOCALE)


def set_locale(locale: str) -> None:
    """Set the current locale.  *locale* must be one of SUPPORTED_LOCALES."""
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale '{locale}'. Choose from: {SUPPORTED_LOCALES}"
        )
    state = _get_session_state()
    state['locale'] = locale


def t(key: str, **kwargs) -> str:
    """Translate *key* (dot-notation) using the current locale.

    Falls back to zh-TW (the default) if the key is missing in the
    active locale.  If the key is missing everywhere, the raw key is
    returned so the UI never shows a ``KeyError``.

    Variable substitution is supported via keyword arguments::

        t('war_room.analyzing', ticker='2330')
        # -> 'AI 正在分析 2330...'
    """
    locale = get_locale()
    strings = LOCALES.get(locale, LOCALES[DEFAULT_LOCALE])
    value = strings.get(key)

    # Fallback chain: current locale -> zh-TW -> raw key
    if value is None and locale != DEFAULT_LOCALE:
        value = LOCALES[DEFAULT_LOCALE].get(key)
    if value is None:
        return key

    if kwargs:
        try:
            value = value.format(**kwargs)
        except (KeyError, IndexError):
            pass  # Return unformatted string rather than crash

    return value


def render_language_selector() -> None:
    """Render a compact language selector in the Streamlit sidebar."""
    import streamlit as st

    options = SUPPORTED_LOCALES
    labels = [LOCALE_LABELS[loc] for loc in options]
    current = get_locale()
    current_idx = options.index(current) if current in options else 0

    choice = st.selectbox(
        '\U0001f310',  # globe emoji as label
        options=options,
        format_func=lambda loc: LOCALE_LABELS.get(loc, loc),
        index=current_idx,
        key='_i18n_language_selector',
    )
    if choice != current:
        set_locale(choice)
        st.rerun()
