"""Locale packages for i18n — zh_TW (default), en, zh_CN."""

from config.locales import zh_TW, en, zh_CN

LOCALES = {
    'zh-TW': zh_TW.STRINGS,
    'en': en.STRINGS,
    'zh-CN': zh_CN.STRINGS,
}

LOCALE_LABELS = {
    'zh-TW': '\U0001f1f9\U0001f1fc \u7e41\u9ad4\u4e2d\u6587',
    'en': '\U0001f1fa\U0001f1f8 English',
    'zh-CN': '\U0001f1e8\U0001f1f3 \u7b80\u4f53\u4e2d\u6587',
}

__all__ = ['LOCALES', 'LOCALE_LABELS']
