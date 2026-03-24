"""
Environment configuration (P1-6)
Supports development / production mode switching.
"""
import os
import logging

logger = logging.getLogger(__name__)

# Read from environment variable or default to 'development'
APP_ENV = os.environ.get('APP_ENV', 'development').lower()

# Validate
if APP_ENV not in ('development', 'staging', 'production'):
    logger.warning(f"Unknown APP_ENV='{APP_ENV}', falling back to 'development'")
    APP_ENV = 'development'


def is_development() -> bool:
    return APP_ENV == 'development'


def is_production() -> bool:
    return APP_ENV == 'production'


# Logging configuration per environment
LOG_LEVELS = {
    'development': logging.DEBUG,
    'staging': logging.INFO,
    'production': logging.WARNING,
}

# Feature flags per environment
FEATURES = {
    'development': {
        'debug_panels': True,
        'telegram_enabled': False,
        'verbose_logging': True,
        'show_experimental': True,
        'auto_trading_enabled': False,
    },
    'staging': {
        'debug_panels': True,
        'telegram_enabled': True,
        'verbose_logging': True,
        'show_experimental': True,
        'auto_trading_enabled': False,
    },
    'production': {
        'debug_panels': False,
        'telegram_enabled': True,
        'verbose_logging': False,
        'show_experimental': False,
        'auto_trading_enabled': True,
    },
}


def get_log_level() -> int:
    return LOG_LEVELS.get(APP_ENV, logging.INFO)


def get_feature(name: str) -> bool:
    """Get feature flag value for current environment."""
    env_features = FEATURES.get(APP_ENV, FEATURES['development'])
    return env_features.get(name, False)


def configure_logging():
    """Configure logging based on current environment."""
    level = get_log_level()
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    if APP_ENV == 'development':
        fmt = '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

    logging.basicConfig(level=level, format=fmt, force=True)
    logger.info(f"Environment: {APP_ENV}, Log level: {logging.getLevelName(level)}")
