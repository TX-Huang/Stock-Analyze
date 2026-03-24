"""Retry decorator and circuit breaker for resilient API calls."""
import time
import logging
import functools
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def retry(max_attempts=3, backoff=2.0, exceptions=(Exception,), on_failure=None):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        backoff: Backoff multiplier (delay = backoff^attempt seconds)
        exceptions: Tuple of exception types to catch
        on_failure: Optional callback(exception, attempt) called on each failure
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if on_failure:
                        on_failure(e, attempt)
                    if attempt < max_attempts:
                        delay = backoff ** attempt
                        logger.warning(
                            f"[Retry] {func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Retry] {func.__name__} failed all {max_attempts} attempts: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, requests blocked
        HALF_OPEN: Testing if service recovered
    """

    def __init__(self, failure_threshold=5, recovery_timeout=600, name="default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds
        self.name = name
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._on_open_callbacks = []

    @property
    def state(self):
        if self._state == "OPEN":
            # Check if recovery timeout has passed
            if (datetime.now() - self._last_failure_time).total_seconds() > self.recovery_timeout:
                self._state = "HALF_OPEN"
                logger.info(f"[CircuitBreaker:{self.name}] OPEN -> HALF_OPEN (recovery timeout elapsed)")
        return self._state

    def on_open(self, callback):
        """Register callback for when circuit opens (e.g., send Telegram alert)."""
        self._on_open_callbacks.append(callback)

    def record_success(self):
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failure_count = 0
            logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN -> CLOSED (service recovered)")

    def record_failure(self, exception=None):
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            if self._state != "OPEN":
                self._state = "OPEN"
                logger.error(
                    f"[CircuitBreaker:{self.name}] -> OPEN after {self._failure_count} failures"
                )
                for cb in self._on_open_callbacks:
                    try:
                        cb(self.name, self._failure_count, exception)
                    except Exception:
                        pass

    def __call__(self, func):
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        return wrapper


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass
