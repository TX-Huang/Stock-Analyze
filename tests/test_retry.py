"""Tests for utils/retry.py -- retry decorator and CircuitBreaker."""
import time
from unittest.mock import MagicMock

import pytest

from utils.retry import retry, CircuitBreaker, CircuitBreakerOpen


# ===================================================================
# retry decorator
# ===================================================================

class TestRetry:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, backoff=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_succeeds_after_failures(self):
        attempts = []

        @retry(max_attempts=3, backoff=0.01)
        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return "done"

        assert flaky() == "done"
        assert len(attempts) == 3

    def test_max_attempts_exceeded(self):
        @retry(max_attempts=2, backoff=0.01)
        def always_fail():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            always_fail()

    def test_only_catches_specified_exceptions(self):
        @retry(max_attempts=3, backoff=0.01, exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raise_type_error()

    def test_on_failure_callback(self):
        callback = MagicMock()

        @retry(max_attempts=2, backoff=0.01, on_failure=callback)
        def fail_once():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            fail_once()

        assert callback.call_count == 2

    def test_backoff_timing(self):
        """Verify that backoff introduces a delay between attempts."""
        attempts = []

        @retry(max_attempts=2, backoff=0.1)
        def slow_fail():
            attempts.append(time.time())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            slow_fail()

        assert len(attempts) == 2
        # backoff^1 = 0.1s delay between attempt 1 and 2
        elapsed = attempts[1] - attempts[0]
        assert elapsed >= 0.05  # allow some tolerance

    def test_preserves_function_name(self):
        @retry(max_attempts=1, backoff=0.01)
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ===================================================================
# CircuitBreaker
# ===================================================================

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")
        assert cb.state == "CLOSED"

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "OPEN"

    def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, name="test")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, name="test")
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_decorator_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, name="test")

        @cb
        def guarded():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            guarded()

        # Now circuit is open
        with pytest.raises(CircuitBreakerOpen):
            guarded()

    def test_decorator_records_success(self):
        cb = CircuitBreaker(failure_threshold=5, name="test")

        @cb
        def ok():
            return 42

        assert ok() == 42
        assert cb._failure_count == 0

    def test_on_open_callback(self):
        cb = CircuitBreaker(failure_threshold=2, name="test")
        callback = MagicMock()
        cb.on_open(callback)

        cb.record_failure()
        cb.record_failure()

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"  # name
        assert args[1] == 2       # failure count

    def test_circuit_breaker_open_exception(self):
        exc = CircuitBreakerOpen("service down")
        assert str(exc) == "service down"
        assert isinstance(exc, Exception)
