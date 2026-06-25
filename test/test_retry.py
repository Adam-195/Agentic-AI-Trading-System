"""
tests/test_retry.py
Tests for the retry decorator.
Run with: pytest test/test_retry.py -v
"""

import pytest
from unittest.mock import MagicMock

from utils.retry import retry


class TestRetry:
    def test_returns_result_on_first_success(self):
        mock_fn = MagicMock(return_value="ok")
        decorated = retry(max_attempts=3)(mock_fn)
        result = decorated()
        assert result == "ok"
        assert mock_fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        mock_fn = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "ok"])
        decorated = retry(max_attempts=3, delay=0)(mock_fn)
        result = decorated()
        assert result == "ok"
        assert mock_fn.call_count == 3

    def test_returns_none_after_max_attempts(self):
        mock_fn = MagicMock(side_effect=Exception("always fails"))
        decorated = retry(max_attempts=3, delay=0)(mock_fn)
        result = decorated()
        assert result is None
        assert mock_fn.call_count == 3

    def test_only_catches_specified_exceptions(self):
        mock_fn = MagicMock(side_effect=ValueError("wrong type"))
        decorated = retry(max_attempts=3, delay=0, exceptions=(TypeError,))(mock_fn)
        with pytest.raises(ValueError):
            decorated()

    def test_preserves_function_name(self):
        def my_function(): pass
        decorated = retry()(my_function)
        assert decorated.__name__ == "my_function"
