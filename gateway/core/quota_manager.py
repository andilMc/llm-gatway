"""Quota and rate limit management."""

import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QuotaManager:
    """
    Detects and handles rate limit and quota errors.

    Handles various error formats from different providers.
    """

    # Common rate limit error patterns
    RATE_LIMIT_PATTERNS = [
        r"rate.?limit",
        r"too.?many.?requests",
        r"429",
        r"throttl",
        r"quota.?exceeded",
    ]

    # Quota exhausted patterns
    QUOTA_PATTERNS = [
        r"quota.?exhausted",
        r"insufficient.?quota",
        r"billing.?limit",
        r"payment.?required",
        r"402",
    ]

    @classmethod
    def is_rate_limit_error(
        cls, error: Exception, response: Optional[Any] = None
    ) -> bool:
        """
        Check if an error is a rate limit error.

        Args:
            error: The exception that occurred
            response: Optional response object

        Returns:
            True if this is a rate limit error
        """
        error_str = str(error).lower()

        # Check error message patterns
        for pattern in cls.RATE_LIMIT_PATTERNS:
            if re.search(pattern, error_str, re.IGNORECASE):
                return True

        # Check HTTP status code from response
        if response and hasattr(response, "status_code"):
            if response.status_code == 429:
                return True

        # Check error dict if available
        if isinstance(error, dict):
            error_type = error.get("error", {}).get("type", "").lower()
            error_code = error.get("error", {}).get("code", "")
            if error_type in ["rate_limit", "rate_limit_exceeded"] or error_code == 429:
                return True

        return False

    @classmethod
    def is_quota_exhausted_error(
        cls, error: Exception, response: Optional[Any] = None
    ) -> bool:
        """
        Check if an error indicates quota exhaustion.

        Args:
            error: The exception that occurred
            response: Optional response object

        Returns:
            True if quota is exhausted
        """
        error_str = str(error).lower()

        # Check error message patterns
        for pattern in cls.QUOTA_PATTERNS:
            if re.search(pattern, error_str, re.IGNORECASE):
                return True

        # Check HTTP status code from response
        if response and hasattr(response, "status_code"):
            if response.status_code in [402, 403]:
                return True

        # Check error dict if available
        if isinstance(error, dict):
            error_type = error.get("error", {}).get("type", "").lower()
            error_code = error.get("error", {}).get("code", "")
            if error_type in ["insufficient_quota", "quota_exceeded"] or error_code in [
                402,
                403,
            ]:
                return True

        return False

    @classmethod
    def extract_retry_after(
        cls, response: Optional[Any] = None, error: Optional[Exception] = None
    ) -> Optional[int]:
        """
        Extract retry-after value from response headers or error.

        Args:
            response: HTTP response object
            error: Exception object

        Returns:
            Seconds to wait before retry, or None if not specified
        """
        # Check response headers
        if response and hasattr(response, "headers"):
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return int(retry_after)
                except ValueError:
                    # Might be a date string, ignore for now
                    pass

        # Check for retry-after in error details
        if error and isinstance(error, dict):
            retry_after = error.get("error", {}).get("retry_after")
            if retry_after:
                return retry_after

        return None

    @classmethod
    def classify_error(
        cls, error: Exception, response: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Classify an error and provide handling recommendations.

        Returns:
            Dict with error classification and recommended action
        """
        result = {
            "is_rate_limit": False,
            "is_quota_exhausted": False,
            "retry_after": None,
            "action": "retry",  # retry, fallback, abort
            "message": str(error),
        }

        if cls.is_quota_exhausted_error(error, response):
            result["is_quota_exhausted"] = True
            result["action"] = "switch_key"
            return result

        if cls.is_rate_limit_error(error, response):
            result["is_rate_limit"] = True
            result["retry_after"] = cls.extract_retry_after(response, error)
            result["action"] = "retry"
            return result

        # Check for other transient errors
        if response and hasattr(response, "status_code"):
            status = response.status_code
            if status in [500, 502, 503, 504]:
                result["action"] = "retry"
                result["message"] = f"Server error {status}"
            elif status == 408:
                result["action"] = "retry"
                result["message"] = "Request timeout"
            elif status >= 400:
                result["action"] = "abort"

        return result
