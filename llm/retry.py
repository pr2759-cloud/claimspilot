"""Retry with exponential backoff for temporary API failures."""
import functools
import random
import time

import anthropic
import openai

# Errors worth retrying: the same request may succeed if we wait.
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,       # 429: too many requests
    anthropic.APITimeoutError,      # the request took too long
    anthropic.APIConnectionError,   # network problem
    anthropic.InternalServerError,  # 5xx: problem on the provider's side
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def with_retries(max_attempts: int = 4, base_delay: float = 1.0, max_delay: float = 20.0):
    """Decorator: retry the wrapped function on RETRYABLE_ERRORS."""

    def decorator(func):
        @functools.wraps(func)  # keeps the original function's name and docstring
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as e:
                    if attempt == max_attempts:
                        raise  # out of attempts: let the error reach the caller
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))  # 1, 2, 4, 8...
                    delay *= random.uniform(0.5, 1.0)  # jitter
                    print(f"[retry] {type(e).__name__} on attempt {attempt}, waiting {delay:.1f}s")
                    time.sleep(delay)

        return wrapper

    return decorator
