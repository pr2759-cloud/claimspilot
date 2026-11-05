"""Prices per million tokens, used to estimate the cost of each call.

Prices change, so look them up on each provider's pricing page and fill them in.
The keys must match the model names in your .env exactly.
"""

# model name -> (USD per 1M input tokens, USD per 1M output tokens)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # Anthropic, standard API rates (checked 2026-11-05)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens; $0.50/$2.50 beyond
    # OpenAI. Sources disagree ($10/$50 vs $5/$25): confirm on openai.com/api/pricing
    "gpt-6-astra": (10.00, 50.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the estimated cost in USD, or None if we don't know this model's price."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
