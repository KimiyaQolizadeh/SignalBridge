"""Configurable estimated OpenAI pricing used for internal observability.

Prices are USD per one million tokens and are estimates that must be reviewed
when provider pricing changes. They are deliberately static: pipeline runs never
perform network requests to discover pricing.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    cached_input: Decimal
    output: Decimal


MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-4.1": ModelPrice(Decimal("2.00"), Decimal("0.50"), Decimal("8.00")),
    "gpt-4.1-mini": ModelPrice(Decimal("0.40"), Decimal("0.10"), Decimal("1.60")),
    "gpt-5.4": ModelPrice(Decimal("2.50"), Decimal("0.25"), Decimal("15.00")),
    "gpt-5.4-mini": ModelPrice(Decimal("0.75"), Decimal("0.075"), Decimal("4.50")),
    "gpt-5.5": ModelPrice(Decimal("5.00"), Decimal("0.50"), Decimal("30.00")),
    "text-embedding-3-small": ModelPrice(Decimal("0.02"), Decimal("0"), Decimal("0")),
}
MILLION = Decimal("1000000")


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> Decimal | None:
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    uncached = max(0, input_tokens - cached_input_tokens)
    return (
        Decimal(uncached) * price.input
        + Decimal(cached_input_tokens) * price.cached_input
        + Decimal(output_tokens) * price.output
    ) / MILLION
