"""Tokenizer-based response-length measurement."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def count_response_tokens(texts: Iterable[str], tokenizer: object) -> NDArray[np.int64]:
    """Count response tokens without prompt text or model-added special tokens."""

    values = list(texts)
    encoded = tokenizer(
        values,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    return np.asarray([len(token_ids) for token_ids in input_ids], dtype=int)
