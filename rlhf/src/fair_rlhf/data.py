"""HelpSteer2 normalization, prompt-level splitting, and data audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import math
import statistics
from typing import Any, Literal

from .token_count import count_response_tokens


SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


class DataAlignmentError(ValueError):
    """Raised when preference rows cannot be safely joined to pointwise ratings."""

    def __init__(self, report: AlignmentReport):
        self.report = report
        super().__init__(
            "HelpSteer2 alignment failed with "
            f"{report.fatal_issue_count} fatal issue(s)"
        )


@dataclass(frozen=True)
class NormalizedPair:
    """A preference pair joined to the pointwise helpfulness annotations."""

    prompt: str
    response_left: str
    response_right: str
    preference_label: int
    preference_strength: int
    helpfulness_left: float
    helpfulness_right: float
    source_split: str
    split: SplitName | None = None
    token_count_left: int | None = None
    token_count_right: int | None = None
    is_swapped: bool = False

    def __post_init__(self) -> None:
        _required_text(self.prompt, "prompt")
        _response_text(self.response_left, "response_left")
        _response_text(self.response_right, "response_right")
        if self.response_left == self.response_right:
            raise ValueError("a preference pair must contain two distinct responses")
        if self.preference_label not in (-1, 0, 1):
            raise ValueError("preference_label must be -1, 0, or +1")
        if self.preference_strength not in (0, 1, 2, 3):
            raise ValueError("preference_strength must be between 0 and 3")
        if (self.preference_label == 0) != (self.preference_strength == 0):
            raise ValueError(
                "preference_label must be zero exactly when preference_strength is zero"
            )
        _finite_number(self.helpfulness_left, "helpfulness_left")
        _finite_number(self.helpfulness_right, "helpfulness_right")
        if self.split not in (*SPLIT_NAMES, None):
            raise ValueError(f"unknown experimental split: {self.split}")
        if (self.token_count_left is None) != (self.token_count_right is None):
            raise ValueError("left and right token counts must be set together")
        for name, value in (
            ("token_count_left", self.token_count_left),
            ("token_count_right", self.token_count_right),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def prompt_id(self) -> str:
        return stable_prompt_id(self.prompt)

    @property
    def pair_id(self) -> str:
        return stable_pair_id(
            self.prompt,
            self.response_left,
            self.response_right,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized on-disk data contract."""

        return {
            "pair_id": self.pair_id,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "response_left": self.response_left,
            "response_right": self.response_right,
            "preference_label": self.preference_label,
            "preference_strength": self.preference_strength,
            "helpfulness_left": self.helpfulness_left,
            "helpfulness_right": self.helpfulness_right,
            "source_split": self.source_split,
            "split": self.split,
            "token_count_left": self.token_count_left,
            "token_count_right": self.token_count_right,
            "is_swapped": self.is_swapped,
        }


@dataclass(frozen=True)
class AlignmentReport:
    """Diagnostics from joining preference and pointwise HelpSteer2 rows."""

    pointwise_rows: int
    unique_pointwise_responses: int
    preference_rows: int
    aligned_pairs: int
    tie_pairs: int
    duplicate_pointwise_rows: int
    conflicting_pointwise_rows: int
    duplicate_preference_pairs: int
    identical_preference_response_rows: int
    invalid_pointwise_rows: int
    invalid_preference_rows: int
    missing_left_ratings: int
    missing_right_ratings: int
    source_split_mismatches: int

    @property
    def fatal_issue_count(self) -> int:
        return sum(
            (
                self.conflicting_pointwise_rows,
                self.duplicate_preference_pairs,
                self.invalid_pointwise_rows,
                self.invalid_preference_rows,
                self.missing_left_ratings,
                self.missing_right_ratings,
                self.source_split_mismatches,
            )
        )

    @property
    def is_valid(self) -> bool:
        return self.fatal_issue_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "fatal_issue_count": self.fatal_issue_count,
            "pointwise_rows": self.pointwise_rows,
            "unique_pointwise_responses": self.unique_pointwise_responses,
            "preference_rows": self.preference_rows,
            "aligned_pairs": self.aligned_pairs,
            "tie_pairs": self.tie_pairs,
            "duplicate_pointwise_rows": self.duplicate_pointwise_rows,
            "conflicting_pointwise_rows": self.conflicting_pointwise_rows,
            "duplicate_preference_pairs": self.duplicate_preference_pairs,
            "identical_preference_response_rows": (
                self.identical_preference_response_rows
            ),
            "invalid_pointwise_rows": self.invalid_pointwise_rows,
            "invalid_preference_rows": self.invalid_preference_rows,
            "missing_left_ratings": self.missing_left_ratings,
            "missing_right_ratings": self.missing_right_ratings,
            "source_split_mismatches": self.source_split_mismatches,
        }


@dataclass(frozen=True)
class NormalizationResult:
    pairs: tuple[NormalizedPair, ...]
    report: AlignmentReport


@dataclass(frozen=True)
class PairAuditReport:
    """Aggregate invariants and imbalance diagnostics for normalized pairs."""

    total_pairs: int
    unique_prompts: int
    unique_responses: int
    duplicate_pair_count: int
    identical_response_pairs: int
    unassigned_pairs: int
    prompt_leakage_count: int
    label_counts: dict[str, int]
    source_split_pair_counts: dict[str, int]
    split_pair_counts: dict[str, int]
    split_prompt_counts: dict[str, int]
    left_preference_rate: float | None
    position_preference_gap: float | None
    helpfulness_preference_counts: dict[str, int]
    token_counted_pairs: int
    missing_token_count_pairs: int
    preferred_length_counts: dict[str, int]
    response_token_count_summary: dict[str, int | float] | None

    @property
    def is_valid(self) -> bool:
        return (
            self.total_pairs > 0
            and self.duplicate_pair_count == 0
            and self.identical_response_pairs == 0
            and self.unassigned_pairs == 0
            and self.prompt_leakage_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_pairs": self.total_pairs,
            "unique_prompts": self.unique_prompts,
            "unique_responses": self.unique_responses,
            "duplicate_pair_count": self.duplicate_pair_count,
            "identical_response_pairs": self.identical_response_pairs,
            "unassigned_pairs": self.unassigned_pairs,
            "prompt_leakage_count": self.prompt_leakage_count,
            "label_counts": self.label_counts,
            "source_split_pair_counts": self.source_split_pair_counts,
            "split_pair_counts": self.split_pair_counts,
            "split_prompt_counts": self.split_prompt_counts,
            "left_preference_rate": self.left_preference_rate,
            "position_preference_gap": self.position_preference_gap,
            "helpfulness_preference_counts": self.helpfulness_preference_counts,
            "token_counted_pairs": self.token_counted_pairs,
            "missing_token_count_pairs": self.missing_token_count_pairs,
            "preferred_length_counts": self.preferred_length_counts,
            "response_token_count_summary": self.response_token_count_summary,
        }


@dataclass(frozen=True)
class _PointwiseRating:
    helpfulness: float
    source_split: str


def stable_prompt_id(prompt: str) -> str:
    """Return a deterministic ID derived from the exact normalized prompt text."""

    normalized = _normalized_text(prompt, "prompt")
    return sha256(f"prompt\0{normalized}".encode()).hexdigest()


def stable_pair_id(prompt: str, response_left: str, response_right: str) -> str:
    """Return an orientation-invariant ID for a response pair."""

    prompt_key = stable_prompt_id(prompt)
    response_keys = sorted(
        (
            sha256(
                _normalized_response(response_left, "response_left").encode()
            ).hexdigest(),
            sha256(
                _normalized_response(response_right, "response_right").encode()
            ).hexdigest(),
        )
    )
    payload = f"pair\0{prompt_key}\0{response_keys[0]}\0{response_keys[1]}"
    return sha256(payload.encode()).hexdigest()


def preference_label_from_strength(preference_strength: object) -> int:
    """Convert the official signed strength to the left-win label convention.

    HelpSteer2 uses negative strengths when response 1 is preferred and positive
    strengths when response 2 is preferred. This project uses +1 for a left win,
    -1 for a right win, and 0 for an audit-only tie.
    """

    strength = _integer(preference_strength, "preference_strength")
    if strength < -3 or strength > 3:
        raise ValueError("preference_strength must be between -3 and 3")
    return 1 if strength < 0 else -1 if strength > 0 else 0


def normalize_helpsteer2_pairs(
    pointwise_rows: Iterable[Mapping[str, object]],
    preference_rows: Iterable[Mapping[str, object]],
    *,
    strict: bool = True,
) -> NormalizationResult:
    """Join HelpSteer2 preference rows to exact pointwise response ratings.

    Rows that cannot be safely normalized are counted and omitted. In strict
    mode, any fatal alignment issue raises :class:`DataAlignmentError`.
    """

    pointwise_index: dict[tuple[str, str], _PointwiseRating] = {}
    pointwise_count = 0
    duplicate_pointwise = 0
    conflicting_pointwise = 0
    invalid_pointwise = 0

    for row in pointwise_rows:
        pointwise_count += 1
        try:
            prompt = _normalized_text(row["prompt"], "prompt")
            response = _normalized_text(row["response"], "response")
            helpfulness = _finite_number(row["helpfulness"], "helpfulness")
            source_split = _row_source_split(row)
        except (KeyError, TypeError, ValueError):
            invalid_pointwise += 1
            continue

        key = (prompt, response)
        rating = _PointwiseRating(helpfulness, source_split)
        existing = pointwise_index.get(key)
        if existing is not None:
            duplicate_pointwise += 1
            if existing != rating:
                conflicting_pointwise += 1
            continue
        pointwise_index[key] = rating

    preference_count = 0
    tie_pairs = 0
    duplicate_preferences = 0
    identical_preference_responses = 0
    invalid_preferences = 0
    missing_left = 0
    missing_right = 0
    source_split_mismatches = 0
    pairs: list[NormalizedPair] = []
    seen_pair_ids: set[str] = set()

    for row in preference_rows:
        preference_count += 1
        try:
            prompt = _normalized_text(row["prompt"], "prompt")
            response_left = _normalized_text(row["response_1"], "response_1")
            response_right = _normalized_text(row["response_2"], "response_2")
            signed_strength = _integer(
                row["preference_strength"],
                "preference_strength",
            )
            label = preference_label_from_strength(signed_strength)
            source_split = _row_source_split(row, field="split")
        except (KeyError, TypeError, ValueError):
            invalid_preferences += 1
            continue

        if response_left == response_right:
            identical_preference_responses += 1
            continue

        left_rating = pointwise_index.get((prompt, response_left))
        right_rating = pointwise_index.get((prompt, response_right))
        if left_rating is None:
            missing_left += 1
        if right_rating is None:
            missing_right += 1
        if left_rating is None or right_rating is None:
            continue

        split_values = {
            value
            for value in (
                source_split,
                left_rating.source_split,
                right_rating.source_split,
            )
            if value != "unknown"
        }
        if len(split_values) > 1:
            source_split_mismatches += 1

        pair_id = stable_pair_id(prompt, response_left, response_right)
        if pair_id in seen_pair_ids:
            duplicate_preferences += 1
            continue
        seen_pair_ids.add(pair_id)

        pair = NormalizedPair(
            prompt=prompt,
            response_left=response_left,
            response_right=response_right,
            preference_label=label,
            preference_strength=abs(signed_strength),
            helpfulness_left=left_rating.helpfulness,
            helpfulness_right=right_rating.helpfulness,
            source_split=source_split,
        )
        pairs.append(pair)
        tie_pairs += int(label == 0)

    report = AlignmentReport(
        pointwise_rows=pointwise_count,
        unique_pointwise_responses=len(pointwise_index),
        preference_rows=preference_count,
        aligned_pairs=len(pairs),
        tie_pairs=tie_pairs,
        duplicate_pointwise_rows=duplicate_pointwise,
        conflicting_pointwise_rows=conflicting_pointwise,
        duplicate_preference_pairs=duplicate_preferences,
        identical_preference_response_rows=identical_preference_responses,
        invalid_pointwise_rows=invalid_pointwise,
        invalid_preference_rows=invalid_preferences,
        missing_left_ratings=missing_left,
        missing_right_ratings=missing_right,
        source_split_mismatches=source_split_mismatches,
    )
    if strict and not report.is_valid:
        raise DataAlignmentError(report)
    return NormalizationResult(tuple(pairs), report)


def assign_prompt_splits(
    pairs: Iterable[NormalizedPair],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[NormalizedPair, ...]:
    """Assign exact prompt groups to deterministic experimental partitions."""

    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not math.isfinite(value) or value < 0 for value in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split fractions must sum to one")

    values = tuple(pairs)
    prompt_ids = sorted({pair.prompt_id for pair in values})
    prompt_ids.sort(
        key=lambda prompt_id: sha256(f"{seed}\0{prompt_id}".encode()).digest()
    )
    counts = _largest_remainder_counts(len(prompt_ids), fractions)

    prompt_splits: dict[str, SplitName] = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for prompt_id in prompt_ids[start : start + count]:
            prompt_splits[prompt_id] = split_name
        start += count

    return tuple(
        replace(pair, split=prompt_splits[pair.prompt_id]) for pair in values
    )


def assign_source_holdout_splits(
    pairs: Iterable[NormalizedPair],
    *,
    seed: int,
    holdout_source_split: str = "validation",
    train_fraction: float = 0.85,
    validation_fraction: float = 0.15,
) -> tuple[NormalizedPair, ...]:
    """Split development prompts while preserving an official source holdout.

    Prompts from ``holdout_source_split`` are always assigned to test. All
    remaining prompts are deterministically divided between train and
    validation. A prompt appearing on both sides of the source boundary is a
    fatal leakage error.
    """

    if not isinstance(holdout_source_split, str) or not holdout_source_split:
        raise ValueError("holdout_source_split must be a non-empty string")
    fractions = (train_fraction, validation_fraction)
    if any(not math.isfinite(value) or value < 0 for value in fractions):
        raise ValueError("development split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("train and validation fractions must sum to one")

    values = tuple(pairs)
    prompt_sources: dict[str, set[str]] = defaultdict(set)
    for pair in values:
        prompt_sources[pair.prompt_id].add(pair.source_split)
    leaked = [
        prompt_id
        for prompt_id, sources in prompt_sources.items()
        if holdout_source_split in sources and len(sources) > 1
    ]
    if leaked:
        raise ValueError(
            "prompt leakage across development and official holdout: "
            f"{len(leaked)} prompt(s)"
        )

    holdout_prompts = {
        prompt_id
        for prompt_id, sources in prompt_sources.items()
        if holdout_source_split in sources
    }
    development_prompts = sorted(set(prompt_sources).difference(holdout_prompts))
    if not holdout_prompts:
        raise ValueError("official holdout contains no prompts")
    if len(development_prompts) < 2:
        raise ValueError("development data must contain at least two prompts")
    development_prompts.sort(
        key=lambda prompt_id: sha256(f"{seed}\0{prompt_id}".encode()).digest()
    )
    train_count, validation_count = _largest_remainder_counts(
        len(development_prompts),
        fractions,
    )
    if train_count == 0 or validation_count == 0:
        raise ValueError("development split must assign prompts to train and validation")

    prompt_splits: dict[str, SplitName] = {
        prompt_id: "test" for prompt_id in holdout_prompts
    }
    prompt_splits.update(
        (prompt_id, "train")
        for prompt_id in development_prompts[:train_count]
    )
    prompt_splits.update(
        (prompt_id, "validation")
        for prompt_id in development_prompts[
            train_count : train_count + validation_count
        ]
    )
    return tuple(
        replace(pair, split=prompt_splits[pair.prompt_id]) for pair in values
    )


def attach_token_counts(
    pairs: Iterable[NormalizedPair],
    tokenizer: object,
    *,
    batch_size: int = 256,
) -> tuple[NormalizedPair, ...]:
    """Attach response-only token counts while tokenizing each unique text once."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = tuple(pairs)
    responses = tuple(
        dict.fromkeys(
            response
            for pair in values
            for response in (pair.response_left, pair.response_right)
        )
    )
    response_counts: dict[str, int] = {}
    for start in range(0, len(responses), batch_size):
        batch = responses[start : start + batch_size]
        counts = count_response_tokens(batch, tokenizer)
        if len(counts) != len(batch):
            raise ValueError("tokenizer returned the wrong number of token sequences")
        response_counts.update(
            (response, int(count))
            for response, count in zip(batch, counts, strict=True)
        )

    return tuple(
        replace(
            pair,
            token_count_left=response_counts[pair.response_left],
            token_count_right=response_counts[pair.response_right],
        )
        for pair in values
    )


def swap_pair(pair: NormalizedPair) -> NormalizedPair:
    """Swap response position without changing prompt, split, or pair identity."""

    return replace(
        pair,
        response_left=pair.response_right,
        response_right=pair.response_left,
        preference_label=-pair.preference_label,
        helpfulness_left=pair.helpfulness_right,
        helpfulness_right=pair.helpfulness_left,
        token_count_left=pair.token_count_right,
        token_count_right=pair.token_count_left,
        is_swapped=not pair.is_swapped,
    )


def audit_pairs(pairs: Iterable[NormalizedPair]) -> PairAuditReport:
    """Check split invariants and summarize position, quality, and length balance."""

    values = tuple(pairs)
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    split_prompt_ids: dict[str, set[str]] = defaultdict(set)
    label_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    split_pair_counts: Counter[str] = Counter()
    helpfulness_counts: Counter[str] = Counter()
    preferred_length_counts: Counter[str] = Counter()
    pair_ids: Counter[str] = Counter()
    response_ids: set[str] = set()
    response_token_counts: dict[str, int] = {}
    identical_response_pairs = 0
    token_counted_pairs = 0

    for pair in values:
        split_name = pair.split or "unassigned"
        prompt_splits[pair.prompt_id].add(split_name)
        split_prompt_ids[split_name].add(pair.prompt_id)
        split_pair_counts[split_name] += 1
        source_split_counts[pair.source_split] += 1
        pair_ids[pair.pair_id] += 1
        left_response_id = sha256(pair.response_left.encode()).hexdigest()
        right_response_id = sha256(pair.response_right.encode()).hexdigest()
        response_ids.update((left_response_id, right_response_id))
        identical_response_pairs += int(pair.response_left == pair.response_right)

        label_name = (
            "left_preferred"
            if pair.preference_label == 1
            else "right_preferred"
            if pair.preference_label == -1
            else "tie"
        )
        label_counts[label_name] += 1

        if pair.preference_label != 0:
            quality_direction = pair.preference_label * (
                pair.helpfulness_left - pair.helpfulness_right
            )
            quality_name = (
                "agrees"
                if quality_direction > 0
                else "disagrees"
                if quality_direction < 0
                else "pointwise_tie"
            )
            helpfulness_counts[quality_name] += 1

        if pair.token_count_left is not None and pair.token_count_right is not None:
            token_counted_pairs += 1
            response_token_counts[left_response_id] = pair.token_count_left
            response_token_counts[right_response_id] = pair.token_count_right
            if pair.preference_label != 0:
                length_direction = pair.preference_label * (
                    pair.token_count_left - pair.token_count_right
                )
                length_name = (
                    "preferred_longer"
                    if length_direction > 0
                    else "preferred_shorter"
                    if length_direction < 0
                    else "equal_length"
                )
                preferred_length_counts[length_name] += 1

    left_count = label_counts["left_preferred"]
    right_count = label_counts["right_preferred"]
    decisive_count = left_count + right_count
    left_rate = left_count / decisive_count if decisive_count else None
    position_gap = (
        (left_count - right_count) / decisive_count if decisive_count else None
    )
    leakage_count = sum(len(splits) > 1 for splits in prompt_splits.values())
    duplicate_pair_count = sum(count - 1 for count in pair_ids.values())

    return PairAuditReport(
        total_pairs=len(values),
        unique_prompts=len(prompt_splits),
        unique_responses=len(response_ids),
        duplicate_pair_count=duplicate_pair_count,
        identical_response_pairs=identical_response_pairs,
        unassigned_pairs=split_pair_counts["unassigned"],
        prompt_leakage_count=leakage_count,
        label_counts=_complete_counts(
            label_counts,
            ("left_preferred", "right_preferred", "tie"),
        ),
        source_split_pair_counts=dict(sorted(source_split_counts.items())),
        split_pair_counts=_complete_counts(
            split_pair_counts,
            (*SPLIT_NAMES, "unassigned"),
        ),
        split_prompt_counts={
            split_name: len(split_prompt_ids.get(split_name, set()))
            for split_name in (*SPLIT_NAMES, "unassigned")
        },
        left_preference_rate=left_rate,
        position_preference_gap=position_gap,
        helpfulness_preference_counts=_complete_counts(
            helpfulness_counts,
            ("agrees", "disagrees", "pointwise_tie"),
        ),
        token_counted_pairs=token_counted_pairs,
        missing_token_count_pairs=len(values) - token_counted_pairs,
        preferred_length_counts=_complete_counts(
            preferred_length_counts,
            ("preferred_longer", "preferred_shorter", "equal_length"),
        ),
        response_token_count_summary=_token_count_summary(
            tuple(response_token_counts.values())
        ),
    )


def _normalized_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    _required_text(normalized, name)
    return normalized


def _normalized_response(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _required_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _response_text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} must be an integer")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _row_source_split(
    row: Mapping[str, object],
    *,
    field: str = "_source_split",
) -> str:
    value = row.get(field, "unknown")
    if value is None:
        return "unknown"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip().lower()
    return "validation" if normalized == "val" else normalized


def _largest_remainder_counts(
    total: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    raw_counts = tuple(total * fraction for fraction in fractions)
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    remainder_order = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw_counts[index] - counts[index]), index),
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _complete_counts(
    counts: Mapping[str, int],
    names: Iterable[str],
) -> dict[str, int]:
    return {name: int(counts.get(name, 0)) for name in names}


def _token_count_summary(
    values: tuple[int, ...],
) -> dict[str, int | float] | None:
    if not values:
        return None
    ordered = tuple(sorted(values))
    return {
        "responses": len(ordered),
        "minimum": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
    }


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]
