"""Core Bradley-Terry helpers for the HelpSteer2 reward-model study."""

from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


TOKENIZER_NATIVE_SERIALIZATION = "tokenizer_native"
PAIRWISE_REWARD_LOSSES = frozenset(
    {"binary_bradley_terry", "scaled_bradley_terry", "binary_hinge", "scaled_hinge"}
)


def configure_phase2_chat_template(
    tokenizer: Any,
    *,
    serialization_profile: str,
) -> dict[str, str]:
    """Record the tokenizer-native chat template used by every HS2 condition."""

    if serialization_profile != TOKENIZER_NATIVE_SERIALIZATION:
        raise ValueError("HelpSteer2 experiments use tokenizer_native serialization")
    template = tokenizer.chat_template
    if not template:
        raise ValueError("the tokenizer has no chat template")
    return {
        "profile": TOKENIZER_NATIVE_SERIALIZATION,
        "source": "tokenizer_config.json",
        "chat_template_sha256": sha256(template.encode()).hexdigest(),
    }


def pairwise_reward_loss_values(
    chosen_rewards: ArrayLike,
    rejected_rewards: ArrayLike,
    preference_strengths: ArrayLike,
    *,
    loss: str,
) -> NDArray[np.float64]:
    """NumPy version of the reward-model objective, used for audits."""

    chosen = np.asarray(chosen_rewards, dtype=float).reshape(-1)
    rejected = np.asarray(rejected_rewards, dtype=float).reshape(-1)
    strength = np.asarray(preference_strengths, dtype=float).reshape(-1)
    margin = chosen - rejected

    if loss == "binary_bradley_terry":
        return np.logaddexp(0.0, -margin)
    if loss == "scaled_bradley_terry":
        return strength * np.logaddexp(0.0, -margin)
    if loss == "binary_hinge":
        return np.maximum(0.0, 1.0 - margin)
    if loss == "scaled_hinge":
        return np.maximum(0.0, strength - margin)
    raise ValueError(f"unknown pairwise loss: {loss}")


def scaled_bradley_terry_loss_values(
    chosen_rewards: ArrayLike,
    rejected_rewards: ArrayLike,
    preference_strengths: ArrayLike,
) -> NDArray[np.float64]:
    return pairwise_reward_loss_values(
        chosen_rewards,
        rejected_rewards,
        preference_strengths,
        loss="scaled_bradley_terry",
    )


def pairwise_reward_loss(
    chosen_rewards: Any,
    rejected_rewards: Any,
    preference_strengths: Any,
    *,
    loss: str,
    reduction: str = "mean",
) -> Any:
    """Torch version used by the Trainer."""

    import torch.nn.functional as F

    margin = chosen_rewards.reshape(-1) - rejected_rewards.reshape(-1)
    strength = preference_strengths.reshape(-1).to(margin)
    if loss == "binary_bradley_terry":
        values = F.softplus(-margin)
    elif loss == "scaled_bradley_terry":
        values = strength * F.softplus(-margin)
    elif loss == "binary_hinge":
        values = F.relu(1.0 - margin)
    elif loss == "scaled_hinge":
        values = F.relu(strength - margin)
    else:
        raise ValueError(f"unknown pairwise loss: {loss}")

    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.mean()


def scaled_bradley_terry_loss(
    chosen_rewards: Any,
    rejected_rewards: Any,
    preference_strengths: Any,
    *,
    reduction: str = "mean",
) -> Any:
    return pairwise_reward_loss(
        chosen_rewards,
        rejected_rewards,
        preference_strengths,
        loss="scaled_bradley_terry",
        reduction=reduction,
    )


def effective_gradient_accumulation_steps(
    *,
    global_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Convert the paper's global pair batch into Trainer accumulation steps."""

    local_global_batch = micro_batch_size * world_size
    steps, remainder = divmod(global_batch_size, local_global_batch)
    if remainder:
        raise ValueError("global batch must be divisible by micro batch × world size")
    return steps
