from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from litellm.types.utils import ChatCompletionTokenLogprob
from pydantic import BaseModel


################################################################################
# LM related
################################################################################
@dataclass
class LMOutput:
    outputs: list[str]
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class LMStats:
    @dataclass
    class TotalUsage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        total_cost: float = 0.0

        def __sub__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens - other.prompt_tokens,
                completion_tokens=self.completion_tokens - other.completion_tokens,
                total_tokens=self.total_tokens - other.total_tokens,
                total_cost=self.total_cost - other.total_cost,
            )

        def __add__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                completion_tokens=self.completion_tokens + other.completion_tokens,
                total_tokens=self.total_tokens + other.total_tokens,
                total_cost=self.total_cost + other.total_cost,
            )

    # Usage stats if there was no caching
    virtual_usage: TotalUsage = field(default_factory=TotalUsage)
    # Actual usage with caching applied
    physical_usage: TotalUsage = field(default_factory=TotalUsage)

    cache_hits: int = 0
    operator_cache_hits: int = 0


@dataclass
class LogprobsForCascade:
    tokens: list[list[str]]
    confidences: list[list[float]]


@dataclass
class LogprobsForFilterCascade:
    true_probs: list[float]
    tokens: list[list[str]]
    confidences: list[list[float]]


################################################################################
# Semantic operation outputs
################################################################################
@dataclass
class SemanticMapPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticMapOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticExtractPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticExtractOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticFilterPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]


@dataclass
class SemanticFilterOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]
    stats: dict[str, Any] | None = None
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class SemanticAggOutput:
    outputs: list[str]


@dataclass
class SemanticJoinOutput:
    join_results: list[tuple[int, int, str | None]]
    filter_outputs: list[bool]
    all_raw_outputs: list[str]
    all_explanations: list[str | None]
    stats: dict[str, Any] | None = None


class ProxyModel(Enum):
    HELPER_LM = "helper_lm"
    EMBEDDING_MODEL = "embedding_model"


class CascadeArgs(BaseModel):
    recall_target: float = 0.8
    precision_target: float = 0.8
    sampling_percentage: float = 0.1
    failure_probability: float = 0.2
    map_instruction: str | None = None
    map_examples: pd.DataFrame | None = None
    proxy_model: ProxyModel = ProxyModel.HELPER_LM

    # Filter cascade args
    cascade_IS_weight: float = 0.9
    cascade_num_calibration_quantiles: int = 50

    # Join cascade args
    min_join_cascade_size: int = 100
    cascade_IS_max_sample_range: int = 200
    cascade_IS_random_seed: int | None = None

    # to enable pandas
    class Config:
        arbitrary_types_allowed = True


@dataclass
class SemanticTopKOutput:
    indexes: list[int]
    stats: dict[str, Any] | None = None


################################################################################
# RM related
################################################################################


@dataclass
class RMOutput:
    distances: list[list[float]]
    indices: list[list[int]]


################################################################################
# Reranker related
################################################################################
@dataclass
class RerankerOutput:
    indices: list[int]


################################################################################
# Serialization related
################################################################################
class SerializationFormat(Enum):
    JSON = "json"
    XML = "xml"
    DEFAULT = "default"


################################################################################
# Utility
################################################################################
@dataclass
class UsageLimit:
    prompt_tokens_limit: float = float("inf")
    completion_tokens_limit: float = float("inf")
    total_tokens_limit: float = float("inf")
    total_cost_limit: float = float("inf")


################################################################################
# Exception related
################################################################################
class LotusException(Exception):
    """Base class for all Lotus exceptions."""

    pass


class LotusUsageLimitException(LotusException):
    """Exception raised when the usage limit is exceeded."""

    pass


################################################################################
# Prompt Strategy
################################################################################
@dataclass
class PromptStrategy:
    """
    Configurable prompt strategy for semantic operations.

    This class encapsulates various prompting techniques including chain-of-thought
    reasoning, demonstrations, and bootstrapping configurations.

    Args:
        cot (bool): Whether to use chain-of-thought reasoning. Defaults to False.
        dems (pd.DataFrame | str | None): Demonstrations to use. Can be:
            - pd.DataFrame: User-provided examples with Answer column
            - "auto": Automatically bootstrap demonstrations
            - None: No demonstrations
        max_dems (int): Maximum number of demonstrations to use. Defaults to 3.
        teacher_lm: Language model to use for bootstrapping demonstrations.
            If None, uses the main model. Defaults to None.

    Example:
        >>> # Chain-of-thought with user-provided demonstrations
        >>> strat = PromptStrategy(cot=True, dems=examples_df)
        >>> df.sem_filter(user_instruction, prompt_strategy=strat)

        >>> # Auto-bootstrap demonstrations with CoT
        >>> strat = PromptStrategy(
        ...     cot=True,
        ...     dems="auto",
        ...     max_dems=2,
        ...     teacher_lm=lotus.models.LM(model="gpt-4o-mini")
        ... )
        >>> df.sem_filter(user_instruction, prompt_strategy=strat)
    """

    cot: bool = False
    dems: pd.DataFrame | str | None = None
    max_dems: int = 3
    teacher_lm: Any = None  # lotus.models.LM type, but avoiding circular import
