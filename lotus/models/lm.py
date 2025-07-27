import hashlib
import logging
import math
import time
import warnings
from typing import Any

import litellm
import numpy as np
from litellm import batch_completion, completion_cost
from litellm.exceptions import AuthenticationError
from litellm.types.utils import ChatCompletionTokenLogprob, Choices, ModelResponse
from litellm.utils import token_counter
from openai._exceptions import OpenAIError
from pydantic import BaseModel
from tokenizers import Tokenizer
from tqdm import tqdm

import lotus
from lotus.cache import CacheFactory
from lotus.types import (
    LMOutput,
    LMStats,
    LogprobsForCascade,
    LogprobsForFilterCascade,
    LotusUsageLimitException,
    UsageLimit,
)

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


class LM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_ctx_len: int = 128000,
        max_tokens: int = 512,
        max_batch_size: int = 64,
        rate_limit: int | None = None,
        tokenizer: Tokenizer | None = None,
        cache=None,
        physical_usage_limit: UsageLimit = UsageLimit(),
        virtual_usage_limit: UsageLimit = UsageLimit(),
        **kwargs: dict[str, Any],
    ):
        """Language Model class for interacting with various LLM providers.

        Args:
            model (str): Name of the model to use. Defaults to "gpt-4o-mini".
            temperature (float): Sampling temperature. Defaults to 0.0.
            max_ctx_len (int): Maximum context length in tokens. Defaults to 128000.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 512.
            max_batch_size (int): Maximum batch size for concurrent requests. Defaults to 64.
            rate_limit (int | None): Maximum requests per minute. If set, caps max_batch_size and adds delays.
            tokenizer (Tokenizer | None): Custom tokenizer instance. Defaults to None.
            cache: Cache instance to use. Defaults to None.
            physical_usage_limit (UsageLimit): Physical usage limits for the model. Defaults to UsageLimit().
            virtual_usage_limit (UsageLimit): Virtual usage limits for the model. Defaults to UsageLimit().
            **kwargs: Additional keyword arguments passed to the underlying LLM API.
        """
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        if rate_limit is not None:
            self._rate_limit_delay = 60 / rate_limit
            if max_batch_size is not None:
                self.max_batch_size = min(rate_limit, max_batch_size)
            else:
                self.max_batch_size = rate_limit
        else:
            self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

        self.stats: LMStats = LMStats()
        self.physical_usage_limit = physical_usage_limit
        self.virtual_usage_limit = virtual_usage_limit

        self.cache = cache or CacheFactory.create_default_cache()

    def __call__(
        self,
        messages: list[list[dict[str, str]]],
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        **kwargs: dict[str, Any],
    ) -> LMOutput:
        all_kwargs = {**self.kwargs, **kwargs}

        # Set top_logprobs if logprobs requested
        if all_kwargs.get("logprobs", False):
            all_kwargs.setdefault("top_logprobs", 10)

        if lotus.settings.enable_cache:
            # Check cache and separate cached and uncached messages
            hashed_messages = [self._hash_messages(msg, all_kwargs) for msg in messages]
            cached_responses = [self.cache.get(hash) for hash in hashed_messages]

        uncached_data = (
            [(msg, hash) for msg, hash, resp in zip(messages, hashed_messages, cached_responses) if resp is None]
            if lotus.settings.enable_cache
            else [(msg, "no-cache") for msg in messages]
        )

        self.stats.cache_hits += len(messages) - len(uncached_data)

        # Process uncached messages in batches
        uncached_responses = self._process_uncached_messages(
            uncached_data, all_kwargs, show_progress_bar, progress_bar_desc
        )

        # Add new responses to cache and update stats
        for resp, (_, hash) in zip(uncached_responses, uncached_data):
            self._update_stats(resp, is_cached=False)
            if lotus.settings.enable_cache:
                self._cache_response(resp, hash)

        # Update virtual stats for cached responses
        if lotus.settings.enable_cache:
            for resp in cached_responses:
                if resp is not None:
                    self._update_stats(resp, is_cached=True)

        # Merge all responses in original order and extract outputs
        all_responses = (
            self._merge_responses(cached_responses, uncached_responses)
            if lotus.settings.enable_cache
            else uncached_responses
        )
        outputs: list[str] = [self._get_top_choice(resp) for resp in all_responses]
        logprobs = (
            [self._get_top_choice_logprobs(resp) for resp in all_responses] if all_kwargs.get("logprobs") else None
        )

        return LMOutput(outputs=outputs, logprobs=logprobs)

    def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        show_progress_bar: bool = True,
        progress_bar_desc: str = "Processing uncached messages",
        response_format: BaseModel | None = None,
        **kwargs: dict[str, Any],
    ) -> str:
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        ]
        output = self(
            messages,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=progress_bar_desc,
            response_format=response_format,  # type: ignore
            **kwargs,
        ).outputs[0]
        if response_format:
            assert isinstance(output, BaseModel)
            return response_format.model_validate_json(output)
        return output

    def _process_uncached_messages(self, uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
        """Processes uncached messages in batches and returns responses."""
        total_calls = len(uncached_data)

        pbar = tqdm(
            total=total_calls,
            desc=progress_bar_desc,
            disable=not show_progress_bar,
            bar_format="{l_bar}{bar} {n}/{total} LM calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        batch = [msg for msg, _ in uncached_data]

        if self.rate_limit is not None:
            uncached_responses = self._process_with_rate_limiting(batch, all_kwargs, pbar)
        else:
            uncached_responses = batch_completion(
                self.model, batch, drop_params=True, max_workers=self.max_batch_size, **all_kwargs
            )
            pbar.update(total_calls)

        pbar.close()
        return uncached_responses

    def _process_with_rate_limiting(self, batch, all_kwargs, pbar):
        responses = []
        num_batches = math.ceil(len(batch) / self.max_batch_size)
        min_interval_per_request = 60 / self.rate_limit  # seconds per request

        for i in range(num_batches):
            start_time = time.time()
            start_idx = i * self.max_batch_size
            end_idx = min((i + 1) * self.max_batch_size, len(batch))
            sub_batch = batch[start_idx:end_idx]
            sub_responses = batch_completion(
                self.model, sub_batch, drop_params=True, max_workers=self.max_batch_size, **all_kwargs
            )
            responses.extend(sub_responses)
            pbar.update(len(sub_batch))
            end_time = time.time()
            elapsed = end_time - start_time

            # Calculate required delay based on number of requests in this batch
            # Each request should be spaced by min_interval_per_request
            required_time_for_batch = len(sub_batch) * min_interval_per_request

            # Only sleep if the batch was faster than the required time
            if i < num_batches - 1:  # Don't sleep after the last batch
                to_sleep = required_time_for_batch - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        return responses

    def _cache_response(self, response, hash):
        """Caches a response and updates stats if successful."""
        if isinstance(response, OpenAIError):
            raise response
        self.cache.insert(hash, response)

    def _hash_messages(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        """Hash messages and kwargs to create a unique key for the cache"""
        to_hash = str(self.model) + str(messages) + str(kwargs)
        return hashlib.sha256(to_hash.encode()).hexdigest()

    def _merge_responses(
        self, cached_responses: list[ModelResponse | None], uncached_responses: list[ModelResponse]
    ) -> list[ModelResponse]:
        """Merge cached and uncached responses, maintaining order"""
        uncached_iter = iter(uncached_responses)
        return [resp if resp is not None else next(uncached_iter) for resp in cached_responses]

    def _check_usage_limit(self, usage: LMStats.TotalUsage, limit: UsageLimit, usage_type: str):
        """Helper to check if usage exceeds limits"""
        if (
            usage.prompt_tokens > limit.prompt_tokens_limit
            or usage.completion_tokens > limit.completion_tokens_limit
            or usage.total_tokens > limit.total_tokens_limit
            or usage.total_cost > limit.total_cost_limit
        ):
            raise LotusUsageLimitException(f"Usage limit exceeded. Current {usage_type} usage: {usage}, Limit: {limit}")

    def _update_usage_stats(self, usage: LMStats.TotalUsage, response: ModelResponse, cost: float | None):
        """Helper to update usage statistics"""
        if hasattr(response, "usage"):
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
            usage.total_tokens += response.usage.total_tokens
            if cost is not None:
                usage.total_cost += cost

    def _update_stats(self, response: ModelResponse, is_cached: bool = False):
        if not hasattr(response, "usage"):
            return

        # Calculate cost once
        try:
            cost = completion_cost(completion_response=response)
        except litellm.exceptions.NotFoundError as e:
            # Sometimes the model's pricing information is not available
            lotus.logger.debug(f"Error updating completion cost: {e}")
            cost = None
        except Exception as e:
            # Handle any other unexpected errors when calculating cost
            lotus.logger.debug(f"Unexpected error calculating completion cost: {e}")
            warnings.warn(
                "Error calculating completion cost - cost metrics will be inaccurate. Enable debug logging for details."
            )

            cost = None

        # Always update virtual usage
        self._update_usage_stats(self.stats.virtual_usage, response, cost)
        self._check_usage_limit(self.stats.virtual_usage, self.virtual_usage_limit, "virtual")

        # Only update physical usage for non-cached responses
        if not is_cached:
            self._update_usage_stats(self.stats.physical_usage, response, cost)
            self._check_usage_limit(self.stats.physical_usage, self.physical_usage_limit, "physical")

    def _get_top_choice(self, response: ModelResponse) -> str:
        # Handle authentication errors and other exceptions
        if isinstance(response, (AuthenticationError, OpenAIError)):
            raise response

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        if choice.message.content is None:
            raise ValueError(f"No content in response: {response}")
        return choice.message.content

    def _get_top_choice_logprobs(self, response: ModelResponse) -> list[ChatCompletionTokenLogprob]:
        # Handle authentication errors and other exceptions
        if isinstance(response, (AuthenticationError, OpenAIError)):
            raise response

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        logprobs = choice.logprobs["content"]
        return logprobs

    def format_logprobs_for_cascade(self, logprobs: list[list[ChatCompletionTokenLogprob]]) -> LogprobsForCascade:
        all_tokens = []
        all_confidences = []
        for resp_logprobs in logprobs:
            tokens = [logprob.token for logprob in resp_logprobs]
            confidences = [np.exp(logprob.logprob) for logprob in resp_logprobs]
            all_tokens.append(tokens)
            all_confidences.append(confidences)
        return LogprobsForCascade(tokens=all_tokens, confidences=all_confidences)

    def format_logprobs_for_filter_cascade(
        self, logprobs: list[list[ChatCompletionTokenLogprob]]
    ) -> LogprobsForFilterCascade:
        # Get base cascade format first
        base_cascade = self.format_logprobs_for_cascade(logprobs)
        all_true_probs = []

        def get_normalized_true_prob(token_probs: dict[str, float]) -> float | None:
            if "True" in token_probs and "False" in token_probs:
                true_prob = token_probs["True"]
                false_prob = token_probs["False"]
                return true_prob / (true_prob + false_prob)
            return None

        # Get true probabilities for filter cascade
        for resp_idx, response_logprobs in enumerate(logprobs):
            true_prob = None
            for logprob in response_logprobs:
                token_probs = {top.token: np.exp(top.logprob) for top in logprob.top_logprobs}
                true_prob = get_normalized_true_prob(token_probs)
                if true_prob is not None:
                    break

            # Default to 1 if "True" in tokens, 0 if not
            if true_prob is None:
                true_prob = 1 if "True" in base_cascade.tokens[resp_idx] else 0

            all_true_probs.append(true_prob)

        return LogprobsForFilterCascade(
            tokens=base_cascade.tokens, confidences=base_cascade.confidences, true_probs=all_true_probs
        )

    def count_tokens(self, messages: list[dict[str, str]] | str) -> int:
        """Count tokens in messages using either custom tokenizer or model's default tokenizer"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        custom_tokenizer: dict[str, Any] | None = None
        if self.tokenizer:
            custom_tokenizer = dict(type="huggingface_tokenizer", tokenizer=self.tokenizer)

        return token_counter(
            custom_tokenizer=custom_tokenizer,
            model=self.model,
            messages=messages,
        )

    def print_total_usage(self):
        print("\n=== Usage Statistics ===")
        print("Virtual  = Total usage if no caching was used")
        print("Physical = Actual usage with caching applied\n")
        print(f"Virtual Cost:     ${self.stats.virtual_usage.total_cost:,.6f}")
        print(f"Physical Cost:    ${self.stats.physical_usage.total_cost:,.6f}")
        print(f"Virtual Tokens:   {self.stats.virtual_usage.total_tokens:,}")
        print(f"Physical Tokens:  {self.stats.physical_usage.total_tokens:,}")
        print(f"Cache Hits:       {self.stats.cache_hits:,}\n")

    def reset_stats(self):
        self.stats = LMStats()

    def reset_cache(self, max_size: int | None = None):
        self.cache.reset(max_size)

    def get_model_name(self) -> str:
        raw_model = self.model
        if not raw_model:
            return ""

        # If a slash is present, assume the model name is after the last slash.
        if "/" in raw_model:
            candidate = raw_model.split("/")[-1]
        else:
            candidate = raw_model

        # If a colon is present, assume the model version is appended and remove it.
        if ":" in candidate:
            candidate = candidate.split(":")[0]

        return candidate.lower()

    def is_deepseek(self) -> bool:
        model_name = self.get_model_name()
        return model_name.startswith("deepseek-r1")
