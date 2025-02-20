import hashlib
import logging
from typing import Any

import litellm
import numpy as np
from litellm import batch_completion, completion_cost
from litellm.types.utils import ChatCompletionTokenLogprob, Choices, ModelResponse
from litellm.utils import token_counter
from openai import OpenAIError
from tokenizers import Tokenizer
from tqdm import tqdm

import lotus
from lotus.cache import CacheFactory
from lotus.types import LMOutput, LMStats, LogprobsForCascade, LogprobsForFilterCascade

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
        tokenizer: Tokenizer | None = None,
        cache=None,
        **kwargs: dict[str, Any],
    ):
        self.model = model
        self.max_ctx_len = max_ctx_len
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

        self.stats: LMStats = LMStats()

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

        self.stats.total_usage.cache_hits += len(messages) - len(uncached_data)

        # Process uncached messages in batches
        uncached_responses = self._process_uncached_messages(
            uncached_data, all_kwargs, show_progress_bar, progress_bar_desc
        )
        if lotus.settings.enable_cache:
            # Add new responses to cache
            for resp, (_, hash) in zip(uncached_responses, uncached_data):
                self._cache_response(resp, hash)

        # Merge all responses in original order and extract outputs
        all_responses = (
            self._merge_responses(cached_responses, uncached_responses)
            if lotus.settings.enable_cache
            else uncached_responses
        )
        outputs = [self._get_top_choice(resp) for resp in all_responses]
        logprobs = (
            [self._get_top_choice_logprobs(resp) for resp in all_responses] if all_kwargs.get("logprobs") else None
        )

        return LMOutput(outputs=outputs, logprobs=logprobs)

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
        uncached_responses = batch_completion(
            self.model, batch, drop_params=True, max_workers=self.max_batch_size, **all_kwargs
        )

        pbar.update(total_calls)
        pbar.close()

        return uncached_responses

    def _cache_response(self, response, hash):
        """Caches a response and updates stats if successful."""
        if isinstance(response, OpenAIError):
            raise response
        self._update_stats(response)
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

    def _update_stats(self, response: ModelResponse):
        if not hasattr(response, "usage"):
            return

        self.stats.total_usage.prompt_tokens += response.usage.prompt_tokens
        self.stats.total_usage.completion_tokens += response.usage.completion_tokens
        self.stats.total_usage.total_tokens += response.usage.total_tokens

        try:
            self.stats.total_usage.total_cost += completion_cost(completion_response=response)
        except litellm.exceptions.NotFoundError as e:
            # Sometimes the model's pricing information is not available
            lotus.logger.debug(f"Error updating completion cost: {e}")

    def _get_top_choice(self, response: ModelResponse) -> str:
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        if choice.message.content is None:
            raise ValueError(f"No content in response: {response}")
        return choice.message.content

    def _get_top_choice_logprobs(self, response: ModelResponse) -> list[ChatCompletionTokenLogprob]:
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        logprobs = choice.logprobs["content"]
        return [ChatCompletionTokenLogprob(**logprob) for logprob in logprobs]

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
        all_true_probs: list[float] = []

        def get_normalized_true_prob(token_probs: dict[str, float]) -> float | None:
            # Normalize keys by converting to lowercase and stripping whitespace
            # Take the max probability for each key (e.g. True and true both map to "true" so we take the max of the two)
            normalized_probs: dict[str, float] = {}
            for k, v in token_probs.items():
                normalized_key = k.lower().strip()
                normalized_probs[normalized_key] = max(v, normalized_probs.get(normalized_key, float("-inf")))

            # Look for true/false in normalized keys
            if "true" in normalized_probs and "false" in normalized_probs:
                true_prob = normalized_probs["true"]
                false_prob = normalized_probs["false"]
                return true_prob / (true_prob + false_prob)
            return None

        for response_logprobs in logprobs:
            true_prob: float = 1  # Default if no true/false token found

            # Find last true/false token by normalizing all tokens and searching in reverse
            cleaned_tokens = [logprob.token.lower().strip() for logprob in response_logprobs]
            true_false_indices = [i for i, token in enumerate(cleaned_tokens) if token in ["true", "false"]]

            if true_false_indices:
                last_true_false_idx = true_false_indices[-1]
                logprob = response_logprobs[last_true_false_idx]
                cleaned_token = cleaned_tokens[last_true_false_idx]

                token_probs = {top.token: np.exp(top.logprob) for top in logprob.top_logprobs}
                normalized_prob = get_normalized_true_prob(token_probs)

                # Fall back to binary true/false if normalization fails
                if normalized_prob is None:
                    true_prob = 1 if cleaned_token == "true" else 0
                else:
                    true_prob = normalized_prob

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
        print(f"Total cost: ${self.stats.total_usage.total_cost:.6f}")
        print(f"Total prompt tokens: {self.stats.total_usage.prompt_tokens}")
        print(f"Total completion tokens: {self.stats.total_usage.completion_tokens}")
        print(f"Total tokens: {self.stats.total_usage.total_tokens}")
        print(f"Total cache hits: {self.stats.total_usage.cache_hits}")

    def reset_stats(self):
        self.stats = LMStats(
            total_usage=LMStats.TotalUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, total_cost=0.0)
        )

    def reset_cache(self, max_size: int | None = None):
        self.cache.reset(max_size)
