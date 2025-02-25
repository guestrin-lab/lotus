from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import CascadeArgs, LMOutput, LogprobsForFilterCascade, ProxyModel, SemanticFilterOutput
from lotus.utils import show_safe_mode

from .cascade_utils import calibrate_llm_logprobs, importance_sampling, learn_cascade_thresholds
from .postprocessors import filter_postprocess


def sem_filter(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
) -> SemanticFilterOutput:
    """
    Filters a list of documents based on a given user instruction using a language model.

    Args:
        docs (list[dict[str, Any]]): The list of documents to filter. Each document is a tuple of text and images.
        model (lotus.models.LM): The language model used for filtering.
        user_instruction (str): The user instruction for filtering.
        default (bool): The default value for filtering in case of parsing errors. Defaults to True.
        examples_multimodal_data (list[dict[str, Any]] | None): The text for examples. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        logprobs (bool): Whether to return log probabilities. Defaults to False.
        additional_cot_instructions (str): Additional instructions for the CoT. Defaults to "".

    Returns:
        SemanticFilterOutput: The True/False outputs, raw outputs, and explanations, and log probabilities.
    """
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            doc,
            user_instruction,
            examples_multimodal_data,
            examples_answers,
            cot_reasoning,
            strategy,
            reasoning_instructions=additional_cot_instructions,
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    kwargs: dict[str, Any] = {"logprobs": logprobs}

    if safe_mode:
        estimated_total_calls = len(docs)
        estimated_total_cost = sum(model.count_tokens(input) for input in inputs)
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    lm_output: LMOutput = model(
        inputs, show_progress_bar=show_progress_bar, progress_bar_desc=progress_bar_desc, **kwargs
    )

    postprocess_output = filter_postprocess(lm_output.outputs, default=default)
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"raw_outputs: {postprocess_output.raw_outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")

    if safe_mode:
        model.print_total_usage()

    return SemanticFilterOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
        logprobs=lm_output.logprobs if logprobs else None,
    )


def learn_filter_cascade_thresholds(
    sample_multimodal_data: list[dict[str, Any]],
    lm: lotus.models.LM,
    formatted_usr_instr: str,
    default: bool,
    cascade_args: CascadeArgs,
    proxy_scores: list[float],
    sample_correction_factors: NDArray[np.float64],
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    additional_cot_instructions: str = "",
) -> tuple[float, float]:
    """Automatically learns the cascade thresholds for a cascade
    filter given a sample of data and doing a search across threshold
    to see what threshold gives the best accuracy."""

    try:
        large_outputs = sem_filter(
            sample_multimodal_data,
            lm,
            formatted_usr_instr,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=False,
            progress_bar_desc="Running oracle for threshold learning",
            additional_cot_instructions=additional_cot_instructions,
        ).outputs

        best_combination, _ = learn_cascade_thresholds(
            proxy_scores=proxy_scores,
            oracle_outputs=large_outputs,
            sample_correction_factors=sample_correction_factors,
            cascade_args=cascade_args,
        )

        lotus.logger.info(f"Learned cascade thresholds: {best_combination}")
        return best_combination

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        raise e


@pd.api.extensions.register_dataframe_accessor("sem_filter")
class SemFilterDataframe:
    """DataFrame accessor for semantic filter."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        # verify that the Series has the correct type
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        return_all: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: pd.DataFrame | None = None,
        helper_examples: pd.DataFrame | None = None,
        strategy: str | None = None,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        return_scores: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Filtering",
        additional_cot_instructions: str = "",
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """
        Applies semantic filter over a dataframe.

        Args:
            user_instruction (str): The user instruction for filtering.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            default (bool): The default value for filtering in case of parsing errors. Defaults to True.
            suffix (str): The suffix for the new columns. Defaults to "_filter".
            examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
            helper_examples (pd.DataFrame | None): The helper examples dataframe. Defaults to None.
            strategy (str | None): The reasoning strategy. Defaults to None.
            cascade_args (CascadeArgs | None): The arguments for join cascade. Defaults to None.
                recall_target (float | None): The target recall. Defaults to None.
                precision_target (float | None): The target precision when cascading. Defaults to None.
                sampling_percentage (float): The percentage of the data to sample when cascading. Defaults to 0.1.
                failure_probability (float): The failure probability when cascading. Defaults to 0.2.
            return_stats (bool): Whether to return statistics. Defaults to False.
            return_scores (bool): Whether to return probabilities. Defaults to False.
            additional_cot_instructions (str): Additional instructions for the CoT. Defaults to "".

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The filtered dataframe or a tuple containing the filtered dataframe and statistics.
        """
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        stats: dict[str, Any] = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)
        helper_strategy = strategy

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        lotus.logger.debug(multimodal_data)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot" and "Reasoning" in examples.columns:
                cot_reasoning = examples["Reasoning"].tolist()

        pos_cascade_threshold, neg_cascade_threshold = None, None
        if cascade_args is not None:
            # Get few-shot examples for small LM
            helper_examples_multimodal_data = None
            helper_examples_answers = None
            helper_cot_reasoning = None
            if helper_examples is not None:
                assert "Answer" in helper_examples.columns, "Answer must be a column in examples dataframe"
                helper_examples_multimodal_data = task_instructions.df2multimodal_info(helper_examples, col_li)
                helper_examples_answers = helper_examples["Answer"].tolist()
                if helper_strategy == "cot" and "Reasoning" in helper_examples.columns:
                    helper_cot_reasoning = helper_examples["Reasoning"].tolist()

        if cascade_args:
            proxy_model = cascade_args.proxy_model
            if (
                cascade_args.recall_target is None
                or cascade_args.precision_target is None
                or cascade_args.failure_probability is None
            ):
                raise ValueError(
                    "Recall target, precision target, and confidence need to be specified for learned thresholds."
                )

            # Get the proxy scores
            if proxy_model == ProxyModel.HELPER_LM:
                if not lotus.settings.helper_lm:
                    raise ValueError("Helper LM must be set in settings")

                if helper_strategy == "cot":
                    raise ValueError("CoT not supported for helper models in cascades.")

                # Run small LM and get logits
                helper_output = sem_filter(
                    multimodal_data,
                    lotus.settings.helper_lm,
                    formatted_usr_instr,
                    default=default,
                    examples_multimodal_data=helper_examples_multimodal_data,
                    examples_answers=helper_examples_answers,
                    cot_reasoning=helper_cot_reasoning,
                    logprobs=True,
                    strategy=helper_strategy,
                    safe_mode=safe_mode,
                    show_progress_bar=True,
                    progress_bar_desc="Running helper LM",
                )
                _, helper_logprobs = helper_output.outputs, helper_output.logprobs
                assert helper_logprobs is not None
                formatted_helper_logprobs: LogprobsForFilterCascade = (
                    lotus.settings.helper_lm.format_logprobs_for_filter_cascade(helper_logprobs)
                )
                proxy_scores = calibrate_llm_logprobs(formatted_helper_logprobs.true_probs, cascade_args)
            elif proxy_model == ProxyModel.EMBEDDING_MODEL:
                if not lotus.settings.rm:
                    raise ValueError("RM must be set in settings")

                # TODO: How to handle multiple columns?
                search_df = self._obj.sem_search(col_li[0], formatted_usr_instr, K=len(self._obj), return_scores=True)
                proxy_scores = search_df["vec_scores_sim_score"].tolist()

            sample_indices, correction_factors = importance_sampling(proxy_scores, cascade_args)
            sample_df = self._obj.loc[sample_indices]
            sample_multimodal_data = task_instructions.df2multimodal_info(sample_df, col_li)
            sample_proxy_scores = [proxy_scores[i] for i in sample_indices]
            sample_correction_factors = correction_factors[sample_indices]

            pos_cascade_threshold, neg_cascade_threshold = learn_filter_cascade_thresholds(
                sample_multimodal_data=sample_multimodal_data,
                lm=lotus.settings.lm,
                formatted_usr_instr=formatted_usr_instr,
                default=default,
                cascade_args=cascade_args,
                proxy_scores=sample_proxy_scores,
                sample_correction_factors=sample_correction_factors,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                additional_cot_instructions=additional_cot_instructions,
            )

            stats["pos_cascade_threshold"] = pos_cascade_threshold
            stats["neg_cascade_threshold"] = neg_cascade_threshold

        if pos_cascade_threshold is not None and neg_cascade_threshold is not None:
            stats["filters_resolved_by_helper_model"] = 0
            stats["filters_resolved_by_large_model"] = 0

            high_conf_idxs = set()
            proxy_outputs = [False] * len(multimodal_data)

            # Set proxy_outputs where confidence is high
            for idx_i in range(len(proxy_scores)):
                true_prob = proxy_scores[idx_i]
                if true_prob >= pos_cascade_threshold or true_prob <= neg_cascade_threshold:
                    high_conf_idxs.add(idx_i)
                    proxy_outputs[idx_i] = (
                        True
                        if true_prob >= pos_cascade_threshold
                        else False
                        if true_prob <= neg_cascade_threshold
                        else proxy_outputs[idx_i]
                    )

            lotus.logger.info(f"Num routed to smaller model: {len(high_conf_idxs)}")
            stats["num_routed_to_helper_model"] = len(high_conf_idxs)

            outputs: list[bool] = [False] * len(multimodal_data)
            raw_outputs: list[str] = [""] * len(multimodal_data)
            explanations: list[str | None] = [None] * len(multimodal_data)
            scores: list[float] = [0.0] * len(multimodal_data)
            score_methods: list[str] = [""] * len(multimodal_data)

            for idx in high_conf_idxs:
                outputs[idx] = proxy_outputs[idx]
                scores[idx] = proxy_scores[idx]
                if proxy_model == ProxyModel.HELPER_LM:
                    score_methods[idx] = "HELPER_LM"
                else:
                    score_methods[idx] = "EMBEDDING_MODEL"

            # If using helper LM, get raw outputs and explanations
            if proxy_model == ProxyModel.HELPER_LM:
                assert all(isinstance(x, str) for x in helper_output.explanations) or all(
                    x is None for x in helper_output.explanations
                )
                for idx in high_conf_idxs:
                    raw_outputs[idx] = helper_output.raw_outputs[idx]
                    explanations[idx] = helper_output.explanations[idx]

            # Send low confidence samples to large LM if any
            low_conf_idxs = sorted([i for i in range(len(proxy_outputs)) if i not in high_conf_idxs])
            low_conf_multimodal_data = [multimodal_data[idx] for idx in low_conf_idxs]
            if low_conf_idxs:
                large_output = sem_filter(
                    low_conf_multimodal_data,
                    lotus.settings.lm,
                    formatted_usr_instr,
                    default=default,
                    examples_multimodal_data=examples_multimodal_data,
                    examples_answers=examples_answers,
                    cot_reasoning=cot_reasoning,
                    strategy=strategy,
                    safe_mode=safe_mode,
                    progress_bar_desc="Running predicate evals with oracle LM",
                    logprobs=return_scores,
                    additional_cot_instructions=additional_cot_instructions,
                )

                if return_scores:
                    assert large_output.logprobs is not None, "Logprobs must be returned if return_scores is True"
                    formatted_logprobs = lotus.settings.lm.format_logprobs_for_filter_cascade(large_output.logprobs)
                    large_probs = formatted_logprobs.true_probs

                for idx, large_idx in enumerate(low_conf_idxs):
                    outputs[large_idx] = large_output.outputs[idx]
                    raw_outputs[large_idx] = large_output.raw_outputs[idx]
                    explanations[large_idx] = large_output.explanations[idx]
                    if return_scores:
                        scores[large_idx] = large_probs[idx]
                        score_methods[large_idx] = "LM"

            stats["filters_resolved_by_helper_model"] += len(high_conf_idxs)
            stats["filters_resolved_by_large_model"] += len(low_conf_idxs)

        else:
            output = sem_filter(
                multimodal_data,
                lotus.settings.lm,
                formatted_usr_instr,
                default=default,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                strategy=strategy,
                safe_mode=safe_mode,
                show_progress_bar=True,
                progress_bar_desc=progress_bar_desc,
                logprobs=return_scores,
                additional_cot_instructions=additional_cot_instructions,
            )
            outputs = output.outputs
            raw_outputs = output.raw_outputs
            explanations = output.explanations

            if return_scores:
                assert output.logprobs is not None, "Logprobs must be returned if return_scores is True"
                formatted_logprobs = lotus.settings.lm.format_logprobs_for_filter_cascade(output.logprobs)
                scores = formatted_logprobs.true_probs
                score_methods = ["LM"] * len(multimodal_data)
            else:
                scores = [0.0] * len(multimodal_data)
                score_methods = [""] * len(multimodal_data)

        if not return_all:
            # find indices where output is True
            ids = [i for i, x in enumerate(outputs) if x]
            idx_ids = [self._obj.index[i] for i, x in enumerate(outputs) if x]
            lotus.logger.debug(f"ids: {ids}")
            lotus.logger.debug(f"idx_ids: {idx_ids}")

            [outputs[i] for i in ids]
            filtered_explanations = [explanations[i] for i in ids]
            filtered_raw_outputs = [raw_outputs[i] for i in ids]
            filtered_scores = [scores[i] for i in ids]
            filtered_score_methods = [score_methods[i] for i in ids]
            lotus.logger.debug(f"filtered_raw_outputs: {filtered_raw_outputs}")

            new_df = self._obj.iloc[ids]
            new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)
        else:

            def get_out_col_name(df, col_name):
                if col_name in df.columns:
                    i = 1
                    while f"{col_name}_{i}" in new_df.columns:
                        i += 1
                    return f"{col_name}_{i}"
                else:
                    return col_name

            new_df = self._obj.copy()
            new_df[get_out_col_name(new_df, "filter_label")] = outputs
            filtered_explanations = explanations
            filtered_raw_outputs = raw_outputs
            filtered_scores = scores
            filtered_score_methods = score_methods

        # return rows where output is True
        if return_explanations and return_raw_outputs:
            new_df["explanation" + suffix] = filtered_explanations
            new_df["raw_output" + suffix] = filtered_raw_outputs
        elif return_explanations:
            new_df["explanation" + suffix] = filtered_explanations
        elif return_raw_outputs:
            new_df["raw_output" + suffix] = filtered_raw_outputs

        if return_scores:
            new_df["score"] = filtered_scores
            new_df["score_method"] = filtered_score_methods

        if return_stats:
            return new_df, stats

        return new_df
