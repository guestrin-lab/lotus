"""
Semantic filter with hybrid optimization.

This module implements a hybrid approach to semantic filtering that combines:
1. Pattern analysis to identify filter patterns
2. Keyword-based filtering for sentiment analysis and other simple text patterns
3. Fallback to full LLM-based filtering for complex cases

This reduces cost and latency for compatible filter patterns
while maintaining F1-score.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import pandas as pd
import io
import re

import lotus
from lotus.cache import operator_cache
from lotus.hybrid_optimizer.code_generator import CodeGenerator
from lotus.templates import task_instructions
from lotus.types import SemanticFilterOutput
from .sem_filter import sem_filter


def sem_filter_hybrid(
    df: pd.DataFrame,
    multimodal_data: list[dict[str, Any]],
    model: lotus.models.LM,
    formatted_instruction: str,
    col_li: List[str],
    default: bool = True,
    examples_multimodal_data: Optional[List[Dict[str, Any]]] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    keyword_generation: bool = True,
    strategy: Optional[str] = None,
    logprobs: bool = False,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Filtering",
    additional_cot_instructions: str = "",
    sample_percentage_for_keywords: float = 0.1,
    num_keyword_calls: int = 1,
    accuracy_cost_preference: float = 0.5,
) -> SemanticFilterOutput:
    """
    Hybrid semantic filtering that combines pattern analysis, code generation, and LLM-based filtering.

    Args:
        df (pd.DataFrame): DataFrame to filter
        multimodal_data (list[dict[str, Any]]): Multimodal data for filtering
        model (lotus.models.LM): Language model
        formatted_instruction (str): Filtering instruction (with column placeholders removed)
        col_li (List[str]): List of columns extracted from original instruction
        default (bool): Default value for filtering
        examples_multimodal_data (Optional[List[Dict[str, Any]]]): Examples for few-shot learning
        examples_answers (Optional[List[bool]]): Answers for examples
        cot_reasoning (Optional[List[str]]): Reasoning for examples
        keyword_generation (bool): Whether to use keyword generation
        strategy (Optional[str]): Reasoning strategy
        logprobs (bool): Whether to return log probabilities
        safe_mode (bool): Whether to operate in safe mode
        show_progress_bar (bool): Whether to show progress bar
        progress_bar_desc (str): Progress bar description
        additional_cot_instructions (str): Additional instructions for chain-of-thought
        sample_percentage_for_keywords (float): Sample percentage for keyword generation example data
        num_keyword_calls (int): Number of LLM calls for keyword generation
        accuracy_cost_preference (float): Balance between accuracy (1.0) and cost savings (0.0). Default 0.5.

    Returns:
        SemanticFilterOutput: Filtering results
    """

    code_generator = CodeGenerator()

    outputs = []
    raw_outputs = []
    explanations = []
    final_mask = None
    method_used = "Unknown"

    # get sample size for keyword generation certain percentage of df, limit to 20
    n_samples = min(int(len(df) * sample_percentage_for_keywords), 20)

    if len(df) < n_samples:
        n_samples = len(df)

    sample_df_keywords = df.sample(n=n_samples, random_state=43)
    example_data_keywords = sample_df_keywords.to_string()


    # 1. keyword generation
    keyword_result_generated = False
    keyword_mask_intermediate = None
    keyword_explanation_intermediate = []
    keyword_matching_method = "Exact" 

    df_copy_for_keywords = df.copy()

    if final_mask is None and keyword_generation:
        keywords = code_generator.generate_keywords(
            df=df_copy_for_keywords,
            instruction=formatted_instruction,
            example_data=example_data_keywords,
            model=model,
            num_keyword_calls=num_keyword_calls
        )

    if keywords and len(keywords) >= 1:
        keyword_matching_method = "Exact"

        text_cols = col_li
        if text_cols:
            target_col = text_cols[0]
            escaped_keywords = [re.escape(kw) for kw in keywords]
            regex_pattern = r'(?i)' + '|'.join(escaped_keywords)

            temp_mask = df_copy_for_keywords[target_col].astype(
                str).str.contains(regex_pattern, na=False)
            if 0 < temp_mask.sum() < len(df):
                keyword_result_generated = True
                keyword_mask_intermediate = temp_mask.to_numpy()
                keyword_explanation_intermediate = [
                    f"Exactly matched keywords: {keywords}" if mask else "No exact keyword match" for mask in temp_mask]

    if keyword_result_generated:
        pref = accuracy_cost_preference
        # ~0.83
        accuracy_preference_threshold = 1 / 1.2  
        if pref < accuracy_preference_threshold:
            final_mask = keyword_mask_intermediate
            method_used = f"Keyword ({keyword_matching_method})"
            explanations = keyword_explanation_intermediate
        else:
            final_mask = None
            method_used = "Discarded Keyword Opt / Pending Fallback"
            explanations = []

    # 2.using sem_filter as fallback
    if final_mask is None:
        method_used = "Semantic Fallback"

        try:
            fallback_kwargs = model.kwargs.copy() if hasattr( model, 'kwargs') and model.kwargs else {}

            logging.debug(
                f"Creating new LM instance for fallback (Model: {model.model}, Kwargs: {fallback_kwargs})")
            fallback_model = lotus.models.LM(
                model=model.model,
                api_base=getattr(model, 'api_base', None),
                api_key=getattr(model, 'api_key', None),
                **fallback_kwargs
            )
            # Ensure custom provider info is copied if present
            if hasattr(model, 'custom_llm_provider') and model.custom_llm_provider:
                fallback_model.custom_llm_provider = model.custom_llm_provider
        except Exception:
            fallback_model = model

        buffer = io.StringIO()
        df.info(buf=buffer) 

        semantic_filter_result = sem_filter(
            multimodal_data,
            fallback_model,
            formatted_instruction,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            logprobs=logprobs,
            safe_mode=safe_mode,
            show_progress_bar=show_progress_bar,
            progress_bar_desc=f"{progress_bar_desc} (Semantic Fallback)",
            additional_cot_instructions=additional_cot_instructions,
        )
        return semantic_filter_result

    outputs = final_mask.tolist()
    raw_outputs = [
        f"{method_used} Match" if mask else f"No {method_used} Match" for mask in final_mask]

    if not explanations or len(explanations) != len(df):
        explanations = [f"{method_used} (No specific details)"] * len(df)

    return SemanticFilterOutput(
        outputs=outputs,
        raw_outputs=raw_outputs,
        explanations=explanations,
        logprobs=None,  
    )


@pd.api.extensions.register_dataframe_accessor("sem_filter_hybrid")
class SemFilterHybridDataframe:
    """DataFrame accessor for hybrid semantic filter."""

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        # verify that the object is a DataFrame
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
        examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Filtering",
        additional_cot_instructions: str = "",
        sample_percentage_for_keywords: float = 0.1,
        num_keyword_calls: int = 1,
        accuracy_cost_preference: float = 0.5,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Applies hybrid semantic filter over a dataframe.
        This uses pattern analysis and code generation to optimize filtering when possible.

        Args:
            user_instruction (str): The user instruction for filtering.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            return_explanations (bool): Whether to return explanations. Defaults to False.
            return_all (bool): Whether to return all outputs. Defaults to False.
            default (bool): The default value for filtering in case of parsing errors. Defaults to True.
            suffix (str): The suffix for the new columns. Defaults to "_filter".
            examples (Optional[pd.DataFrame]): The examples dataframe. Defaults to None.
            strategy (Optional[str]): The reasoning strategy. Defaults to None.
            return_stats (bool): Whether to return statistics. Defaults to False.
            safe_mode (bool): Whether to operate in safe mode. Defaults to False.
            progress_bar_desc (str): Progress bar description. Defaults to "Filtering".
            additional_cot_instructions (str): Additional instructions for the CoT. Defaults to "".
            sample_percentage_for_keywords (float): Sample percentage for keyword generation example data
            num_keyword_calls (int): Number of LLM calls for keyword generation
            accuracy_cost_preference (float): Balance between accuracy (1.0) and cost savings (0.0). Default 0.5.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]: The filtered dataframe or a tuple containing the filtered dataframe and statistics.
        """
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        stats: Dict[str, float] = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(
            self._obj, col_li)
        lotus.logger.debug(multimodal_data)
        formatted_usr_instr = lotus.nl_expression.nle2str(
            user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(
                examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot" and "Reasoning" in examples.columns:
                cot_reasoning = examples["Reasoning"].tolist()

        start_time = pd.Timestamp.now()

        semantic_filter_output = sem_filter_hybrid(
            self._obj,
            multimodal_data,
            lotus.settings.lm,
            formatted_usr_instr,
            col_li,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            keyword_generation=True,
            strategy=strategy,
            logprobs=False,
            safe_mode=safe_mode,
            show_progress_bar=True,
            progress_bar_desc=progress_bar_desc,
            additional_cot_instructions=additional_cot_instructions,
            sample_percentage_for_keywords=sample_percentage_for_keywords,
            num_keyword_calls=num_keyword_calls,
            accuracy_cost_preference=accuracy_cost_preference,
        )

        # performance
        end_time = pd.Timestamp.now()
        execution_time = (end_time - start_time).total_seconds()
        stats["execution_time"] = execution_time
        stats["num_docs"] = len(multimodal_data)

        outputs, raw_outputs, explanations = (
            semantic_filter_output.outputs,
            semantic_filter_output.raw_outputs,
            semantic_filter_output.explanations,
        )

        keep_idxs = []
        new_raw_outputs = []
        new_explanations = []

        for i, (output, raw_output, explanation) in enumerate(zip(outputs, raw_outputs, explanations)):
            if output:
                keep_idxs.append(i)
                new_raw_outputs.append(raw_output)
                new_explanations.append(explanation)

        result_df = self._obj.iloc[keep_idxs].copy()

        def get_out_col_name(df, col_name):
            if col_name in df.columns:
                return f"{col_name}{suffix}"
            return col_name

        # had some issues with the outputs so this is a workaround
        if return_raw_outputs:
            result_df[get_out_col_name(
                result_df, "raw_output")] = new_raw_outputs

        if return_explanations:
            result_df[get_out_col_name(
                result_df, "explanation")] = new_explanations

        if return_all:
            result_df[get_out_col_name(result_df, "outputs")] = [
                True] * len(result_df)
            non_keep_idxs = [i for i in range(
                len(self._obj)) if i not in keep_idxs]
            non_keep_df = self._obj.iloc[non_keep_idxs].copy()
            non_keep_raw_outputs = [raw_outputs[i] for i in non_keep_idxs]
            non_keep_explanations = [explanations[i] for i in non_keep_idxs]
            non_keep_df[get_out_col_name(non_keep_df, "outputs")] = [
                False] * len(non_keep_df)

            if return_raw_outputs:
                non_keep_df[get_out_col_name(
                    non_keep_df, "raw_output")] = non_keep_raw_outputs

            if return_explanations:
                non_keep_df[get_out_col_name(
                    non_keep_df, "explanation")] = non_keep_explanations

            result_df = pd.concat([result_df, non_keep_df])

        stats["num_results"] = len(result_df)
        stats["percentage"] = len(result_df) / \
            len(self._obj) if len(self._obj) > 0 else 0

        if return_stats:
            return result_df, stats
        return result_df
