from typing import Any, Callable

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticMapOutput, SemanticMapPostprocessOutput
from lotus.utils import show_safe_mode

from .postprocessors import map_postprocess


def sem_map(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    postprocessor: Callable[[list[str], bool, str], SemanticMapPostprocessOutput] = map_postprocess,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: str | None = None,
    safe_mode: bool = False,
    progress_bar_desc: str = "Mapping",
    model_type: str = "default",
) -> SemanticMapOutput:
    """
    Maps a list of documents to a list of outputs using a model.

    Args:
        docs (list[dict[str, Any]]): The list of documents to map.
        model (lotus.models.LM): The model to use.
        user_instruction (str): The user instruction for map.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to map_postprocess.
        examples_multimodal_data (list[dict[str, Any]] | None): The text for examples. Defaults to None.
        examples_answers (list[str] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.

    Returns:
        SemanticMapOutput: The outputs, raw outputs, and explanations.
    """
    # prepare model inputs
    inputs = []
    for doc in docs:
        if model_type == "deepseek":
            prompt = lotus.templates.task_instructions.deepseek_map_formatter(
                doc,
                user_instruction,
                examples_multimodal_data,
                examples_answers,
                cot_reasoning,
                strategy=strategy,
            )
        else:
            prompt = lotus.templates.task_instructions.map_formatter(
                doc, user_instruction, examples_multimodal_data, examples_answers, cot_reasoning, strategy=strategy
            )
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # check if safe_mode is enabled
    if safe_mode:
        estimated_cost = sum(model.count_tokens(input) for input in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    # call model
    lm_output: LMOutput = model(inputs, progress_bar_desc=progress_bar_desc)

    # post process results
    postprocess_output = postprocessor(lm_output.outputs, strategy in ["cot", "zs-cot"], model_type)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")
    if safe_mode:
        model.print_total_usage()

    return SemanticMapOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
    )


@pd.api.extensions.register_dataframe_accessor("sem_map")
class SemMapDataframe:
    """DataFrame accessor for semantic map."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        postprocessor: Callable[[list[str], bool, str], SemanticMapPostprocessOutput] = map_postprocess,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        strategy: str | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Mapping",
        model_type: str = "default",
    ) -> pd.DataFrame:
        """
        Applies semantic map over a dataframe.

        Args:
            user_instruction (str): The user instruction for map.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to map_postprocess.
            return_explanations (bool): Whether to return explanations. Defaults to False.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.
            suffix (str): The suffix for the new columns. Defaults to "_map".
            examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
            strategy (str | None): The reasoning strategy. Defaults to None.

        Returns:
            pd.DataFrame: The dataframe with the new mapped columns.
        """
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        col_li = lotus.nl_expression.parse_cols(user_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot":
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        output = sem_map(
            multimodal_data,
            lotus.settings.lm,
            formatted_usr_instr,
            postprocessor=postprocessor,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            model_type=model_type,
        )

        new_df = self._obj.copy()
        new_df[suffix] = output.outputs
        if return_explanations:
            new_df["explanation" + suffix] = output.explanations
        if return_raw_outputs:
            new_df["raw_output" + suffix] = output.raw_outputs

        return new_df
