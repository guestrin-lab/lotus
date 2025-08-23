from typing import Any, Callable

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import (
    LMOutput,
    PromptStrategy,
    SemanticMapOutput,
    SemanticMapPostprocessOutput,
)
from lotus.utils import show_safe_mode

from .postprocessors import map_postprocess


def sem_map(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    prompt_strategy: PromptStrategy | None = None,
    safe_mode: bool = False,
    progress_bar_desc: str = "Mapping",
) -> SemanticMapOutput:
    """
    Maps a list of documents to a list of outputs using a language model.

    This function applies a natural language instruction to each document in the
    input list, transforming them into new outputs. It supports few-shot learning
    through examples and various reasoning strategies including chain-of-thought.

    Args:
        docs (list[dict[str, Any]]): The list of documents to map. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for mapping.
            Must be properly configured with appropriate API keys and settings.
        user_instruction (str): The natural language instruction that guides the
            mapping process. This instruction tells the model how to transform
            each input document.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, use_cot) and return
            SemanticMapPostprocessOutput. Defaults to map_postprocess.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            documents for few-shot learning. Each example should have the same
            structure as the input docs. Defaults to None.
        examples_answers (list[str] | None, optional): Expected outputs for the
            example documents. Should have the same length as examples_multimodal_data.
            Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example documents. Used when strategy includes COT reasoning.
            Defaults to None.
        prompt_strategy (PromptStrategy | None, optional): The prompt strategy to use.
            Configures chain-of-thought, demonstrations, and bootstrapping. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Mapping".

    Returns:
        SemanticMapOutput: An object containing the processed outputs, raw outputs,
            and explanations (if applicable).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> docs = [{"text": "Document 1"}, {"text": "Document 2"}]
        >>> model = LM(model="gpt-4o")
        >>> result = sem_map(docs, model, "Summarize the text in one sentence")
        >>> print(result.outputs)
    """

    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.map_formatter(
            model,
            doc,
            user_instruction,
            examples_multimodal_data,
            examples_answers,
            cot_reasoning,
            prompt_strategy=prompt_strategy,
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
    postprocess_output = postprocessor(lm_output.outputs, model, prompt_strategy is not None and prompt_strategy.cot)
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
    """
    Apply semantic mapping over a DataFrame.

    This method performs semantic mapping on the DataFrame content using
    a natural language instruction. It can process specific columns identified
    in the instruction and supports few-shot learning through examples.

    Args:
        user_instruction (str): The natural language instruction that guides
            the mapping process. Should describe how to transform each row.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, use_cot) and return
            SemanticMapPostprocessOutput. Defaults to map_postprocess.
        return_explanations (bool, optional): Whether to include explanations
            in the output DataFrame. Useful for debugging and understanding
            model reasoning, when strategy is COT or ZS_COT. Defaults to False.
        return_raw_outputs (bool, optional): Whether to include raw model
            outputs in the output DataFrame. Useful for debugging.
            Defaults to False.
        suffix (str, optional): The suffix for the output column names.
            Defaults to "_map".
        examples (pd.DataFrame | None, optional): Example DataFrame for
            few-shot learning. Should have the same column structure as the
            input DataFrame plus an "Answer" column. Defaults to None.
        prompt_strategy (PromptStrategy | None, optional): The prompt strategy
            to use. Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Mapping".

    Returns:
        pd.DataFrame: A DataFrame containing the original data plus the mapped
            outputs. Additional columns may be added for explanations and raw
            outputs if requested.

    Raises:
        ValueError: If the language model is not configured, if specified
            columns don't exist in the DataFrame, or if the examples DataFrame
            doesn't have the required "Answer" column.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM, SentenceTransformersRM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))
        >>> df = pd.DataFrame({
        ...     'document': ['Harry is happy and love cats', 'Harry is feeling nauseous']
        ... })
        # Example 1: simple mapping
        >>> result1 = df.sem_map("Label the sentiment of Harry in the {document} as positive/negative/neutral. Answer in one word.")
        Mapping: 100%|████████████████████████████████████████████████████████████████████ 2/2 LM calls [00:00<00:00,  3.18it/s]
        document      _map
        0  Harry is happy and love cats  Positive
        1     Harry is feeling nauseous  Negative

        # Example 2: with zero-shot chain-of-thought (ZS-COT) reasoning
        >>> from lotus.types import PromptStrategy
        >>> df.sem_map("Label the sentiment of Harry in the {document} as positive/negative/neutral. Answer in one word.", return_explanations=True, prompt_strategy=PromptStrategy(cot=True))
        Mapping: 100%|████████████████████████████████████████████████████████████████████ 2/2 LM calls [00:02<00:00,  1.04s/it]
        document       _map                                    explanation_map
        0  Harry is happy and love cats   positive  Reasoning: The document states that "Harry is ...
        1  Harry is feeling nauseous   negative  Reasoning: The phrase "Harry is feeling nauseo...
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialize the semantic mapping accessor.

        Args:
            pandas_obj (pd.DataFrame): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (pd.DataFrame): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticMapPostprocessOutput] = map_postprocess,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        suffix: str = "_map",
        examples: pd.DataFrame | None = None,
        prompt_strategy: PromptStrategy | None = None,
        safe_mode: bool = False,
        progress_bar_desc: str = "Mapping",
    ) -> pd.DataFrame:
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

        # Handle examples and demonstrations
        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None

        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if prompt_strategy is not None and prompt_strategy.cot:
                return_explanations = True
                if "Reasoning" in examples.columns:
                    cot_reasoning = examples["Reasoning"].tolist()
                else:
                    cot_reasoning = ["Reasoning omitted"] * len(examples_answers)

        output = sem_map(
            multimodal_data,
            lotus.settings.lm,
            formatted_usr_instr,
            postprocessor=postprocessor,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            prompt_strategy=prompt_strategy,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
        )

        new_df = self._obj.copy()
        new_df[suffix] = output.outputs
        if return_explanations:
            new_df["explanation" + suffix] = output.explanations
        if return_raw_outputs:
            new_df["raw_output" + suffix] = output.raw_outputs

        return new_df
