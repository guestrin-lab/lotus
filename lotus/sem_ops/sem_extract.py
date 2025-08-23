from typing import Any, Callable

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import LM
from lotus.templates import task_instructions
from lotus.types import LMOutput, PromptStrategy, SemanticExtractOutput, SemanticExtractPostprocessOutput
from lotus.utils import show_safe_mode

from .postprocessors import extract_postprocess


def sem_extract(
    docs: list[dict[str, Any]],
    model: LM,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
    return_explanations: bool = False,
    prompt_strategy: PromptStrategy | None = None,
) -> SemanticExtractOutput:
    """
    Extracts structured attributes and values from a list of documents using a language model.

    This function uses a language model to extract specific information from documents
    and return it in a structured format. It can extract multiple attributes at once
    and optionally include quotes from the source text.

    Args:
        docs (list[dict[str, Any]]): The list of documents to extract from. Each
            document should be a dictionary containing multimodal information
            (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for extraction.
            Must be properly configured with appropriate API keys and settings.
        output_cols (dict[str, str | None]): A mapping from desired output column
            names to optional descriptions. The descriptions help guide the model
            on what to extract. For example: {"sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"}.
        extract_quotes (bool, optional): Whether to extract supporting quotes from
            the source text for each extracted value. Defaults to False.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, cot_reasoning) and return
            SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Extracting".
        return_explanations (bool, optional): Whether to return explanations for
            the extraction decisions. Useful for debugging and understanding
            model reasoning. Defaults to False.
        prompt_strategy (PromptStrategy | None, optional): The prompt strategy to use.
            Configures chain-of-thought, demonstrations, and bootstrapping. Defaults to None.


    Returns:
        SemanticExtractOutput: An object containing the extracted outputs, raw
            outputs, and explanations (if requested).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> docs = [{"text": "The product is excellent with 5 stars"}]
        >>> model = LM(model="gpt-4o")
        >>> output_cols = {"sentiment": "positive/negative/neutral", "rating": "1-5 scale"}
        >>> result = sem_extract(docs, model, output_cols)
        >>> print(result.outputs)  # [{"sentiment": "positive", "rating": "5"}]

        >>> # Using PromptStrategy with chain-of-thought
        >>> from lotus.types import PromptStrategy
        >>> strat = PromptStrategy(cot=True)
        >>> result = sem_extract(docs, model, output_cols, prompt_strategy=strat)

        >>> # Using PromptStrategy with demonstrations
        >>> import pandas as pd
        >>> examples = pd.DataFrame({
        ...     'text': ['Great product!', 'Terrible service'],
        ...     'Answer': [{'sentiment': 'positive'}, {'sentiment': 'negative'}]
        ... })
        >>> strat = PromptStrategy(cot=True, dems=examples, max_dems=2)
        >>> result = sem_extract(docs, model, output_cols, prompt_strategy=strat)
    """
    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.extract_formatter(model, doc, output_cols, extract_quotes, prompt_strategy)
        lotus.logger.debug(f"input to model: {prompt}")
        lotus.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # check if safe_mode is enabled
    if safe_mode:
        estimated_cost = sum(model.count_tokens(input) for input in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    # call model
    lm_output: LMOutput = model(inputs, response_format={"type": "json_object"}, progress_bar_desc=progress_bar_desc)

    # post process results
    cot_reasoning = prompt_strategy is not None and prompt_strategy.cot
    postprocess_output = postprocessor(lm_output.outputs, model, cot_reasoning)
    lotus.logger.debug(f"raw_outputs: {lm_output.outputs}")
    lotus.logger.debug(f"outputs: {postprocess_output.outputs}")
    lotus.logger.debug(f"explanations: {postprocess_output.explanations}")
    if safe_mode:
        model.print_total_usage()

    return SemanticExtractOutput(
        raw_outputs=postprocess_output.raw_outputs,
        outputs=postprocess_output.outputs,
        explanations=postprocess_output.explanations,
    )


@pd.api.extensions.register_dataframe_accessor("sem_extract")
class SemExtractDataFrame:
    """
    Extract structured attributes and values from a DataFrame.

    This method performs structured information extraction on the DataFrame
    content using specified input columns and output column definitions.
    It can extract multiple attributes simultaneously and add them as new
    columns to the DataFrame.

    Args:
        input_cols (list[str]): The columns that the model should extract
            information from. These columns will be used as input to the
            language model.
        output_cols (dict[str, str | None]): A mapping from desired output
            column names to optional descriptions. The descriptions help guide
            the model on what to extract. For example: {"sentiment": "positive/negative/neutral",
            "confidence": "0-1 scale"}.
        extract_quotes (bool, optional): Whether to extract supporting quotes
            from the source text for each extracted value. Defaults to False.
        postprocessor (Callable, optional): A function to post-process the model
            outputs. Should take (outputs, model, return_explanations) and return
            SemanticExtractPostprocessOutput. Defaults to extract_postprocess.
        return_raw_outputs (bool, optional): Whether to include raw model
            outputs in the output DataFrame. Useful for debugging.
            Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Extracting".
        return_explanations (bool, optional): Whether to include explanations
            in the output DataFrame. Useful for debugging and understanding
            model reasoning. Defaults to False.
        prompt_strategy (PromptStrategy | None, optional): The prompt strategy to use.
            Configures chain-of-thought, demonstrations, and bootstrapping. Defaults to None.


    Returns:
        pd.DataFrame: A DataFrame containing the original data plus the
            extracted attributes as new columns.

    Raises:
        ValueError: If the language model is not configured, if specified
            input columns don't exist in the DataFrame, or if there are
            other configuration issues.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

        >>> df = pd.DataFrame({
        ...     'text': ['Great product!', 'Terrible service'],
        ...     'rating': [5, 1]
        ... })

        >>> df.sem_extract(
        ...     ['text'],
        ...     {'sentiment': 'positive/negative/neutral', 'emotion': 'joy/anger/sadness'}
        ... )
                    text  rating sentiment emotion
        0    Great product!    5  positive     joy
        1  Terrible service    1  negative   anger

        >>> # Using PromptStrategy with chain-of-thought
        >>> from lotus.types import PromptStrategy
        >>> strat = PromptStrategy(cot=True)
        >>> df.sem_extract(['text'], {'sentiment': 'positive/negative/neutral'}, prompt_strategy=strat)
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialize the semantic extraction accessor.

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
        input_cols: list[str],
        output_cols: dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable[
            [list[str], lotus.models.LM, bool], SemanticExtractPostprocessOutput
        ] = extract_postprocess,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
        return_explanations: bool = False,
        prompt_strategy: PromptStrategy | None = None,
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        # check that column exists
        for column in input_cols:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, input_cols)

        out = sem_extract(
            docs=multimodal_data,
            model=lotus.settings.lm,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            return_explanations=return_explanations,
            prompt_strategy=prompt_strategy,
        )

        new_df = self._obj.copy()
        indices = new_df.index.to_list()
        for i, output_dict in enumerate(out.outputs):
            if i >= len(indices):
                break
            for key, value in output_dict.items():
                if key not in new_df.columns:
                    new_df[key] = None
                new_df.loc[indices[i], key] = value

        if return_raw_outputs:
            new_df["raw_output"] = out.raw_outputs

        if return_explanations:
            new_df["explanation"] = out.explanations

        return new_df
