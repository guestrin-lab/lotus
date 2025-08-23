from typing import Any

import pandas as pd
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import CascadeArgs, PromptStrategy, SemanticJoinOutput
from lotus.utils import show_safe_mode

from .cascade_utils import calibrate_sem_sim_join, importance_sampling, learn_cascade_thresholds
from .sem_filter import sem_filter


def sem_join(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    prompt_strategy: PromptStrategy | None = None,
    safe_mode: bool = False,
    show_progress_bar: bool = True,
    progress_bar_desc: str = "Join comparisons",
) -> SemanticJoinOutput:
    """
    Joins two pandas Series using a language model based on semantic similarity.

    This function performs a semantic join between two Series by evaluating each
    pair of elements using a natural language instruction. It returns pairs that
    satisfy the join condition as determined by the language model.

    Args:
        l1 (pd.Series): The left Series to join. Contains the first set of
            elements to be compared.
        l2 (pd.Series): The right Series to join. Contains the second set of
            elements to be compared.
        ids1 (list[int]): The IDs corresponding to elements in l1. Used to
            track which elements match in the join results.
        ids2 (list[int]): The IDs corresponding to elements in l2. Used to
            track which elements match in the join results.
        col1_label (str): The label/name for the first column. Used in
            formatting the input to the language model.
        col2_label (str): The label/name for the second column. Used in
            formatting the input to the language model.
        model (lotus.models.LM): The language model instance to use for
            evaluating join conditions. Must be properly configured.
        user_instruction (str): The natural language instruction that defines
            the join condition. Should describe when two elements should be
            considered a match.
        examples_multimodal_data (list[dict[str, Any]] | None, optional): Example
            pairs for few-shot learning. Each example should contain both
            left and right elements. Defaults to None.
        examples_answers (list[bool] | None, optional): Expected boolean outputs
            for the example pairs. Should have the same length as
            examples_multimodal_data. Defaults to None.
        cot_reasoning (list[str] | None, optional): Chain-of-thought reasoning
            for the example pairs. Used when strategy includes COT reasoning.
            Defaults to None.
        default (bool, optional): The default value to use when the model
            output cannot be parsed as a boolean. Defaults to True.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy
            to use. Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        show_progress_bar (bool, optional): Whether to show a progress bar
            during processing. Defaults to True.
        progress_bar_desc (str, optional): Description for the progress bar.
            Defaults to "Join comparisons".

    Returns:
        SemanticJoinOutput: An object containing the join results (matching pairs),
            filter outputs, raw outputs, and explanations (if applicable).

    Raises:
        ValueError: If the model is not properly configured or if there are
            issues with the input parameters.

    Example:
        >>> l1 = pd.Series(['Machine learning', 'Data science'])
        >>> l2 = pd.Series(['AI', 'Statistics'])
        >>> model = LM(model="gpt-4o")
        >>> result = sem_join(l1, l2, [0, 1], [0, 1], 'left', 'right',
        ...                   model, "Are these topics related?")
        >>> print(result.join_results)  # List of matching pairs
    """
    filter_outputs = []
    all_raw_outputs = []
    all_explanations = []

    join_results = []

    left_multimodal_data = task_instructions.df2multimodal_info(l1.to_frame(col1_label), [col1_label])
    right_multimodal_data = task_instructions.df2multimodal_info(l2.to_frame(col2_label), [col2_label])

    if safe_mode:
        sample_docs = task_instructions.merge_multimodal_info([left_multimodal_data[0]], right_multimodal_data)
        estimated_tokens_per_call = model.count_tokens(
            lotus.templates.task_instructions.filter_formatter(
                model,
                sample_docs[0],
                user_instruction,
                examples_multimodal_data,
                examples_answers,
                cot_reasoning,
                prompt_strategy,
            )
        )
        estimated_total_calls = len(l1) * len(l2)
        estimated_total_cost = estimated_tokens_per_call * estimated_total_calls
        print("Sem_Join:")
        show_safe_mode(estimated_total_cost, estimated_total_calls)
    if show_progress_bar:
        pbar = tqdm(
            total=len(l1) * len(l2),
            desc=progress_bar_desc,
            bar_format="{l_bar}{bar} {n}/{total} LM Calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    all_docs = []
    all_ids1 = []
    all_ids2 = []
    for id1, i1 in zip(ids1, left_multimodal_data):
        modified_docs = task_instructions.merge_multimodal_info([i1], right_multimodal_data)
        all_docs.extend(modified_docs)
        all_ids1.extend([id1] * len(modified_docs))
        all_ids2.extend(ids2)

    output = sem_filter(
        all_docs,
        model,
        user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        prompt_strategy=prompt_strategy,
        show_progress_bar=False,
    )

    outputs = output.outputs
    raw_outputs = output.raw_outputs
    explanations = output.explanations

    filter_outputs.extend(outputs)
    all_raw_outputs.extend(raw_outputs)
    all_explanations.extend(explanations)

    join_results.extend(
        [
            (all_ids1[i], all_ids2[i], explanation)
            for i, (output, explanation) in enumerate(zip(outputs, explanations))
            if output
        ]
    )

    if show_progress_bar:
        pbar.update(len(l1) * len(l2))
        pbar.close()

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
    )


def sem_join_cascade(
    l1: pd.Series,
    l2: pd.Series,
    ids1: list[int],
    ids2: list[int],
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    prompt_strategy: PromptStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticJoinOutput:
    """
    Joins two series using a cascade helper model and a oracle model.

    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        ids1 (list[int]): The ids for the first series.
        ids2 (list[int]): The ids for the second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        user_instruction (str): The user instruction for join.
        cascade_args (CascadeArgs): The cascade arguments.
        examples_multimodal_data (list[dict[str, Any]] | None): The examples multimodal data. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        map_instruction (str | None): The map instruction. Defaults to None.
        map_examples (pd.DataFrame | None): The map examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (str | None): The reasoning strategy. Defaults to None.

    Returns:
        SemanticJoinOutput: The join results, filter outputs, all raw outputs, all explanations, and stats.

        Note that filter_outputs, all_raw_outputs, and all_explanations are empty list because
        the helper model do not generate these outputs.

        SemanticJoinOutput.stats:
            join_resolved_by_helper_model: total number of join records resolved by the helper model
            join_helper_positive: number of high confidence positive results from the helper model
            join_helper_negative: number of high confidence negative results from the helper model
            join_resolved_by_large_model: total number of joins resolved by the oracle model
            optimized_join_cost: number of LM calls from finding optimal join plan
            total_LM_calls: the total number of LM calls from join cascade, ie: optimized_join_cost + join_resolved_by_helper_model
    """
    filter_outputs: list[bool] = []
    all_raw_outputs: list[str] = []
    all_explanations: list[str | None] = []

    join_results: list[tuple[int, int, str | None]] = []
    num_helper = 0
    num_large = 0

    # Determine the join plan
    helper_high_conf, helper_low_conf, num_helper_high_conf_neg, join_optimization_cost = join_optimizer(
        l1,
        l2,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        map_instruction=map_instruction,
        map_examples=map_examples,
        cot_reasoning=cot_reasoning,
        default=default,
        prompt_strategy=prompt_strategy,
    )

    num_helper = len(helper_high_conf)
    num_large = len(helper_low_conf)

    if safe_mode:
        # TODO: implement safe mode
        lotus.logger.warning("Safe mode is not implemented yet.")

    # Accept helper results with high confidence
    join_results = [(row["_left_id"], row["_right_id"], None) for _, row in helper_high_conf.iterrows()]

    pbar = tqdm(
        total=num_large,
        desc="Running predicate evals with oracle model",
        bar_format="{l_bar}{bar} {n}/{total} LM calls [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    # Send low confidence rows to large LM
    left_multimodal_data = task_instructions.df2multimodal_info(
        helper_low_conf[[col1_label]].drop_duplicates(), [col1_label]
    )

    all_docs = []
    all_ids1 = []
    all_ids2 = []
    for id1, i1 in zip(helper_low_conf["_left_id"].unique(), left_multimodal_data):
        rows_for_l1 = helper_low_conf[helper_low_conf["_left_id"] == id1]
        modified_docs = task_instructions.merge_multimodal_info(
            [i1], task_instructions.df2multimodal_info(rows_for_l1[[col2_label]], [col2_label])
        )
        all_docs.extend(modified_docs)
        all_ids1.extend([id1] * len(modified_docs))
        all_ids2.extend(rows_for_l1["_right_id"].tolist())

    output = sem_filter(
        all_docs,
        model,
        user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        prompt_strategy=prompt_strategy,
        show_progress_bar=True,
    )

    pbar.update(num_large)
    pbar.close()

    join_results.extend(
        [
            (all_ids1[i], all_ids2[i], explanation)
            for i, (output, explanation) in enumerate(zip(output.outputs, output.explanations))
            if output
        ]
    )

    lotus.logger.debug(f"outputs: {filter_outputs}")
    lotus.logger.debug(f"explanations: {all_explanations}")

    # Log join cascade stats:
    stats = {
        "join_resolved_by_helper_model": num_helper + num_helper_high_conf_neg,
        "join_helper_positive": num_helper,
        "join_helper_negative": num_helper_high_conf_neg,
        "join_resolved_by_large_model": num_large,
        "optimized_join_cost": join_optimization_cost,
        "total_LM_calls": join_optimization_cost + num_large,
    }

    return SemanticJoinOutput(
        join_results=join_results,
        filter_outputs=filter_outputs,
        all_raw_outputs=all_raw_outputs,
        all_explanations=all_explanations,
        stats=stats,
    )


def run_sem_sim_join(l1: pd.Series, l2: pd.Series, col1_label: str, col2_label: str) -> pd.DataFrame:
    """
    Wrapper function to run sem_sim_join in sem_join then calibrate the scores for approximate join

    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.

    Returns:
        pd.DataFrame: The similarity join results.
    """
    # Transform the series into DataFrame
    if isinstance(l1, pd.Series):
        l1_df = l1.to_frame(name=col1_label)
    elif isinstance(l1, pd.DataFrame):
        l1_df = l1
    else:
        lotus.logger.error("l1 must be a pandas Series or DataFrame")

    l2_df = l2.to_frame(name=col2_label)
    l2_df = l2_df.sem_index(col2_label, f"{col2_label}_index")

    K = len(l2)
    # Run sem_sim_join as helper on the sampled data
    out = l1_df.sem_sim_join(l2_df, left_on=col1_label, right_on=col2_label, K=K, keep_index=True)

    # Correct helper scores
    out["_scores"] = calibrate_sem_sim_join(out["_scores"].tolist())
    return out


def map_l1_to_l2(
    l1: pd.Series,
    col1_label: str,
    col2_label: str,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Wrapper function to run sem_map in sem_join.

    Args:
        l1 (pd.Series): The first series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        map_instruction (str): The map instruction. Defaults to None.
        map_examples (pd.DataFrame): The map examples. Defaults to None.

    Returns:
        tuple[pd.DataFrame, str]: The mapped DataFrame and the mapped column name.
    """
    if ":left" in col1_label:
        real_left_on = col1_label.split(":left")[0]
    else:
        real_left_on = col1_label

    if ":right" in col2_label:
        real_right_on = col2_label.split(":right")[0]
    else:
        real_right_on = col2_label

    inst = ""
    if map_instruction:
        inst = map_instruction
    else:
        default_map_instruction = f"Given {{{real_left_on}}}, identify the most relevant {real_right_on}. Always write your answer as a list of 2-10 comma-separated {real_right_on}."
        inst = default_map_instruction

    # Transform l1 into DataFrame for sem_map
    l1_df = l1.to_frame(name=real_left_on)
    mapped_col1_name = f"_{col1_label}"

    # Map l1 to l2
    out = l1_df.sem_map(inst, suffix=mapped_col1_name, examples=map_examples, progress_bar_desc="Mapping examples")
    out = out.rename(columns={real_left_on: col1_label})

    return out, mapped_col1_name


def join_optimizer(
    l1: pd.Series,
    l2: pd.Series,
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    map_instruction: str | None = None,
    map_examples: pd.DataFrame | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    prompt_strategy: PromptStrategy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Find most cost-effective join plan between Search-Filter and Map-Search-Filter
    while satisfying the recall and precision target.

    Args:
        l1 (pd.Series): The first series.
        l2 (pd.Series): The second series.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        user_instruction (str): The user instruction for join.
        cascade_args (CascadeArgs): The cascade arguments.
        examples_multimodal_data (list[dict[str, Any]] | None): The examples multimodal data. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        map_instruction (str | None): The map instruction. Defaults to None.
        map_examples (pd.DataFrame | None): The map examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (str | None): The reasoning strategy. Defaults to None.

    returns:
        tuple[pd.DataFrame, pd.DataFrame]: The high confidence and low confidence join results.
        int: The number of high confidence negative results.
        int: The number of LM calls from optimizing join plan.
    """

    # Helper is currently default to similiarity join
    if lotus.settings.helper_lm is not None:
        lotus.logger.debug("Helper model is not supported yet. Default to similarity join.")

    # Learn search-filter thresholds
    sf_helper_join = run_sem_sim_join(l1, l2, col1_label, col2_label)
    sf_t_pos, sf_t_neg, sf_learn_cost = learn_join_cascade_threshold(
        sf_helper_join,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        prompt_strategy=prompt_strategy,
    )
    sf_high_conf = sf_helper_join[sf_helper_join["_scores"] >= sf_t_pos]
    sf_high_conf_neg = len(sf_helper_join[sf_helper_join["_scores"] <= sf_t_neg])
    sf_low_conf = sf_helper_join[(sf_helper_join["_scores"] < sf_t_pos) & (sf_helper_join["_scores"] > sf_t_neg)]
    sf_cost = len(sf_low_conf)

    # Learn map-search-filter thresholds
    mapped_l1, mapped_col1_label = map_l1_to_l2(
        l1, col1_label, col2_label, map_instruction=map_instruction, map_examples=map_examples
    )
    msf_helper_join = run_sem_sim_join(mapped_l1, l2, mapped_col1_label, col2_label)
    msf_t_pos, msf_t_neg, msf_learn_cost = learn_join_cascade_threshold(
        msf_helper_join,
        col1_label,
        col2_label,
        model,
        user_instruction,
        cascade_args,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        cot_reasoning=cot_reasoning,
        default=default,
        prompt_strategy=prompt_strategy,
    )
    msf_high_conf = msf_helper_join[msf_helper_join["_scores"] >= msf_t_pos]
    msf_high_conf_neg = len(msf_helper_join[msf_helper_join["_scores"] <= msf_t_neg])
    msf_low_conf = msf_helper_join[(msf_helper_join["_scores"] < msf_t_pos) & (msf_helper_join["_scores"] > msf_t_neg)]
    msf_cost = len(msf_low_conf)
    msf_learn_cost += len(l1)  # cost from map l1 to l2

    # Select the cheaper join plan
    lotus.logger.info("Join Optimizer: plan cost analysis:")
    lotus.logger.info(f"    Search-Filter: {sf_cost} LLM calls.")
    lotus.logger.info(
        f"    Search-Filter: accept {len(sf_high_conf)} helper positive results, {sf_high_conf_neg} helper negative results."
    )
    lotus.logger.info(f"    Map-Search-Filter: {msf_cost} LLM calls.")
    lotus.logger.info(
        f"    Map-Search-Filter: accept {len(msf_high_conf)} helper positive results, {msf_high_conf_neg} helper negative results."
    )

    learning_cost = sf_learn_cost + msf_learn_cost
    if sf_cost < msf_cost:
        lotus.logger.info("Proceeding with Search-Filter")
        sf_high_conf = sf_high_conf.sort_values(by="_scores", ascending=False)
        sf_low_conf = sf_low_conf.sort_values(by="_scores", ascending=False)
        return sf_high_conf, sf_low_conf, sf_high_conf_neg, learning_cost
    else:
        lotus.logger.info("Proceeding with Map-Search-Filter")
        msf_high_conf = msf_high_conf.sort_values(by="_scores", ascending=False)
        msf_low_conf = msf_low_conf.sort_values(by="_scores", ascending=False)
        return msf_high_conf, msf_low_conf, msf_high_conf_neg, learning_cost


def learn_join_cascade_threshold(
    helper_join: pd.DataFrame,
    col1_label: str,
    col2_label: str,
    model: lotus.models.LM,
    user_instruction: str,
    cascade_args: CascadeArgs,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answers: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    default: bool = True,
    prompt_strategy: PromptStrategy | None = None,
) -> tuple[float, float, int]:
    """
    Extract a small sample of the data and find the optimal threshold pair that satisfies the recall and
    precision target.

    Args:
        helper_join (pd.DataFrame): The helper join results.
        cascade_args (CascadeArgs): The cascade arguments.
        col1_label (str): The label for the first column.
        col2_label (str): The label for the second column.
        model (lotus.models.LM): The language model.
        user_instruction (str): The user instruction for join.
        cascade_args (CascadeArgs): The cascade arguments.
        examples_multimodal_data (list[dict[str, Any]] | None): The examples multimodal data. Defaults to None.
        examples_answers (list[bool] | None): The answers for examples. Defaults to None.
        cot_reasoning (list[str] | None): The reasoning for CoT. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        strategy (str | None): The reasoning strategy. Defaults to None.
    Returns:
        tuple: The positive threshold, negative threshold, and the number of LM calls from learning thresholds.
    """
    # Sample a small subset of the helper join result
    helper_scores = helper_join["_scores"].tolist()

    sample_indices, correction_factors = importance_sampling(helper_scores, cascade_args)
    lotus.logger.info(f"Sampled {len(sample_indices)} out of {len(helper_scores)} helper join results.")

    sample_df = helper_join.iloc[sample_indices]
    sample_scores = sample_df["_scores"].tolist()
    sample_correction_factors = correction_factors[sample_indices]

    col_li = [col1_label, col2_label]
    sample_multimodal_data = task_instructions.df2multimodal_info(sample_df, col_li)

    try:
        output = sem_filter(
            sample_multimodal_data,
            model,
            user_instruction,
            default=default,
            examples_multimodal_data=examples_multimodal_data,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            prompt_strategy=prompt_strategy,
            progress_bar_desc="Running oracle for threshold learning",
        )

        (pos_threshold, neg_threshold), _ = learn_cascade_thresholds(
            proxy_scores=sample_scores,
            oracle_outputs=output.outputs,
            sample_correction_factors=sample_correction_factors,
            cascade_args=cascade_args,
        )

        lotus.logger.info(f"Learned cascade thresholds: {(pos_threshold, neg_threshold)}")

    except Exception as e:
        lotus.logger.error(f"Error while learning filter cascade thresholds: {e}")
        lotus.logger.error("Default to full join.")
        return 1.0, 0.0, len(sample_indices)

    return pos_threshold, neg_threshold, len(sample_indices)


@pd.api.extensions.register_dataframe_accessor("sem_join")
class SemJoinDataframe:
    """
    Applies semantic join over a dataframe.

    Args:
        other (pd.DataFrame | pd.Series): The other dataframe or series to join with.
        join_instruction (str): The user instruction for join.
        return_explanations (bool): Whether to return explanations. Defaults to False.
        how (str): The type of join to perform. Defaults to "inner".
        suffix (str): The suffix for the new columns. Defaults to "_join".
        examples (pd.DataFrame | None): The examples dataframe. Defaults to None.
        strategy (str | None): The reasoning strategy. Defaults to None.
        default (bool): The default value for the join in case of parsing errors. Defaults to True.
        cascade_args (CascadeArgs | None): The arguments for join cascade. Defaults to None.
            recall_target (float | None): The target recall. Defaults to None.
            precision_target (float | None): The target precision when cascading. Defaults to None.
            sampling_percentage (float): The percentage of the data to sample when cascading. Defaults to 0.1.
            failure_probability (float): The failure probability when cascading. Defaults to 0.2.
            map_instruction (str): The map instruction when cascading. Defaults to None.
            map_examples (pd.DataFrame): The map examples when cascading. Defaults to None.
        return_stats (bool): Whether to return stats. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe with the new joined columns.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

        >>> df1 = pd.DataFrame({
                'article': ['Machine learning tutorial', 'Data science guide', 'Python basics', 'AI in finance', 'Cooking healthy food', "Recipes for the holidays"],
            })
        >>> df2 = pd.DataFrame({
                'category': ['Computer Science', 'AI', 'Cooking'],
            })

        >>> df1.sem_join(df2, "the {article} belongs to the {category}.")
        Join comparisons: 100%|█████████████████████████████████████████████████████████ 18/18 LM Calls [00:05<00:00,  3.57it/s]
                                article          category
        0  Machine learning tutorial  Computer Science
        0  Machine learning tutorial                AI
        1         Data science guide  Computer Science
        1         Data science guide                AI
        2              Python basics  Computer Science
        2              Python basics                AI
        3              AI in finance  Computer Science
        3              AI in finance                AI
        4       Cooking healthy food           Cooking
        5   Recipes for the holidays           Cooking
    """

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        other: pd.DataFrame | pd.Series,
        join_instruction: str,
        return_explanations: bool = False,
        how: str = "inner",
        suffix: str = "_join",
        examples: pd.DataFrame | None = None,
        prompt_strategy: PromptStrategy | None = None,
        default: bool = True,
        cascade_args: CascadeArgs | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Join comparisons",
    ) -> pd.DataFrame:
        model = lotus.settings.lm
        if model is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        if how != "inner":
            raise NotImplementedError("Only inner join is currently supported")

        cols = lotus.nl_expression.parse_cols(join_instruction)
        left_on = None
        right_on = None
        for col in cols:
            if ":left" in col:
                left_on = col
                real_left_on = col.split(":left")[0]
            elif ":right" in col:
                right_on = col
                real_right_on = col.split(":right")[0]

        if left_on is None:
            for col in cols:
                if col in self._obj.columns:
                    left_on = col
                    real_left_on = col

                    if col in other.columns:
                        raise ValueError("Column found in both dataframes")
                    break
        if right_on is None:
            for col in cols:
                if col in other.columns:
                    right_on = col
                    real_right_on = col

                    if col in self._obj.columns:
                        raise ValueError("Column found in both dataframes")
                    break

        assert left_on is not None, "Column not found in left dataframe"
        assert right_on is not None, "Column not found in right dataframe"

        examples_multimodal_data = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_multimodal_data = task_instructions.df2multimodal_info(examples, [real_left_on, real_right_on])
            examples_answers = examples["Answer"].tolist()

            if prompt_strategy is not None and prompt_strategy.cot:
                return_explanations = True
                cot_reasoning = examples["Reasoning"].tolist()

        num_full_join = len(self._obj) * len(other)

        if (
            (cascade_args is not None)
            and (cascade_args.recall_target is not None or cascade_args.precision_target is not None)
            and (num_full_join >= cascade_args.min_join_cascade_size)
        ):
            cascade_args.recall_target = 1.0 if cascade_args.recall_target is None else cascade_args.recall_target
            cascade_args.precision_target = (
                1.0 if cascade_args.precision_target is None else cascade_args.precision_target
            )
            output = sem_join_cascade(
                self._obj[real_left_on],
                other[real_right_on],
                self._obj.index,
                other.index,
                left_on,
                right_on,
                model,
                join_instruction,
                cascade_args,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                map_instruction=cascade_args.map_instruction,
                map_examples=cascade_args.map_examples,
                cot_reasoning=cot_reasoning,
                default=default,
                prompt_strategy=prompt_strategy,
                safe_mode=safe_mode,
            )
        else:
            output = sem_join(
                self._obj[real_left_on],
                other[real_right_on],
                self._obj.index,
                other.index,
                left_on,
                right_on,
                model,
                join_instruction,
                examples_multimodal_data=examples_multimodal_data,
                examples_answers=examples_answers,
                cot_reasoning=cot_reasoning,
                default=default,
                prompt_strategy=prompt_strategy,
                safe_mode=safe_mode,
                progress_bar_desc=progress_bar_desc,
            )
        join_results = output.join_results
        all_raw_outputs = output.all_raw_outputs

        lotus.logger.debug(f"join_results: {join_results}")
        lotus.logger.debug(f"all_raw_outputs: {all_raw_outputs}")

        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = self._obj.index
        df2["_right_id"] = other.index
        # add suffix to column names
        for col in df1.columns:
            if col in df2.columns:
                df1.rename(columns={col: col + ":left"}, inplace=True)
                df2.rename(columns={col: col + ":right"}, inplace=True)

        if return_explanations:
            temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", f"explanation{suffix}"])
        else:
            temp_df = pd.DataFrame([(jr[0], jr[1]) for jr in join_results], columns=["_left_id", "_right_id"])

        joined_df = (
            df1.join(temp_df.set_index("_left_id"), how="right", on="_left_id")
            .join(df2.set_index("_right_id"), how="left", on="_right_id")
            .drop(columns=["_left_id", "_right_id"])
        )

        if output.stats and return_stats:
            return joined_df, output.stats

        return joined_df
