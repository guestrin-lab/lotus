import heapq
import re
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, PromptStrategy, SemanticTopKOutput
from lotus.utils import show_safe_mode


def get_match_prompt_binary(
    doc1: dict[str, Any],
    doc2: dict[str, Any],
    user_instruction: str,
    model: lotus.models.LM,
    prompt_strategy: PromptStrategy | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a binary comparison prompt for two documents.

    This function creates a prompt that asks the language model to compare two
    documents and select the one that better matches the user's instruction.
    It supports different reasoning strategies including chain-of-thought.

    Args:
        doc1 (dict[str, Any]): The first document to compare. Should contain
            multimodal information (text, images, etc.).
        doc2 (dict[str, Any]): The second document to compare. Should contain
            multimodal information (text, images, etc.).
        user_instruction (str): The natural language instruction that defines
            the comparison criteria.
        model (lotus.models.LM): The language model instance to use for comparison.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, CoT, or Demonstrations. Defaults to None.

    Returns:
        list[dict[str, Any]]: A list of message dictionaries formatted for the
            language model API.

    Example:
        >>> doc1 = {"text": "Machine learning tutorial"}
        >>> doc2 = {"text": "Data science guide"}
        >>> model = LM(model="gpt-4o")
        >>> prompt = get_match_prompt_binary(doc1, doc2, "Which is more relevant to AI?", model)
    """
    if prompt_strategy is not None and prompt_strategy.cot:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'First give your reasoning. Then you MUST end your output with "Answer: Document 1 or Document 2"\n'
            'You must pick a number and cannot say things like "None" or "Neither"\n'
            'Remember to explicitly state "Answer:" at the end before your choice.'
        )
    else:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'Respond only with the label of the document such as "Document NUMBER".\n'
            "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
            'You must pick a number and cannot say things like "None" or "Neither"'
        )
    prompt = [{"type": "text", "text": f"Question: {user_instruction}\n"}]
    for idx, doc in enumerate([doc1, doc2]):
        content_text, content_image_inputs = task_instructions.context_formatter(doc)
        prompt += [{"type": "text", "text": f"\nDocument {idx+1}:\n{content_text}"}, *content_image_inputs]

    if prompt_strategy is not None and prompt_strategy.cot and model.is_deepseek():
        deepseek_instructions = """Please think through your reasoning step by step, then provide your final answer.
        You must put your reasoning insdie the <think></think> tags, then provide your 
        final answer after the </think> tag with the format: Answer: your answer."""
        prompt += [{"type": "text", "text": f"\n{deepseek_instructions}"}]

    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    lotus.logger.debug(f"Prompt: {messages}")
    return messages


def parse_ans_binary(answer: str) -> tuple[bool, str]:
    """
    Parse a binary comparison answer from the language model.

    This function extracts the model's choice (Document 1 or Document 2) and any
    chain-of-thought reasoning from the response.

    Args:
        answer (str): The raw response from the language model.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if Document 1 was selected, False if Document 2 was selected
            - str: Any chain-of-thought reasoning found in the response

    Example:
        >>> parse_ans_binary("Document 1 is more relevant because it focuses on AI.")
        (True, "")
        >>> parse_ans_binary("<think>Both are relevant but Document 1 is more specific</think>Answer: Document 1")
        (True, "Both are relevant but Document 1 is more specific")
    """
    lotus.logger.debug(f"Response from model: {answer}")
    cot_explanation = ""
    try:
        think_start = answer.find("<think>")
        think_end = answer.find("</think>")

        if think_start != -1 and think_end != -1:
            cot_explanation = answer[think_start + len("<think>") : think_end].strip()
            answer = answer[think_end + len("</think>") :].strip()
        else:
            answer_idx = answer.lower().find("answer:")
            if answer_idx != -1:
                cot_explanation = answer[:answer_idx].strip()
                answer = answer[answer_idx:].strip()

        matches = list(re.finditer(r"Document[\s*](\d+)", answer, re.IGNORECASE))
        if len(matches) == 0:
            matches = list(re.finditer(r"(\d+)", answer, re.IGNORECASE))
        ans = int(matches[-1].group(1)) - 1
        if ans not in [0, 1]:
            lotus.logger.info(f"Could not parse {answer}")
            return True, cot_explanation
        return ans == 0, cot_explanation
    except Exception:
        lotus.logger.info(f"Could not parse {answer}")
        return True, cot_explanation


def compare_batch_binary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: lotus.models.LM,
    user_instruction: str,
    prompt_strategy: PromptStrategy | None = None,
) -> tuple[list[bool], list[str], int]:
    """
    Compare multiple pairs of documents using binary classification.

    This function processes a batch of document pairs, comparing each pair
    according to the user's instruction and returning the results.

    Args:
        pairs (list[tuple[dict[str, Any], dict[str, Any]]]): List of document
            pairs to compare. Each pair should contain two documents.
        model (lotus.models.LM): The language model instance to use for comparison.
        user_instruction (str): The natural language instruction that defines
            the comparison criteria.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.

    Returns:
        tuple[list[bool], list[str], int]: A tuple containing:
            - list[bool]: Results for each pair (True if first document wins, False otherwise)
            - list[str]: Explanations for each comparison
            - int: Total number of tokens used

    Example:
        >>> pairs = [({"text": "AI guide"}, {"text": "ML tutorial"})]
        >>> model = LM(model="gpt-4o")
        >>> results, explanations, tokens = compare_batch_binary(pairs, model, "Which is more relevant to beginners?")
    """
    match_prompts = []
    tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(
            get_match_prompt_binary(doc1, doc2, user_instruction, prompt_strategy=prompt_strategy, model=model)
        )
        tokens += model.count_tokens(match_prompts[-1])
    lm_results: LMOutput = model(match_prompts, show_progress_bar=False)
    result_explanations = list(map(parse_ans_binary, lm_results.outputs))
    results = [r[0] for r in result_explanations]
    explanations = [r[1] for r in result_explanations]
    return results, explanations, tokens


def compare_batch_binary_cascade(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: lotus.models.LM,
    user_instruction: str,
    cascade_threshold: float,
    prompt_strategy: PromptStrategy | None = None,
) -> tuple[list[bool], list[str], int, int, int]:
    """
    Compare multiple pairs of documents using a cascade approach.

    This function uses a two-stage approach: first a smaller/faster model makes
    predictions, then a larger/more accurate model is used for low-confidence cases.

    Args:
        pairs (list[tuple[dict[str, Any], dict[str, Any]]]): List of document
            pairs to compare. Each pair should contain two documents.
        model (lotus.models.LM): The large language model instance to use for
            high-confidence cases.
        user_instruction (str): The natural language instruction that defines
            the comparison criteria.
        cascade_threshold (float): Confidence threshold for using the large model.
            Cases below this threshold will use the helper model.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.

    Returns:
        tuple[list[bool], list[str], int, int, int]: A tuple containing:
            - list[bool]: Results for each pair
            - list[str]: Explanations for each comparison
            - int: Total tokens used by small model
            - int: Total tokens used by large model
            - int: Number of calls to large model

    Raises:
        ValueError: If the helper language model is not configured.

    Example:
        >>> pairs = [({"text": "AI guide"}, {"text": "ML tutorial"})]
        >>> model = LM(model="gpt-4o")
        >>> results, explanations, small_tokens, large_tokens, large_calls = compare_batch_binary_cascade(
        ...     pairs, model, "Which is more relevant?", 0.8
        ... )
    """
    match_prompts = []
    small_tokens = 0
    for doc1, doc2 in pairs:
        match_prompts.append(
            get_match_prompt_binary(doc1, doc2, user_instruction, prompt_strategy=prompt_strategy, model=model)
        )
        small_tokens += model.count_tokens(match_prompts[-1])

    helper_lm = lotus.settings.helper_lm
    if helper_lm is None:
        raise ValueError(
            "The helper language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
        )

    helper_output = helper_lm(match_prompts, kwargs={"logprobs": True})
    results = helper_output.outputs
    helper_logprobs = helper_output.logprobs
    assert helper_logprobs is not None
    formatted_logprobs = helper_lm.format_logprobs_for_cascade(helper_logprobs)
    helper_tokens = formatted_logprobs.tokens
    helper_confidences = formatted_logprobs.confidences

    parsed_results = []
    explanations = [""] * len(results)
    high_conf_idxs = set()
    for idx, res in enumerate(results):
        parsed_res = parse_ans_binary(res)
        parsed_results.append(parsed_res[0])
        explanations[idx] = parsed_res[1]

        # Find where docunent number is said and look at confidence
        for idx_j in range(len(helper_tokens[idx]) - 1, -1, -1):
            if helper_tokens[idx][idx_j].strip(" \n").isnumeric():
                conf = helper_confidences[idx][idx_j]
                if conf >= cascade_threshold:
                    high_conf_idxs.add(idx)

    large_tokens = 0
    num_large_calls = 0
    if len(high_conf_idxs) != len(helper_logprobs):
        # Send low confidence samples to large LM
        low_conf_idxs = sorted([i for i in range(len(helper_logprobs)) if i not in high_conf_idxs])

        large_match_prompts = []
        for i in low_conf_idxs:
            large_match_prompts.append(match_prompts[i])
            large_tokens += model.count_tokens(large_match_prompts[-1])

        large_lm_results: LMOutput = model(large_match_prompts)
        for idx, res in enumerate(large_lm_results.outputs):
            new_idx = low_conf_idxs[idx]
            parsed_res = parse_ans_binary(res)
            parsed_results[new_idx] = parsed_res[0]
            explanations[new_idx] = parsed_res[1]

        num_large_calls = len(low_conf_idxs)
    return parsed_results, explanations, small_tokens, large_tokens, num_large_calls


def llm_naive_sort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    prompt_strategy: PromptStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sort documents using a naive quadratic comparison approach.

    This function implements a simple sorting algorithm that compares every pair
    of documents and uses voting to determine the final order. While simple, it
    requires O(n²) comparisons and is not efficient for large datasets.

    Args:
        docs (list[dict[str, Any]]): The list of documents to sort. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for comparisons.
        user_instruction (str): The natural language instruction that defines
            the sorting criteria.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.

    Returns:
        SemanticTopKOutput: An object containing the sorted indexes and statistics.

    Example:
        >>> docs = [{"text": "AI guide"}, {"text": "ML tutorial"}, {"text": "Data science intro"}]
        >>> model = LM(model="gpt-4o")
        >>> result = llm_naive_sort(docs, model, "Sort by relevance to beginners")
        >>> print(result.indexes)  # [2, 1, 0] - most to least relevant
    """
    N = len(docs)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((docs[i], docs[j]))

    llm_calls = len(pairs)
    pbar = tqdm(
        total=llm_calls,
        desc="All-pairs comparisons",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} LM calls [{elapsed}<{remaining}]",
    )
    comparisons, explanations, tokens = compare_batch_binary(
        pairs, model, user_instruction, prompt_strategy=prompt_strategy
    )
    pbar.update(len(pairs))
    pbar.close()
    if safe_mode:
        show_safe_mode(tokens, llm_calls)
    votes = [0] * N
    idx = 0

    explanations_dict: dict[int, list[str]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if comparisons[idx]:
                votes[i] += 1
                explanations_dict[i].append(explanations[idx])
            else:
                votes[j] += 1
                explanations_dict[j].append(explanations[idx])
            idx += 1

    indexes = sorted(range(len(votes)), key=lambda i: votes[i], reverse=True)

    stats = {"total_tokens": tokens, "total_llm_calls": llm_calls, "explanations": explanations_dict}
    return SemanticTopKOutput(indexes=indexes, stats=stats)


def llm_quicksort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    K: int,
    embedding: bool = False,
    prompt_strategy: PromptStrategy | None = None,
    cascade_threshold: float | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sort documents using a quicksort-based approach optimized for top-K retrieval.

    This function implements a modified quicksort algorithm that only sorts the
    top K elements, making it more efficient than full sorting for top-K queries.
    It can also use embedding-based optimization for improved performance.

    Args:
        docs (list[dict[str, Any]]): The list of documents to sort. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for comparisons.
        user_instruction (str): The natural language instruction that defines
            the sorting criteria.
        K (int): The number of top documents to return.
        embedding (bool, optional): Whether to use embedding optimization for
            pivot selection. Defaults to False.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        cascade_threshold (float | None, optional): Confidence threshold for cascade
            filtering. If provided, uses a two-stage model approach. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.

    Returns:
        SemanticTopKOutput: An object containing the sorted indexes and statistics.

    Example:
        >>> docs = [{"text": "AI guide"}, {"text": "ML tutorial"}, {"text": "Data science intro"}]
        >>> model = LM(model="gpt-4o")
        >>> result = llm_quicksort(docs, model, "Sort by relevance to beginners", K=2)
        >>> print(result.indexes[:2])  # Top 2 most relevant documents
    """
    stats: dict[str, Any] = {}
    stats["total_tokens"] = 0
    stats["total_llm_calls"] = 0
    stats["explanations"] = {}
    if safe_mode:
        sample_prompt = get_match_prompt_binary(
            docs[0], docs[1], user_instruction, prompt_strategy=prompt_strategy, model=model
        )
        estimated_quickselect_calls = 2 * K
        estimated_quicksort_calls = 2 * len(docs) * np.log(len(docs))
        estimated_total_calls = estimated_quickselect_calls + estimated_quicksort_calls
        estimated_total_tokens = model.count_tokens(sample_prompt) * estimated_total_calls
        show_safe_mode(estimated_total_tokens, estimated_total_calls)

    if cascade_threshold is not None:
        stats["total_small_tokens"] = 0
        stats["total_large_tokens"] = 0
        stats["total_small_calls"] = 0
        stats["total_large_calls"] = 0

    def partition(indexes: list[int], low: int, high: int, K: int) -> int:
        nonlocal stats
        i = low - 1

        if embedding:
            # With embedding optimization
            if K <= high - low:
                pivot_value = heapq.nsmallest(K, indexes[low : high + 1])[-1]
            else:
                pivot_value = heapq.nsmallest(int((high - low + 1) / 2), indexes[low : high + 1])[-1]
            pivot_index = indexes.index(pivot_value)
        else:
            # Without embedding optimization
            pivot_index = np.random.randint(low, high + 1)
            pivot_value = indexes[pivot_index]

        pivot = docs[pivot_value]
        indexes[pivot_index], indexes[high] = indexes[high], indexes[pivot_index]

        pairs = [(docs[indexes[j]], pivot) for j in range(low, high)]
        if cascade_threshold is None:
            comparisons, explanations, tokens = compare_batch_binary(
                pairs, model, user_instruction, prompt_strategy=prompt_strategy
            )
            stats["total_tokens"] += tokens
            stats["total_llm_calls"] += len(pairs)

            for j, (doc1_is_better, explanation) in enumerate(zip(comparisons, explanations), start=low):
                doc_idx = indexes[j]
                if doc_idx not in stats["explanations"]:
                    stats["explanations"][doc_idx] = []
                stats["explanations"][doc_idx].append(explanation)
        else:
            comparisons, explanations, small_tokens, large_tokens, num_large_calls = compare_batch_binary_cascade(
                pairs,
                model,
                user_instruction,
                cascade_threshold,
                prompt_strategy=prompt_strategy,
            )

            stats["total_small_tokens"] += small_tokens
            stats["total_large_tokens"] += large_tokens
            stats["total_small_calls"] += len(pairs)
            stats["total_large_calls"] += num_large_calls

            for j, (doc1_is_better, explanation) in enumerate(zip(comparisons, explanations), start=low):
                doc_idx = indexes[j]
                if doc_idx not in stats["explanations"]:
                    stats["explanations"][doc_idx] = []
                stats["explanations"][doc_idx].append(explanation)

        for j, doc1_is_better in enumerate(comparisons, start=low):
            if doc1_is_better:
                i += 1
                indexes[i], indexes[j] = indexes[j], indexes[i]

        indexes[i + 1], indexes[high] = indexes[high], indexes[i + 1]
        return i + 1

    def quicksort_recursive(indexes: list[int], low: int, high: int, K: int) -> None:
        if high <= low:
            return

        num_comparisons = high - low
        pbar = tqdm(
            total=num_comparisons,
            desc="Quicksort comparisons",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} LM calls [{elapsed}<{remaining}]",
        )
        pi = partition(indexes, low, high, K)
        pbar.update(num_comparisons)
        pbar.close()
        left_size = pi - low
        if left_size + 1 >= K:
            quicksort_recursive(indexes, low, pi - 1, K)
        else:
            quicksort_recursive(indexes, low, pi - 1, left_size)
            quicksort_recursive(indexes, pi + 1, high, K - left_size - 1)

    indexes = list(range(len(docs)))
    quicksort_recursive(indexes, 0, len(indexes) - 1, K)

    return SemanticTopKOutput(indexes=indexes, stats=stats)


class HeapDoc:
    """
    Document wrapper for heap-based sorting operations.

    This class wraps documents for use in heap-based sorting algorithms.
    It tracks comparison statistics and provides a custom comparison method
    that uses language model calls to determine document ordering.

    Attributes:
        num_calls (int): Class variable tracking total number of LM calls.
        total_tokens (int): Class variable tracking total tokens used.
        prompt_strategy (PromptStrategy | None): Class variable for reasoning strategy.
        model (lotus.models.LM | None): Class variable for the language model.
        explanations (dict[int, list[str]]): Class variable storing explanations.
    """

    num_calls: int = 0
    total_tokens: int = 0
    prompt_strategy: PromptStrategy | None = None
    model: lotus.models.LM | None = None
    explanations: dict[int, list[str]] = {}

    def __init__(self, doc: dict[str, Any], user_instruction: str, idx: int) -> None:
        """
        Initialize a HeapDoc instance.

        Args:
            doc (dict[str, Any]): The document to wrap.
            user_instruction (str): The instruction for comparison.
            idx (int): The index of the document in the original list.
        """
        self.doc = doc
        self.user_instruction = user_instruction
        self.idx = idx

    def __lt__(self, other: "HeapDoc") -> bool:
        """
        Compare this document with another using language model.

        This method performs a binary comparison between two documents using
        the configured language model and user instruction.

        Args:
            other (HeapDoc): The other document to compare against.

        Returns:
            bool: True if this document is "less than" (worse than) the other.

        Raises:
            AssertionError: If the model is not configured.
        """
        assert HeapDoc.model is not None
        prompt = get_match_prompt_binary(
            self.doc, other.doc, self.user_instruction, prompt_strategy=HeapDoc.prompt_strategy, model=HeapDoc.model
        )
        HeapDoc.num_calls += 1
        HeapDoc.total_tokens += HeapDoc.model.count_tokens(prompt)
        result: LMOutput = HeapDoc.model([prompt], progress_bar_desc="Heap comparisons")
        is_better, explanation = parse_ans_binary(result.outputs[0])

        if self.idx not in HeapDoc.explanations:
            HeapDoc.explanations[self.idx] = []
        if other.idx not in HeapDoc.explanations:
            HeapDoc.explanations[other.idx] = []
        HeapDoc.explanations[self.idx].append(explanation)
        HeapDoc.explanations[other.idx].append(explanation)
        return is_better


def llm_heapsort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    K: int,
    prompt_strategy: PromptStrategy | None = None,
    safe_mode: bool = False,
) -> SemanticTopKOutput:
    """
    Sort documents using a heap-based approach for top-K retrieval.

    This function uses a min-heap to efficiently find the top K documents.
    It's particularly efficient for finding the top K elements without
    fully sorting the entire dataset.

    Args:
        docs (list[dict[str, Any]]): The list of documents to sort. Each document
            should be a dictionary containing multimodal information (text, images, etc.).
        model (lotus.models.LM): The language model instance to use for comparisons.
        user_instruction (str): The natural language instruction that defines
            the sorting criteria.
        K (int): The number of top documents to return.
        prompt_strategy (PromptStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.
        safe_mode (bool, optional): Whether to enable safe mode with cost estimation.
            Defaults to False.

    Returns:
        SemanticTopKOutput: An object containing the sorted indexes and statistics.

    Example:
        >>> docs = [{"text": "AI guide"}, {"text": "ML tutorial"}, {"text": "Data science intro"}]
        >>> model = LM(model="gpt-4o")
        >>> result = llm_heapsort(docs, model, "Sort by relevance to beginners", K=2)
        >>> print(result.indexes[:2])  # Top 2 most relevant documents
    """

    if safe_mode:
        sample_prompt = get_match_prompt_binary(
            docs[0], docs[1], user_instruction, prompt_strategy=prompt_strategy, model=model
        )
        estimated_heap_construction_calls = len(docs) * np.log(len(docs))
        estimated_top_k_extraction_calls = K * np.log(len(docs))
        estimated_total_calls = estimated_heap_construction_calls + estimated_top_k_extraction_calls
        estimated_total_cost = model.count_tokens(sample_prompt) * estimated_total_calls
        show_safe_mode(estimated_total_cost, estimated_total_calls)

    HeapDoc.num_calls = 0
    HeapDoc.total_tokens = 0
    HeapDoc.prompt_strategy = prompt_strategy
    HeapDoc.model = model
    HeapDoc.explanations = {}
    N = len(docs)
    heap = [HeapDoc(docs[idx], user_instruction, idx) for idx in range(N)]

    heap = heapq.nsmallest(K, heap)
    indexes = [heapq.heappop(heap).idx for _ in range(len(heap))]

    stats = {
        "total_tokens": HeapDoc.total_tokens,
        "total_llm_calls": HeapDoc.num_calls,
        "explanations": HeapDoc.explanations,
    }
    return SemanticTopKOutput(indexes=indexes, stats=stats)


@pd.api.extensions.register_dataframe_accessor("sem_topk")
class SemTopKDataframe:
    """
    Apply semantic top-K sorting over a DataFrame.

    This method performs semantic sorting on the DataFrame content using
    a natural language instruction and returns the top K most relevant rows.
    It supports multiple sorting algorithms and group-by operations.

    Args:
        user_instruction (str): The natural language instruction that defines
            the sorting criteria. Should describe how to rank the rows.
        K (int): The number of top rows to return.
        method (str, optional): The sorting algorithm to use. Options are:
            - "quick": Quicksort-based approach (default)
            - "heap": Heap-based approach
            - "naive": Naive quadratic approach
            - "quick-sem": Quicksort with semantic embedding optimization. Requires the passed column to be indexed with sem_index.
            Defaults to "quick".
        prompt_strategy (PromptStrategy | None, optional): The prompt strategy
            to use. Configures chain-of-thought, demonstrations, and bootstrapping. Defaults to None.
        group_by (list[str] | None, optional): Column names to group by before
            sorting. Each group will be sorted separately. Defaults to None.
        cascade_threshold (float | None, optional): Confidence threshold for
            cascade filtering. If provided, uses a two-stage model approach.
            Defaults to None.
        return_stats (bool, optional): Whether to return sorting statistics
            along with the results. Defaults to False.
        safe_mode (bool, optional): Whether to enable safe mode with cost
            estimation. Defaults to False.
        return_explanations (bool, optional): Whether to include explanations
            in the output DataFrame. Useful for debugging and understanding
            model reasoning. Defaults to False.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: A DataFrame
            containing the top K rows, or a tuple containing the DataFrame
            and statistics if return_stats is True.

    Raises:
        ValueError: If the language model is not configured, if specified
            columns don't exist in the DataFrame, or if an invalid method
            is specified.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"))
        >>> df = pd.DataFrame({
                'title': ['AI guide', 'ML tutorial', 'Data science intro'],
                'category': ['AI', 'ML', 'DS']
            })
        >>> df.sem_topk("The tutorial {title} is best for beginners", K=3)
                        title category
        0  Data science intro       DS
        1         ML tutorial       ML
        2            AI guide       AI
    """

    def __init__(self, pandas_obj: Any) -> None:
        """
        Initialize the semantic top-K accessor.

        Args:
            pandas_obj (Any): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (Any): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        pass

    @staticmethod
    def process_group(args):
        """
        Process a group of data for semantic top-K operations.

        This static method is used for parallel processing of grouped data.
        It applies semantic top-K to each group and returns the results.

        Args:
            args (tuple): A tuple containing (group, user_instruction, K, method,
                         prompt_strategy, group_by, cascade_threshold, return_stats).

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The top-K results
                for the group, optionally with statistics.
        """
        group, user_instruction, K, method, prompt_strategy, group_by, cascade_threshold, return_stats = args
        return group.sem_topk(
            user_instruction,
            K,
            method=method,
            prompt_strategy=prompt_strategy,
            group_by=None,
            cascade_threshold=cascade_threshold,
            return_stats=return_stats,
        )

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        K: int,
        method: str = "quick",
        prompt_strategy: PromptStrategy | None = None,
        group_by: list[str] | None = None,
        cascade_threshold: float | None = None,
        return_stats: bool = False,
        safe_mode: bool = False,
        return_explanations: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        model = lotus.settings.lm
        if model is None:
            raise ValueError(
                "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
            )

        lotus.logger.debug(f"Sorting DataFrame with user instruction: {user_instruction}")
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(f"Columns: {col_li}")

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        # Separate code path for grouping
        if group_by:
            grouped = self._obj.groupby(group_by)
            group_args = [
                (group, user_instruction, K, method, prompt_strategy, None, cascade_threshold, return_stats)
                for _, group in grouped
            ]

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=lotus.settings.parallel_groupby_max_threads) as executor:
                results = list(executor.map(SemTopKDataframe.process_group, group_args))

            if return_stats:
                new_df = pd.concat([res[0] for res in results])
                stats = {name: res[1] for name, res in zip(grouped.groups.keys(), results)}
                return new_df, stats
            else:
                return pd.concat(results)

        if method == "quick-sem":
            assert len(col_li) == 1, "Only one column can be used for embedding optimization"
            col_name = col_li[0]
            # Sort the dataframe by the column to be used for embedding optimization
            self._obj = self._obj.sem_index(col_name, f"{col_name}_lotus_index").sem_search(
                col_name, user_instruction, len(self._obj)
            )

        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        lotus.logger.debug(f"multimodal_data: {multimodal_data}")
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        if method in ["quick", "quick-sem"]:
            output = llm_quicksort(
                multimodal_data,
                model,
                formatted_usr_instr,
                K,
                embedding=method == "quick-sem",
                prompt_strategy=prompt_strategy,
                cascade_threshold=cascade_threshold,
                safe_mode=safe_mode,
            )
        elif method == "heap":
            output = llm_heapsort(
                multimodal_data,
                model,
                formatted_usr_instr,
                K,
                prompt_strategy=prompt_strategy,
                safe_mode=safe_mode,
            )
        elif method == "naive":
            output = llm_naive_sort(
                multimodal_data,
                model,
                formatted_usr_instr,
                prompt_strategy=prompt_strategy,
                safe_mode=safe_mode,
            )
        else:
            raise ValueError(f"Method {method} not recognized")

        new_df = self._obj.reset_index(drop=True)
        new_df = new_df.reindex(output.indexes).reset_index(drop=True)
        new_df = new_df.head(K)

        if return_explanations and prompt_strategy is not None and prompt_strategy.cot:
            explanations = []
            for idx in output.indexes[:K]:
                explanation = "No Comparison Made"
                if output.stats is not None:
                    # Retrieve the explanations dictionary safely
                    explanations_dict = output.stats.get("explanations", {})
                    if idx in explanations_dict:
                        explanation = "\n".join(explanations_dict[idx])
                explanations.append(explanation)
            new_df["explanation"] = explanations

        if return_stats:
            if output.stats is None:
                output.stats = {"explanations": {}}
            else:
                output.stats["explanations"] = {}
            return new_df, output.stats
        return new_df
