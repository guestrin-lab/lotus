
from lotus.models.reranker import Reranker
from lotus.types import RerankerOutput


class CrossEncoderReranker(Reranker):
    """
    CrossEncoder reranker model for document reranking.

    This class provides functionality to rerank documents using CrossEncoder models
    from Sentence Transformers. It supports batch processing for efficient reranking.

    Attributes:
        max_batch_size (int): Maximum batch size for reranking requests.
        model (CrossEncoder): The CrossEncoder model instance.
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        device: str | None = None,
        max_batch_size: int = 64,
    ) -> None:
        """
        Initialize the CrossEncoderReranker.

        Args:
            model: Name of the CrossEncoder model to use.
                   Defaults to "mixedbread-ai/mxbai-rerank-large-v1".
            device: Device to run the model on (e.g., "cuda", "cpu").
                    If None, uses default device. Defaults to None.
            max_batch_size: Maximum batch size for reranking requests. Defaults to 64.
        """
        self.max_batch_size: int = max_batch_size
        self._model_name = model
        self._device = device
        self._model = None  # Initialize model as None for lazy loading

    @property
    def model(self):
        """Lazy load the model when it's first accessed."""
        if self._model is None:
            # Only import CrossEncoder when needed
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)  # type: ignore # CrossEncoder has wrong type stubs
        return self._model

    def __call__(self, query: str, docs: list[str], K: int) -> RerankerOutput:
        """
        Rerank documents based on their relevance to the query.

        This method uses the CrossEncoder model to score and rerank documents
        based on their relevance to the given query. It returns the top K
        most relevant documents.

        Args:
            query: The query string to use for reranking.
            docs: List of document strings to rerank.
            K: Number of top documents to return after reranking.

        Returns:
            RerankerOutput: Object containing indices of the top K reranked documents.

        Raises:
            Exception: If the reranking process fails.
        """
        results = self.model.rank(query, docs, top_k=K, batch_size=self.max_batch_size, show_progress_bar=False)
        indices = [int(result["corpus_id"]) for result in results]
        return RerankerOutput(indices=indices)
