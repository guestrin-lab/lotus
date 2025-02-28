import os
import pickle
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS


class FaissVS(VS):
    def __init__(self, factory_string: str = "Flat", metric=faiss.METRIC_INNER_PRODUCT):
        super().__init__()
        self.factory_string = factory_string
        self.metric = metric
        self.index_dir: str | None = None
        self.faiss_index: faiss.Index | None = None
        self.vecs: NDArray[np.float64] | None = None

    def index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]) -> None:
        self.faiss_index = faiss.index_factory(embeddings.shape[1], self.factory_string, self.metric)
        self.faiss_index.add(embeddings)
        self.index_dir = index_dir

        os.makedirs(index_dir, exist_ok=True)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(embeddings, fp)
        faiss.write_index(self.faiss_index, f"{index_dir}/index")

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.faiss_index = faiss.read_index(f"{index_dir}/index")
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs: NDArray[np.float64] = pickle.load(fp)
        return vecs[ids]

    def __call__(
        self, query_vectors: NDArray[np.float64], K: int, ids: list[int] | None = None, **kwargs: dict[str, Any]
    ) -> RMOutput:
        """
        Search for nearest neighbors using pre-embedded query vectors.

        Note:
          - The query vectors are expected to be pre-computed (e.g. via the RM module).
          - If the `ids` parameter is provided, the search is performed only on the subset of vectors
            corresponding to these ids.
        """
        if self.faiss_index is None or self.index_dir is None:
            raise ValueError("Index not loaded")

        if ids is not None:
            # Get the subset of vectors corresponding to the provided ids.
            subset_vecs = self.get_vectors_from_index(self.index_dir, ids)

            # Create a temporary FAISS index for the subset. This means we assume the same
            # dimensionality, factory, and metric as the main index.
            tmp_index = faiss.index_factory(subset_vecs.shape[1], self.factory_string, self.metric)
            tmp_index.add(subset_vecs)

            # Perform search on the temporary index.
            distances, sub_indices = tmp_index.search(query_vectors, K)

            # Remap the sub-indices to the original global ids.
            # Here we convert the list of filtered ids into a NumPy array so that we can index it.
            subset_ids = np.array(ids)
            indices = np.array([subset_ids[sub_indices[i]] for i in range(len(sub_indices))]).tolist()
        else:
            # Otherwise, search against the entire index.
            distances, indices = self.faiss_index.search(query_vectors, K)

        return RMOutput(distances=distances, indices=indices)  # type: ignore
