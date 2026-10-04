"""Fixed local BGE-M3 dense embeddings, loaded only when first needed."""
import math
from threading import Lock

DIMENSION = 1024
MODEL_ID = 'BAAI/bge-m3'


def validate_vector(vector):
    values = [float(value) for value in vector]
    if len(values) != DIMENSION or not all(math.isfinite(value) for value in values):
        raise ValueError('Expected 1024 finite dense embedding values')
    return values


class BgeM3Embedder:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self._model = None
        self._lock = Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_ID, cache_folder=self.cache_dir)
            vectors = self._model.encode(texts, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
        if len(vectors) != len(texts):
            raise ValueError('Embedding count does not match input count')
        return [validate_vector(vector) for vector in vectors]
