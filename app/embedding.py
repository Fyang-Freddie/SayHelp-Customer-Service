"""Fixed local BGE-M3 dense embeddings, loaded only when first needed."""
import math
from threading import Lock

DIMENSION = 1024
MODEL_ID = 'BAAI/bge-m3'


class EmbeddingInputTooLongError(ValueError):
    """Trusted input-length diagnostic containing no source body/library error."""
    def __init__(self, input_index, token_count, limit, *, chunk_id=None,
                 content_type=None, section_path=None):
        self.input_index = input_index
        self.token_count = token_count
        self.limit = limit
        source = '' if chunk_id is None else (
            f'chunk {chunk_id} ({content_type or "unknown"}, '
            f'section {section_path or "<none>"}): ')
        super().__init__(f'{source}input {input_index}: {token_count} tokens exceeds '
                         f'BGE-M3 limit {limit}; text was not truncated')


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
            # Count the complete input, including special tokens. encode() would
            # otherwise silently truncate at max_seq_length before inference.
            tokenized = self._model.tokenizer(texts, truncation=False,
                                              add_special_tokens=True, padding=False)
            for index, tokens in enumerate(tokenized['input_ids'], start=1):
                if len(tokens) > self._model.max_seq_length:
                    raise EmbeddingInputTooLongError(index, len(tokens), self._model.max_seq_length)
            vectors = self._model.encode(texts, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
        if len(vectors) != len(texts):
            raise ValueError('Embedding count does not match input count')
        return [validate_vector(vector) for vector in vectors]
