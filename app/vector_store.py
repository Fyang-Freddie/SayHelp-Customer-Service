"""Milvus contains only MySQL-aligned IDs and dense vectors."""
from dataclasses import dataclass
from threading import Lock
from app.embedding import DIMENSION, validate_vector


@dataclass(frozen=True)
class SearchHit:
    id: int
    score: float


class MilvusKnowledgeStore:
    collection_name = 'knowledge'

    def __init__(self, uri='http://127.0.0.1:19530', *, client=None):
        self.uri = uri
        self._client = client
        self._owns_client = client is None
        self._client_lock = Lock()
        self._closed = False

    @property
    def client(self):
        with self._client_lock:
            if self._closed:
                raise RuntimeError('Milvus knowledge store is closed')
            if self._client is None:
                from pymilvus import MilvusClient
                self._client = MilvusClient(uri=self.uri)
            return self._client

    def close(self):
        """Release only an owned client, without initializing an unused store."""
        with self._client_lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_client and self._client is not None:
                self._client.close()

    def ensure_collection(self):
        from pymilvus import DataType
        client = self.client
        if client.has_collection(self.collection_name):
            schema = client.describe_collection(self.collection_name)
            fields = {field['name']: field for field in schema['fields']}
            if (set(fields) != {'id', 'vector'} or schema.get('auto_id')
                    or fields['id']['type'] != DataType.INT64
                    or not fields['id'].get('is_primary')
                    or fields['vector']['type'] != DataType.FLOAT_VECTOR
                    or int(fields['vector']['params']['dim']) != DIMENSION
                    or schema.get('enable_dynamic_field')):
                raise ValueError('Existing knowledge collection has incompatible schema')
            indexes = client.list_indexes(self.collection_name)
            if not any(client.describe_index(self.collection_name, name).get('metric_type') == 'COSINE'
                       for name in indexes):
                raise ValueError('Existing knowledge collection requires a COSINE index')
            client.load_collection(self.collection_name)
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field('id', DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field('vector', DataType.FLOAT_VECTOR, dim=DIMENSION)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='COSINE')
        client.create_collection(collection_name=self.collection_name, schema=schema,
                                 index_params=indexes, consistency_level='Strong')

    def upsert(self, id: int, vector: list[float]) -> int:
        if type(id) is not int or not 0 < id < 2**63:
            raise ValueError('Knowledge primary key must fit positive Milvus INT64')
        result = self.client.upsert(collection_name=self.collection_name,
                                    data=[{'id': id, 'vector': validate_vector(vector)}])
        if result.get('upsert_count') != 1 or result.get('ids') != [id]:
            raise ValueError('Milvus did not acknowledge the expected primary key')
        return id

    def search(self, vector: list[float], limit: int) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError('limit must be positive')
        result = self.client.search(collection_name=self.collection_name,
                                    data=[validate_vector(vector)], anns_field='vector', limit=limit,
                                    search_params={'metric_type': 'COSINE'}, consistency_level='Strong')
        return [SearchHit(id=int(hit['id']), score=float(hit['distance'])) for hit in result[0]]
