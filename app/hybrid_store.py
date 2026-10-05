"""Milvus-native Chinese BM25 and dense RRF with parameterized prefilters."""
import json
import math
import re
from time import perf_counter

from app.embedding import DIMENSION, validate_vector
from app.knowledge_types import KnowledgeFilters, PreparedQuery, StageHit, StoreResult
from app.vector_store import MilvusKnowledgeStore

_LENGTHS={'category':255,'product_category':64,'content_type':32,'source_digest':64}
_OUTPUT=['id','text',*_LENGTHS]


def _id(value):
    if type(value) is not int or not 0<value<2**63:
        raise ValueError('Knowledge primary key must fit positive Milvus INT64')
    return value


def _text(value,limit,name,*,allow_empty=False):
    if not isinstance(value,str) or (not value and not allow_empty) or len(value.encode('utf-8'))>limit:
        raise ValueError(f'{name} must be valid UTF-8 within {limit} bytes')
    return value


def compile_filters(filters):
    if not isinstance(filters,KnowledgeFilters):
        raise ValueError('Filters require the validated KnowledgeFilters contract')
    params=filters.model_dump(exclude_none=True)
    return ' and '.join(f'{name} == {{{name}}}' for name in params),params


def _json(value):
    return json.loads(value) if isinstance(value,str) else value


class HybridMilvusStore(MilvusKnowledgeStore):
    def __init__(self,uri='http://127.0.0.1:19530',*,collection_name='knowledge_ch04',client=None):
        if not isinstance(collection_name,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',collection_name) or collection_name=='knowledge':
            raise ValueError('Chapter 4 requires an independent named collection')
        super().__init__(uri,client=client)
        self.collection_name=collection_name

    def ensure_collection(self):
        from pymilvus import DataType,Function,FunctionType
        client=self.client
        if client.has_collection(self.collection_name):
            schema=client.describe_collection(self.collection_name)
            try:
                fields={f['name']:f for f in schema['fields']}
                valid=(set(fields)=={'id','vector','text','sparse',*_LENGTHS}
                    and not schema.get('auto_id') and not schema.get('enable_dynamic_field')
                    and fields['id']['type']==DataType.INT64 and fields['id'].get('is_primary')
                    and fields['vector']['type']==DataType.FLOAT_VECTOR and int(fields['vector']['params']['dim'])==DIMENSION
                    and fields['sparse']['type']==DataType.SPARSE_FLOAT_VECTOR
                    and fields['text']['type']==DataType.VARCHAR and int(fields['text']['params']['max_length'])==65535
                    and _json(fields['text']['params'].get('enable_analyzer',False)) is True
                    and _json(fields['text']['params'].get('analyzer_params',{}))=={'type':'chinese'}
                    and all(fields[name]['type']==DataType.VARCHAR and int(fields[name]['params']['max_length'])==length for name,length in _LENGTHS.items())
                    and len(schema.get('functions',[]))==1
                    and schema['functions'][0]['type']==FunctionType.BM25
                    and schema['functions'][0]['input_field_names']==['text']
                    and schema['functions'][0]['output_field_names']==['sparse']
                    and schema.get('consistency_level_name')=='Strong')
                indexes=[client.describe_index(self.collection_name,name) for name in client.list_indexes(self.collection_name)] if valid else []
                valid=valid and all(any(i.get('field_name')==field and i.get('metric_type')==metric and i.get('index_type')==kind for i in indexes)
                    for field,metric,kind in [('vector','COSINE','AUTOINDEX'),('sparse','BM25','SPARSE_INVERTED_INDEX')])
            except (KeyError,TypeError,ValueError):
                valid=False
            if not valid:
                raise ValueError('Existing chapter 4 collection has incompatible schema/functions/indexes')
            client.load_collection(self.collection_name)
            return
        schema=client.create_schema(auto_id=False,enable_dynamic_field=False)
        schema.add_field('id',DataType.INT64,is_primary=True,auto_id=False)
        schema.add_field('vector',DataType.FLOAT_VECTOR,dim=DIMENSION)
        schema.add_field('text',DataType.VARCHAR,max_length=65535,enable_analyzer=True,analyzer_params={'type':'chinese'})
        schema.add_field('sparse',DataType.SPARSE_FLOAT_VECTOR)
        for name,length in _LENGTHS.items(): schema.add_field(name,DataType.VARCHAR,max_length=length)
        schema.add_function(Function(name='text_bm25',function_type=FunctionType.BM25,input_field_names=['text'],output_field_names=['sparse']))
        indexes=client.prepare_index_params()
        indexes.add_index(field_name='vector',index_type='AUTOINDEX',metric_type='COSINE')
        indexes.add_index(field_name='sparse',index_type='SPARSE_INVERTED_INDEX',metric_type='BM25')
        client.create_collection(collection_name=self.collection_name,schema=schema,index_params=indexes,consistency_level='Strong')

    def upsert(self,chunk_id,vector,text,metadata):
        id_=_id(chunk_id)
        text=_text(text,65535,'text')
        if not isinstance(metadata,dict) or set(metadata)!=set(_LENGTHS):
            raise ValueError('Index metadata requires exactly the declared scalar fields')
        values={name:_text(metadata[name],limit,name,allow_empty=name=='category') for name,limit in _LENGTHS.items()}
        result=self.client.upsert(collection_name=self.collection_name,data=[{'id':id_,'vector':validate_vector(vector),'text':text,**values}])
        if result.get('upsert_count')!=1 or result.get('ids')!=[id_]:
            raise ValueError('Milvus did not acknowledge the expected primary key')
        return id_

    def read_entities(self,ids=None):
        if ids is not None:
            ids=[_id(value) for value in ids]
            if not ids: return {}
            rows=[]
            for start in range(0,len(ids),1000):
                rows.extend(self.client.get(self.collection_name,ids=ids[start:start+1000],output_fields=_OUTPUT,consistency_level='Strong'))
        else:
            iterator=self.client.query_iterator(self.collection_name,filter='id > 0',batch_size=1000,output_fields=_OUTPUT,consistency_level='Strong')
            rows=[]
            try:
                while True:
                    try: batch=iterator.next()
                    except StopIteration: break
                    if not batch: break
                    rows.extend(batch)
            finally:
                iterator.close()
        return {_id(row['id']):dict(row) for row in rows}

    def delete_ids(self,ids):
        ids=[_id(value) for value in ids]
        for start in range(0,len(ids),1000):
            self.client.delete(self.collection_name,ids=ids[start:start+1000])

    def retrieve(self,query:PreparedQuery,vector,strategy,filters):
        from pymilvus import AnnSearchRequest,RRFRanker
        if strategy not in {'dense','bm25','hybrid','hybrid_rerank'}:
            raise ValueError('Unknown retrieval strategy')
        if not isinstance(query,PreparedQuery): raise ValueError('Prepared query required')
        search_text=_text(query.search_text,65535,'query')
        expr,params=compile_filters(filters)
        dense=validate_vector(vector) if strategy!='bm25' else None
        requests=[]
        if dense is not None:
            requests.append(AnnSearchRequest(data=[dense],anns_field='vector',param={'metric_type':'COSINE'},limit=50,expr=expr,expr_params=params))
        if strategy!='dense':
            requests.append(AnnSearchRequest(data=[search_text],anns_field='sparse',param={'metric_type':'BM25'},limit=50,expr=expr,expr_params=params))
        stages,entities,timings={},{},{}
        def parse(raw,stage):
            if not isinstance(raw,list) or len(raw)!=1: raise ValueError('Milvus returned wrong query count')
            hits=[]; seen=set()
            for rank,hit in enumerate(raw[0],1):
                id_=_id(hit['id']); score=float(hit['distance'])
                if id_ in seen or not math.isfinite(score): raise ValueError('Milvus returned invalid hit')
                seen.add(id_); hits.append(StageHit(id_,rank,score,stage))
                entities[id_]={'id':id_,**dict(hit.get('entity',{}))}
            stages[stage]=hits
            return hits
        if strategy in {'hybrid','hybrid_rerank'}:
            start=perf_counter()
            fused=parse(self.client.hybrid_search(collection_name=self.collection_name,reqs=requests,ranker=RRFRanker(k=60),limit=50,output_fields=_OUTPUT,consistency_level='Strong'),'rrf')
            timings['rrf_ms']=(perf_counter()-start)*1000
        else: fused=None
        for request in requests:
            stage='dense' if request.anns_field=='vector' else 'bm25'
            start=perf_counter()
            hits=parse(self.client.search(collection_name=self.collection_name,data=request.data,anns_field=request.anns_field,search_params=request.param,limit=50,filter=expr,filter_params=params,output_fields=_OUTPUT,consistency_level='Strong'),stage)
            timings[('diagnostic_' if fused is not None else '')+stage+'_ms']=(perf_counter()-start)*1000
            if fused is None: selected=hits
        return StoreResult(fused if fused is not None else selected,stages,entities,timings)
