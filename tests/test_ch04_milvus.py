"""Native BM25/RRF tests: spies for contracts plus isolated real Milvus."""
import importlib
import json
import math
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pymilvus import DataType, MilvusClient
from app.knowledge_types import KnowledgeFilters, PreparedQuery


def store_class():
    return importlib.import_module('app.hybrid_store').HybridMilvusStore


def vector():
    return [1/math.sqrt(1024)]*1024


class Spy:
    def __init__(self): self.calls=[]; self.schema=None; self.indexes=None
    def has_collection(self,name): return False
    def create_schema(self,**kwargs): return MilvusClient.create_schema(**kwargs)
    def prepare_index_params(self): return MilvusClient.prepare_index_params()
    def create_collection(self,**kwargs): self.schema=kwargs['schema'].to_dict(); self.indexes=kwargs['index_params']; self.calls.append(('create',kwargs))
    def upsert(self,**kwargs): self.calls.append(('upsert',kwargs)); return {'upsert_count':1,'ids':[kwargs['data'][0]['id']]}
    def search(self,**kwargs): self.calls.append(('search',kwargs)); return [[{'id':1,'distance':.8,'entity':{'text':'source'}}]]
    def hybrid_search(self,**kwargs): self.calls.append(('hybrid',kwargs)); return [[{'id':1,'distance':.03,'entity':{'text':'native RRF'}}]]


def test_schema_native_chinese_bm25_and_no_client_sparse_vector():
    spy=Spy(); store=store_class()('unused',client=spy)
    store.ensure_collection()
    fields={f['name']:f for f in spy.schema['fields']}
    assert set(fields)=={'id','vector','text','sparse','category','product_category','content_type','source_digest'}
    assert fields['vector']['params']['dim']==1024
    assert fields['text']['params']['max_length']==65535
    assert fields['text']['params']['enable_analyzer'] is True
    assert json.loads(fields['text']['params']['analyzer_params'])=={'type':'chinese'}
    functions=spy.schema['functions']
    assert len(functions)==1 and functions[0]['input_field_names']==['text'] and functions[0]['output_field_names']==['sparse']
    metadata={'category':'商品规格手册','product_category':'猫砂盆','content_type':'manual','source_digest':'a'*64}
    assert store.upsert(1,vector(),'MH-LP50 原文',metadata)==1
    row=spy.calls[-1][1]['data'][0]
    assert 'sparse' not in row and row['text']=='MH-LP50 原文' and row['source_digest']=='a'*64
    create=spy.calls[0][1]
    assert create['consistency_level']=='Strong'


def test_hybrid_uses_native_rrf_and_same_parameterized_prefilter():
    spy=Spy(); store=store_class()('unused',client=spy)
    filters=KnowledgeFilters(product_category='猫砂盆',content_type='manual')
    result=store.retrieve(PreparedQuery('MH-LP50','MH-LP50','MH-LP50'),vector(),'hybrid',filters)
    hybrid=next(kwargs for name,kwargs in spy.calls if name=='hybrid')
    assert hybrid['limit']==50 and hybrid['ranker'].dict()=={'strategy':'rrf','params':{'k':60}}
    reqs=hybrid['reqs']
    assert len(reqs)==2 and all(r.limit==50 for r in reqs)
    assert {r.anns_field for r in reqs}=={'vector','sparse'}
    assert reqs[0].expr==reqs[1].expr and reqs[0].expr_params==reqs[1].expr_params
    assert reqs[0].expr_params=={'product_category':'猫砂盆','content_type':'manual'}
    assert '猫砂盆' not in reqs[0].expr
    assert result.hits[0].score==.03 and result.hits[0].stage=='rrf'
    assert set(result.stage_hits)=={'dense','bm25','rrf'}
    assert len([1 for name,_ in spy.calls if name=='search'])==2


def test_bm25_needs_no_vector_and_injection_is_only_bound_value():
    spy=Spy(); store=store_class()('unused',client=spy)
    attack='猫砂盆" or id > 0 or category == "'
    result=store.retrieve(PreparedQuery('型号','型号','MH-LP50'),None,'bm25',KnowledgeFilters(category=attack))
    assert result.hits and len(spy.calls)==1
    kwargs=spy.calls[0][1]
    assert kwargs['anns_field']=='sparse' and kwargs['data']==['MH-LP50'] and kwargs['limit']==50
    assert attack not in kwargs['filter'] and kwargs['filter_params']=={'category':attack}
    with pytest.raises(ValueError):
        store.retrieve(PreparedQuery('x','x','x'),None,'bm25',{'unsafe':'x'})


@pytest.mark.parametrize('id_,text,metadata',[
    (True,'x',{}),(0,'x',{}),(2**63,'x',{}),(1,'中'*22000,{}),
    (1,'x',{'category':'中'*100,'product_category':'猫砂盆','content_type':'manual','source_digest':'a'*64}),
], ids=['bool','zero','overflow','text_bytes','category_bytes'])
def test_write_rejects_invalid_id_or_utf8_lengths(id_,text,metadata):
    spy=Spy(); store=store_class()('unused',client=spy)
    with pytest.raises(ValueError): store.upsert(id_,vector(),text,metadata)
    assert spy.calls==[]


def test_existing_dense_only_schema_is_rejected():
    class Wrong(Spy):
        def has_collection(self,name): return True
        def describe_collection(self,name): return {'auto_id':False,'enable_dynamic_field':False,'fields':[]}
    with pytest.raises(ValueError,match='[Ii]ncompatible'):
        store_class()('unused',client=Wrong()).ensure_collection()


def test_lazy_owned_client_is_closed_once_and_injected_client_remains_owned_by_caller(monkeypatch):
    created=[]
    class Client:
        def __init__(self,**kwargs): created.append(self); self.closed=0
        def close(self): self.closed+=1
    monkeypatch.setattr('pymilvus.MilvusClient',Client)
    store=store_class()('unused'); store.close(); store.close()
    assert created==[]
    store=store_class()('unused'); assert store.client is store.client
    store.close(); store.close(); assert created[0].closed==1
    injected=Client(); store=store_class()('unused',client=injected); store.close()
    assert injected.closed==0
    with pytest.raises(RuntimeError): _=store.client


@pytest.fixture
def live():
    values=dotenv_values('D:/llm_agent/contact_agent/.env')
    client=MilvusClient(uri=values.get('MILVUS_URI','http://127.0.0.1:19530'))
    name='sayhelp_ch04_test_'+uuid.uuid4().hex
    before={n:client.query(n,filter='',output_fields=['count(*)'])[0]['count(*)'] for n in client.list_collections()}
    try:
        yield client,name,before
    finally:
        if client.has_collection(name): client.drop_collection(name)
        after={n:client.query(n,filter='',output_fields=['count(*)'])[0]['count(*)'] for n in before}
        client.close()
        assert before==after


def test_real_milvus_exact_model_filters_schema_and_rrf(live):
    from app.knowledge_chunking import chunk_markdown
    client,name,_=live
    store=store_class()('unused',collection_name=name,client=client)
    store.ensure_collection(); store.ensure_collection()
    body=(Path(__file__).resolve().parents[1]/'knowledge_db/product-specs.md').read_text(encoding='utf-8')
    chunks=chunk_markdown(body,content_type='manual')
    selected=[c for c in chunks if 'MH-LP50' in c.section_path or 'MH-LP100' in c.section_path]
    assert len(selected)==2
    for id_,chunk in enumerate(selected,1):
        store.upsert(id_,vector(),chunk.section_path+'\n'+chunk.answer,{'category':'商品规格手册','product_category':'猫砂盆','content_type':'manual','source_digest':'a'*64})
    store.upsert(3,vector(),'MH-LP50 对照干扰材料',{'category':'干扰','product_category':'摄像头','content_type':'faq','source_digest':'b'*64})
    query=PreparedQuery('MH-LP50多大','MH-LP50尺寸','MH-LP50尺寸')
    filters=KnowledgeFilters(product_category='猫砂盆',content_type='manual')
    bm=store.retrieve(query,None,'bm25',filters)
    assert bm.hits and 'MH-LP50' in bm.entities[bm.hits[0].chunk_id]['text']
    assert all(h.chunk_id!=3 for h in bm.hits)
    hybrid=store.retrieve(query,vector(),'hybrid',filters)
    assert hybrid.hits and all(h.chunk_id!=3 for h in hybrid.hits)
    assert set(hybrid.stage_hits)=={'dense','bm25','rrf'}
    assert all(h.chunk_id!=3 for hits in hybrid.stage_hits.values() for h in hits)
    entities=store.read_entities()
    assert set(entities)=={1,2,3} and store.read_entities([3])[3]['source_digest']=='b'*64
    store.delete_ids([3]); assert set(store.read_entities())=={1,2}
    fields={f['name']:f for f in client.describe_collection(name)['fields']}
    assert fields['text']['params']['enable_analyzer']=='true'
    indexes=[client.describe_index(name,n) for n in client.list_indexes(name)]
    assert any(i['field_name']=='sparse' and i['metric_type']=='BM25' for i in indexes)
    assert any(i['field_name']=='vector' and i['metric_type']=='COSINE' for i in indexes)


def test_ch04_collection_configuration(monkeypatch,tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for name,value in {'CHAT_BASE_URL':'https://example.invalid','CHAT_MODEL':'test','CHAT_API_KEY':'test','DATABASE_URL':'mysql+pymysql://test:test@localhost/test'}.items():
        monkeypatch.setenv(name,value)
    assert Settings.from_env().knowledge_collection=='knowledge_ch04'
    monkeypatch.setenv('KNOWLEDGE_COLLECTION','knowledge_ch04_custom')
    assert Settings.from_env().knowledge_collection=='knowledge_ch04_custom'
    monkeypatch.setenv('KNOWLEDGE_COLLECTION','knowledge')
    with pytest.raises(ValueError,match='KNOWLEDGE_COLLECTION'):
        Settings.from_env()


@pytest.mark.parametrize('change',['analyzer','function','dimension','index'])
def test_real_existing_incompatible_collection_is_rejected(live,change):
    import copy
    client,name,_=live
    store=store_class()('unused',collection_name=name,client=client)
    store.ensure_collection()
    class Incompatible:
        def has_collection(self,name): return True
        def describe_collection(self,name):
            schema=copy.deepcopy(client.describe_collection(name))
            fields={f['name']:f for f in schema['fields']}
            if change=='analyzer': fields['text']['params']['analyzer_params']='{"type":"standard"}'
            if change=='function': schema['functions'][0]['input_field_names']=['category']
            if change=='dimension': fields['vector']['params']['dim']=384
            return schema
        def list_indexes(self,name): return client.list_indexes(name)
        def describe_index(self,name,index):
            result=client.describe_index(name,index)
            if change=='index' and result['field_name']=='vector': result['metric_type']='IP'
            return result
        def load_collection(self,name): pytest.fail('Incompatible collection must not be loaded')
    with pytest.raises(ValueError,match='incompatible'):
        store_class()('unused',collection_name=name,client=Incompatible()).ensure_collection()


def test_root_preamble_may_have_empty_legacy_category():
    spy=Spy()
    store=store_class()('unused',client=spy)
    metadata={'category':'','product_category':'商品','content_type':'manual','source_digest':'a'*64}
    assert store.upsert(32,vector(),'商品规格手册\n以下是规格概览。',metadata)==32
    assert spy.calls[-1][1]['data'][0]['category']==''
