import asyncio,json
from importlib import import_module
from types import SimpleNamespace
from langchain_core.messages import AIMessage
import pytest

class Model:
    def __init__(self,payload): self.payload=payload
    async def expand_query(self,messages): return AIMessage(content=json.dumps(self.payload,ensure_ascii=False))

def expand(payload,query='耳机未拆封，能退吗？'):
    try: cls=import_module('app.query_expansion').QueryExpander
    except ModuleNotFoundError: pytest.fail('QueryExpander missing')
    return asyncio.run(cls(Model(payload)).expand(query,SimpleNamespace(to_dict=lambda:{'product_name':'耳机','variant':'黑色'})))

def test_three_distinct_queries_only():
    result,_=expand({'queries':['未拆封耳机退货资格','未拆封耳机退款条件','未拆封耳机退货运费']})
    assert len(result.queries)==3 and set(result.model_dump())=={'queries'}

@pytest.mark.parametrize('payload',[
 {'queries':['耳机退货']}, {'queries':['未拆封耳机退货']*3},
 {'queries':['未拆封耳机退货','未拆封耳机退款','未拆封耳机退货流程'],'intent':'退款退货'},
 {'queries':['耳机退货','未拆封耳机退款','未拆封耳机运费']},
 {'queries':['未拆封耳机已使用30天退货','未拆封耳机退款','未拆封耳机运费']},
])
def test_invalid_expansion_stops_before_retrieval(payload):
    try: module=import_module('app.query_expansion')
    except ModuleNotFoundError: pytest.fail('QueryExpander missing')
    with pytest.raises(module.QueryExpansionError): expand(payload)
