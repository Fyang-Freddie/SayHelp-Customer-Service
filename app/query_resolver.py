"""One bounded query-understanding call with strict JSON and factual guards."""
import json,re
from langchain_core.messages import AIMessage
from app.ch06_types import Resolution
from app.ch06_prompts import resolution_messages
from app.jev_classifier import strict_json
from app.workflow_types import Usage

class QueryResolutionError(ValueError):
    def __init__(self,reason,usage=None):
        super().__init__('Query understanding requires fallback')
        self.reason,self.usage=reason,usage or Usage()


def bounded_context(history,confirmed):
    cid=confirmed.get('conversation_id')
    rows=[]; size=0
    for row in reversed(history):
        if not isinstance(row,dict) or row.get('role') not in ('user','assistant') or row.get('completed') is False: continue
        if row.get('conversation_id') is not None and row['conversation_id']!=cid: continue
        text=row.get('content')
        if not isinstance(text,str) or not text or len(text)>2000: continue
        if size+len(text)>6000 or len(rows)>=8: break
        rows.append({'role':row['role'],'content':text}); size+=len(text)
    known={k:v for k,v in confirmed.items() if k in ('customer_id','order_id','item_id','product_name','variant') and isinstance(v,str) and len(v)<=300}
    return list(reversed(rows)),known


class QueryResolver:
    def __init__(self,model): self.model=model

    async def resolve(self,raw: str,history: list,confirmed: dict):
        if not isinstance(raw,str) or not raw.strip() or len(raw)>4000:
            raise QueryResolutionError('invalid_query')
        rows,known=bounded_context(history,confirmed)
        messages=resolution_messages(raw,rows,known)
        estimate=sum(len(str(m.content)) for m in messages)
        try: response=await self.model.resolve_query(messages)
        except Exception: raise QueryResolutionError('query_provider_error',Usage(estimated=estimate)) from None
        if not isinstance(response,AIMessage): raise QueryResolutionError('query_invalid_response',Usage(estimated=estimate))
        usage=(Usage(response.usage_metadata['input_tokens'],response.usage_metadata['output_tokens'])
               if response.usage_metadata else Usage(estimated=estimate+len(str(response.content))))
        try:
            if not isinstance(response.content,str) or len(response.content)>16000 or response.tool_calls or response.invalid_tool_calls: raise ValueError()
            decision=Resolution.model_validate(strict_json(response.content))
            if any(not ref.strip() or ref not in raw for ref in decision.unresolved_references): raise ValueError()
            if not decision.changed:
                return decision.model_copy(update={'query':raw}),usage
            query=decision.query
            if not query.strip(): raise ValueError()
            if not set(re.findall(r'\d+(?:\.\d+)?',raw))<=set(re.findall(r'\d+(?:\.\d+)?',query)): raise ValueError()
            if any(token in raw and token not in query for token in ('不','没','未','别','无')): raise ValueError()
            source=raw+'\n'+'\n'.join(r['content'] for r in rows if r['role']=='user')+'\n'+json.dumps(known,ensure_ascii=False)
            pattern=r'[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|(?<!\d)\d{6,}(?!\d)'
            ids=set(re.findall(pattern,query))
            if not ids<=set(re.findall(pattern,source)): raise ValueError()
            return decision,usage
        except (ValueError,TypeError): raise QueryResolutionError('query_invalid_response',usage) from None
