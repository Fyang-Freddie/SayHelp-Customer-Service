"""Exact application request payloads, without client credentials or hidden reasoning."""
from datetime import datetime, UTC
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock


class ContextLog:
    def __init__(self, path='log/app.log'):
        self.path=Path(path)
        self.lock=Lock()

    def write(self,event:str,payload:dict)->None:
        row={'at':datetime.now(UTC).isoformat(),'event':event,**payload}
        line=json.dumps(row,ensure_ascii=False,allow_nan=False,default=str)
        with self.lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf8') as stream:stream.write(line+'\n')

    def model_request(self,cid,turn,phase,messages,tools,ledger,summary='',
                      *,max_output_tokens=None,summary_upto_msg_id=None,layer1_from_msg_id=None):
        self.write('model_ctx',{'conversation_id':cid,'turn_id':turn,'phase':phase,
            'summary':summary,'messages':[m.model_dump(mode='json') for m in messages],
            'tools':tools,'tools_bound':bool(tools),'window_count':len(messages),
            'tools_digest':hashlib.sha256(json.dumps(tools,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'tokens':asdict(ledger) if is_dataclass(ledger) else ledger,
            'max_output_tokens':max_output_tokens,
            'summary_upto_msg_id':summary_upto_msg_id,
            'layer1_from_msg_id':layer1_from_msg_id})

    def history(self,cid,turn,view):
        self.write('history_ctx',{'conversation_id':cid,'turn_id':turn,
            'summary':view.summary_text,'messages':[m.model_dump(mode='json') for m in (*view.layer2,*view.layer1)],
            'window_count':len(view.layer2)+len(view.layer1),'tokens':asdict(view.ledger),
            'summary_upto_msg_id':view.snapshot.summary_upto_msg_id,
            'layer1_from_msg_id':view.layer1_from_msg_id})
