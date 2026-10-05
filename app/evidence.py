"""Keep whole chunks within the complete prompt budget and cite final positions."""
import json
from urllib.parse import quote
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from app.rag_prompts import SYSTEM_PROMPT
from app.knowledge_types import PromptEvidence


def select_prompt_evidence(query,result,settings):
    selected=list(result.ranked[:10])
    while True:
        ordered=selected[::2]+selected[1::2][::-1]
        citations=[]
        for n,item in enumerate(ordered,1):
            c=item.chunk
            citations.append(dict(n=n,chunk_id=str(c.id),section_path=c.section_path,question=c.question,answer=c.answer,
                source_file=c.source_file,source_start_line=c.source_start_line,source_end_line=c.source_end_line,
                source_digest=c.source_digest,source_url='/v1/knowledge/documents/'+quote(c.source_file.split('/')[-1]) if c.source_file else None))
        body={'raw_question':query.raw_question,'standard_question':query.standard_question,
              'evidence':[{k:c[k] for k in ('n','chunk_id','section_path','question','answer')} for c in citations]}
        messages=[SystemMessage(SYSTEM_PROMPT),HumanMessage(json.dumps(body,ensure_ascii=False,separators=(',',':')))]
        tokens=count_tokens_approximately(messages,chars_per_token=1.0)
        if tokens+settings.response_token_reserve<=settings.context_token_budget or not selected:
            return PromptEvidence(messages,citations,[item.rank for item in ordered],tokens)
        selected.pop()
