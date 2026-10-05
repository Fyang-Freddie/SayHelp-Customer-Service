"""Frozen annotations with stable anchors, source digests and leakage checks."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field,PrivateAttr,StringConstraints,model_validator
from sqlalchemy import select
from sqlalchemy.orm import undefer
from app.ch04_ingest import DOCUMENTS,row_evidence,verify_source_versions
from app.knowledge_db import KnowledgeChunk
from app.knowledge_types import CorpusSnapshot,KnowledgeFilters

Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=12000)]
Fact=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=512)]
_BUCKETS={'A_policy','B_model','C_colloquial','D_unknown','E_multi'}
_MODEL=re.compile(r'[a-z]+-[a-z]+\d+')


class EvidenceAnchor(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    source_file: Text
    section_path: Text
    evidence_quote: Text
    @model_validator(mode='after')
    def allowed_source(self):
        if self.source_file not in DOCUMENTS: raise ValueError('Gold source outside explicit real corpus')
        return self


class EvalCase(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    eval_id: Annotated[str,StringConstraints(min_length=1,max_length=16)]
    bucket: Literal['A_policy','B_model','C_colloquial','D_unknown','E_multi']
    difficulty: Literal['easy','medium','hard']
    query: Fact
    filters: KnowledgeFilters
    answerable: bool
    ground_truth: Text
    required_facts: list[Fact]=Field(max_length=24)
    forbidden_claims: list[Fact]=Field(max_length=24)
    evidence: list[EvidenceAnchor]=Field(max_length=12)
    _source_digests: dict=PrivateAttr(default_factory=dict)
    _split: str=PrivateAttr(default='')
    @model_validator(mode='after')
    def coherent_annotation(self):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,15}',self.eval_id): raise ValueError('Invalid eval ID')
        if self.answerable and (not self.evidence or not self.required_facts): raise ValueError('Answerable case needs all required facts and evidence')
        if not self.answerable and (self.evidence or self.required_facts): raise ValueError('Unknown case cannot invent gold')
        if self.bucket=='D_unknown' and self.answerable: raise ValueError('Unknown bucket must be unanswerable')
        anchors=[(a.source_file,a.section_path,a.evidence_quote) for a in self.evidence]
        if len(set(anchors))!=len(anchors): raise ValueError('Duplicate gold anchor')
        return self


@dataclass(frozen=True)
class BoundCase:
    case: EvalCase
    gold_ids: list[int]
    corpus_digest: str


def _canonical(value):
    value=unicodedata.normalize('NFKC',value).casefold().replace('邮费','运费').replace('退钱','退款')
    return ''.join(c for c in value if c.isalnum())


def _near(left,right):
    a,b=_canonical(left),_canonical(right)
    if a==b: return True
    if set(_MODEL.findall(left.casefold()))!=set(_MODEL.findall(right.casefold())): return False
    x={a[i:i+2] for i in range(len(a)-1)};y={b[i:i+2] for i in range(len(b)-1)}
    return bool(x and y and len(x&y)/len(x|y)>=.90)


def _cross(left,right):
    for a in left:
        for b in right:
            same_facts=a.required_facts and {_canonical(f) for f in a.required_facts}=={_canonical(f) for f in b.required_facts}
            if a.eval_id==b.eval_id or _near(a.query,b.query) or same_facts:
                raise ValueError(f'Calibration/test split leakage or overlap: {a.eval_id}/{b.eval_id}')


def _read(path):
    try: data=json.loads(Path(path).read_text(encoding='utf-8'))
    except (ValueError,OSError): raise ValueError('Evaluation dataset cannot be read') from None
    if not isinstance(data,dict) or set(data)!={'version','split','source_digests','cases'} or type(data['version']) is not int or data['version']!=1:
        raise ValueError('Evaluation dataset requires version/split/source_digests/cases')
    if data['split'] not in {'calibration','test'} or not isinstance(data['cases'],list) or not data['cases']:
        raise ValueError('Evaluation dataset has invalid split or no cases')
    cases=[EvalCase.model_validate(row) for row in data['cases']]
    if len({c.eval_id for c in cases})!=len(cases): raise ValueError('Duplicate evaluation ID')
    hashes=data['source_digests']
    if not isinstance(hashes,dict) or any(name not in DOCUMENTS or not isinstance(digest,str) or not re.fullmatch(r'[a-f0-9]{64}',digest) for name,digest in hashes.items()):
        raise ValueError('Invalid annotated source digests')
    for case in cases:
        case._source_digests=dict(hashes);case._split=data['split']
    return cases


def load_cases(path:Path)->list[EvalCase]:
    path=Path(path)
    cases=_read(path)
    sibling=path.with_name('test.json' if path.name=='calibration.json' else 'calibration.json')
    if path.name in {'calibration.json','test.json'} and sibling.is_file(): _cross(cases,_read(sibling))
    return cases


def validate_splits(calibration,test):
    _cross(calibration,test)
    if len(calibration)!=12 or len(test)!=60: raise ValueError('Frozen evaluation requires calibration12/test60')
    counts=Counter((c.bucket,c.difficulty) for c in test)
    if counts!={(bucket,difficulty):4 for bucket in _BUCKETS for difficulty in ('easy','medium','hard')}:
        raise ValueError('Frozen test requires five buckets with 4 easy/medium/hard each')
    if {c.bucket for c in calibration}!=_BUCKETS: raise ValueError('Calibration must cover all five buckets')


def bind_gold(cases,corpus:CorpusSnapshot,session_factory)->list[BoundCase]:
    if any(c._source_digests!=corpus.files for c in cases): raise ValueError('Annotated source version differs from current corpus')
    try: verify_source_versions(corpus)
    except (RuntimeError,OSError): raise ValueError('Corpus source changed before gold binding') from None
    bound=[]
    for case in cases:
        ids=[]
        for anchor in case.evidence:
            matches=[chunk.id for chunk in corpus.chunks.values() if chunk.source_file==anchor.source_file
                and chunk.section_path==anchor.section_path and anchor.evidence_quote in chunk.answer]
            if len(matches)!=1: raise ValueError(f'Gold anchor missing or ambiguous: {case.eval_id}')
            if matches[0] not in ids: ids.append(matches[0])
        bound.append(BoundCase(case,ids,corpus.corpus_digest))
    with session_factory() as session:
        rows=list(session.scalars(select(KnowledgeChunk).options(undefer('*')).where(KnowledgeChunk.id.in_(list(corpus.chunks)))))
        current={row.id:row_evidence(row) for row in rows}
    if current!=corpus.chunks: raise ValueError('MySQL source changed; re-ingest before binding gold')
    return bound
