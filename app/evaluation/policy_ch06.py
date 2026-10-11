"""Real policy/FAQ calibration and acceptance against frozen section labels."""
import asyncio,json
from pathlib import Path
from types import SimpleNamespace
from app.query_expansion import QueryExpander,QueryExpansionError
from app.policy_retrieval import PolicyRetrieval
from app.workflow_knowledge import KnowledgeGate,create_live_retrieval
from app.evaluation.calibration import ConfidencePolicy
from app.knowledge_policy import knowledge_fingerprints,calibrate_knowledge,load_policy
from app.workflow_log import WorkflowLog

ROOT=Path(__file__).resolve().parents[2]

def cases_for(split):
    return [c for c in json.loads((ROOT/'eval/ch06/policy-cases.json').read_text(encoding='utf-8')) if c['split']==split]

async def evaluate_policy_cases(cases,settings,model,budget,output,split):
    class Counted:
        async def expand_query(self,messages):
            budget.reserve('chat'); return await model.expand_query(messages)
    expander=QueryExpander(Counted())
    rows=[]
    with create_live_retrieval(settings) as retrieval:
        expected=knowledge_fingerprints(retrieval.corpus,settings)
        policies={mode:ConfidencePolicy({'hybrid_rerank':-1e100},expected) if split=='calibration' else load_policy(ROOT/settings.ch06_knowledge_policy,expected,mode) for mode in ('policy','faq')}
        policy_gate=PolicyRetrieval(retrieval,policies['policy'],settings)
        faq_gate=KnowledgeGate(retrieval,policies['faq'],settings,log=WorkflowLog(ROOT/'.runtime/ch06/policy-eval'))
        for case in cases:
            usage=None; queries=[]
            try:
                if case['mode']=='policy':
                    item=SimpleNamespace(issues=(),to_dict=lambda:case['item'])
                    expansion,usage=await expander.expand(case['query'],item)
                    queries=expansion.queries
                    result=await policy_gate.prepare(case['query'],queries,item)
                    confidence=result['knowledge_confidence']
                else:
                    result=await faq_gate.prepare(case['query'])
                    confidence=result['confidence']
                sections=[citation.get('section_path') or '' for citation in result['citations']]
                found=bool(case['required_sections']) and all(any(term in section for section in sections) for term in case['required_sections'])
                row={**case,'queries':queries,'score':confidence['score'],'gold_found':found,
                     'sufficient':confidence['sufficient'],'citations':result['citations'],
                     'error':result.get('stop_reason'),'usage':usage.to_dict() if usage else None}
            except QueryExpansionError as error:
                row={**case,'queries':[],'score':None,'gold_found':False,'sufficient':False,'citations':[],
                     'error':error.reason,'usage':error.usage.to_dict()}
            rows.append(row)
            report={'split':split,'fingerprints':expected,'cases':rows,'cumulative_attempts':budget.counts}
            Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'case':case['id'],'score':row['score'],'gold_found':row['gold_found'],'error':row['error']},ensure_ascii=True),flush=True)
        if split=='calibration':
            artifact=calibrate_knowledge(rows,expected)
            (ROOT/settings.ch06_knowledge_policy).write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            report['calibration']=artifact
        else:
            report['correct']=sum((r['sufficient'] and r['gold_found']) if r['answerable'] else not r['sufficient'] for r in rows)
            report['passed']=report['correct']==len(rows) and all(r['error'] not in ('retrieval_error','index_stale','expansion_provider_error','expansion_invalid_response') for r in rows)
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return report
