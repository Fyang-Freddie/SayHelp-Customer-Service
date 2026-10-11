"""Chapter-six retrieval calibration, with distinct policy and FAQ score distributions."""
import hashlib,json,math
from pathlib import Path
from app.evaluation.calibration import ConfidencePolicy,ConfidenceConfigurationError,choose_threshold,fingerprints
from app.query_expansion import EXPANSION_SYSTEM
from app.config import POLICY_RRF_K,POLICY_CANDIDATES,POLICY_TOP_K

ROOT=Path(__file__).resolve().parents[1]

def knowledge_fingerprints(corpus,settings):
    path=ROOT/'eval/ch06/policy-cases.json'
    cases=json.loads(path.read_text(encoding='utf-8'))
    expected=fingerprints(corpus,settings,path)
    expected.update(calibration_digest=hashlib.sha256(json.dumps([c for c in cases if c['split']=='calibration'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        expansion_prompt=hashlib.sha256(EXPANSION_SYSTEM.encode()).hexdigest(),
        query_mode='policy:original+3-expansions/source-filter/RRF/one-rerank;faq:original-only',
        policy_rrf_k=POLICY_RRF_K,policy_candidates=POLICY_CANDIDATES,policy_top_k=POLICY_TOP_K)
    return expected

def calibrate_knowledge(rows,expected):
    if not rows or any(r['split']!='calibration' for r in rows): raise ValueError('Only calibration rows may tune knowledge gates')
    modes={}
    for mode in ('policy','faq'):
        samples=[(r['score'],bool(r['answerable'] and r['gold_found'])) for r in rows if r['mode']==mode and r['score'] is not None and not r['error']]
        if not samples:
            modes[mode]={'enabled':False,'threshold':None,'reason':'no_observed_scores'}; continue
        cut=choose_threshold(samples)
        enabled=any(label and score>=cut['threshold'] for score,label in samples)
        modes[mode]={**cut,'enabled':enabled,'observations':len(samples)}
    return {'version':1,'fingerprints':expected,'modes':modes}

def load_policy(path,expected,mode):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        entry=data['modes'][mode]; threshold=entry['threshold']
        if data['version']!=1 or data['fingerprints']!=expected or entry['enabled'] is not True: raise ValueError()
        if type(threshold) not in (float,int) or not math.isfinite(threshold): raise ValueError()
        return ConfidencePolicy({'hybrid_rerank':threshold},expected)
    except (OSError,ValueError,TypeError,KeyError):
        raise ConfidenceConfigurationError('Chapter 6 knowledge calibration missing, disabled or incompatible') from None
