"""Normalize one current question and reject lost constraints before retrieval."""
import re
import unicodedata
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from app.knowledge_types import PreparedQuery

Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=4000)]
Term=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=32)]
_MODEL=re.compile(r'[A-Za-z]{1,8}\s*-\s*[A-Za-z]{1,8}\s*\d+[A-Za-z0-9]*')
_NUMBER=re.compile(r'\d+(?:\.\d+)?')
_UNIT=re.compile(r'(\d+(?:\.\d+)?)\s*(公斤|千克|厘米|毫米|毫升|分钟|小时|个月|工作日|积分|ghz|kg|cm|mm|ml|元|℃|天|月|克|米|升|g|m|l)')
_UNIT_ALIASES={'公斤':'kg','千克':'kg','克':'g','厘米':'cm','毫米':'mm','米':'m','升':'l','毫升':'ml','个月':'月','ghz':'g'}
_NEGATIVE=re.compile(r'不支持|不含|不能|不是|不影响|不在|不允许|不可以|没有|没|无|非|不')
_NON_QUALITY=re.compile(r'非质量|不是质量|没(?:有)?质量问题|质量(?:没(?:有)?|无)问题')
_NO_APP=re.compile(r'(?:没有|没|无|不含|不支持)(?:任何)?(?:app|联网|远程)')


class QueryDecision(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    standard_question: Text
    synonyms: list[Term]=Field(max_length=8)
    intent: Literal['knowledge','operation','chitchat','clarify']
    clarification: Text | None


class QueryUnderstandingError(RuntimeError):
    pass


def _normalized(value):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',value).casefold())


def _unique(values): return list(dict.fromkeys(values))

def _units(value):
    return {(number,_UNIT_ALIASES.get(unit,unit)) for number,unit in _UNIT.findall(value)}


def _negations(value):
    # Interrogatives are not negative assertions: preserve their question meaning.
    value=_normalized(value)
    for expression in ('能不能','可不可以','是不是','有没有','要不要','支不支持'):
        value=value.replace(expression,'')
    value=re.sub(r'[没不][？?]?$', '', value)
    return _unique(_NEGATIVE.findall(value))


def _needs_context(raw,models):
    if re.search(r'上(?:一)?(?:款|个|轮)|刚才|上一|前面|上面',raw): return True
    return bool(not models and re.search(r'它们?|那款|这款|那个|这个',raw)
                and not re.search(r'退货|退款|运费|质保|订单|饮水机|猫砂盆|喂食器|摄像头|加热垫|猫爬架',raw))


class QueryUnderstanding:
    def __init__(self,model_service): self.model_service=model_service

    async def prepare(self,raw_question:str)->PreparedQuery:
        if not isinstance(raw_question,str) or not raw_question.strip() or len(raw_question)>4000:
            raise ValueError('Question must be nonempty within 4000 characters')
        try:
            decision=await self.model_service.understand_query(raw_question)
            if not isinstance(decision,QueryDecision):
                decision=QueryDecision.model_validate(decision)
        except Exception:
            raise QueryUnderstandingError('问题理解服务暂时不可用，请稍后重试。') from None
        normalized_raw=_normalized(raw_question)
        models=_unique(_MODEL.findall(raw_question))
        numbers=_unique(_NUMBER.findall(unicodedata.normalize('NFKC',raw_question)))
        negatives=_negations(raw_question)
        standard=decision.standard_question
        normalized_standard=_normalized(standard)
        reasons=[]
        if {_normalized(model) for model in models} != set(_MODEL.findall(normalized_standard)): reasons.append('型号')
        if set(numbers)!=set(_NUMBER.findall(normalized_standard)): reasons.append('数字')
        if _units(normalized_raw)!=_units(normalized_standard): reasons.append('单位')
        if negatives and not _negations(standard): reasons.append('否定条件')
        if bool(_NON_QUALITY.search(normalized_raw)) != bool(_NON_QUALITY.search(normalized_standard)): reasons.append('质量条件')
        if _NO_APP.search(normalized_raw) and not _NO_APP.search(normalized_standard): reasons.append('无App条件')
        intent=decision.intent
        clarification=decision.clarification
        if reasons:
            intent='clarify'
            standard=raw_question
            clarification='请确认原问题中的'+ '、'.join(reasons)+'，以便按这些限制查询。'
        elif _needs_context(raw_question,models):
            intent='clarify'; clarification='请提供这次要咨询的商品完整型号或具体问题。'
        elif intent=='operation' and re.search(r'库存|多少钱|当前价格|现在.*价格',raw_question) and not re.search(r'订单|物流|取消|申请',raw_question):
            intent='knowledge'; clarification=None
        if intent=='clarify' and not clarification:
            clarification='请提供商品完整型号或更具体的问题。'
        synonyms=[]
        for term in decision.synonyms:
            # Retrieval expansion is lexical, never an opportunity to add facts.
            if not set(_NUMBER.findall(_normalized(term)))<=set(numbers): continue
            if any(_normalized(m) not in normalized_raw for m in _MODEL.findall(term)): continue
            if term not in synonyms: synonyms.append(term)
        if reasons or intent=='clarify': synonyms=[]
        search_text='\n'.join(_unique([standard,raw_question,*models,*synonyms]))
        return PreparedQuery(raw_question,standard,search_text,synonyms,intent,clarification,models,numbers,negatives)
