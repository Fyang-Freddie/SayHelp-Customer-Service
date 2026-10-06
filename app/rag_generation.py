"""Self-evaluate evidence and validate citations before a customer answer exists."""
import re
import unicodedata
from typing import Annotated
from pydantic import BaseModel,ConfigDict,StringConstraints,ValidationError
from app.knowledge_types import KnowledgeAnswer

REFUSAL='现有知识库证据不足，无法确认这个问题。请补充具体型号或条件，或联系人工客服核实。'
class GenerationFormatError(ValueError): pass
class GenerationServiceError(RuntimeError): pass

class GenerationDecision(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')
    useful: bool
    answer: Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=12000)]
    reason: Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=1000)]

_NEGATIVE=re.compile(r'(?:不|不能|无法|不会|不可|不应|没法|不得|不是|未|尚未|没有|还没)(?:给您|为您|向您)?(?:作出|提供)?(?:任何)?(?:承诺|保证|说)?$|(?:并不|不)意味着$')
_PROHIBITED=re.compile(r'已(?:经)?(?:帮(?:你|您)|为(?:你|您))?(?:退款|退钱|办理退款|提交退款|完成退款)|已为(?:你|您)(?:办理|申请)|保证.{0,18}(?:到账|送到|送达|到货|免费|修好|安全|库存|有货|赔偿|审核通过)|(?:明天|今天|后天|\d+个?工作日内|\d+小时内)(?:一定|必定|保证|必然)?到账|(?:绝对安全|审核必过|肯定通过|一定送达|必定送达)')


def forbidden_claim(answer):
    value=unicodedata.normalize('NFKC',answer).replace(' ','').replace('“','').replace('”','').replace('「','').replace('」','')
    for match in _PROHIBITED.finditer(value):
        prefix=value[max(0,match.start()-12):match.start()]
        if not _NEGATIVE.search(prefix): return True
    return False


def validate_citations(answer,citations):
    labels=re.findall(r'\[([^\[\]]*)\]',answer)
    allowed={c['n'] for c in citations}
    if not labels: return False
    for label in labels:
        if not re.fullmatch(r'[0-9]{1,3}',label) or int(label)<=0 or int(label) not in allowed: return False
    return True


class RagGenerator:
    def __init__(self,model_service): self.model_service=model_service
    async def generate(self,query,prompt,confidence):
        def refusal(reason,source): return KnowledgeAnswer(False,REFUSAL,reason,source,prompt.citations)
        if not confidence.sufficient: return refusal(confidence.reason,'retrieval_low_conf')
        if not prompt.citations: return refusal('上下文预算内没有完整证据','retrieval_low_conf')
        try:
            payload=await self.model_service.generate_knowledge(prompt.messages)
        except GenerationFormatError:
            return refusal('生成自评结构无效，无法验证证据充分性','self_check')
        except Exception:
            raise GenerationServiceError('Knowledge generation unavailable; retry later') from None
        try:
            decision=payload if isinstance(payload,GenerationDecision) else GenerationDecision.model_validate(payload)
        except (ValueError,TypeError,ValidationError):
            return refusal('生成自评结构无效，无法验证证据充分性','self_check')
        if not decision.useful: return refusal(decision.reason,'self_check')
        if not validate_citations(decision.answer,prompt.citations):
            return refusal('回答引用缺失或编号不属于本轮证据','self_check')
        if forbidden_claim(decision.answer):
            return refusal('回答含禁止承诺或未执行操作的表述','self_check')
        return KnowledgeAnswer(True,decision.answer,decision.reason,None,prompt.citations)
