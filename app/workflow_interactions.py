"""Recognize explicit execution replies only; business intent still belongs to Jev."""
import re

IDENTITY=r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}'

async def resolve_execution_input(text,pending):
    if not isinstance(text,str) or not text.strip():return {'kind':'unclear'}
    value=text.strip();kind=pending.get('kind')
    if kind=='customer_input':
        match=re.fullmatch(r'(?:客户(?:ID|id|编号)?[：: ]*)?('+IDENTITY+r')',value)
        if match:return {'kind':'slot_reply','value':{'customer_id':match[1]}}
    if kind=='order_selection':
        ids=re.findall(IDENTITY,value)
        explicit=re.fullmatch(r'(?:订单(?:号)?[：: ]*)?'+IDENTITY+r'(?:[ ,，/]+(?:商品项(?:ID|id)?[：: ]*)?'+IDENTITY+r')?',value)
        if explicit:
            matches=[item for item in pending.get('orders',[]) if item['order_id'] in ids and (len(ids)==1 or item['item_id'] in ids)]
            if len(matches)==1:
                return {'kind':'slot_reply','value':{'customer_id':pending['customer_id'],'order_id':matches[0]['order_id'],'item_id':matches[0]['item_id'],'data_digest':pending['data_digest']}}
            return {'kind':'unclear'}
    # Only a change-request signal, never an eight-class intent prediction.
    if any(word in value for word in ('先不','不退了','取消','物流','发票','投诉','我要','我想','换个问题','查订单')):
        return {'kind':'new_request'}
    if kind=='clarification' and len(value)<=2000:return {'kind':'slot_reply','value':{'text':value}}
    return {'kind':'unclear'}
