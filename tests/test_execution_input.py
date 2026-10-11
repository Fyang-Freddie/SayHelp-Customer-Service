import asyncio,importlib

def resolve(text,pending):return asyncio.run(importlib.import_module('app.workflow_interactions').resolve_execution_input(text,pending))

def test_customer_reply_is_only_explicit_id():
    assert resolve('客户ID：C001',{'kind':'customer_input'})=={'kind':'slot_reply','value':{'customer_id':'C001'}}
    assert resolve('我忘了',{'kind':'customer_input'})['kind']=='unclear'

def test_new_request_is_not_mistaken_for_order_selection():
    assert resolve('先不退了，物流到哪了',{'kind':'order_selection'})['kind']=='new_request'
    assert resolve('选第一个',{'kind':'order_selection'})['kind']=='unclear'

def test_order_id_selects_only_when_one_item_in_pending_cards():
    pending={'kind':'order_selection','customer_id':'C1','data_digest':'v1','orders':[{'order_id':'O1','item_id':'I1'}]}
    assert resolve('O1',pending)=={'kind':'slot_reply','value':{'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'v1'}}
    pending['orders'].append({'order_id':'O1','item_id':'I2'})
    assert resolve('O1',pending)['kind']=='unclear'
