"""Business records must never be guessed or cross the selected customer scope."""
from decimal import Decimal
from importlib import import_module
import pytest

ORDERS = '''# 订单数据
| 客户ID | 订单号 | 商品项ID | 商品名称 | 型号/规格 | 数量 | 实付金额（元） | 下单时间 | 签收时间 | 订单状态 | 售后状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | O01 | I01 | 鼠标 | 黑色 | 2 | 99.00 | 2026-10-01 10:00:00 | 2026-10-03 12:00:00 | 已签收 | 无售后 |
| C01 | O01 | I02 | 毛毯 | 灰色 | 1 | 132.05 | 2026-10-01 10:00:00 | 2026-10-03 12:00:00 | 已签收 | 无售后 |
| C02 | O02 | I03 | 键盘 | 红轴 | 1 | 200.00 | 2026-10-02 10:00:00 | 未签收 | 运输中 | 无售后 |
'''
LOGISTICS = '''# 物流数据
| 订单号 | 承运商 | 物流状态 | 最新更新时间 | 轨迹 |
|---|---|---|---|---|
| O01 | 快递 | 已签收 | 2026-10-03 12:00:00 | 已签收 |
| O02 | 快递 | 运输中 | 2026-10-03 12:00:00 | 运输中 |
'''
REASONS = '''# 退款原因
| 原因编码 | 显示名称 |
|---|---|
| quality_issue | 商品质量问题 |
| other | 其他 |
'''


def api():
    try:
        return import_module('app.business_data')
    except ModuleNotFoundError:
        pytest.fail('Missing exact business-data implementation')


@pytest.fixture
def root(tmp_path):
    for name,body in [('orders.md',ORDERS),('logistics.md',LOGISTICS),('refund-reasons.md',REASONS)]:
        (tmp_path/name).write_text(body,encoding='utf-8')
    return tmp_path


def test_multiline_order_selects_exact_item(root):
    data=api().MarkdownBusinessData(root)
    assert [i.item_id for i in data.list_items('C01')] == ['I01','I02']
    item=data.get_item('C01','O01','I02')
    assert (item.product_name,item.paid_amount)==('毛毯',Decimal('132.05'))
    assert data.get_logistics('C01','O01')['status']=='已签收'
    assert data.refund_reasons()==[{'code':'quality_issue','label':'商品质量问题'},{'code':'other','label':'其他'}]


def test_wrong_customer_rejected(root):
    module=api(); data=module.MarkdownBusinessData(root)
    with pytest.raises(module.BusinessDataError,match='not_found'):
        data.get_item('C02','O01','I01')
    with pytest.raises(module.BusinessDataError,match='not_found'):
        data.get_logistics('C02','O01')
    assert data.list_items('unknown')==[]


def test_paid_amount_is_item_total(root):
    item=api().MarkdownBusinessData(root).get_item('C01','O01','I01')
    assert item.paid_amount==Decimal('99.00') and item.quantity==2
    assert item.ordered_at.utcoffset().total_seconds()==28800
    assert api().MarkdownBusinessData(root).get_item('C02','O02','I03').signed_at is None


@pytest.mark.parametrize('before,after',[
    ('| I02 |','| I01 |'), ('| C01 | O01 | I02','| C02 | O01 | I02'),
    ('| 99.00 |','| NaN |'), ('| 99.00 |','| -1.00 |'),
    ('| 2 | 99.00','| -2 | 99.00'), ('2026-10-01 10:00:00','2026-13-01 10:00:00'),
    ('| 鼠标 | 黑色 |','| 鼠标 |'),
])
def test_invalid_row_reports_source_line(root,before,after):
    (root/'orders.md').write_text(ORDERS.replace(before,after,1),encoding='utf-8')
    module=api()
    with pytest.raises(module.BusinessDataError) as caught:
        module.MarkdownBusinessData(root).load()
    assert caught.value.source=='orders.md' and caught.value.line>=4


def test_orphan_logistics_rejected(root):
    (root/'logistics.md').write_text(LOGISTICS.replace('O02','O99'),encoding='utf-8')
    module=api()
    with pytest.raises(module.BusinessDataError,match='orphan_logistics'):
        module.MarkdownBusinessData(root).load()


def test_source_change_is_visible_and_invalid_change_never_uses_cached_order(root):
    module=api(); data=module.MarkdownBusinessData(root)
    digest=data.load().digest
    (root/'orders.md').write_text(ORDERS.replace('99.00','89.00'),encoding='utf-8')
    assert data.get_item('C01','O01','I01').paid_amount==Decimal('89.00')
    assert data.load().digest!=digest
    (root/'orders.md').write_text('broken',encoding='utf-8')
    with pytest.raises(module.BusinessDataError): data.list_items('C01')


def test_inconsistent_dates_are_flagged_without_rewriting_source(root):
    body=ORDERS.replace('2026-10-03 12:00:00','2026-09-30 12:00:00',1)
    (root/'orders.md').write_text(body,encoding='utf-8')
    item=api().MarkdownBusinessData(root).get_item('C01','O01','I01')
    assert 'signed_before_ordered' in item.issues
    assert (root/'orders.md').read_text(encoding='utf-8')==body


def test_order_tools_use_captured_customer_and_exact_records(root):
    from app.tools import build_tools
    data=api().MarkdownBusinessData(root)
    tools=build_tools(None,123,None,business_data=data,customer_id='C01')
    order=tools['query_order'].invoke({'order_id':'O01'})
    assert order['available'] is True
    assert [(i['item_id'],i['paid_amount']) for i in order['items']]==[('I01','99.00'),('I02','132.05')]
    assert tools['query_order'].invoke({'order_id':'O02'})['reason']=='not_found'
    assert tools['query_logistics'].invoke({'order_id':'O01'})['status']=='已签收'


def test_order_tool_requests_customer_without_guessing(root):
    from app.tools import build_tools
    tools=build_tools(None,123,None,business_data=api().MarkdownBusinessData(root))
    assert tools['query_order'].invoke({'order_id':'O01'})['reason']=='customer_id_required'
    assert tools['query_logistics'].invoke({'order_id':'O01'})['reason']=='customer_id_required'


def test_missing_table_end_is_rejected(root):
    (root/'orders.md').write_text(ORDERS.replace('无售后 |','无售后',1),encoding='utf-8')
    module=api()
    with pytest.raises(module.BusinessDataError): module.MarkdownBusinessData(root).load()

NEW_LOGISTICS = '''# 物流数据
| 订单号 | 商品项ID | 运单号 | 物流类型 | 承运商 | 物流状态 | 最新更新时间 | 是否最新 | 轨迹 |
|---|---|---|---|---|---|---|---|---|
| O01 | I01、I02 | W01 | 正向物流 | 快递 | 运输中 | 2026-10-02 12:00:00 | 否 | 在途 |
| O01 | I01、I02 | W01 | 正向物流 | 快递 | 已签收 | 2026-10-03 12:00:00 | 是 | 已签收 |
| O02 | I03 | W02 | 退货物流 | 快递 | 已签收 | 2026-10-03 12:00:00 | 是 | 商家收货 |
'''


def test_waybill_history_and_return_direction(root):
    (root/'logistics.md').write_text(NEW_LOGISTICS,encoding='utf-8')
    data=api().MarkdownBusinessData(root)
    shipment=data.get_logistics('C01','O01')['shipments'][0]
    assert shipment['status']=='已签收' and shipment['waybill_id']=='W01'
    assert shipment['item_ids']==['I01','I02'] and shipment['direction']=='正向物流'
    assert len(shipment['history'])==2
    assert 'logistics_signature_conflict' not in data.get_item('C02','O02','I03').issues


@pytest.mark.parametrize('before,after,code',[
    ('| I01、I02 |','| I03 |','logistics_item_mismatch'),
    ('| 否 |','| 是 |','invalid_latest_snapshot'),
    ('| 是 |','| 否 |','invalid_latest_snapshot'),
    ('2026-10-02 12:00:00','2026-10-04 12:00:00','invalid_latest_snapshot'),
    ('| 退货物流 |','| 错误方向 |','invalid_logistics_direction'),
])
def test_waybill_inconsistencies_rejected(root,before,after,code):
    (root/'logistics.md').write_text(NEW_LOGISTICS.replace(before,after,1),encoding='utf-8')
    with pytest.raises(api().BusinessDataError,match=code):
        api().MarkdownBusinessData(root).load()


def test_not_applicable_signature_is_preserved_and_not_missing(root):
    (root/'orders.md').write_text(ORDERS.replace('2026-10-03 12:00:00','不适用').replace('已签收','交易完成'),encoding='utf-8')
    (root/'logistics.md').write_text('\n'.join(LOGISTICS.splitlines()[:3])+'\n',encoding='utf-8')
    item=api().MarkdownBusinessData(root).get_item('C01','O01','I01')
    assert item.signed_at is None and item.signature_applicable is False
    assert item.issues==() and item.to_dict()['signature_applicable'] is False
