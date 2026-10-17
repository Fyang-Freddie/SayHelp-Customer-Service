"""Exact, customer-scoped business records from the supplied Markdown tables."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re

TZ = timezone(timedelta(hours=8), 'Asia/Shanghai')
HEADERS = {
    'orders.md': ('客户ID','订单号','商品项ID','商品名称','型号/规格','数量','实付金额（元）','下单时间','签收时间','订单状态','售后状态'),
    'logistics.md': ('订单号','承运商','物流状态','最新更新时间','轨迹'),
    'refund-reasons.md': ('原因编码','显示名称'),
}


class BusinessDataError(ValueError):
    def __init__(self, code: str, source: str = '', line: int = 0):
        self.code, self.source, self.line = code, source, line
        super().__init__(f'{code}: {source}:{line}')


@dataclass(frozen=True)
class OrderItem:
    customer_id: str
    order_id: str
    item_id: str
    product_name: str
    variant: str
    quantity: int
    paid_amount: Decimal
    ordered_at: datetime | None
    signed_at: datetime | None
    order_status: str
    aftersales_status: str
    issues: tuple[str, ...] = ()
    signature_applicable: bool = True

    def to_dict(self) -> dict:
        row = asdict(self)
        row['paid_amount'] = str(self.paid_amount)
        for field in ('ordered_at', 'signed_at'):
            row[field] = row[field].isoformat() if row[field] is not None else None
        row['issues'] = list(self.issues)
        return row


@dataclass(frozen=True)
class BusinessSnapshot:
    items: tuple[OrderItem, ...]
    logistics: dict[str, dict]
    reasons: tuple[dict, ...]
    digest: str


LOGISTICS_HEADER = ("订单号","商品项ID","运单号","物流类型","承运商","物流状态","最新更新时间","是否最新","轨迹")


def _table(body: str, source: str, header=None):
    header = header or HEADERS[source]
    for n, line in enumerate(body.splitlines(), 1):
        if line.strip().startswith('|') and not line.strip().endswith('|'):
            raise BusinessDataError('invalid_row', source, n)
    rows = [(n, [c.strip() for c in line.strip()[1:-1].split('|')])
            for n, line in enumerate(body.splitlines(), 1) if line.strip().startswith('|')]
    if len(rows) < 2 or tuple(rows[0][1]) != header:
        raise BusinessDataError('invalid_header', source, rows[0][0] if rows else 1)
    if len(rows[1][1]) != len(header) or not all(re.fullmatch(r':?-{3,}:?', c) for c in rows[1][1]):
        raise BusinessDataError('invalid_separator', source, rows[1][0])
    for line, cells in rows[2:]:
        if len(cells) != len(header) or any(not cell for cell in cells):
            raise BusinessDataError('invalid_row', source, line)
        yield line, cells


def _date(value, source, line):
    if value in ('未知', '未签收'):
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
    except ValueError:
        raise BusinessDataError('invalid_date', source, line) from None


class MarkdownBusinessData:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def load(self) -> BusinessSnapshot:
        bodies, digest = {}, hashlib.sha256()
        for name in HEADERS:
            path = self.root / name
            try:
                if path.is_symlink() or path.resolve().parent != self.root or path.stat().st_size > 2_000_000:
                    raise BusinessDataError('invalid_source', name, 1)
                raw = path.read_bytes()
                bodies[name] = raw.decode('utf-8-sig')
            except (OSError, UnicodeError):
                raise BusinessDataError('unavailable_source', name, 1) from None
            digest.update(name.encode()); digest.update(b'\0'); digest.update(raw); digest.update(b'\0')
        items, seen, owners = [], set(), {}
        for line, row in _table(bodies['orders.md'], 'orders.md'):
            customer, order, item, product, variant, quantity, amount, ordered, signed, status, aftersales = row
            if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity) for identity in (customer, order, item)):
                raise BusinessDataError('invalid_identity', 'orders.md', line)
            if item in seen:
                raise BusinessDataError('duplicate_item', 'orders.md', line)
            if order in owners and owners[order] != customer:
                raise BusinessDataError('conflicting_owner', 'orders.md', line)
            seen.add(item); owners[order] = customer
            try:
                if not re.fullmatch(r'[1-9]\d{0,5}', quantity): raise ValueError()
                paid = Decimal(amount)
                if not paid.is_finite() or not 0 <= paid <= Decimal('999999999.99') or paid.as_tuple().exponent < -2:
                    raise ValueError()
            except (ValueError, InvalidOperation):
                raise BusinessDataError('invalid_amount_or_quantity', 'orders.md', line) from None
            signature_applicable = signed != '不适用'
            ordered_at = _date(ordered, 'orders.md', line)
            signed_at = _date(signed, 'orders.md', line) if signature_applicable else None
            issues = []
            if ordered_at is None: issues.append('unknown_order_time')
            if signed_at and ordered_at and signed_at < ordered_at: issues.append('signed_before_ordered')
            if signature_applicable and signed_at is None and status in ('已签收', '交易完成'): issues.append('missing_signature_time')
            if signed_at is not None and status in ('待发货', '未发货', '运输中', '派送中'): issues.append('signature_status_conflict')
            items.append(OrderItem(customer,order,item,product,variant,int(quantity),paid,ordered_at,signed_at,status,aftersales,tuple(issues),signature_applicable))
        logistics = {}
        first = next((line for line in bodies['logistics.md'].splitlines() if line.strip().startswith('|')), '')
        extended = '运单号' in first
        if extended:
            groups = {}
            by_item = {item.item_id:item for item in items}
            for line, row in _table(bodies['logistics.md'], 'logistics.md', LOGISTICS_HEADER):
                order, item_text, waybill, direction, carrier, status, updated, latest, trajectory = row
                if order not in owners: raise BusinessDataError('orphan_logistics','logistics.md',line)
                identities = item_text.split('、')
                if len(set(identities)) != len(identities) or any(i not in by_item or by_item[i].order_id != order for i in identities):
                    raise BusinessDataError('logistics_item_mismatch','logistics.md',line)
                if direction not in ('正向物流','退货物流'):
                    raise BusinessDataError('invalid_logistics_direction','logistics.md',line)
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',waybill) or latest not in ('是','否'):
                    raise BusinessDataError('invalid_waybill','logistics.md',line)
                at = _date(updated,'logistics.md',line)
                if at is None: raise BusinessDataError('invalid_date','logistics.md',line)
                record = {'order_id':order,'item_ids':identities,'waybill_id':waybill,'direction':direction,
                          'carrier':carrier,'status':status,'updated_at':at.isoformat(),'trajectory':trajectory}
                groups.setdefault(waybill,[]).append((line,latest,record))
            for snapshots in groups.values():
                line, _, first_record = snapshots[0]
                keys = ('order_id','item_ids','direction','carrier')
                if any(any(row[k]!=first_record[k] for k in keys) for _,_,row in snapshots):
                    raise BusinessDataError('conflicting_waybill','logistics.md',line)
                current = [row for _,flag,row in snapshots if flag=='是']
                times = [row['updated_at'] for _,_,row in snapshots]
                if len(current)!=1 or len(set(times))!=len(times) or current[0]['updated_at']!=max(times):
                    raise BusinessDataError('invalid_latest_snapshot','logistics.md',line)
                shipment = {**current[0], 'history':sorted((row for _,_,row in snapshots),key=lambda r:r['updated_at'])}
                order = shipment['order_id']
                logistics.setdefault(order,{'order_id':order,'shipments':[]})['shipments'].append(shipment)
        else:
            for line, row in _table(bodies['logistics.md'], 'logistics.md'):
                order, carrier, status, updated, trajectory = row
                if order not in owners: raise BusinessDataError('orphan_logistics', 'logistics.md', line)
                if order in logistics: raise BusinessDataError('duplicate_logistics', 'logistics.md', line)
                at = _date(updated, 'logistics.md', line)
                logistics[order] = {'order_id':order,'carrier':carrier,'status':status,
                                    'updated_at':at.isoformat() if at else None,'trajectory':trajectory}
        reasons, codes = [], set()
        for line, row in _table(bodies['refund-reasons.md'], 'refund-reasons.md'):
            code, label = row
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', code) or code in codes:
                raise BusinessDataError('invalid_reason', 'refund-reasons.md', line)
            codes.add(code); reasons.append({'code':code,'label':label})
        if not reasons: raise BusinessDataError('missing_reasons', 'refund-reasons.md', 1)
        checked = []
        for item in items:
            log = logistics.get(item.order_id)
            issues = list(item.issues)
            forward_signed = log and (log.get('status') == '已签收' or any(
                row['direction']=='正向物流' and row['status']=='已签收' and item.item_id in row['item_ids']
                for row in log.get('shipments',[])))
            if forward_signed and item.signed_at is None:
                issues.append('logistics_signature_conflict')
            checked.append(replace(item,issues=tuple(issues)))
        return BusinessSnapshot(tuple(checked),logistics,tuple(reasons),digest.hexdigest())

    def list_items(self, customer_id: str) -> list[OrderItem]:
        return sorted((i for i in self.load().items if i.customer_id == customer_id),
                      key=lambda i: (-(i.ordered_at.timestamp() if i.ordered_at else 0),i.order_id,i.item_id))

    def get_item(self, customer_id: str, order_id: str, item_id: str) -> OrderItem:
        item = next((i for i in self.load().items if (i.customer_id,i.order_id,i.item_id)==(customer_id,order_id,item_id)),None)
        if item is None: raise BusinessDataError('not_found')
        return item

    def get_logistics(self, customer_id: str, order_id: str) -> dict:
        snapshot = self.load()
        if not any((i.customer_id,i.order_id)==(customer_id,order_id) for i in snapshot.items):
            raise BusinessDataError('not_found')
        if order_id not in snapshot.logistics: raise BusinessDataError('logistics_not_found')
        return dict(snapshot.logistics[order_id])

    def refund_reasons(self) -> list[dict[str,str]]:
        return [dict(row) for row in self.load().reasons]
