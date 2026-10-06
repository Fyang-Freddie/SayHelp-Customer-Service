"""Versioned keyset cursors for pinned and ordinary history groups."""
import base64
import binascii
import json
import re
from datetime import datetime

MAX_ID = 18446744073709551615
FIELDS = {'version', 'pin_group', 'pinned_at', 'activity_id', 'id'}
ERROR = 'Invalid history cursor; refresh the history list'


def encode_cursor(group: int, pinned_at: datetime | None, activity_id: int, id: int) -> str:
    payload = {'version': 1, 'pin_group': group,
               'pinned_at': pinned_at.isoformat() if pinned_at else None,
               'activity_id': activity_id, 'id': id}
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).rstrip(b'=').decode()


def decode_cursor(token: str) -> dict:
    try:
        if not isinstance(token, str) or not 2 <= len(token) <= 512 or not re.fullmatch(r'[A-Za-z0-9_-]+', token):
            raise ValueError(ERROR)
        data = base64.b64decode(token + '=' * (-len(token) % 4), altchars=b'-_', validate=True)
        if base64.urlsafe_b64encode(data).rstrip(b'=').decode() != token:
            raise ValueError(ERROR)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError(ERROR)
                result[key] = value
            return result
        payload = json.loads(data, object_pairs_hook=unique)
        if not isinstance(payload, dict) or set(payload) != FIELDS:
            raise ValueError(ERROR)
        if type(payload['version']) is not int or payload['version'] != 1:
            raise ValueError(ERROR)
        if type(payload['pin_group']) is not int or payload['pin_group'] not in (0, 1):
            raise ValueError(ERROR)
        for key in ('activity_id', 'id'):
            if type(payload[key]) is not int or not 1 <= payload[key] <= MAX_ID:
                raise ValueError(ERROR)
        stamp = payload['pinned_at']
        if payload['pin_group'] == 0:
            if stamp is not None: raise ValueError(ERROR)
        else:
            if not isinstance(stamp, str): raise ValueError(ERROR)
            parsed = datetime.fromisoformat(stamp)
            if parsed.tzinfo is not None or parsed.microsecond or parsed.year < 1000 or parsed.isoformat() != stamp:
                raise ValueError(ERROR)
            payload['pinned_at'] = parsed
        return payload
    except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError):
        raise ValueError(ERROR) from None
