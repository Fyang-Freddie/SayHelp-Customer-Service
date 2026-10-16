"""Opaque visitor ownership, including the legacy conversation write routes."""
import asyncio
from hashlib import sha256
import re
import secrets
from starlette.requests import Request
from starlette.responses import JSONResponse,Response
from sqlalchemy.exc import SQLAlchemyError

COOKIE='sayhelp_visitor'


def get_or_create_visitor(request):
    token=request.cookies.get(COOKIE,'')
    created=re.fullmatch(r'[A-Za-z0-9_-]{43}',token) is None
    if created:token=secrets.token_urlsafe(32)
    return token,created


class VisitorMiddleware:
    def __init__(self,app,repository,enabled=True):
        self.app,self.repository,self.enabled=app,repository,enabled

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        request=Request(scope)
        token,created=get_or_create_visitor(request)
        visitor=sha256(token.encode()).hexdigest()
        scope.setdefault('state',{})['visitor_id']=visitor
        headers=[]
        if created:
            response=Response()
            response.set_cookie(COOKIE,token,httponly=True,samesite='lax',
                secure=scope.get('scheme')=='https',max_age=31536000)
            headers=[h for h in response.raw_headers if h[0]==b'set-cookie']
        async def send_headers(message):
            if message['type']=='http.response.start':
                message={**message,'headers':[*message.get('headers',[]),*headers]}
            await send(message)
        path=scope['path']
        match=re.match(r'^/(?:api|v1)/conversations/(\d+)(?:/|$)',path)
        if match and (self.enabled or path.startswith('/api/')):
            try:await asyncio.to_thread(self.repository.assert_owner,int(match[1]),visitor)
            except KeyError:
                return await JSONResponse({'detail':'Conversation not found'},status_code=404)(scope,receive,send_headers)
            except SQLAlchemyError:
                return await JSONResponse({'detail':'Conversation storage unavailable'},status_code=503)(scope,receive,send_headers)
        await self.app(scope,receive,send_headers)
