"""Bound HTTP JSON bytes before parsing, including chunked transfer bodies."""
from starlette.responses import JSONResponse


class RequestBodyLimit:
    def __init__(self, app, max_bytes=131072):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        size, chunks = 0, []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            part = message.get('body', b'')
            size += len(part)
            if size > self.max_bytes:
                return await JSONResponse({'detail': 'Request exceeds 128 KiB'}, status_code=413)(scope, receive, send)
            chunks.append(part)
            if not message.get('more_body', False):
                break
        delivered = False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()
        return await self.app(scope, bounded_receive, send)
