"""Pass-through proxy between Claude Code and api.anthropic.com.

Forwards every request unchanged, streams the response back as bytes arrive,
and hands both directions to capture.py (disk + Phoenix). See PLAN.md.
"""

import asyncio
import os
import sys

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

import capture

UPSTREAM = os.environ.get("CLAUDE_PROXY_UPSTREAM", "https://api.anthropic.com")  # override for testing only
PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "8787"))  # override for running a second instance

# Server: default body cap is 1 MiB; long conversations exceed that per request.
CLIENT_MAX_SIZE = 1024 ** 3

# Client: default total timeout is 300s, which would cut long streams. Read timeout
# is set just above Claude Code's own 300s silence limit: upstream pings keep the
# socket alive during thinking pauses, and a fully silent upstream is bounded instead
# of leaking the handler forever (aiohttp does not cancel handlers on client disconnect).
CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=330)

# Headers aiohttp would add if absent. Upstream must see exactly what Claude Code sent.
SKIP_AUTO_HEADERS = {"Accept-Encoding", "User-Agent", "Accept"}

# Hop-by-hop headers. aiohttp sets its own framing and errors if these are set manually.
DROP_REQUEST_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}
DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection", "keep-alive"}


def is_inference(path: str) -> bool:
    return path == "/v1/messages"


async def handle(request: web.Request) -> web.StreamResponse:
    session: aiohttp.ClientSession = request.app["session"]

    body = await request.read()  # bounded by CLIENT_MAX_SIZE
    # CIMultiDict keeps repeated header names instead of collapsing them to one.
    headers = CIMultiDict(
        (k, v) for k, v in request.headers.items() if k.lower() not in DROP_REQUEST_HEADERS
    )
    url = UPSTREAM + request.path_qs

    # Write the request to disk before forwarding so failed/aborted calls leave a file.
    cap = None
    if is_inference(request.path):
        cap = _safe_capture(capture.begin, request.method, request.path_qs, request.headers, body)

    try:
        upstream = await session.request(
            request.method,
            url,
            headers=headers,
            data=body if body else None,
            allow_redirects=False,
        )
    except aiohttp.ClientError as exc:
        if cap:
            _safe_capture(capture.finish, cap, None, None, b"", error=f"upstream connection failed: {exc}")
        return web.Response(status=502, text=f"proxy: upstream connection failed: {exc}")

    response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
    for k, v in upstream.headers.items():
        if k.lower() not in DROP_RESPONSE_HEADERS:
            response.headers.add(k, v)

    response_copy = bytearray()
    aborted, error = False, None
    try:
        await response.prepare(request)
        if request.method != "HEAD":
            async for chunk in upstream.content.iter_any():
                response_copy += chunk
                await response.write(chunk)
        await response.write_eof()
    except ConnectionResetError:
        aborted, error = True, "client disconnected"  # Claude Code went away (Escape)
    except aiohttp.ClientError as exc:
        aborted, error = True, f"upstream error mid-stream: {exc}"
    except asyncio.CancelledError:
        aborted, error = True, "cancelled"  # shutdown; record, then let it propagate
        upstream.release()
        if cap:
            _safe_capture(capture.finish, cap, upstream.status, upstream.headers, bytes(response_copy), aborted=True, error=error)
        raise
    finally:
        upstream.release()

    if cap:
        _safe_capture(capture.finish, cap, upstream.status, upstream.headers, bytes(response_copy), aborted=aborted, error=error)

    return response


def _safe_capture(fn, *args, **kwargs):
    """Capture must never break the relay. Failures go to stderr; the call proceeds."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"claude-proxy: capture error in {fn.__name__}: {exc!r}", file=sys.stderr)
        return None


async def on_startup(app: web.Application) -> None:
    capture.setup()
    app["session"] = aiohttp.ClientSession(
        timeout=CLIENT_TIMEOUT,
        auto_decompress=False,
        skip_auto_headers=SKIP_AUTO_HEADERS,
    )


async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()
    capture.shutdown()  # flush pending spans or the last ones are lost


def main() -> None:
    app = web.Application(client_max_size=CLIENT_MAX_SIZE)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    print(f"claude-proxy listening on http://localhost:{PORT} -> {UPSTREAM}", file=sys.stderr)
    print(f"  captures: {capture.CAPTURE_DIR} (mode={capture.STORE_MODE})  phoenix: {capture.PHOENIX_ENDPOINT or 'disabled'} project={capture.PROJECT_NAME}", file=sys.stderr)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
