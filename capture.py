"""Capture step for claude-proxy: parse, classify, write to disk, emit a Phoenix span.

Two-phase so a call that fails or is aborted still leaves a file:
    cap = begin(request, body)          # before forwarding: writes request + "pending"
    finish(cap, status, headers, ...)   # after stream ends: rewrites with response + span
"""

import hashlib
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode

CAPTURE_DIR = Path(os.environ.get("CLAUDE_PROXY_CAPTURE_DIR", "~/.claude-proxy/captures")).expanduser()
STORE_MODE = os.environ.get("CLAUDE_PROXY_STORE", "full")  # full | dedupe
PHOENIX_ENDPOINT = os.environ.get("CLAUDE_PROXY_PHOENIX", "http://localhost:6006/v1/traces")  # "" disables
PROJECT_NAME = os.environ.get("CLAUDE_PROXY_PROJECT", "claude-proxy")

# Never written to disk or to Phoenix.
SECRET_HEADERS = {"authorization", "x-api-key", "cookie", "proxy-authorization"}

# Claude Code's compaction request asks the model for a conversation summary.
# Refine after seeing real captures.
COMPACTION_MARKERS = ("summary of the conversation", "summarize the conversation", "summary of our conversation")

_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None
_seq = 0


# ----------------------------------------------------------------------------- setup


def setup() -> None:
    """Create capture dir (mode 700) and the Phoenix tracer. Call once at proxy startup."""
    global _provider, _tracer
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CAPTURE_DIR, 0o700)
    if not PHOENIX_ENDPOINT:
        return
    resource = Resource.create({ResourceAttributes.PROJECT_NAME: PROJECT_NAME})
    # Default limit is 128 attributes per span and the oldest get evicted. One call with
    # 160 tools and a long message history is far more than that. Neither None nor UNSET
    # lifts the count limit in this SDK (both resolve to 128); an explicit large int does.
    limits = SpanLimits(max_attributes=1_000_000, max_attribute_length=None)
    _provider = TracerProvider(resource=resource, span_limits=limits)
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)))
    _tracer = _provider.get_tracer("claude-proxy")


def shutdown() -> None:
    """Flush pending spans. Call at proxy shutdown or the last spans are lost."""
    if _provider is not None:
        _provider.shutdown()


# ----------------------------------------------------------------------------- request


@dataclass
class Capture:
    path: Path
    started_ns: int
    meta: dict
    request: dict  # parsed body, or {"_raw": "..."} if not JSON
    raw_request: bytes
    blobs: dict = field(default_factory=dict)  # dedupe mode: name -> hash


def begin(method: str, path_qs: str, headers: Any, body: bytes) -> Capture:
    """Write the request to disk before forwarding. Returns a handle for finish()."""
    global _seq
    _seq += 1
    now = datetime.now(timezone.utc)
    hdrs = {k: v for k, v in headers.items() if k.lower() not in SECRET_HEADERS}
    lc = {k.lower(): v for k, v in hdrs.items()}  # header names are case-insensitive on the wire
    session_id = lc.get("x-claude-code-session-id") or "no-session"

    try:
        req = json.loads(body) if body else {}
    except ValueError:
        req = {"_raw": body.decode("utf-8", "replace")}

    meta = {
        "captured_at": now.isoformat(),
        "method": method,
        "path": path_qs,
        "status": "pending",
        "session_id": lc.get("x-claude-code-session-id"),
        "agent_id": lc.get("x-claude-code-agent-id"),
        "parent_agent_id": lc.get("x-claude-code-parent-agent-id"),
        "anthropic_version": lc.get("anthropic-version"),
        "anthropic_beta": [b.strip() for b in lc.get("anthropic-beta", "").split(",") if b.strip()],
        "request_headers": hdrs,
        "request_bytes": len(body),
        "call_class": classify(req, lc),
        "aborted": False,
        "error": None,
    }

    session_dir = CAPTURE_DIR / _safe(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{now.strftime('%Y%m%dT%H%M%S.%f')}-{_seq:04d}.json"
    cap = Capture(path=session_dir / fname, started_ns=time.time_ns(), meta=meta, request=req, raw_request=body)
    _write(cap, response=None)
    return cap


def classify(req: dict, hdrs: dict) -> str:
    if hdrs.get("x-claude-code-agent-id"):
        return "subagent"
    if _mentions_compaction(req):
        return "compaction"
    if not req.get("tools"):
        return "side"
    return "main"


def _mentions_compaction(req: dict) -> bool:
    texts = []
    sys_ = req.get("system")
    if isinstance(sys_, str):
        texts.append(sys_)
    elif isinstance(sys_, list):
        texts += [b.get("text", "") for b in sys_ if isinstance(b, dict)]
    msgs = req.get("messages") or []
    if msgs:
        last = msgs[-1]
        c = last.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            texts += [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
    joined = "\n".join(texts).lower()
    return any(m in joined for m in COMPACTION_MARKERS)


# ----------------------------------------------------------------------------- response


def finish(
    cap: Capture,
    status: Optional[int],
    response_headers: Any,
    response_body: bytes,
    aborted: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record the response, rewrite the capture file, emit the span."""
    ended_ns = time.time_ns()
    rh = {k: v for k, v in (response_headers.items() if response_headers is not None else [])}
    rlc = {k.lower(): v for k, v in rh.items()}
    cap.meta.update(
        {
            "status": status,
            "duration_ms": round((ended_ns - cap.started_ns) / 1e6, 1),
            "request_id": rh.get("request-id"),
            "response_headers": rh,
            "response_bytes": len(response_body),
            "aborted": aborted,
            "error": error,
        }
    )
    decoded, decode_error = _decode(response_body, rlc.get("content-encoding", ""))
    if decode_error:
        cap.meta["error"] = (error + "; " if error else "") + decode_error
    cap.meta["response_encoding"] = rlc.get("content-encoding")
    response = parse_response(decoded, rlc.get("content-type", ""))
    if status == 200 and response_body and not aborted and _parsed_nothing(response):
        cap.meta["error"] = (cap.meta.get("error") + "; " if cap.meta.get("error") else "") + "200 response parsed to nothing"
    cap.meta["has_compaction_block"] = _has_compaction_block(cap.request, response)
    _write(cap, response)
    _emit_span(cap, response, ended_ns)


def _decode(body: bytes, encoding: str) -> tuple[bytes, Optional[str]]:
    """Undo Content-Encoding on our copy only. Claude Code advertises gzip, deflate, br, zstd.
    gzip/deflate use the stdlib; br and zstd need extra packages, so they are reported, not decoded."""
    enc = encoding.strip().lower()
    if not body or enc in ("", "identity"):
        return body, None
    try:
        # decompressobj returns whatever it can decode from a truncated (aborted) stream.
        if enc == "gzip":
            return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body), None
        if enc == "deflate":
            return zlib.decompressobj().decompress(body), None
        if enc == "br":
            import brotli  # type: ignore
            return brotli.Decompressor().process(body), None
        if enc == "zstd":
            import zstandard  # type: ignore
            return zstandard.ZstdDecompressor().decompressobj().decompress(body), None
    except ImportError:
        return body, f"response is {enc}-encoded and no decoder is installed"
    except Exception as exc:  # truncated stream on abort, etc.
        return body, f"{enc} decode failed: {exc}"
    return body, f"unknown content-encoding {enc!r}"


def parse_response(body: bytes, content_type: str) -> dict:
    """SSE stream -> reassembled message. Plain JSON -> as is. Else raw text."""
    text = body.decode("utf-8", "replace")
    if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
        return reassemble_sse(text)
    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        return {"_raw": text}


def reassemble_sse(text: str) -> dict:
    """Rebuild the final message from Anthropic's SSE events.

    Traps handled: tool_use input arrives as partial_json fragments; thinking arrives as
    thinking_delta + signature_delta; input/cache tokens are in message_start, output
    tokens in message_delta; an error event can arrive mid-stream after a 200.
    """
    message: dict = {"content": [], "_events": 0, "_event_types": {}}
    blocks: dict = {}
    partial_json: dict = {}
    for raw in text.split("\n\n"):
        data_lines = [l[5:].strip() for l in raw.split("\n") if l.startswith("data:")]
        if not data_lines:
            continue
        try:
            ev = json.loads("\n".join(data_lines))
        except ValueError:
            continue
        t = ev.get("type")
        message["_events"] += 1
        message["_event_types"][t] = message["_event_types"].get(t, 0) + 1

        if t == "message_start":
            m = ev.get("message", {})
            message.update({k: v for k, v in m.items() if k != "content"})
        elif t == "content_block_start":
            i = ev["index"]
            blocks[i] = dict(ev.get("content_block", {}))
            if blocks[i].get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                partial_json[i] = ""
        elif t == "content_block_delta":
            i = ev["index"]
            d = ev.get("delta", {})
            dt = d.get("type")
            b = blocks.setdefault(i, {})
            if dt == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            elif dt == "signature_delta":
                b["signature"] = b.get("signature", "") + d.get("signature", "")
            elif dt == "input_json_delta":
                partial_json[i] = partial_json.get(i, "") + d.get("partial_json", "")
            elif dt == "citations_delta":
                b.setdefault("citations", []).append(d.get("citation"))
            else:
                b.setdefault("_other_deltas", []).append(d)
        elif t == "content_block_stop":
            i = ev["index"]
            if i in partial_json:
                pj = partial_json.pop(i)
                try:
                    blocks[i]["input"] = json.loads(pj) if pj.strip() else {}
                except ValueError:
                    blocks[i]["input"] = {"_unparsed": pj}
        elif t == "message_delta":
            message.update(ev.get("delta", {}))
            if "usage" in ev:
                usage = dict(message.get("usage") or {})
                usage.update({k: v for k, v in ev["usage"].items() if v is not None})
                message["usage"] = usage
        elif t == "error":
            message["error"] = ev.get("error")
        # ping, message_stop: nothing to keep

    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


def _parsed_nothing(resp: dict) -> bool:
    if "_raw" in resp:
        return True
    if "_events" in resp:
        return resp["_events"] == 0
    return not resp


def _has_compaction_block(req: dict, resp: dict) -> bool:
    def scan(content: Any) -> bool:
        return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "compaction" for b in content)

    if scan(resp.get("content")):
        return True
    return any(scan(m.get("content")) for m in (req.get("messages") or []) if isinstance(m, dict))


# ----------------------------------------------------------------------------- disk


def _write(cap: Capture, response: Optional[dict]) -> None:
    req = cap.request
    if STORE_MODE == "dedupe" and isinstance(req, dict) and "_raw" not in req:
        req = dict(req)
        for key in ("system", "tools"):
            if key in req:
                h = _blob(cap.path.parent, key, req[key])
                cap.blobs[key] = h
                req[key] = {"_blob": h}
    doc = {"meta": cap.meta, "request": req, "response": response}
    tmp = cap.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    os.chmod(tmp, 0o600)
    tmp.replace(cap.path)


def _blob(session_dir: Path, name: str, value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(data.encode()).hexdigest()[:16]
    bdir = session_dir / "blobs"
    bdir.mkdir(exist_ok=True)
    p = bdir / f"{name}-{h}.json"
    if not p.exists():
        p.write_text(data)
        os.chmod(p, 0o600)
    return h


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


# ----------------------------------------------------------------------------- phoenix

# Tree shape in Phoenix. Every HTTP call is independent on the wire: Claude Code sends a
# session id and, on sub-agent calls, an agent id, but nothing that says which call
# launched the agent. The tree is rebuilt here from what the calls contain:
#   - main (and side) calls are top-level rows.
#   - when a response contains a tool call carrying a task brief as `input.prompt` (the
#     Agent tool), that prompt is remembered against the call's span.
#   - a sub-agent's first call opens with that same brief verbatim in its first message.
#     Matching the text identifies the launching call exactly. Every call that agent makes
#     is then parented directly to that launching call, as siblings, never to each other.
#   - a sub-agent that launches its own sub-agent is handled the same way, one level down.
# A sub-agent whose launching call was never seen (proxy started mid-session, or the
# race where the first sub-agent request lands before the parent's span is emitted) is
# top-level for that call, marked `claude.spawned_by = "unmatched"`, and is re-tried on
# its later calls since the first message stays the same.


@dataclass
class _SessionState:
    agent_parent: dict = field(default_factory=dict)  # agent_id -> SpanContext of the launching call
    pending_spawns: list = field(default_factory=list)  # [(prompt_text, SpanContext)] not yet claimed


_sessions: dict[str, _SessionState] = {}
_MAX_PENDING_SPAWNS = 200  # per session; oldest unclaimed spawn is dropped past this
_MIN_SPAWN_PROMPT_LEN = 40  # shorter prompts are too likely to appear by coincidence


def _plain_text(content: Any) -> str:
    """Text blocks of an Anthropic message `content`; other block types are skipped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _spawn_prompts(content: Any) -> list[str]:
    """`input.prompt` of every tool call in an assistant response. The Agent tool hands
    the new agent its whole brief this way; the brief reappears in that agent's first call."""
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
            p = (b.get("input") or {}).get("prompt")
            if isinstance(p, str) and len(p) >= _MIN_SPAWN_PROMPT_LEN:
                out.append(p)
    return out


def _launching_call(sess: _SessionState, agent_id: str, req: Any) -> Optional[SpanContext]:
    """Span of the call that launched `agent_id`, or None if it was never seen."""
    if agent_id in sess.agent_parent:
        return sess.agent_parent[agent_id]
    if not isinstance(req, dict):
        return None
    msgs = req.get("messages") or []
    first_text = _plain_text(msgs[0].get("content")) if msgs and isinstance(msgs[0], dict) else ""
    for i, (prompt, ctx) in enumerate(sess.pending_spawns):
        if prompt in first_text:
            del sess.pending_spawns[i]
            sess.agent_parent[agent_id] = ctx
            return ctx
    return None


def _emit_span(cap: Capture, resp: dict, ended_ns: int) -> None:
    if _tracer is None:
        return
    req = cap.request
    meta = cap.meta
    model = req.get("model") if isinstance(req, dict) else None

    sess = _sessions.setdefault(meta.get("session_id") or "no-session", _SessionState())
    agent_id = meta.get("agent_id")
    parent_ctx = _launching_call(sess, agent_id, req) if meta["call_class"] == "subagent" and agent_id else None
    span = _tracer.start_span(
        name=f"{meta['call_class']}: {resp.get('model') or model or 'unknown'}",
        start_time=cap.started_ns,
        context=trace.set_span_in_context(NonRecordingSpan(parent_ctx)) if parent_ctx else None,
    )
    a: dict = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_PROVIDER: "anthropic",
        SpanAttributes.LLM_SYSTEM: "anthropic",
        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        SpanAttributes.INPUT_VALUE: cap.raw_request.decode("utf-8", "replace"),
        SpanAttributes.OUTPUT_VALUE: json.dumps(resp, ensure_ascii=False),
        "claude.call_class": meta["call_class"],
        "claude.capture_path": str(cap.path),
        "claude.aborted": meta["aborted"],
        "claude.http_status": meta.get("status") if isinstance(meta.get("status"), int) else 0,
        "claude.request_bytes": meta["request_bytes"],
        "claude.response_bytes": meta.get("response_bytes", 0),
        "claude.anthropic_beta": json.dumps(meta["anthropic_beta"]),
        "claude.compaction_beta": any(b.startswith("compact-") for b in meta["anthropic_beta"]),
        "claude.has_compaction_block": meta.get("has_compaction_block", False),
    }
    if meta["call_class"] == "subagent" and agent_id:
        a["claude.spawned_by"] = "matched" if parent_ctx else "unmatched"
    for k, key in (("session_id", SpanAttributes.SESSION_ID), ("agent_id", "claude.agent_id"),
                   ("parent_agent_id", "claude.parent_agent_id"), ("request_id", "claude.request_id"),
                   ("error", "claude.error")):
        if meta.get(k):
            a[key] = meta[k]
    if model:
        a[SpanAttributes.LLM_MODEL_NAME] = resp.get("model") or model
        a[SpanAttributes.LLM_REQUEST_MODEL_NAME] = model
    if resp.get("model"):
        a[SpanAttributes.LLM_RESPONSE_MODEL_NAME] = resp["model"]
    if resp.get("stop_reason"):
        a[SpanAttributes.LLM_FINISH_REASON] = resp["stop_reason"]

    if isinstance(req, dict) and "_raw" not in req:
        inv = {k: req[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "thinking", "output_config",
                                   "context_management", "tool_choice", "stream", "metadata") if k in req}
        inv["anthropic_beta"] = meta["anthropic_beta"]
        a[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(inv, ensure_ascii=False)
        for i, tool in enumerate(req.get("tools") or []):
            a[f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}"] = json.dumps(tool, ensure_ascii=False)
        idx = 0
        if req.get("system") is not None:
            _msg_attrs(a, SpanAttributes.LLM_INPUT_MESSAGES, idx, {"role": "system", "content": req["system"]})
            idx += 1
        for m in req.get("messages") or []:
            _msg_attrs(a, SpanAttributes.LLM_INPUT_MESSAGES, idx, m)
            idx += 1

    if resp.get("content") is not None:
        _msg_attrs(a, SpanAttributes.LLM_OUTPUT_MESSAGES, 0, {"role": resp.get("role", "assistant"), "content": resp["content"]})

    u = resp.get("usage") or {}
    if u:
        p_in = u.get("input_tokens") or 0
        cr = u.get("cache_read_input_tokens") or 0
        cw = u.get("cache_creation_input_tokens") or 0
        out = u.get("output_tokens") or 0
        a[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = p_in + cr + cw
        a[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cr
        a[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] = cw
        a[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = out
        a[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = p_in + cr + cw + out

    for k, v in a.items():
        if v is not None:
            span.set_attribute(k, v)
    status = meta.get("status") if isinstance(meta.get("status"), int) else 0
    if meta["aborted"] or meta.get("error") or status >= 400 or resp.get("error"):
        span.set_status(Status(StatusCode.ERROR, meta.get("error") or json.dumps(resp.get("error")) or f"HTTP {status}"))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end(end_time=ended_ns)

    # Remember any agent briefs this response handed out, so the agents they launch can
    # be parented to this call when their first request arrives.
    spawned = _spawn_prompts(resp.get("content"))
    if spawned:
        ctx = span.get_span_context()
        sess.pending_spawns.extend((p, ctx) for p in spawned)
        del sess.pending_spawns[: max(0, len(sess.pending_spawns) - _MAX_PENDING_SPAWNS)]


def _msg_attrs(a: dict, prefix: str, i: int, m: dict) -> None:
    """Flatten one Anthropic message into OpenInference indexed attributes."""
    p = f"{prefix}.{i}."
    a[p + MessageAttributes.MESSAGE_ROLE] = m.get("role", "")
    c = m.get("content")
    if isinstance(c, str):
        a[p + MessageAttributes.MESSAGE_CONTENT] = c
        return
    if not isinstance(c, list):
        return
    texts, n_tc, n_cc, tool_result_ids = [], 0, 0, []
    for b in c:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text", ""))
            cp = f"{p}{MessageAttributes.MESSAGE_CONTENTS}.{n_cc}."
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TYPE] = "text"
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TEXT] = b.get("text", "")
            n_cc += 1
        elif t in ("tool_use", "server_tool_use", "mcp_tool_use"):
            tp = f"{p}{MessageAttributes.MESSAGE_TOOL_CALLS}.{n_tc}."
            a[tp + ToolCallAttributes.TOOL_CALL_ID] = b.get("id", "")
            a[tp + ToolCallAttributes.TOOL_CALL_FUNCTION_NAME] = b.get("name", "")
            a[tp + ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON] = json.dumps(b.get("input", {}), ensure_ascii=False)
            n_tc += 1
        elif t == "tool_result":
            # Anthropic bundles parallel tool results into one user message; one attribute
            # per result, and the single tool_call_id attribute holds all ids joined.
            tool_result_ids.append(b.get("tool_use_id", ""))
            rc = b.get("content")
            rtext = rc if isinstance(rc, str) else json.dumps(rc, ensure_ascii=False)
            texts.append(rtext)
            cp = f"{p}{MessageAttributes.MESSAGE_CONTENTS}.{n_cc}."
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TYPE] = "tool_result"
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_ID] = b.get("tool_use_id", "")
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TEXT] = rtext
            n_cc += 1
        elif t in ("thinking", "redacted_thinking"):
            cp = f"{p}{MessageAttributes.MESSAGE_CONTENTS}.{n_cc}."
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TYPE] = t
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TEXT] = b.get("thinking", "")
            if b.get("signature"):
                a[cp + MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE] = b["signature"]
            n_cc += 1
        else:
            cp = f"{p}{MessageAttributes.MESSAGE_CONTENTS}.{n_cc}."
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TYPE] = t or "unknown"
            a[cp + MessageContentAttributes.MESSAGE_CONTENT_TEXT] = json.dumps(b, ensure_ascii=False)
            n_cc += 1
    if tool_result_ids:
        a[p + MessageAttributes.MESSAGE_TOOL_CALL_ID] = ",".join(tool_result_ids)
    if texts:
        a[p + MessageAttributes.MESSAGE_CONTENT] = "\n".join(texts)
