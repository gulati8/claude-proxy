# claude-proxy

A local pass-through proxy between Claude Code and `api.anthropic.com`. It forwards every
request unchanged, streams the response back, and saves what went over the wire: the full
request (system prompt, tools, messages) and the reassembled response, one JSON file per call.
Optionally it also sends each call to a local [Phoenix](https://github.com/Arize-ai/phoenix)
instance as an LLM span.

Use it to see exactly what Claude Code sends to the model.

## Run it

Python 3.11 or newer.

```
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python proxy.py
```

It listens on `http://localhost:8787`. Keep it running in the background however you prefer
(a service manager, tmux, a terminal tab). If it is down, Claude Code cannot reach the model.

## Point Claude Code at it

In `~/.claude/settings.json`:

```json
"model": "fable",
"env": {
  "ANTHROPIC_BASE_URL": "http://localhost:8787",
  "ENABLE_TOOL_SEARCH": "true",
  "ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-fable-5-1[1m]",
  "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-5-5[1m]",
  "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5[1m]"
}
```

All of the `env` lines matter. When Claude Code sees a custom base URL it switches into
"gateway mode": it stops deferring MCP tools (every tool gets inlined into every request) and
it assumes every model has a 200K context window. `ENABLE_TOOL_SEARCH` fixes the first.
The `[1m]` suffix on the three model defaults fixes the second for the main conversation
and for every sub-agent, whichever model alias they use. Claude Code strips the suffix
before sending. Set `"model"` to whichever alias you want as your default; `/model` still works.

Login passes straight through. Nothing about auth changes.

To skip the proxy for one session: `ANTHROPIC_BASE_URL=https://api.anthropic.com claude`.

## Where things land

| What | Where |
|---|---|
| One JSON file per call | `~/.claude-proxy/captures/<session-id>/<timestamp>-<n>.json` |
| Phoenix spans (optional) | project `claude-proxy` at `http://localhost:6006` |

Each capture file has three parts:

```
meta      status, duration, call_class (main / subagent / compaction / side), session_id,
          agent_id, request_id, beta headers, request/response headers, aborted, error
request   the request body as sent: model, system, tools, messages, thinking, ...
response  the reassembled message: content blocks, stop_reason, usage
```

`Authorization`, `x-api-key`, and `cookie` headers are never written.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_PROXY_PORT` | `8787` | Listen port |
| `CLAUDE_PROXY_CAPTURE_DIR` | `~/.claude-proxy/captures` | Where capture files go |
| `CLAUDE_PROXY_STORE` | `full` | `full` stores the whole request per call. `dedupe` stores `system` and `tools` once per session. |
| `CLAUDE_PROXY_PHOENIX` | `http://localhost:6006/v1/traces` | Phoenix endpoint. Set to an empty string to disable. |
| `CLAUDE_PROXY_PROJECT` | `claude-proxy` | Phoenix project name |

## Tests

```
./.venv/bin/python -m unittest discover -s tests
```
