"""Unit tests for capture.py. Run: ./.venv/bin/python -m unittest discover -s tests -v"""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["CLAUDE_PROXY_PHOENIX"] = ""  # no exporter in tests

import capture  # noqa: E402


def load(path):
    with open(path) as f:
        return json.load(f)


def text(path):
    with open(path) as f:
        return f.read()


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


STREAM = sse(
    {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-fable-5-1",
     "content": [], "stop_reason": None, "usage": {"input_tokens": 12, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 90000, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "let me "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "think"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "abc"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Running "}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "ls."}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls'}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ' -la"}'}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 57}},
    {"type": "message_stop"},
)


class ReassembleSSE(unittest.TestCase):
    def test_full_stream(self):
        m = capture.reassemble_sse(STREAM)
        self.assertEqual(m["model"], "claude-fable-5-1")
        self.assertEqual(m["stop_reason"], "tool_use")
        self.assertEqual(m["usage"], {"input_tokens": 12, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 90000, "output_tokens": 57})
        self.assertEqual(m["content"][0], {"type": "thinking", "thinking": "let me think", "signature": "abc"})
        self.assertEqual(m["content"][1], {"type": "text", "text": "Running ls."})
        self.assertEqual(m["content"][2]["name"], "Bash")
        self.assertEqual(m["content"][2]["input"], {"command": "ls -la"})
        self.assertEqual(m["_events"], 17)
        self.assertEqual(m["_event_types"]["ping"], 1)

    def test_mid_stream_error_kept(self):
        m = capture.reassemble_sse(sse({"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
                                       {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
        self.assertEqual(m["error"]["type"], "overloaded_error")

    def test_unparseable_tool_input_kept_raw(self):
        m = capture.reassemble_sse(sse({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "X", "input": {}}},
                                       {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a": '}},
                                       {"type": "content_block_stop", "index": 0}))
        self.assertEqual(m["content"][0]["input"], {"_unparsed": '{"a": '})


class ParseResponse(unittest.TestCase):
    def test_sse_by_content_type(self):
        self.assertEqual(capture.parse_response(STREAM.encode(), "text/event-stream; charset=utf-8")["stop_reason"], "tool_use")

    def test_plain_json(self):
        r = capture.parse_response(b'{"type":"error","error":{"type":"authentication_error"}}', "application/json")
        self.assertEqual(r["error"]["type"], "authentication_error")

    def test_garbage_sse_parses_to_nothing(self):
        r = capture.parse_response(b"\x1f\x8bnot really", "text/event-stream")
        self.assertTrue(capture._parsed_nothing(r))

    def test_garbage_json_kept_raw(self):
        self.assertIn("_raw", capture.parse_response(b"\x1f\x8bnot really", "application/json"))

    def test_parsed_nothing(self):
        self.assertTrue(capture._parsed_nothing({"content": [], "_events": 0, "_event_types": {}}))
        self.assertTrue(capture._parsed_nothing({"_raw": "x"}))
        self.assertFalse(capture._parsed_nothing(capture.reassemble_sse(STREAM)))


class Decode(unittest.TestCase):
    RAW = STREAM.encode()

    def test_gzip(self):
        out, err = capture._decode(gzip.compress(self.RAW), "gzip")
        self.assertEqual(out, self.RAW); self.assertIsNone(err)

    def test_truncated_gzip_returns_partial(self):
        out, err = capture._decode(gzip.compress(self.RAW)[:200], "gzip")
        self.assertIsNone(err)
        self.assertTrue(out.startswith(b"event: message_start"), out[:40])
        self.assertLess(len(out), len(self.RAW))

    def test_identity(self):
        self.assertEqual(capture._decode(self.RAW, ""), (self.RAW, None))
        self.assertEqual(capture._decode(self.RAW, "identity"), (self.RAW, None))

    def test_unknown(self):
        out, err = capture._decode(b"abc", "lz77")
        self.assertEqual(out, b"abc"); self.assertIn("unknown", err)

    def test_brotli_and_zstd_roundtrip(self):
        import brotli, zstandard
        self.assertEqual(capture._decode(brotli.compress(self.RAW), "br")[0], self.RAW)
        self.assertEqual(capture._decode(zstandard.ZstdCompressor().compress(self.RAW), "zstd")[0], self.RAW)


class Classify(unittest.TestCase):
    def test_main(self):
        self.assertEqual(capture.classify({"tools": [{"name": "Bash"}], "messages": [{"role": "user", "content": "hi"}]}, {}), "main")

    def test_subagent_by_header(self):
        self.assertEqual(capture.classify({"tools": [{"name": "Bash"}], "messages": []}, {"x-claude-code-agent-id": "a1"}), "subagent")

    def test_side_without_tools(self):
        self.assertEqual(capture.classify({"messages": [{"role": "user", "content": "Write a 3 word title"}]}, {}), "side")

    def test_compaction_by_phrasing(self):
        req = {"tools": [], "messages": [{"role": "user", "content": [{"type": "text", "text": "Your task is to create a detailed summary of the conversation so far"}]}]}
        self.assertEqual(capture.classify(req, {}), "compaction")

    def test_compaction_block_detection(self):
        self.assertTrue(capture._has_compaction_block({"messages": [{"role": "user", "content": [{"type": "compaction", "content": "..."}]}]}, {}))
        self.assertTrue(capture._has_compaction_block({"messages": []}, {"content": [{"type": "compaction"}]}))
        self.assertFalse(capture._has_compaction_block({"messages": []}, {"content": [{"type": "text", "text": "x"}]}))


class Disk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        capture.CAPTURE_DIR = Path(self.tmp.name)
        capture.setup()
        self.req = {"model": "m", "system": [{"type": "text", "text": "S"}], "tools": [{"name": "Bash"}], "messages": [{"role": "user", "content": "hi"}]}
        self.body = json.dumps(self.req).encode()
        self.hdrs = {"X-Claude-Code-Session-Id": "sess1", "anthropic-beta": "a, b", "Authorization": "Bearer SECRET", "x-api-key": "SECRET2"}

    def tearDown(self):
        capture.STORE_MODE = "full"
        self.tmp.cleanup()

    def test_two_phase_write_and_secrets(self):
        c = capture.begin("POST", "/v1/messages?beta=true", self.hdrs, self.body)
        pending = load(c.path)
        self.assertEqual(pending["meta"]["status"], "pending"); self.assertIsNone(pending["response"])
        self.assertEqual(pending["meta"]["session_id"], "sess1"); self.assertEqual(pending["meta"]["anthropic_beta"], ["a", "b"])
        self.assertEqual(c.path.parent.name, "sess1")
        capture.finish(c, 200, {"Content-Type": "text/event-stream", "Content-Encoding": "gzip", "request-id": "req_1"}, gzip.compress(STREAM.encode()))
        d = load(c.path)
        self.assertEqual(d["meta"]["status"], 200); self.assertEqual(d["meta"]["request_id"], "req_1")
        self.assertEqual(d["meta"]["response_encoding"], "gzip"); self.assertIsNone(d["meta"]["error"])
        self.assertEqual(d["response"]["stop_reason"], "tool_use")
        body_text = text(c.path)
        self.assertNotIn("SECRET", body_text); self.assertNotIn("Authorization", body_text); self.assertNotIn("x-api-key", body_text)
        self.assertEqual(oct(os.stat(c.path).st_mode)[-3:], "600")
        self.assertEqual(oct(os.stat(capture.CAPTURE_DIR).st_mode)[-3:], "700")

    def test_200_that_parses_to_nothing_is_an_error(self):
        c = capture.begin("POST", "/v1/messages", self.hdrs, self.body)
        capture.finish(c, 200, {"Content-Type": "text/event-stream"}, gzip.compress(STREAM.encode()))  # gzip bytes, no encoding header
        self.assertIn("parsed to nothing", load(c.path)["meta"]["error"])

    def test_aborted_partial_gzip_keeps_partial_content(self):
        c = capture.begin("POST", "/v1/messages", self.hdrs, self.body)
        capture.finish(c, 200, {"Content-Type": "text/event-stream", "Content-Encoding": "gzip"}, gzip.compress(STREAM.encode())[:300], aborted=True, error="client disconnected")
        d = load(c.path)
        self.assertTrue(d["meta"]["aborted"]); self.assertGreater(d["response"]["_events"], 0)

    def test_no_session_header(self):
        c = capture.begin("POST", "/v1/messages", {}, self.body)
        self.assertEqual(c.path.parent.name, "no-session")

    def test_dedupe_mode(self):
        capture.STORE_MODE = "dedupe"
        c1 = capture.begin("POST", "/v1/messages", self.hdrs, self.body); capture.finish(c1, 200, {}, b"{}")
        c2 = capture.begin("POST", "/v1/messages", self.hdrs, self.body); capture.finish(c2, 200, {}, b"{}")
        d = load(c1.path)
        self.assertEqual(d["request"]["system"], {"_blob": c1.blobs["system"]})
        self.assertEqual(d["request"]["tools"], {"_blob": c1.blobs["tools"]})
        self.assertEqual(d["request"]["messages"], self.req["messages"])
        self.assertEqual(c1.blobs, c2.blobs)
        self.assertEqual(len(os.listdir(c1.path.parent / "blobs")), 2)


class SpanAttributes(unittest.TestCase):
    def test_multiple_tool_results_in_one_message(self):
        a = {}
        capture._msg_attrs(a, "llm.input_messages", 0, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "one"},
            {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "two"}]},
        ]})
        self.assertEqual(a["llm.input_messages.0.message.tool_call_id"], "t1,t2")
        self.assertEqual(a["llm.input_messages.0.message.contents.0.message_content.id"], "t1")
        self.assertEqual(a["llm.input_messages.0.message.contents.1.message_content.id"], "t2")
        self.assertEqual(a["llm.input_messages.0.message.contents.1.message_content.type"], "tool_result")

    def test_tool_use_and_text(self):
        a = {}
        capture._msg_attrs(a, "llm.output_messages", 0, {"role": "assistant", "content": [
            {"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]})
        self.assertEqual(a["llm.output_messages.0.message.content"], "hi")
        self.assertEqual(a["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"], "Bash")
        self.assertEqual(json.loads(a["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]), {"command": "ls"})


if __name__ == "__main__":
    unittest.main()
