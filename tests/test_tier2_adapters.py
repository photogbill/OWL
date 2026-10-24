"""Reasoner adapters. The HTTP one is tested against a real local server --
a stub of the OpenAI chat-completions shape on 127.0.0.1 -- so what is
exercised is the actual request on the wire, not a mock of urllib."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from owl.adapters.openai_http import OpenAICompatReasoner, ReasonerUnavailable


class Stub:
    def __init__(self):
        self.requests = []
        self.replies = []          # list of (status, body-dict)

    def __enter__(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                stub.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(n) or b"{}")})
                status, body = (stub.replies.pop(0) if stub.replies else
                                (200, {"choices": [{"message": {
                                    "content": '{"answer": "ok"}'}}],
                                    "usage": {"prompt_tokens": 5,
                                              "completion_tokens": 3}}))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


def test_request_shape_and_reply():
    with Stub() as s:
        r = OpenAICompatReasoner(s.url, "qwen-24b", api_key="sekret")
        out = r.complete("hello", grammar="root ::= x", max_tokens=99,
                         temperature=0.3)
    assert out == '{"answer": "ok"}'
    req = s.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sekret"
    b = req["body"]
    assert b["model"] == "qwen-24b" and b["max_tokens"] == 99
    assert b["temperature"] == 0.3 and b["messages"][-1]["content"] == "hello"
    assert b["response_format"] == {"type": "json_object"}
    assert "grammar" not in b
    assert r.name == f"qwen-24b@{s.url.split('//')[1].split('/')[0]}"
    assert "sekret" not in json.dumps(r.describe())


def test_grammar_mode_sends_the_grammar_and_none_sends_neither():
    with Stub() as s:
        OpenAICompatReasoner(s.url, "m", constrain="grammar").complete(
            "p", grammar="root ::= x")
        OpenAICompatReasoner(s.url, "m", constrain="none").complete(
            "p", grammar="root ::= x")
        OpenAICompatReasoner(s.url, "m").complete("p")      # no grammar asked
    g, n, plain = (q["body"] for q in s.requests)
    assert g["grammar"] == "root ::= x" and "response_format" not in g
    assert "grammar" not in n and "response_format" not in n
    assert "response_format" not in plain


def test_a_client_error_is_not_retried():
    with Stub() as s:
        s.replies = [(400, {"error": "bad field"})] * 3
        r = OpenAICompatReasoner(s.url, "m", retries=2)
        with pytest.raises(ReasonerUnavailable, match="HTTP 400"):
            r.complete("p")
    assert len(s.requests) == 1


def test_a_server_error_is_retried_then_succeeds():
    with Stub() as s:
        s.replies = [(503, {"error": "loading"})]
        r = OpenAICompatReasoner(s.url, "m", retries=1)
        assert r.complete("p") == '{"answer": "ok"}'
    assert len(s.requests) == 2 and r.stats["errors"] == 1


def test_unreachable_server_raises_a_sentence():
    r = OpenAICompatReasoner("http://127.0.0.1:9/v1", "m", retries=0,
                             timeout=2)
    with pytest.raises(ReasonerUnavailable, match="unreachable"):
        r.complete("p")


def test_a_reply_without_choices_is_unusable():
    with Stub() as s:
        s.replies = [(200, {"oops": True})]
        with pytest.raises(ReasonerUnavailable, match="no choices"):
            OpenAICompatReasoner(s.url, "m", retries=0).complete("p")


def test_bad_configuration_is_refused():
    with pytest.raises(ValueError):
        OpenAICompatReasoner("ftp://x", "m")
    with pytest.raises(ValueError):
        OpenAICompatReasoner("http://x", "m", constrain="regex")


def test_drives_a_full_tier2_question_over_http(mind):
    """End to end: the engine's prompt goes over the wire and the verified
    answer comes back as a decontext node."""
    from owl import Tier2Config
    a = mind.observe("Ahmed from Tanaka Logistics called about the pump gasket.")
    b = mind.observe("He said the pump gasket was delayed.")
    reply = json.dumps({"answer": "Ahmed said the pump gasket was delayed.",
                        "citations": [
                            {"node": a, "quote": "Ahmed from Tanaka Logistics"},
                            {"node": b, "quote": "said the pump gasket was "
                                                 "delayed"}]})
    with Stub() as s:
        s.replies = [(200, {"choices": [{"message": {"content": reply}}]})] * 3
        mind.reasoner = OpenAICompatReasoner(s.url, "local-24b")
        mind.tier2 = Tier2Config()
        res = mind.reason_decontext(b)
    assert res["outcome"] == "written"
    assert mind._s.one("SELECT producer_model FROM derived WHERE id=?",
                       (res["derived"],))["producer_model"].startswith(
        "local-24b@127.0.0.1:")


def test_llama_adapter_imports_only_when_present():
    pytest.importorskip("llama_cpp")
    from owl.adapters.llama_reasoner import LlamaCppReasoner
    with pytest.raises(FileNotFoundError):
        LlamaCppReasoner("/nonexistent/model.gguf")
