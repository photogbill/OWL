"""A Reasoner for any OpenAI-compatible server -- stdlib only.

llama.cpp's `llama-server`, vLLM, LM Studio, Ollama's /v1 endpoint, and the
hosted APIs all speak `POST {base_url}/chat/completions`. This adapter uses
`urllib` and nothing else, so Tier 0's zero-dependency install still gives a
working Tier 2 against a local server.

Constraining the output. Small models produce far more parseable JSON when
the server forces the shape. Three ways, chosen by `constrain`:

  * "grammar"     -- send the GBNF grammar the engine passes (llama.cpp's
                     server honours a `grammar` field; most others reject
                     unknown fields, so this is opt-in);
  * "json_object" -- send `response_format: {"type": "json_object"}`, which
                     llama-server, vLLM and the hosted APIs accept;
  * "none"        -- send neither and rely on the prompt.

Whatever the server does, the engine re-parses and re-verifies every answer.
The constraint improves the hit rate; it is never what makes output safe.

Air-gapped by default: proxies from the environment are IGNORED unless
`use_env_proxy=True`, because an analyst workstation pointed at
`http://127.0.0.1:8080` must not have its prompts -- which contain memory
content -- routed through whatever HTTPS_PROXY the shell happened to set.
The API key, if any, is never logged.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("owl.adapters.openai_http")

CONSTRAIN_MODES = ("grammar", "json_object", "none")


class ReasonerUnavailable(RuntimeError):
    """The server could not be reached or returned something unusable."""


class OpenAICompatReasoner:
    """`complete()` over HTTP. `name` (model@host) is recorded as A6 provenance."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 timeout: float = 120.0, constrain: str = "json_object",
                 system: str | None = None, retries: int = 1,
                 use_env_proxy: bool = False, extra_body: dict | None = None):
        if constrain not in CONSTRAIN_MODES:
            raise ValueError(f"constrain must be one of {CONSTRAIN_MODES}")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be http(s): {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.constrain = constrain
        self.system = system
        self.retries = max(0, int(retries))
        self.extra_body = dict(extra_body or {})
        self._api_key = api_key
        host = urllib.parse.urlparse(self.base_url).netloc or self.base_url
        self.name = f"{model}@{host}"
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self.stats = {"requests": 0, "errors": 0, "ms_total": 0.0}
        log.info("OpenAI-compatible reasoner: %s (constrain=%s, timeout=%.0fs,"
                 " env proxy %s)", self.name, constrain, self.timeout,
                 "on" if use_env_proxy else "off")

    def _body(self, prompt: str, grammar: str | None, max_tokens: int,
              temperature: float) -> dict:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": msgs,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature), "stream": False}
        if self.constrain == "grammar" and grammar:
            body["grammar"] = grammar
        elif self.constrain == "json_object" and grammar:
            # The engine only passes a grammar when it wants JSON back.
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        return body

    def complete(self, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(self._body(prompt, grammar, max_tokens,
                                     temperature)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            self.stats["requests"] += 1
            try:
                req = urllib.request.Request(url, data=data, headers=headers,
                                             method="POST")
                with self._opener.open(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                ms = (time.perf_counter() - t0) * 1000
                self.stats["ms_total"] += ms
                text = _extract_text(payload)
                usage = payload.get("usage") or {}
                log.info("reasoner %s: %d prompt chars -> %d chars in %.0f ms"
                         " (tokens in=%s out=%s, temp=%.2f)", self.name,
                         len(prompt), len(text), ms,
                         usage.get("prompt_tokens", "?"),
                         usage.get("completion_tokens", "?"), temperature)
                return text
            except urllib.error.HTTPError as e:
                self.stats["errors"] += 1
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception as read_err:                  # noqa: BLE001
                    log.debug("could not read HTTP error body: %s", read_err)
                last = ReasonerUnavailable(
                    f"{self.name} returned HTTP {e.code}: {detail}")
                log.warning("reasoner %s HTTP %s (attempt %d/%d): %s",
                            self.name, e.code, attempt + 1, self.retries + 1,
                            detail)
                if 400 <= e.code < 500 and e.code != 429:
                    break          # the request itself is wrong; retrying won't help
            except (urllib.error.URLError, TimeoutError, OSError,
                    ValueError) as e:
                self.stats["errors"] += 1
                last = ReasonerUnavailable(
                    f"{self.name} unreachable: {e.__class__.__name__}: {e}")
                log.warning("reasoner %s unreachable (attempt %d/%d): %s",
                            self.name, attempt + 1, self.retries + 1, e)
            if attempt < self.retries:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        assert last is not None
        raise last

    def describe(self) -> dict:
        """For doctor(): configuration only -- never makes a network call."""
        return {"adapter": "openai_http", "name": self.name,
                "base_url": self.base_url, "model": self.model,
                "constrain": self.constrain, "timeout": self.timeout,
                "api_key": "set" if self._api_key else "none",
                **self.stats}


def _extract_text(payload: dict) -> str:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ReasonerUnavailable(
            f"response has no choices: {str(payload)[:200]}") from e
    msg = choice.get("message") or {}
    text = msg.get("content")
    if text is None:
        text = choice.get("text")           # legacy /completions shape
    if text is None:
        raise ReasonerUnavailable(f"choice has no text: {str(choice)[:200]}")
    return str(text)
