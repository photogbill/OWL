"""Tier 2 under discipline -- the model proposes, deterministic code disposes.

Part XII of the plan says it plainly: below about 7B, model-driven memory
operations can be NEGATIVE value, and even at 24-40B a model will sometimes
invent a name, round a number, or resolve "he" to the wrong person. The
answer here is not to trust the model less. It is to give it no authority.
Every Tier 2 feature obeys the same four rules (plan Part XIV):

T1  CITE OR IT DOESN'T COUNT. Every output carries the exact text it relied
    on: a source id and a verbatim quote. `verify()` checks each quote
    character-for-character against the source (whitespace runs collapsed,
    nothing else), then checks that every number, identifier and proper name
    in the output appears in some verified quote. Output that passes is
    written as its natural kind. Output that fails is written ONLY as a
    `hypothesis` -- epistemic HYPOTHESIZED, never presentable as fact, with
    the producing model recorded (A6) and a falsifier saying what to check.
    The check is conservative on purpose: a correct rewrite that capitalises
    a word the sources never used is quarantined, which costs a little
    recall and never costs a wrong fact.

T2  SELF-DISAGREEMENT IS IGNORANCE. Ask N times (default 3) at a non-zero
    temperature. No majority means nothing is written and the run is logged
    as a disagreement. A majority is recorded with its agreement ratio,
    which scales the derived node's confidence and discounts its A7 opinion
    -- the mass that leaves belief goes to ignorance, not disbelief.

T3  COMPRESSION IS EARNED. The original is hidden; the model must rebuild it
    from a cue and its neighbours, and EVERY sample must clear the fidelity
    floor (numbers all-or-nothing). Modes: off | report | apply. `apply`
    writes the proven cue as a `summary` and demotes the original in
    ranking. The original observation is never altered -- it cannot be.

T4  RUNS IN A BUDGET. `tend(budget_seconds=)` and a per-pass call cap stop
    Tier 2 work; the report says where it stopped and how much was deferred.
    Deferred work is not recorded as attempted, so the next pass resumes it.

Two further rules that are not in the plan text but follow from the threat
model (Part II): content that the write screen QUARANTINED is never put in a
prompt -- it may be a prompt-injection payload -- and a model's verdict on a
conflict is only ever a PROPOSAL; accepting it re-runs the same supersession
authority rules a human write would face.

Every feature is OFF by default (`Tier2Config`). Explicit method calls on
`Owl` (`reason_decontext`, `propose_resolutions`, `compress`) run when a
Reasoner is attached whatever the flags say -- calling one is the opt-in.
The flags govern only what `tend()` does on its own.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from . import decontext
from .protocols import Epistemic, OwlError

log = logging.getLogger("owl.tier2")
logging.getLogger("owl").addHandler(logging.NullHandler())

COMPRESSION_MODES = ("off", "report", "apply")
VERDICTS = ("keep_old", "adopt_new", "both_hold", "undetermined")
# Outcomes that mean "this subject has been dealt with" -- anything else
# (errors, budget stops) is retried on a later pass.
FINAL_OUTCOMES = ("written", "quarantined", "disagreed", "no_answer",
                  "proposed", "proven", "failed", "skipped")

PROMPT_VERSION = "t2-2026-10-24"


# ── configuration ────────────────────────────────────────────────────────

@dataclass
class Tier2Config:
    """What `tend()` may do with a Reasoner. Every switch starts OFF."""
    decontext: bool = False          # B2, Tier 2 half
    resolutions: bool = False        # conflict-resolution PROPOSALS
    compression: str = "off"         # C1/T3: off | report | apply
    samples: int = 3                 # T2: answers per question
    temperature: float = 0.7         # non-zero, or disagreement can't show
    max_tokens: int = 384
    agreement: float = 0.6           # majority fraction; 2 of 3 clears it
    max_calls_per_pass: int = 48     # T4: hard cap per tend() pass
    context_nodes: int = 6           # decontext: episode-mates in the prompt
    min_quote_chars: int = 4         # shorter "quotes" prove nothing
    compress_min_chars: int = 160    # compressing a short note saves nothing
    compress_limit: int = 8          # compression candidates per pass

    def __post_init__(self) -> None:
        if self.compression not in COMPRESSION_MODES:
            raise ValueError(
                f"compression must be one of {COMPRESSION_MODES}, "
                f"not {self.compression!r}")
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if not 0.0 < self.agreement <= 1.0:
            raise ValueError("agreement must be in (0, 1]")

    @property
    def any_enabled(self) -> bool:
        return self.decontext or self.resolutions or self.compression != "off"

    def as_dict(self) -> dict:
        return asdict(self)


# ── budget (T4) ──────────────────────────────────────────────────────────

class Budget:
    """Wall-clock deadline plus a call cap. Checked BEFORE every call."""

    def __init__(self, *, deadline: float | None = None,
                 calls: int | None = None):
        self.deadline = deadline          # time.monotonic() value, or None
        self.calls_cap = calls
        self.calls = 0
        self.stopped: str | None = None

    @classmethod
    def from_seconds(cls, seconds: float | None, *, calls: int | None,
                     started: float | None = None) -> Budget:
        if seconds is None:
            return cls(calls=calls)
        base = time.monotonic() if started is None else started
        return cls(deadline=base + max(0.0, float(seconds)), calls=calls)

    def can_spend(self, n: int = 1) -> bool:
        if self.stopped:
            return False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stopped = "time budget reached"
        elif self.calls_cap is not None and self.calls + n > self.calls_cap:
            self.stopped = f"call cap of {self.calls_cap} reached"
        return self.stopped is None

    def spend(self) -> None:
        self.calls += 1

    def as_dict(self) -> dict:
        left = None
        if self.deadline is not None:
            left = round(max(0.0, self.deadline - time.monotonic()), 3)
        return {"calls_made": self.calls, "calls_cap": self.calls_cap,
                "seconds_left": left, "stopped": self.stopped}


# ── T1: parsing and the citation verifier ────────────────────────────────

@dataclass
class Citation:
    node: str
    quote: str


@dataclass
class Verification:
    ok: bool
    verified: list[Citation] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = self.failures[:3]
        if self.ungrounded:
            bits.append("not in any verified quote: "
                        + ", ".join(self.ungrounded[:6]))
        return "; ".join(bits) or "verified"


def parse_json(raw: str) -> tuple[dict | None, str]:
    """The first JSON object in a model's output, or (None, why not).

    Models wrap JSON in prose and code fences even under instruction. The
    first decodable object wins; nothing else is guessed at.
    """
    if not raw or not raw.strip():
        return None, "empty output"
    dec = json.JSONDecoder()
    tries = 0
    for m in re.finditer(r"\{", raw):
        tries += 1
        if tries > 20:
            break
        try:
            obj, _ = dec.raw_decode(raw[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj, ""
    return None, "no JSON object in output"


def _ws(s: str) -> str:
    return " ".join(str(s).split())


_QUOTE_STRIP = "\"'“”‘’ \t\n"

_NUM_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_CAP_RE = re.compile(r"(?<![A-Za-z0-9_\-])[A-Z][A-Za-z]+(?:['’]s)?")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Capitalised words that are NOT names. Deliberately short: a missing entry
# costs a false quarantine (safe); an extra one could let an invented name
# through (unsafe).
_NOT_NAMES = frozenset(["a", "an", "the", "he", "she", "it", "they", "we", "i", "you", "his", "her", "hers", "its", "their", "theirs", "our", "ours", "your", "yours", "my", "mine", "me", "him", "them", "us", "this", "that", "these", "those", "there", "here", "then", "now", "when", "where", "what", "which", "who", "whom", "whose", "why", "how", "if", "but", "and", "or", "nor", "so", "yet", "not", "no", "yes", "on", "in", "at", "by", "for", "from", "to", "of", "with", "without", "into", "onto", "upon", "over", "under", "after", "before", "during", "since", "until", "while", "as", "about", "above", "below", "between", "among", "across", "according", "also", "all", "any", "some", "each", "every", "both", "either", "neither", "none", "one", "two", "three", "several", "many", "much", "more", "most", "less", "least", "few", "other", "others", "another", "such", "same", "only", "just", "very", "still", "already", "again", "once", "ever", "never", "always", "often", "sometimes", "perhaps", "maybe", "however", "although", "though", "because", "therefore", "thus", "said", "says", "say", "told", "tells", "asked", "asks", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "do", "does", "did", "done", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "please", "note", "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "unknown", "unclear", "unresolved"])


def _norm_num(tok: str) -> str:
    t = tok
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "")
    if t.isdigit():
        t = t.lstrip("0") or "0"
    return t


def key_tokens(text: str) -> dict[str, set[str]]:
    """Numbers, identifiers and proper names -- what T1 requires a quote for."""
    nums = {_norm_num(m.group(0)) for m in _NUM_RE.finditer(text)}
    idents = set()
    for tok in _TOKEN_RE.findall(text):
        if (any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)):
            idents.add(tok.lower())
    names = set()
    for m in _CAP_RE.finditer(text):
        w = re.sub(r"['’]s$", "", m.group(0))
        if w.lower() not in _NOT_NAMES:
            names.add(w.lower())
    return {"numbers": nums, "identifiers": idents, "names": names}


def _allowed(quotes: Iterable[str]) -> dict[str, set[str]]:
    nums: set[str] = set()
    idents: set[str] = set()
    words: set[str] = set()
    for q in quotes:
        for m in _NUM_RE.finditer(q):
            tok = m.group(0)
            nums.add(_norm_num(tok))
            for part in re.findall(r"\d+", tok):
                nums.add(_norm_num(part))
        for tok in _TOKEN_RE.findall(q):
            idents.add(tok.lower())
        for w in _WORD_RE.findall(q):
            words.add(w.lower())
    return {"numbers": nums, "identifiers": idents, "words": words}


def verify(answer: str, citations: Any, sources: dict[str, str], *,
           min_quote_chars: int = 4, require_citation: bool = True
           ) -> Verification:
    """T1. Deterministic, no model. `sources` maps citable id -> text.

    A citation is verified when its id is one of the sources given to the
    model and its quote occurs in that source verbatim (whitespace runs
    collapsed, case kept). The answer is grounded when every number,
    identifier and proper name in it occurs in some verified quote.
    """
    v = Verification(ok=False)
    if not isinstance(answer, str) or not answer.strip():
        v.failures.append("no answer text")
        return v
    cites = citations if isinstance(citations, list) else []
    if citations is not None and not isinstance(citations, list):
        v.failures.append("citations is not a list")
    for c in cites:
        if not isinstance(c, dict):
            v.failures.append("a citation is not an object")
            continue
        node = str(c.get("node", "")).strip()
        quote = str(c.get("quote", "")).strip(_QUOTE_STRIP)
        if node not in sources:
            v.failures.append(f"cited {node or '(nothing)'!r}, which was not "
                              "a source")
            continue
        if len(_ws(quote)) < min_quote_chars:
            v.failures.append(f"quote from {node} too short to prove anything")
            continue
        if _ws(quote) not in _ws(sources[node]):
            v.failures.append(f"quote not found verbatim in {node}: "
                              f"{quote[:60]!r}")
            continue
        v.verified.append(Citation(node, quote))
    if require_citation and not v.verified:
        v.failures.append("no verified citation -- cite or it doesn't count")

    allowed = _allowed(c.quote for c in v.verified)
    need = key_tokens(answer)
    missing = sorted(n for n in need["numbers"] if n not in allowed["numbers"])
    missing += sorted(i for i in need["identifiers"]
                      if i not in allowed["identifiers"])
    missing += sorted(n for n in need["names"] if n not in allowed["words"])
    v.ungrounded = missing
    v.ok = not missing and not v.failures
    return v


# ── T2: sampling and consensus ───────────────────────────────────────────

@dataclass
class Sample:
    raw: str
    obj: dict | None
    error: str = ""
    verification: Verification | None = None
    key: Any = None

    @property
    def verified(self) -> bool:
        return bool(self.verification and self.verification.ok)


@dataclass
class Consensus:
    samples: list[Sample]
    n: int
    agreeing: int = 0
    key: Any = None
    representative: Sample | None = None
    outcome: str = "disagreed"       # agreed | disagreed | no_answer | error

    @property
    def agreement(self) -> float:
        return self.agreeing / self.n if self.n else 0.0

    def detail(self) -> dict:
        keys = Counter(repr(s.key) for s in self.samples if s.key is not None)
        return {"keys": dict(keys),
                "errors": [s.error for s in self.samples if s.error][:5],
                "failures": [s.verification.summary() for s in self.samples
                             if s.verification and not s.verification.ok][:5]}


class Unfinished(Exception):
    """Budget ran out mid-question, or the host asked to wait. Nothing is
    written, nothing is recorded as attempted; the subject is retried."""


class ReasonerBusy(RuntimeError):
    """A Reasoner may raise this -- or any exception with `owl_defer = True`,
    so a host need not import OWL -- to say "not now": the model is busy,
    or not loaded. The pass stops as if its budget ran out (plan T4's host
    contract). A generic exception is a failure; this is a deferral."""
    owl_defer = True


def _call(reasoner, prompt: str, *, grammar: str | None, max_tokens: int,
          temperature: float) -> tuple[str, str]:
    """One completion as (text, error). A host deferral raises Unfinished."""
    try:
        return reasoner.complete(prompt, grammar=grammar,
                                 max_tokens=max_tokens,
                                 temperature=temperature), ""
    except Exception as e:
        if getattr(e, "owl_defer", False):
            log.info("reasoner deferred: %s", e)
            raise Unfinished(f"reasoner deferred: {e}") from e
        log.warning("reasoner call failed: %s: %s", e.__class__.__name__, e)
        return "", f"{e.__class__.__name__}: {e}"[:200]


def sample(reasoner, prompt: str, *, n: int, temperature: float,
           max_tokens: int, grammar: str | None, budget: Budget,
           first_temperature: float | None = None) -> list[tuple[str, str]]:
    """N raw answers as (text, error). Raises Unfinished if the budget can't
    cover all N -- a partial vote is not a vote."""
    if not budget.can_spend(n):
        raise Unfinished(budget.stopped or "budget")
    out: list[tuple[str, str]] = []
    for i in range(n):
        temp = (first_temperature if (i == 0 and first_temperature is not None)
                else temperature)
        budget.spend()
        out.append(_call(reasoner, prompt, grammar=grammar,
                         max_tokens=max_tokens, temperature=temp))
    return out


def consensus(samples: list[Sample], *, threshold: float) -> Consensus:
    n = len(samples)
    c = Consensus(samples=samples, n=n)
    keyed = [s for s in samples if s.key is not None]
    if not keyed:
        c.outcome = "error" if all(s.error for s in samples) else "no_answer"
        return c
    counts = Counter(s.key for s in keyed)
    key, top = counts.most_common(1)[0]
    tied = [k for k, v in counts.items() if v == top]
    need = max(1, -(-int(threshold * n * 1000) // 1000))   # ceil, float-safe
    if n >= 2:
        need = max(need, 2)
    if len(tied) > 1 or top < need:
        c.outcome = "disagreed"
        c.agreeing = top
        return c
    group = [s for s in keyed if s.key == key]
    c.key, c.agreeing, c.outcome = key, top, "agreed"
    c.representative = next((s for s in group if s.verified), group[0])
    return c


# ── grammars and prompts ─────────────────────────────────────────────────

_GBNF_COMMON = r'''
cites  ::= "[" ws ( cite ( ws "," ws cite )* )? ws "]"
cite   ::= "{" ws "\"node\"" ws ":" ws string ws "," ws "\"quote\"" ws ":" ws string ws "}"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""
ws     ::= [ ]?
'''
GBNF_CITED = (r'root ::= "{" ws "\"answer\"" ws ":" ws string ws "," ws '
              r'"\"citations\"" ws ":" ws cites ws "}"' + _GBNF_COMMON)
GBNF_VERDICT = (r'root ::= "{" ws "\"verdict\"" ws ":" ws verdict ws "," ws '
                r'"\"rationale\"" ws ":" ws string ws "," ws '
                r'"\"citations\"" ws ":" ws cites ws "}"' + "\n"
                r'verdict ::= "\"keep_old\"" | "\"adopt_new\"" | '
                r'"\"both_hold\"" | "\"undetermined\""' + _GBNF_COMMON)
GBNF_CUE = r'root ::= "{" ws "\"cue\"" ws ":" ws string ws "}"' + _GBNF_COMMON
GBNF_TEXT = r'root ::= "{" ws "\"text\"" ws ":" ws string ws "}"' + _GBNF_COMMON


def _sources_block(sources: dict[str, str]) -> str:
    return "\n".join(f"[{k}] {_ws(v)}" for k, v in sources.items())


def prompt_decontext(target: str, sources: dict[str, str]) -> str:
    return (
        "SOURCES (each starts with its id in brackets):\n"
        f"{_sources_block(sources)}\n\n"
        f"TASK: Rewrite source [{target}] so it can be understood on its own, "
        "with no other context. Replace pronouns and relative dates with "
        "what they refer to, using ONLY the sources. Keep the original "
        "wording otherwise. If a reference cannot be resolved from the "
        "sources, leave it as it is. Do not add facts.\n"
        "For every name, number and date you use, cite the source it came "
        "from, copying the words exactly.\n"
        'Reply with JSON only: {"answer": "<rewritten text>", "citations": '
        '[{"node": "<source id>", "quote": "<exact words from that source>"}]}')


def prompt_resolution(old_id: str, new_id: str, sources: dict[str, str],
                      meta: dict[str, str]) -> str:
    return (
        "Two records in a memory store conflict.\n"
        f"[{old_id}] ({meta.get(old_id, '')}) {_ws(sources[old_id])}\n"
        f"[{new_id}] ({meta.get(new_id, '')}) {_ws(sources[new_id])}\n\n"
        "TASK: Using only these two records, choose one verdict:\n"
        f"  keep_old     - [{old_id}] still holds; [{new_id}] does not replace it\n"
        f"  adopt_new    - [{new_id}] replaces [{old_id}]\n"
        "  both_hold    - they do not really contradict (different time, "
        "place or scope)\n"
        "  undetermined - the records alone cannot settle it\n"
        "Give a one-sentence rationale and cite the exact words you relied "
        "on.\n"
        'Reply with JSON only: {"verdict": "<verdict>", "rationale": '
        '"<one sentence>", "citations": [{"node": "<record id>", '
        '"quote": "<exact words>"}]}')


def prompt_cue(original: str) -> str:
    return (
        f"RECORD:\n{_ws(original)}\n\n"
        "TASK: Write the shortest note from which this record could be "
        "rebuilt exactly. Keep every number, name, identifier and unit, "
        "exactly as written. Drop only wording that can be inferred.\n"
        'Reply with JSON only: {"cue": "<note>"}')


def prompt_reconstruct(cue: str, neighbours: list[str]) -> str:
    nb = "\n".join(f"- {_ws(x)}" for x in neighbours) or "- (none)"
    return (
        f"NOTE:\n{_ws(cue)}\n\nRELATED RECORDS:\n{nb}\n\n"
        "TASK: The note is a compressed form of one original record. Rewrite "
        "the original record in full from the note. Keep every number, "
        "name and identifier exactly.\n"
        'Reply with JSON only: {"text": "<original record>"}')


# ── shared bookkeeping ───────────────────────────────────────────────────

def _model(mind) -> str | None:
    return getattr(mind.reasoner, "name", None) or mind.model_id


def _record_run(mind, *, task: str, subject: str, samples: int,
                agreeing: int, verified: int, outcome: str,
                result_node: str | None = None, detail: dict | None = None,
                ms: float = 0.0) -> int:
    now = mind.clock.now()
    model = _model(mind)
    payload = json.dumps(detail or {}, separators=(",", ":"), default=str)

    def _w(c: sqlite3.Connection) -> int:
        return c.execute(
            "INSERT INTO reasoner_run(at,task,subject,model,prompt_version,"
            "samples,agreeing,verified,outcome,result_node,detail,ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, task, subject, model, PROMPT_VERSION, samples, agreeing,
             verified, outcome, result_node, payload, ms)).lastrowid
    rid = mind._s.write(_w)
    log.info("tier2 %s on %s: %s (agree %d/%d, verified %d, model %s)%s",
             task, subject, outcome, agreeing, samples, verified, model,
             f" -> {result_node}" if result_node else "")
    return rid


def _attempted(mind, task: str, subject: str) -> bool:
    marks = ",".join("?" * len(FINAL_OUTCOMES))
    return bool(mind._s.one(
        f"SELECT 1 FROM reasoner_run WHERE task=? AND subject=? AND outcome "
        f"IN ({marks}) LIMIT 1", (task, subject, *FINAL_OUTCOMES)))


def _quarantine(mind, *, text: str, parents: list[str], task: str,
                falsifier: str, partition: str, confidence: float) -> str:
    """T1's only exit for output that failed verification."""
    model = _model(mind) or "unknown model"
    return mind.derive(
        f"[UNVERIFIED {task} by {model}] {text}", parents=parents,
        kind="hypothesis", producer=f"tier2:{task}", producer_model=model,
        confidence=min(0.3, confidence), epistemic=Epistemic.HYPOTHESIZED,
        partition=partition, falsifier=falsifier[:500])


def _human_date(ts: float) -> str:
    d = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)
    return f"{d:%Y-%m-%d} ({d:%A} {d.day} {d:%B %Y})"


def _calendar(text: str, at: float) -> str:
    """Relative dates resolved by CODE, so the model can quote rather than
    compute. Arithmetic is the deterministic layer's job."""
    lines = [f"Said on {_human_date(at)}."]
    seen: set[str] = set()
    for m in decontext._REL_DATE.finditer(text):
        phrase = m.group(0)
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        iso = decontext.resolve_date(phrase, at)
        if not iso:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso):
            d = _dt.datetime.strptime(iso, "%Y-%m-%d").replace(
                tzinfo=_dt.timezone.utc)
            iso = _human_date(d.timestamp())
        lines.append(f'"{phrase}" here means {iso}.')
    return " ".join(lines)


# ── B2, Tier 2 half: decontextualisation ─────────────────────────────────

def decontextualise(mind, node_id: str, cfg: Tier2Config, budget: Budget
                    ) -> dict:
    """Rewrite one observation to stand alone, as a DERIVED node.

    The observation is never touched. Verified + agreed -> `decontext` node.
    Agreed but unverifiable -> quarantined hypothesis. No agreement ->
    nothing, and the disagreement is recorded.
    """
    t0 = time.perf_counter()
    row = mind._s.one(
        "SELECT id,partition,content,observed_at,episode_id,trust,"
        "rowid AS seq FROM observation WHERE id=?", (node_id,))
    if row is None:
        raise OwlError(f"decontext: {node_id!r} is not an observation")
    if row["trust"] != "trusted":
        return {"node": node_id, "outcome": "skipped",
                "reason": f"{row['trust']} content is never put in a prompt"}
    if not decontext.needs_context(row["content"]):
        return {"node": node_id, "outcome": "skipped",
                "reason": "already standalone"}

    sources: dict[str, str] = {node_id: row["content"]}
    # Context is what was written BEFORE this, in write order. Timestamps
    # tie (a batch import, a coarse clock), and a tie let later records into
    # the prompt -- the model then resolved "he" against the future.
    ctx = []
    if row["episode_id"]:
        ctx = mind._s.query(
            "SELECT id,content FROM observation WHERE episode_id=? AND "
            "rowid<? AND trust='trusted' ORDER BY rowid DESC LIMIT ?",
            (row["episode_id"], row["seq"], cfg.context_nodes))
    if not ctx:
        ctx = mind._s.query(
            "SELECT id,content FROM observation WHERE partition=? AND "
            "rowid<? AND trust='trusted' ORDER BY rowid DESC LIMIT ?",
            (row["partition"], row["seq"], min(3, cfg.context_nodes)))
    for r in reversed(list(ctx)):
        sources[r["id"]] = r["content"]
    sources["when"] = _calendar(row["content"], row["observed_at"])

    prompt = prompt_decontext(node_id, sources)
    raw = sample(mind.reasoner, prompt, n=cfg.samples,
                 temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                 grammar=GBNF_CITED, budget=budget)
    original = _ws(row["content"]).lower()
    samples = []
    for text, err in raw:
        s = Sample(raw=text, obj=None, error=err)
        if not err:
            s.obj, s.error = parse_json(text)
        if s.obj is not None:
            ans = s.obj.get("answer")
            s.verification = verify(ans, s.obj.get("citations"), sources,
                                    min_quote_chars=cfg.min_quote_chars)
            if isinstance(ans, str) and ans.strip():
                if _ws(ans).lower() == original:
                    s.key = "UNCHANGED"
                else:
                    s.key = _referent_key(ans, row["content"])
        samples.append(s)
    c = consensus(samples, threshold=cfg.agreement)
    ms = (time.perf_counter() - t0) * 1000
    verified = sum(1 for s in samples if s.verified)
    detail = c.detail()

    if c.outcome != "agreed":
        outcome = c.outcome
        rid = _record_run(mind, task="decontext", subject=node_id,
                          samples=c.n, agreeing=c.agreeing, verified=verified,
                          outcome=outcome, detail=detail, ms=ms)
        return {"node": node_id, "outcome": outcome, "run": rid,
                "agreement": round(c.agreement, 3)}
    if c.key == "UNCHANGED":
        rid = _record_run(mind, task="decontext", subject=node_id,
                          samples=c.n, agreeing=c.agreeing, verified=verified,
                          outcome="no_answer", detail=detail, ms=ms)
        return {"node": node_id, "outcome": "no_answer", "run": rid,
                "reason": "the model could not resolve anything either"}

    rep = c.representative
    assert rep is not None and rep.obj is not None
    answer = _ws(rep.obj["answer"])
    base_conf = 1.0
    if rep.verified:
        parents = [node_id] + [k for k in dict.fromkeys(
            ci.node for ci in rep.verification.verified)
            if k not in ("when", node_id)]
        did = mind.derive(
            answer, parents=parents, kind="decontext",
            producer="tier2:decontext", producer_model=_model(mind),
            confidence=base_conf * c.agreement * 0.9,
            epistemic=Epistemic.INFERRED, partition=row["partition"])
        outcome = "written"
    else:
        did = _quarantine(
            mind, text=answer, parents=[node_id], task="decontext",
            falsifier=(f"Check every name, number and date in this rewrite "
                       f"against {node_id} and its episode; it failed the "
                       f"citation check: {rep.verification.summary()}"),
            partition=row["partition"], confidence=c.agreement * 0.5)
        outcome = "quarantined"
    detail["citations"] = [asdict(ci) for ci in
                           (rep.verification.verified if rep.verification
                            else [])]
    rid = _record_run(mind, task="decontext", subject=node_id, samples=c.n,
                      agreeing=c.agreeing, verified=verified, outcome=outcome,
                      result_node=did, detail=detail, ms=ms)
    return {"node": node_id, "outcome": outcome, "derived": did,
            "text": answer, "agreement": round(c.agreement, 3), "run": rid}


def _referent_key(answer: str, original: str) -> frozenset:
    """What a rewrite RESOLVED: its names, numbers and identifiers, plus the
    content words it added. Two samples agree when they resolved the same
    references, however they phrased the rest."""
    kt = key_tokens(answer)
    before = {w.lower() for w in _WORD_RE.findall(original)}
    added = {w.lower() for w in _WORD_RE.findall(answer)
             if len(w) > 2 and w.lower() not in before
             and w.lower() not in _NOT_NAMES}
    return frozenset(kt["numbers"] | kt["identifiers"] | kt["names"] | added)


def decontext_candidates(mind, partition: str, limit: int) -> list[str]:
    rows = mind._s.query(
        "SELECT o.id,o.content FROM observation o WHERE o.partition=? AND "
        "o.trust='trusted' AND NOT EXISTS (SELECT 1 FROM derivation_edge e "
        "JOIN derived d ON d.id=e.child_id WHERE e.parent_id=o.id AND "
        "(d.kind='decontext' OR d.producer='tier2:decontext')) "
        "ORDER BY o.observed_at DESC LIMIT ?", (partition, limit * 4))
    out = []
    for r in rows:
        if len(out) >= limit:
            break
        if decontext.needs_context(r["content"]) and not _attempted(
                mind, "decontext", r["id"]):
            out.append(r["id"])
    return out


# ── conflict resolution: proposals only ──────────────────────────────────

def _conflict_sides(mind, conflict_id: str) -> tuple[str, str] | None:
    parents = [r["parent_id"] for r in mind._s.query(
        "SELECT parent_id FROM derivation_edge WHERE child_id=?",
        (conflict_id,))]
    obs = []
    for p in parents:
        r = mind._s.one("SELECT id,observed_at,rowid AS seq FROM observation "
                        "WHERE id=?", (p,))
        if r is not None:
            obs.append(r)
    if len(obs) != 2:
        return None
    # Which side is OLD matters: it decides which one a verdict would let
    # supersede the other. A blocked supersession names its target in
    # supersession_attempt, which is authoritative. Otherwise write order:
    # observed_at, then insertion order -- never the id, which is random.
    ids = [r["id"] for r in obs]
    named = mind._s.one(
        "SELECT old_node FROM supersession_attempt WHERE old_node IN (?,?) "
        "AND allowed=0 ORDER BY id DESC LIMIT 1", tuple(ids))
    if named is not None:
        old = named["old_node"]
        return old, next(i for i in ids if i != old)
    obs.sort(key=lambda r: (r["observed_at"], r["seq"]))
    return obs[0]["id"], obs[1]["id"]


def resolution_candidates(mind, partition: str, limit: int) -> list[str]:
    rows = mind._s.query(
        "SELECT d.id FROM derived d WHERE d.kind='conflict' AND "
        "d.partition=? AND NOT EXISTS (SELECT 1 FROM resolution_proposal p "
        "WHERE p.conflict_id=d.id) ORDER BY d.created_at LIMIT ?",
        (partition, limit * 4))
    return [r["id"] for r in rows
            if not _attempted(mind, "resolution", r["id"])][:limit]


def propose_resolution(mind, conflict_id: str, cfg: Tier2Config,
                       budget: Budget) -> dict:
    """Ask which side of a recorded conflict the records support.

    The answer is a PROPOSAL row and nothing else. Accepting it is a
    separate step that re-checks supersession authority.
    """
    t0 = time.perf_counter()
    sides = _conflict_sides(mind, conflict_id)
    if sides is None:
        rid = _record_run(mind, task="resolution", subject=conflict_id,
                          samples=0, agreeing=0, verified=0,
                          outcome="skipped",
                          detail={"reason": "conflict is not between two "
                                            "observations"})
        return {"conflict": conflict_id, "outcome": "skipped", "run": rid}
    old_id, new_id = sides
    rows = {r["id"]: r for r in mind._s.query(
        "SELECT id,content,trust,reliability,credibility,observed_at,"
        "partition FROM observation WHERE id IN (?,?)", (old_id, new_id))}
    if any(rows[i]["trust"] == "quarantined" for i in sides):
        rid = _record_run(mind, task="resolution", subject=conflict_id,
                          samples=0, agreeing=0, verified=0,
                          outcome="skipped",
                          detail={"reason": "one side was quarantined by the "
                                            "write screen; it is never put "
                                            "in a prompt"})
        return {"conflict": conflict_id, "outcome": "skipped", "run": rid,
                "reason": "quarantined content"}
    sources = {i: rows[i]["content"] for i in sides}
    meta = {}
    for i in sides:
        rel, cred = mind.effective_grade(i)
        meta[i] = (f"recorded {_human_date(rows[i]['observed_at'])}, source "
                   f"grade {rel}{cred}, {rows[i]['trust']}")
    prompt = prompt_resolution(old_id, new_id, sources, meta)
    raw = sample(mind.reasoner, prompt, n=cfg.samples,
                 temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                 grammar=GBNF_VERDICT, budget=budget)
    samples = []
    for text, err in raw:
        s = Sample(raw=text, obj=None, error=err)
        if not err:
            s.obj, s.error = parse_json(text)
        if s.obj is not None:
            verdict = str(s.obj.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                s.key = verdict
                s.verification = verify(
                    s.obj.get("rationale"), s.obj.get("citations"), sources,
                    min_quote_chars=cfg.min_quote_chars)
            else:
                s.error = f"verdict {verdict[:30]!r} is not one of {VERDICTS}"
        samples.append(s)
    c = consensus(samples, threshold=cfg.agreement)
    ms = (time.perf_counter() - t0) * 1000
    verified = sum(1 for s in samples if s.verified)
    detail = c.detail()
    partition = rows[new_id]["partition"]
    if c.outcome != "agreed":
        rid = _record_run(mind, task="resolution", subject=conflict_id,
                          samples=c.n, agreeing=c.agreeing, verified=verified,
                          outcome=c.outcome, detail=detail, ms=ms)
        return {"conflict": conflict_id, "outcome": c.outcome, "run": rid,
                "agreement": round(c.agreement, 3)}
    rep = c.representative
    assert rep is not None and rep.obj is not None
    rationale = _ws(rep.obj.get("rationale", ""))
    if not rep.verified:
        did = _quarantine(
            mind, text=f"{c.key}: {rationale}", parents=[old_id, new_id],
            task="resolution",
            falsifier=(f"Compare {old_id} with {new_id} directly; the "
                       f"proposal failed the citation check: "
                       f"{rep.verification.summary() if rep.verification else 'unparsed'}"),
            partition=partition, confidence=c.agreement * 0.5)
        rid = _record_run(mind, task="resolution", subject=conflict_id,
                          samples=c.n, agreeing=c.agreeing, verified=verified,
                          outcome="quarantined", result_node=did,
                          detail=detail, ms=ms)
        return {"conflict": conflict_id, "outcome": "quarantined",
                "derived": did, "run": rid}
    cites = [asdict(ci) for ci in rep.verification.verified]
    detail["citations"] = cites
    rid = _record_run(mind, task="resolution", subject=conflict_id,
                      samples=c.n, agreeing=c.agreeing, verified=verified,
                      outcome="proposed", detail=detail, ms=ms)
    now = mind.clock.now()
    pid = mind._s.write(lambda cn: cn.execute(
        "INSERT INTO resolution_proposal(conflict_id,partition,old_node,"
        "new_node,verdict,rationale,citations,agreement,model,run_id,"
        "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (conflict_id, partition, old_id, new_id, c.key, rationale,
         json.dumps(cites), c.agreement, _model(mind), rid, now)).lastrowid)
    log.info("tier2 resolution proposal #%d for %s: %s (agreement %.2f)",
             pid, conflict_id, c.key, c.agreement)
    return {"conflict": conflict_id, "outcome": "proposed", "proposal": pid,
            "verdict": c.key, "rationale": rationale,
            "agreement": round(c.agreement, 3), "run": rid}


def accept_resolution(mind, proposal_id: int, *, by: str) -> dict:
    """Act on a proposal -- under exactly the rules a human write faces.

    `adopt_new` becomes a supersession only if `defence.may_supersede` and
    the coup check both pass for the CURRENT grades. Otherwise the proposal
    is marked refused with the reason, and the beliefs do not move. The
    other verdicts record the decision and change no belief: the older
    record already stands.
    """
    from . import defence
    p = mind._s.one("SELECT * FROM resolution_proposal WHERE id=?",
                    (proposal_id,))
    if p is None:
        raise OwlError(f"no resolution proposal #{proposal_id}")
    if p["status"] != "pending":
        raise OwlError(f"proposal #{proposal_id} is already {p['status']}")
    now = mind.clock.now()
    old_id, new_id = p["old_node"], p["new_node"]
    if p["verdict"] != "adopt_new":
        _set_status(mind, proposal_id, "accepted", by,
                    f"{p['verdict']}: no belief changed", now)
        return {"proposal": proposal_id, "status": "accepted",
                "verdict": p["verdict"], "superseded": False}

    new = mind._s.one("SELECT trust,source_ref FROM observation WHERE id=?",
                      (new_id,))
    old = mind._s.one("SELECT observed_at,claim_class FROM observation "
                      "WHERE id=?", (old_id,))
    if new is None or old is None:
        raise OwlError(f"proposal #{proposal_id} refers to a missing node")
    ok, why = defence.may_supersede(mind.effective_grade(new_id),
                                    mind.effective_grade(old_id),
                                    trust=new["trust"])
    if ok:
        recent = mind._scalar(
            "SELECT COUNT(*) FROM supersession_attempt WHERE source_ref=? "
            "AND at > ? AND allowed=1",
            (new["source_ref"], now - defence.COUP_WINDOW))
        coup, coup_why = defence.is_coup(recent)
        if coup:
            ok, why = False, coup_why
    reason = f"tier2 proposal #{proposal_id} accepted by {by}"

    def _w(c: sqlite3.Connection) -> None:
        c.execute("INSERT INTO supersession_attempt(at,source_ref,old_node,"
                  "allowed,reason) VALUES(?,?,?,?,?)",
                  (now, new["source_ref"], old_id, 1 if ok else 0,
                   reason if ok else f"{reason}; refused: {why}"))
        if ok:
            c.execute("INSERT OR REPLACE INTO supersession(old_node,new_node,"
                      "claim_class,survived,at) VALUES(?,?,?,?,?)",
                      (old_id, new_id, old["claim_class"],
                       now - old["observed_at"], now))
            c.execute("UPDATE mem_index SET tier='cold' WHERE node_id=?",
                      (old_id,))
    mind._s.write(_w)
    if not ok:
        _set_status(mind, proposal_id, "refused", by, why, now)
        log.info("tier2 proposal #%d refused at acceptance: %s",
                 proposal_id, why)
        return {"proposal": proposal_id, "status": "refused", "reason": why,
                "superseded": False,
                "remedy": "a source with sufficient authority can supersede "
                          "directly with observe(..., supersedes=)"}
    _set_status(mind, proposal_id, "accepted", by, "superseded", now)
    mind.raise_impacts(old_id, cause="superseded")
    log.info("tier2 proposal #%d accepted by %s: %s supersedes %s",
             proposal_id, by, new_id, old_id)
    return {"proposal": proposal_id, "status": "accepted",
            "verdict": "adopt_new", "superseded": True, "old": old_id,
            "new": new_id}


def _set_status(mind, pid: int, status: str, by: str, reason: str,
                now: float) -> None:
    mind._s.write(lambda c: c.execute(
        "UPDATE resolution_proposal SET status=?,decided_by=?,"
        "status_reason=?,decided_at=? WHERE id=?",
        (status, by, reason, now, pid)))


# ── C1 / T3: compression, earned ─────────────────────────────────────────

def compression_candidates(mind, partition: str, cfg: Tier2Config,
                           limit: int) -> list[str]:
    rows = mind._s.query(
        "SELECT o.id FROM observation o JOIN mem_index m ON m.node_id=o.id "
        "WHERE o.partition=? AND m.tier IN ('cold','warm') AND "
        "o.claim_class<>'verbatim' AND o.trust='trusted' AND "
        "length(o.content)>=? AND NOT EXISTS (SELECT 1 FROM compression k "
        "WHERE k.node_id=o.id) ORDER BY m.last_review LIMIT ?",
        (partition, cfg.compress_min_chars, limit * 4))
    return [r["id"] for r in rows
            if not _attempted(mind, "compression", r["id"])][:limit]


def prove_compression(mind, node_id: str, cfg: Tier2Config,
                      budget: Budget) -> dict:
    """The fill-in-the-blank test: hide the record, rebuild it, every time.

    One call proposes a cue (T1-grounded in the original: it may not
    introduce a single number or name the record does not contain). Then
    `samples` independent rebuilds from cue + neighbours, the original
    hidden; ALL must clear the fidelity floor. A host Reasoner exposing
    `reconstruct(cue, neighbours)` -- an infilling model such as
    DiffusionGemma -- is used for the rebuild in place of a prompt.
    """
    from . import reconstructive as rc
    t0 = time.perf_counter()
    row = mind._s.one("SELECT id,content,claim_class,partition FROM "
                      "observation WHERE id=?", (node_id,))
    if row is None:
        raise OwlError(f"compression: {node_id!r} is not an observation")
    original = row["content"]
    if row["claim_class"] == "verbatim":
        return {"node": node_id, "outcome": "skipped",
                "reason": "verbatim content is never compressed"}
    if not budget.can_spend(1 + cfg.samples):
        raise Unfinished(budget.stopped or "budget")
    budget.spend()
    raw_cue, err = _call(mind.reasoner, prompt_cue(original), grammar=GBNF_CUE,
                         max_tokens=cfg.max_tokens, temperature=0.2)
    obj, perr = parse_json(raw_cue) if not err else (None, err)
    cue = _ws(obj.get("cue", "")) if obj else ""
    reason = ""
    if not cue:
        reason = f"no usable cue ({perr or 'empty'})"
    else:
        need = key_tokens(cue)
        allowed = _allowed([original])
        extra = sorted([n for n in need["numbers"]
                        if n not in allowed["numbers"]]
                       + [i for i in need["identifiers"]
                          if i not in allowed["identifiers"]]
                       + [n for n in need["names"]
                          if n not in allowed["words"]])
        if extra:
            reason = "cue introduces content the record lacks: " + ", ".join(
                extra[:6])
        elif len(cue) > 0.7 * len(_ws(original)):
            reason = (f"cue is {len(cue)} chars against {len(original)}; "
                      "not worth compressing")
    if reason:
        if err and not obj:
            rid = _record_run(mind, task="compression", subject=node_id,
                              samples=1, agreeing=0, verified=0,
                              outcome="error", detail={"reason": reason})
            return {"node": node_id, "outcome": "error", "reason": reason,
                    "run": rid}
        _store_compression(mind, node_id, row["partition"], cue, [], "failed",
                           reason)
        rid = _record_run(mind, task="compression", subject=node_id,
                          samples=1, agreeing=0, verified=0,
                          outcome="failed", detail={"reason": reason},
                          ms=(time.perf_counter() - t0) * 1000)
        return {"node": node_id, "outcome": "failed", "reason": reason,
                "run": rid}

    nb = [r["content"] for r in mind._s.query(
        "SELECT o2.content FROM assoc_edge a JOIN observation o2 "
        "ON o2.id=a.dst WHERE a.src=? AND o2.trust='trusted' LIMIT 5",
        (node_id,))]
    recon = getattr(mind.reasoner, "reconstruct", None)
    scores: list[float] = []
    for _ in range(cfg.samples):
        budget.spend()
        if callable(recon):
            try:
                text = recon(cue, nb) or ""
            except Exception as e:
                if getattr(e, "owl_defer", False):
                    raise Unfinished(f"reasoner deferred: {e}") from e
                log.warning("reconstruction failed for %s: %s", node_id, e)
                text = ""
        else:
            raw, _err = _call(
                mind.reasoner, prompt_reconstruct(cue, nb), grammar=GBNF_TEXT,
                max_tokens=max(cfg.max_tokens, len(original) // 2),
                temperature=cfg.temperature)
            o2, _e = parse_json(raw)
            text = str(o2.get("text", "")) if o2 else ""
        scores.append(rc.fidelity(original, text))
    floor = rc.FIDELITY_FLOOR
    passed = all(s >= floor for s in scores)
    status = "proven" if passed else "failed"
    why = (f"all {len(scores)} rebuilds cleared {floor:.2f} "
           f"(min {min(scores):.2f})" if passed else
           f"rebuild fidelity {', '.join(f'{s:.2f}' for s in scores)} "
           f"against a floor of {floor:.2f} -- kept verbatim")
    _store_compression(mind, node_id, row["partition"], cue, scores, status,
                       why)
    rid = _record_run(mind, task="compression", subject=node_id,
                      samples=len(scores),
                      agreeing=sum(1 for s in scores if s >= floor),
                      verified=1, outcome=status,
                      detail={"fidelity": scores, "cue_chars": len(cue),
                              "original_chars": len(original)},
                      ms=(time.perf_counter() - t0) * 1000)
    return {"node": node_id, "outcome": status, "cue": cue,
            "fidelity": scores, "reason": why, "run": rid}


def _store_compression(mind, node_id: str, partition: str, cue: str,
                       scores: list[float], status: str, reason: str) -> None:
    now = mind.clock.now()
    mind._s.write(lambda c: c.execute(
        "INSERT OR REPLACE INTO compression(node_id,partition,cue,fidelity,"
        "min_fidelity,status,reason,model,proved_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (node_id, partition, cue, json.dumps(scores),
         min(scores) if scores else 0.0, status, reason, _model(mind), now)))


def apply_compression(mind, node_id: str) -> dict:
    """Spend a proof: the cue becomes a `summary`, the original is demoted.

    The original observation is untouched and remains the evidence; `why()`
    on the summary leads back to it. Dimensional integrity is checked again
    by `derive()` -- a second, independent guard on numbers and units.
    """
    k = mind._s.one("SELECT * FROM compression WHERE node_id=?", (node_id,))
    if k is None or k["status"] != "proven":
        raise OwlError(f"{node_id} has no proven compression to apply")
    try:
        sid = mind.derive(k["cue"], parents=[node_id], kind="summary",
                          producer="tier2:compression",
                          producer_model=k["model"],
                          confidence=float(k["min_fidelity"]),
                          epistemic=Epistemic.INFERRED,
                          partition=k["partition"])
    except OwlError as e:
        why = f"refused at apply: {e}"
        mind._s.write(lambda c: c.execute(
            "UPDATE compression SET status='failed',reason=? WHERE node_id=?",
            (why, node_id)))
        log.warning("compression of %s %s", node_id, why)
        return {"node": node_id, "applied": False, "reason": str(e)}
    now = mind.clock.now()
    mind._s.write(lambda c: c.execute(
        "UPDATE compression SET status='applied',summary_node=?,applied_at=? "
        "WHERE node_id=?", (sid, now, node_id)))
    # The summary REPLACES the original in ranking; it is not a second copy.
    # Leaving both live put near-duplicates in every result set and pushed
    # OTHER memories out of the budget -- the ablation bench caught recall
    # falling for records that were never compressed. Suppression is ranking
    # demotion, reversible and logged; the original stays addressable.
    mind.suppress(node_id, reason=f"compressed into {sid} (tier2 proof, "
                                  f"min fidelity {float(k['min_fidelity']):.2f})")
    log.info("compression applied: %s -> summary %s (original kept, demoted)",
             node_id, sid)
    return {"node": node_id, "applied": True, "summary": sid}


# ── the tend() hook ──────────────────────────────────────────────────────

def run_passes(mind, *, partition: str, budget: Budget) -> dict:
    """Every Tier 2 pass tend() may run, each reported as ran or skipped."""
    cfg: Tier2Config = mind.tier2
    out: dict[str, Any] = {"config": {k: v for k, v in cfg.as_dict().items()
                                      if k in ("decontext", "resolutions",
                                               "compression", "samples")},
                           "model": _model(mind), "skipped": [],
                           "deferred": 0}
    passes = (("decontext", cfg.decontext),
              ("resolutions", cfg.resolutions),
              ("compression", cfg.compression != "off"))
    if mind.reasoner is None:
        for name, on in passes:
            out["skipped"].append(
                f"tier2.{name} ({'no Reasoner configured' if on else 'off'})")
        return out
    if mind._s.readonly:
        out["skipped"].append("tier2 (store opened read-only)")
        return out
    failures = 0

    def _guard(fn: Callable, subject: str) -> dict | None:
        nonlocal failures
        try:
            res = fn(subject)
        except Unfinished as e:
            out["deferred"] += 1
            # A host deferral ("the model is busy") stops every later pass
            # too -- asking again a millisecond later would be the same no.
            budget.stopped = budget.stopped or str(e)
            return None
        except OwlError as e:
            log.warning("tier2 %s: %s", subject, e)
            return {"outcome": "error", "reason": str(e)}
        if res.get("outcome") == "error":
            failures += 1
        else:
            failures = 0
        return res

    for name, on in passes:
        if not on:
            out["skipped"].append(f"tier2.{name} (off)")
            continue
        if failures >= 2:
            out["skipped"].append(f"tier2.{name} (reasoner unavailable: "
                                  "consecutive errors)")
            continue
        t0 = time.perf_counter()
        stats: Counter = Counter()
        if name == "decontext":
            todo = decontext_candidates(mind, partition, 40)
            fn = lambda s: decontextualise(mind, s, cfg, budget)
        elif name == "resolutions":
            todo = resolution_candidates(mind, partition, 20)
            fn = lambda s: propose_resolution(mind, s, cfg, budget)
        else:
            todo = compression_candidates(mind, partition, cfg,
                                          cfg.compress_limit)
            fn = lambda s: prove_compression(mind, s, cfg, budget) 
        stats["candidates"] = len(todo)
        for i, subject in enumerate(todo):
            if failures >= 2:
                stats["abandoned"] = len(todo) - i
                break
            if budget.stopped:
                out["deferred"] += len(todo) - i
                break
            res = _guard(fn, subject)
            if res is None:
                out["deferred"] += len(todo) - i - 1
                break
            stats[res.get("outcome", "?")] += 1
        if name == "compression" and cfg.compression == "apply":
            for r in mind._s.query(
                    "SELECT node_id FROM compression WHERE partition=? AND "
                    "status='proven' LIMIT ?", (partition, 50)):
                res = apply_compression(mind, r["node_id"])
                stats["applied" if res["applied"] else "apply_refused"] += 1
        stats_d = dict(stats)
        stats_d["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out[name] = stats_d
        log.info("tier2 pass %s on %s: %s", name, partition, stats_d)
    out["budget"] = budget.as_dict()
    if budget.stopped and out["deferred"]:
        out["skipped"].append(f"tier2: {out['deferred']} item(s) deferred "
                              f"({budget.stopped}); the next pass resumes them")
    return out


# ── reading back ─────────────────────────────────────────────────────────

def agreement_for(mind, node_id: str) -> float | None:
    """The T2 agreement behind a Tier 2 node, or None for anything else."""
    try:
        r = mind._s.one("SELECT samples,agreeing FROM reasoner_run WHERE "
                        "result_node=? ORDER BY id DESC LIMIT 1", (node_id,))
    except sqlite3.OperationalError:      # a pre-Tier-2 store opened read-only
        return None
    if r is None or not r["samples"]:
        return None
    return r["agreeing"] / r["samples"]


def run_stats(mind) -> dict:
    try:
        rows = mind._s.query("SELECT task,outcome,COUNT(*) AS n FROM "
                             "reasoner_run GROUP BY task,outcome")
    except sqlite3.OperationalError:
        return {"available": False}
    out: dict[str, Any] = {"available": True, "by_task": {}}
    for r in rows:
        out["by_task"].setdefault(r["task"], {})[r["outcome"]] = r["n"]
    return out


def _log_crash(exc: BaseException) -> None:
    log.error("tier2 pass crashed; Tier 0 maintenance completed",
              exc_info=(type(exc), exc, exc.__traceback__))
