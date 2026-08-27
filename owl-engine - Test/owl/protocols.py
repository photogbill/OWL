"""Protocols and value types. Nothing here imports numpy, torch, or a model."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable


class Origin(str, Enum):
    """Where a piece of content came from. Primary sources only."""
    USER = "user_utterance"
    DOCUMENT = "document"
    TOOL = "tool_output"


class Epistemic(str, Enum):
    """Monotone lattice. A node is at least as speculative as its parents."""
    OBSERVED = "observed"
    REPORTED = "reported"
    INFERRED = "inferred"
    HYPOTHESIZED = "hypothesized"

    @property
    def rank(self) -> int:
        return _EPI_RANK[self]


_EPI_RANK = {
    Epistemic.OBSERVED: 0,
    Epistemic.REPORTED: 1,
    Epistemic.INFERRED: 2,
    Epistemic.HYPOTHESIZED: 3,
}


class State(str, Enum):
    """Feeling-of-Knowing triage. Check this BEFORE consuming chunks.

    Six states, not four. The last two are things every other memory system is
    structurally incapable of saying, because they delete rows:

      FAMILIAR            "I have seen this before but cannot place it" --
                           dual-process recognition: familiarity is fast and
                           context-free; recollection retrieves the details
                           AND their context. Conflating them loses an honest
                           answer that is neither KNOW_WHERE nor DONT_KNOW.
      KNEW_ONCE           "you told me, I no longer hold the detail, here is
                           the source" -- which is NOT the same as never told,
                           and is a completely different instruction to the user.
      SEARCHED_AND_ABSENT "I looked on the 14th and it was not there" -- absence
                           established at real cost and worth storing, instead
                           of re-derived every time it is asked.
    """
    KNOW = "know"
    KNOW_WHERE = "know_where"
    FAMILIAR = "familiar"
    TIP_OF_TONGUE = "tip_of_tongue"
    KNEW_ONCE = "knew_once"
    SEARCHED_AND_ABSENT = "searched_and_absent"
    DONT_KNOW = "dont_know"


class Space(str, Enum):
    """Two embedding spaces. Write separates; read completes."""
    WRITE = "write"   # pattern separation: push near-duplicates apart
    READ = "read"     # pattern completion: retrieve a neighbourhood


@dataclass(frozen=True)
class Provenance:
    origin: str
    source_ref: str
    epistemic: Epistemic
    observed_at: float
    valid_from: float | None = None
    valid_to: float | None = None
    derivation: tuple[str, ...] = ()

    def is_primary(self) -> bool:
        return not self.derivation


@dataclass(frozen=True)
class Chunk:
    node_id: str
    content: str
    provenance: Provenance
    score: float
    retrievability: float          # can I still FIND this?
    staleness: float = 0.0         # should I still BELIEVE it? -- different curve
    claim_class: str = "unknown"
    affect: float = 0.0
    reliability: str = "F"
    credibility: int = 6
    trust: str = "trusted"

    @property
    def trustworthy(self) -> bool:
        return (self.staleness < 0.5
                and self.provenance.epistemic.rank <= 1
                and self.trust == "trusted")

    @property
    def presentable_as_fact(self) -> bool:
        """Quarantined content is retrievable but never authoritative."""
        return self.trust == "trusted" and self.provenance.epistemic.rank <= 1


@dataclass
class Recall:
    state: State
    chunks: list[Chunk] = field(default_factory=list)
    query: str = ""
    reason: str = ""
    latency_ms: float = 0.0
    tokens: int = 0            # approximate cost of what was returned

    def __bool__(self) -> bool:
        """True when there is content to consume."""
        return self.state in (State.KNOW, State.KNOW_WHERE, State.FAMILIAR,
                              State.TIP_OF_TONGUE)

    @property
    def informative(self) -> bool:
        """True when the answer is USEFUL even without content.

        'I looked and it is not there' and 'you told me and I lost the detail'
        both carry real information. Only DONT_KNOW is genuinely empty.
        """
        return self.state is not State.DONT_KNOW

    def texts(self) -> list[str]:
        return [c.content for c in self.chunks]


# ── Protocols the host supplies ──────────────────────────────────────────

@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    __slots__ = ()

    def now(self) -> float:
        return time.time()


@runtime_checkable
class Embedder(Protocol):
    dim: int
    # Cosine below which a match is noise. This is a property of the ENCODER,
    # not a global constant: BGE-M3 puts unrelated short texts around
    # 0.45-0.55, while a bag-of-concepts projection puts them near 0. Hard-
    # coding one number makes an engine that works with one model and
    # silently fails with another.
    noise_floor: float
    # Where to stop LOOKING. Deliberately well below `noise_floor`: a strict
    # noise floor used as a search cutoff starves the candidate set, and a
    # gate that needs a distribution then has none to work with.
    search_floor: float

    def embed(self, texts: Sequence[str], space: Space) -> list[list[float]]: ...


@runtime_checkable
class Reasoner(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


class OwlError(RuntimeError):
    pass


class MonotonicityError(OwlError):
    """A derived node claimed more certainty than its parents allow."""


class PartitionError(OwlError):
    """An information-flow violation. Never caught internally; always fatal."""
