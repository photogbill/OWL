"""Heterogeneous graph: entities and observations in one structure.

Why this is worth having, given OWL already has co-occurrence edges:

`assoc_edge` connects observations that were retrieved together or that share
vocabulary. It cannot connect two notes written five weeks apart that share no
wording but concern the same person -- and that is precisely the multi-hop
case a field analyst cares about. An entity node bridges them in one hop.

MiniRAG's contribution is that this works *without* asking the model to be
clever: the graph does the reasoning, the model only reads the result. Their
benchmark shows why that matters (see docs/MINIRAG_NOTES.md) -- with a 3B
model, LightRAG scores 21.9% against naive RAG's 39.5%. Sophisticated
pipelines that lean on model comprehension go NEGATIVE at small scale.

OWL takes the structure and adds provenance to every edge.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Answer-type prediction. MiniRAG steers graph traversal by predicting what
# KIND of thing the answer is; OWL uses the same signal to sharpen the FOK
# gate. If a query asks for a person and the store holds no person entity
# anywhere near it, DONT_KNOW is a much better-founded answer.
_TYPE_CUES = (
    ("person", r"\b(who|whose|whom|which (person|man|woman)|contact|runs|"
               r"leads|manages|in charge)\b"),
    ("place", r"\b(where|which (place|town|village|site|location)|located|"
              r"route to|address)\b"),
    ("time", r"\b(when|what time|which (day|date|week|month)|how long|"
             r"deadline|schedule)\b"),
    ("quantity", r"\b(how (many|much)|what (number|amount|quantity|capacity)|"
                 r"cost|price|litres|liters|beds|percent)\b"),
    ("identifier", r"\b(serial|reference|call ?sign|tail number|registration|"
                   r"code|id|imei|plate|frequency)\b|\b[A-Z]{2,}-?\d{2,}\b"),
    ("org", r"\b(which (organisation|organization|company|agency|ngo|unit)|"
            r"who (supplies|provides|funds))\b"),
)


def predict_answer_type(query: str) -> str | None:
    low = query.lower()
    for kind, pat in _TYPE_CUES:
        if re.search(pat, low, re.IGNORECASE):
            return kind
    return None


# Does a candidate actually CONTAIN something of the type being asked for?
# MiniRAG steers graph traversal by predicted answer type; the same signal
# works on raw content, which matters because OWL does not extract entities
# itself and a host may supply none.
#
# The case this exists for: "who is in charge of the medical centre" against
#   "Dr Warsame runs the Bardera clinic and speaks Somali."   <- has a person
#   "The clinic generator runs on depot fuel."                <- does not
# A bi-encoder finds these nearly equally similar (both are about a clinic,
# both use "runs"). Only one can answer the question.
_HAS = {
    # A bare Title Case pair is NOT enough: "Route Alpha", "Km Forty",
    # "North Well" all match it. Either an explicit title, or a pair where
    # neither word is a common operational/geographic capitalised term.
    "person": re.compile(
        r"\b(?:Dr|Mr|Mrs|Ms|Prof|Capt|Lt|Sgt|Sheikh|Imam)\.?\s+[A-Z][a-z]+"
        r"|\b(?!(?:Route|Road|Km|North|South|East|West|Camp|Sector|Grid|"
        r"Alpha|Bravo|Charlie|Delta|Echo|Zone|Block|Phase|Team|Unit)\b)"
        r"[A-Z][a-z]{2,}\s+"
        r"(?!(?:Route|Road|Km|North|South|East|West|Camp|Sector|Grid|"
        r"Alpha|Bravo|Charlie|Delta|Echo|Zone|Block|Phase|Team|Unit|"
        r"Clinic|Depot|Hospital|Ridge|Valley|Well)\b)"
        r"[A-Z][a-z]{2,}\b"),
    "place": re.compile(
        r"\b(?:in|at|near|from|to)\s+[A-Z][a-z]{2,}"
        r"|\b[A-Z][a-z]{2,}\s+(?:ridge|valley|road|route|camp|town|village)\b",
        re.I),
    "time": re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)"
        r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|january|february|march|april|may|june|july|august|september"
        r"|october|november|december|today|tomorrow|yesterday|weekly|daily)\b",
        re.I),
    "quantity": re.compile(r"\b\d+(?:[.,]\d+)?\s*\w*\b"),
    "identifier": re.compile(r"\b[A-Z]{2,}-?\d{2,}\b|\b\d{3,}\b"),
    "org": re.compile(
        r"\b(?:clinic|depot|hospital|agency|ministry|office|company|"
        r"organisation|organization|NGO|UN[A-Z]*)\b", re.I),
}


def content_affinity(content: str, want: str | None) -> float:
    """1.0 if the text plausibly contains the kind of thing being asked for.

    Deliberately a mild multiplier, not a filter: the type predictor is a
    regex heuristic and must never be able to veto a genuine match.
    """
    if not want:
        return 1.0
    pat = _HAS.get(want)
    if pat is None:
        return 1.0
    return 1.25 if pat.search(content) else 0.85


def canonicalise(name: str) -> str:
    """Normalise for dedup. Deliberately conservative.

    Aggressive normalisation merges distinct people -- and in a casework
    context merging two individuals is a far worse error than carrying a
    duplicate, which the interference sweep will surface anyway.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an|dr|mr|mrs|ms|prof|sr|jr)\.?\s+", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str


@dataclass(frozen=True)
class PathStep:
    src: str
    kind: str
    dst: str
    evidence_node: str


@dataclass(frozen=True)
class Path:
    steps: tuple[PathStep, ...]

    @property
    def evidence(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.evidence_node for s in self.steps))

    def render(self) -> str:
        """Compact form. A path is denser than the notes it came from.

        ATK measured roughly 300-600 tokens for relationship paths against
        2000+ for the equivalent chunk RAG. On a 16 GB box with a slow local
        model that difference is most of the latency budget -- and it composes
        with the 4-7 chunk cap, because a path IS a dense chunk.
        """
        if not self.steps:
            return ""
        out = [self.steps[0].src]
        for s in self.steps:
            out.append(f" --[{s.kind}]--> {s.dst}")
        return "".join(out)

    def __len__(self) -> int:
        return len(self.steps)
