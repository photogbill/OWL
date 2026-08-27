"""Decontextualisation — make a memory stand on its own.

"He said it'd arrive Thursday" is useless six weeks later. A large fraction of
conversational memory is like this, and every system in the field stores the
raw utterance and hopes retrieval sorts it out. LycheeMemory expands
pronouns and context-dependent phrases at write time so each record is
self-contained; this is the deterministic Tier 0 half of that idea.

Two rules that shape everything here:

  * **Never rewrite the observation.** The raw utterance is evidence. The
    expansion is a DERIVED node -- which is also why the append-only trigger
    would reject the alternative.

  * **Refuse rather than guess.** A confidently wrong resolution is worse
    than an unresolved pronoun: the reader can see "he" is ambiguous, but
    cannot see that "Ahmed" was substituted for the wrong person. Every
    ambiguous case is left alone and reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DAY = 86400.0

# Pronoun -> the entity kinds it can legally refer to.
PRONOUNS: dict[str, tuple[str, ...]] = {
    "he": ("person",), "him": ("person",), "his": ("person",),
    "she": ("person",), "her": ("person",), "hers": ("person",),
    "they": ("person", "org"), "them": ("person", "org"),
    "their": ("person", "org"), "theirs": ("person", "org"),
    # Ordered by preference: things get delivered, organisations deliver.
    "it": ("artifact", "place", "event", "org"),
    "its": ("artifact", "place", "event", "org"),
}
POSSESSIVE = {"his", "her", "hers", "their", "theirs", "its"}

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")

_REL_DATE = re.compile(
    r"\b(today|tonight|this morning|this afternoon|this evening|"
    r"tomorrow|yesterday|last night|"
    r"next (?:week|month)|last (?:week|month)|"
    r"(?:next |last |this )?(?:" + "|".join(_WEEKDAYS) + r"))\b", re.I)

_DEICTIC = re.compile(r"\b(here|there|now|then|soon|recently|earlier|later)\b",
                      re.I)


@dataclass
class Expansion:
    text: str
    substitutions: list[tuple[str, str]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.substitutions)

    @property
    def standalone(self) -> bool:
        """Can this be read cold? Unresolved deixis means no."""
        return not self.unresolved


def _iso(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _weekday_index(ts: float) -> int:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(ts).weekday()


def resolve_date(phrase: str, at: float) -> str | None:
    """Turn a relative date into an absolute one, anchored to when it was said."""
    p = phrase.strip().lower()
    if p in ("today", "this morning", "this afternoon", "this evening",
             "tonight"):
        return _iso(at)
    if p == "tomorrow":
        return _iso(at + DAY)
    if p in ("yesterday", "last night"):
        return _iso(at - DAY)
    if p == "next week":
        return f"week of {_iso(at + 7 * DAY)}"
    if p == "last week":
        return f"week of {_iso(at - 7 * DAY)}"
    if p == "next month":
        return f"month after {_iso(at)}"
    if p == "last month":
        return f"month before {_iso(at)}"

    m = re.match(r"(next |last |this )?(" + "|".join(_WEEKDAYS) + ")", p)
    if m:
        direction, day = (m.group(1) or "").strip(), m.group(2)
        target = _WEEKDAYS.index(day)
        cur = _weekday_index(at)
        if direction == "last":
            delta = -((cur - target) % 7 or 7)
        else:
            delta = (target - cur) % 7
            if delta == 0 and direction == "next":
                delta = 7
        return _iso(at + delta * DAY)
    return None


def expand(text: str, *, at: float,
           candidates: list[tuple[str, str]] | None = None,
           speaker: str | None = None) -> Expansion:
    """Rewrite `text` so it can be read cold.

    `candidates` is [(name, kind)] in RECENCY ORDER, most recent first --
    normally the entities mentioned earlier in the same episode.
    """
    cands = candidates or []
    out = text
    subs: list[tuple[str, str]] = []
    unresolved: list[str] = []

    # ── relative dates ───────────────────────────────────────────────
    def _date_sub(m: re.Match) -> str:
        abs_date = resolve_date(m.group(0), at)
        if abs_date is None:
            unresolved.append(m.group(0))
            return m.group(0)
        subs.append((m.group(0), abs_date))
        return f"{m.group(0)} ({abs_date})"

    out = _REL_DATE.sub(_date_sub, out)

    # ── pronouns ─────────────────────────────────────────────────────
    used: set[str] = set()

    def _pron_sub(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        allowed = PRONOUNS[low]
        # An entity already substituted in this sentence is not available for
        # a different pronoun. Without this, "they will deliver it" resolved
        # both pronouns to the same org and produced "the depot will deliver
        # the depot".
        pool = [(n, k) for n, k in cands if n not in used]
        matches = [n for n, k in pool if k in allowed]
        if low in ("he", "him", "his", "she", "her", "hers") and speaker \
                and not matches and speaker not in used:
            matches = [speaker]
        if len(matches) > 1:
            # Prefer the most specific allowed kind, in the order declared
            # for this pronoun. Ties stay ambiguous and are refused.
            for kind in allowed:
                tier = [n for n, k in pool if k == kind]
                if len(tier) == 1:
                    matches = tier
                    break
                if len(tier) > 1:
                    break
        if len(matches) != 1:
            # Zero candidates, or still more than one -- refuse. A wrong
            # substitution is invisible to the reader; an unresolved
            # pronoun is not.
            unresolved.append(word)
            return word
        name = matches[0]
        used.add(name)
        repl = f"{name}'s" if low in POSSESSIVE else name
        subs.append((word, repl))
        return repl

    if cands or speaker:
        out = re.sub(r"\b(" + "|".join(PRONOUNS) + r")\b", _pron_sub, out,
                     flags=re.I)
    else:
        for m in re.finditer(r"\b(" + "|".join(PRONOUNS) + r")\b", out, re.I):
            unresolved.append(m.group(0))

    for m in _DEICTIC.finditer(text):
        unresolved.append(m.group(0))

    return Expansion(out, subs, sorted(set(unresolved), key=str.lower))


def needs_context(text: str) -> bool:
    """Cheap pre-check: is this worth trying to expand at all?"""
    return bool(
        re.search(r"\b(" + "|".join(PRONOUNS) + r")\b", text, re.I)
        or _REL_DATE.search(text)
        or _DEICTIC.search(text))
