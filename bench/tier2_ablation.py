"""Tier 2 ablation -- what each discipline buys, measured against ground truth.

Plan Part XIV says the model proposes and deterministic code disposes. This
measures whether that is true, arm by arm, on scenarios where the right
answer is known:

  DECONTEXT  "Ahmed ... / He said ..." pairs with a known referent.
    raw      what an undisciplined system stores: the first answer, as fact
    T1       the citation verifier, one sample
    T1+T2    verifier plus 3-sample self-agreement (the default)
  COMPRESSION  long records with numbers, compressed and rebuilt.
    ungated  what a "summarise old memories" pass stores: the first rebuild
    T3       the engine's gate: every one of 3 rebuilds must clear the floor

The headline number per arm is WRONG FACTS: model output stored as
presentable fact whose referent or figures are not the truth. Quarantined
hypotheses are not facts and are counted separately.

    python bench/tier2_ablation.py                 scripted model profiles
    python bench/tier2_ablation.py --json
    python bench/tier2_ablation.py --endpoint http://127.0.0.1:8080/v1 \\
        --model qwen2.5-14b-instruct               a real model, same scenarios

The scripted profiles are DEMONSTRATIONS of each failure mode, not
measurements of any real model. Publishing a Tier 2 claim needs the
--endpoint run on the model you actually ship, with these numbers beside it.
One limit is shown on purpose: a model that consistently picks the WRONG
person who is genuinely in the sources passes T1 (the quote is real) and
passes T2 (it agrees with itself). Only a better model, or a human, catches
that -- the `consistent-wrong` profile exists so nobody forgets it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from owl import Owl, Tier2Config
from owl import tier2 as t2
from owl.adapters.scripted import ScriptedReasoner

PEOPLE = ["Ahmed", "Bilal", "Chen", "Dawit", "Eba", "Fikre", "Hodan", "Imran",
          "Jamila", "Kofi", "Layla", "Musa"]
INVENTED = ["Karim", "Omar", "Tariq", "Zainab"]
ITEMS = ["pump gasket", "generator belt", "water filter", "fuel pump",
         "radio battery", "tyre set"]


class Clock:
    def __init__(self, t=1_760_000_000.0):
        self.t = t

    def now(self):
        return self.t


def _answer(text, cites):
    return json.dumps({"answer": text, "citations": [
        {"node": n, "quote": q} for n, q in cites]})


def _parse_case(prompt):
    src = {m.group(2).strip(): m.group(1) for m in re.finditer(
        r"^\[([a-z_0-9]+)\] (.*)$", prompt, re.MULTILINE)}
    # Sources run target first, then context oldest -> newest: the case's
    # own context line is the LAST "called about" line before "when".
    ctxs = [(c, i) for c, i in src.items() if " called about the " in c]
    ctx = ctxs[-1] if ctxs else None
    tgt = next(((c, i) for c, i in src.items() if c.startswith("He said")),
               None)
    return ctx, tgt


def profile(name, rng, p=0.3):
    """A scripted model that fails in one specific way, `p` of the time."""
    def fn(prompt, temperature):
        ctx, tgt = _parse_case(prompt)
        if ctx is None or tgt is None:
            return _answer("He said it.", [])
        person = ctx[0].split()[0]
        other = re.search(r"with (\w+)", ctx[0]).group(1)
        item = re.search(r"about the (.+?)\.", ctx[0]).group(1)
        good_q = [(ctx[1], ctx[0].rstrip(".")), (tgt[1], tgt[0].rstrip("."))]
        roll = rng.random()
        if name == "faithful" or roll >= p:
            return _answer(f"{person} said the {item} was delayed.", good_q)
        if name == "confabulating":
            return _answer(f"{rng.choice(INVENTED)} said the {item} was "
                           "delayed.", [(tgt[1], tgt[0].rstrip("."))])
        if name == "misquoting":
            return _answer(f"{person} said the {item} was delayed.",
                           [(ctx[1], f"{person} rang about the {item}")])
        if name == "unstable":
            return _answer(f"{rng.choice([person, other] + INVENTED)} said "
                           f"the {item} was delayed.", good_q)
        if name == "consistent-wrong":
            return _answer(f"{other} said the {item} was delayed.", good_q)
        raise ValueError(name)
    return fn


PROFILES = ("faithful", "confabulating", "misquoting", "unstable")
# consistent-wrong is run separately: T1/T2 cannot catch it, by design.
LIMIT_PROFILE = "consistent-wrong"


def _build_decontext(mind, n, rng):
    truth = {}
    for i in range(n):
        a, b = rng.sample(PEOPLE, 2)
        item = ITEMS[i % len(ITEMS)]
        mind.observe(f"{a} called about the {item}, with {b} on the line.")
        tgt = mind.observe(f"He said the {item} was delayed (case {i}).")
        truth[tgt] = a
    return truth


def _referent(text):
    m = re.match(r"\[UNVERIFIED[^\]]*\]\s*(\w+)|(\w+)", text)
    return (m.group(1) or m.group(2)) if m else ""


def run_decontext(reasoner_factory, *, n=24, seed=7, arms=None):
    arms = arms or {"T1": 1, "T1+T2": 3}
    out = {}
    # raw: first answer stored as fact, no checks
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        mind = Owl.open(os.path.join(d, "raw.owl"), clock=Clock())
        truth = _build_decontext(mind, n, rng)
        r = reasoner_factory()
        wrong = correct = 0
        for tgt, who in truth.items():
            row = mind._s.one("SELECT content,observed_at,episode_id FROM "
                              "observation WHERE id=?", (tgt,))
            ctx = mind._s.query("SELECT id,content FROM observation WHERE "
                                "rowid < (SELECT rowid FROM observation WHERE "
                                "id=?) ORDER BY rowid DESC LIMIT 1", (tgt,))
            sources = {c["id"]: c["content"] for c in ctx}
            sources[tgt] = row["content"]
            raw = r.complete(t2.prompt_decontext(tgt, sources),
                             grammar=t2.GBNF_CITED, temperature=0.7)
            obj, _ = t2.parse_json(raw)
            said = _referent((obj or {}).get("answer", "") or "")
            correct += said == who
            wrong += said != who
        mind.close()
    out["raw"] = {"facts": n, "wrong_facts": wrong, "correct_facts": correct,
                  "quarantined": 0, "nothing_written": 0, "calls": n}
    for arm, samples in arms.items():
        rng = random.Random(seed)
        with tempfile.TemporaryDirectory() as d:
            mind = Owl.open(os.path.join(d, f"{arm}.owl"), clock=Clock(),
                            reasoner=reasoner_factory(),
                            tier2=Tier2Config(samples=samples))
            truth = _build_decontext(mind, n, rng)
            res = {"facts": 0, "wrong_facts": 0, "correct_facts": 0,
                   "quarantined": 0, "nothing_written": 0, "calls": 0}
            t0 = time.perf_counter()
            for tgt, who in truth.items():
                rr = mind.reason_decontext(tgt)
                if rr["outcome"] == "written":
                    res["facts"] += 1
                    ok = _referent(rr["text"]) == who
                    res["correct_facts" if ok else "wrong_facts"] += 1
                elif rr["outcome"] == "quarantined":
                    res["quarantined"] += 1
                else:
                    res["nothing_written"] += 1
            res["calls"] = sum(x["samples"] for x in mind.reasoner_runs(
                limit=10_000))
            res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            mind.close()
        out[arm] = res
    return out


# ── compression ──────────────────────────────────────────────────────────

def _record(i, rng):
    trucks, tonnes, litres = rng.randint(2, 9), rng.randint(5, 40), \
        rng.randint(5, 60) * 100
    return (f"Convoy {i} left the depot at 0{rng.randint(5, 9)}:{rng.randint(10, 59)} "
            f"with {trucks} trucks carrying {tonnes} tonnes of maize meal and "
            f"{litres} litres of diesel, escorted by two vehicles, expected at "
            "the camp by the afternoon.")


def lossy_rebuilder(rng, p_drop=0.25):
    """Rebuilds faithfully, except `p_drop` of the time it loses a figure."""
    def fn(prompt, temperature):
        if prompt.startswith("RECORD:"):
            rec = prompt.split("\n", 2)[1]
            return json.dumps({"cue": rec.replace(" escorted by two vehicles,",
                                                  "").replace(
                " expected at the camp by the afternoon", "")})
        cue = re.search(r"NOTE:\n(.*)\n", prompt).group(1)
        full = cue.rstrip(".") + (", escorted by two vehicles, expected at "
                                  "the camp by the afternoon.")
        if rng.random() < p_drop:
            full = re.sub(r"\d+ tonnes", "several tonnes", full)
        return json.dumps({"text": full})
    return fn


def run_compression(*, n=16, seed=11, p_drop=0.25):
    out = {}
    rng = random.Random(seed)
    records = [_record(i, rng) for i in range(n)]
    # ungated: accept the first rebuild, whatever it lost
    rng2 = random.Random(seed + 1)
    fn = lossy_rebuilder(rng2, p_drop)
    lost = 0
    for rec in records:
        cue = json.loads(fn(t2.prompt_cue(rec), 0.2))["cue"]
        text = json.loads(fn(t2.prompt_reconstruct(cue, []), 0.7))["text"]
        lost += any(tok not in text for tok in re.findall(r"\d+", rec))
    out["ungated"] = {"compressed": n, "lost_figures": lost}
    with tempfile.TemporaryDirectory() as d:
        mind = Owl.open(os.path.join(d, "c.owl"), clock=Clock(),
                        reasoner=ScriptedReasoner(lossy_rebuilder(
                            random.Random(seed + 1), p_drop)),
                        tier2=Tier2Config(compression="apply"))
        ids = []
        for rec in records:
            nid = mind.observe(rec)
            ids.append(nid)
        mind._s.write(lambda c: c.execute("UPDATE mem_index SET tier='cold'"))

        def recall_hits():
            found = 0
            for rec in records:
                q = re.search(r"(\d+) tonnes", rec).group(0) + " maize meal"
                hit = mind.recall(q, budget=5)
                found += any(re.search(r"\b" + re.escape(q.split()[0])
                                       + r" tonnes", c.content)
                             for c in hit.chunks)
            return found
        before = recall_hits()
        # Recall reinforces what it touches (the testing effect), which
        # lifts memories out of the cold tier compression draws from.
        mind._s.write(lambda c: c.execute("UPDATE mem_index SET tier='cold'"))
        res = mind.compress(mode="apply", limit=n)
        applied = [a for a in res["applied"] if a.get("applied")]
        lost_g = 0
        for a in applied:
            s = mind._s.one("SELECT content FROM derived WHERE id=?",
                            (a["summary"],))["content"]
            rec = mind._s.one("SELECT content FROM observation WHERE id=?",
                              (a["node"],))["content"]
            lost_g += any(tok not in s for tok in re.findall(r"\d+", rec))
        # XI.2's sleep ablation in miniature: a figure-specific query, before
        # and after the proofs are spent. Apply must not lose recall.
        after = recall_hits()
        out["T3"] = {"compressed": len(applied), "kept_verbatim":
                     n - len(applied), "lost_figures": lost_g,
                     "recall_before": f"{before}/{n}",
                     "recall_after": f"{after}/{n}"}
        mind.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL (…/v1)")
    ap.add_argument("--model", default="local")
    ap.add_argument("--constrain", default="json_object",
                    choices=("grammar", "json_object", "none"))
    a = ap.parse_args(argv)
    report = {"decontext": {}, "compression": {}}
    if a.endpoint:
        from owl.adapters.openai_http import OpenAICompatReasoner
        report["decontext"][a.model] = run_decontext(
            lambda: OpenAICompatReasoner(a.endpoint, a.model,
                                         constrain=a.constrain), n=a.n)
    else:
        for name in PROFILES + (LIMIT_PROFILE,):
            rng = random.Random(name)
            report["decontext"][name] = run_decontext(
                lambda name=name, rng=rng: ScriptedReasoner(profile(name, rng)),
                n=a.n)
        report["compression"] = run_compression()
    if a.json:
        print(json.dumps(report, indent=2))
        return report
    print("DECONTEXT -- wrong facts stored (lower is better); "
          "quarantined/nothing are not facts")
    print(f"{'model':18} {'arm':7} {'facts':>6} {'WRONG':>6} {'right':>6} "
          f"{'quar.':>6} {'none':>5} {'calls':>6}")
    for model, arms in report["decontext"].items():
        for arm, r in arms.items():
            print(f"{model:18} {arm:7} {r['facts']:6} {r['wrong_facts']:6} "
                  f"{r['correct_facts']:6} {r['quarantined']:6} "
                  f"{r['nothing_written']:5} {r['calls']:6}")
    if report["compression"]:
        print("\nCOMPRESSION -- records whose stored form lost a figure")
        for arm, r in report["compression"].items():
            print(f"  {arm:8} " + ", ".join(f"{k}={v}" for k, v in r.items()))
    if not a.endpoint:
        print("\nScripted profiles demonstrate failure modes; they measure no "
              "real model.\n'consistent-wrong' is the limit: a real quote and "
              "a self-consistent wrong answer pass T1 and T2.")
    return report


if __name__ == "__main__":
    main()
