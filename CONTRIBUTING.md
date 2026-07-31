# Contributing to O.W.L.

## The one hard rule

**The correctness suite runs fast, with no GPU, no model, and no network.**
A memory system whose tests don't get run is one that corrupts itself quietly
for six months. If a change would require hardware or the network, it belongs
behind an optional extra and a marker.

```bat
run_tests.bat -m "not slow"     :: engine correctness, ~6 s
run_tests.bat                   :: everything including tooling smoke tests
```

*This rule originally said "under one second", set when the suite was 24
tests. It is now 127 covering far more, and the honest number is a few
seconds. Recorded rather than quietly dropped — and note that the biggest
single win came from not gaming it: executing ~500 lines of schema DDL per
test cost 22 ms x 116 = 2.6 s of pure setup, so `conftest` now builds the
schema once and copies the file. That is a real fix; raising the threshold
would not have been.*

## Layering

Three layers, and mixing them is the mistake to watch for:

| Layer | Table | Mutability |
|---|---|---|
| **Substrate** | `observation` | append-only, enforced by a SQL trigger |
| **Index** | `mem_index` | freely mutable — all forgetting lives here |
| **Derivation** | `derived` | rewritten via `supersedes`, never in place |

If you're adding a field, ask which layer it belongs to. Suppression
(`"stop bringing this up"`) is a *forgetting* operation, so it lives in the
index — it was originally added to `observation` and the trigger rejected it
within the hour. That's the layering check working.

## Invariants that must not regress

1. `confidence(node) <= min(confidence(parents))`
2. `epistemic(node) >= max(epistemic(parents))`
3. Nothing calls `time.time()` — the clock is injected
4. Nothing writes to SQLite except `SqliteStore.write()`
4b. OWL never makes a network call. Fetching models is a separate opt-in
   command, deliberately outside the library's runtime path.
5. Sealed partitions never export and never flow outward
6. `kind='hypothesis'` requires a falsifier and forces `epistemic='hypothesized'`
7. Quarantined content never corroborates, fuses, supersedes, or presents as fact
8. A source assessment is revisable; the ingest-time grade on an observation is not
9. Discrediting demotes and never deletes — having believed something is itself evidence

**The layering check has now caught three real errors during development**
(`suppressed_at`, then `reliability`, both wrongly placed on the append-only
`observation` table). If a write is rejected by the immutability trigger, the
answer is almost never to relax the trigger — it is that the field belongs in
the index or assessment layer.

Each has a test. If you break one, the fix is the invariant, not the test.

## What's most wanted

- **`Reasoner` adapter** — llama.cpp + GBNF, or OpenAI-compatible. Unlocks Tier 2
- **ANN index** behind `VectorIndex` — the one place that knows about similarity
- **Benchmarks** — LoCoMo, LongMemEval
- **Postgres `Store`** — the protocol is small; SQLite-specific SQL is in one file
- **Adversarial tests** — if you can make OWL assert something it was never
  told, that's the single most valuable issue you can file

## Style

Type hints everywhere. Comments explain *why*, not *what* — particularly when
a choice looks wrong at first glance (the read and write embedding spaces
being different, for instance, or the segmenter deliberately not clearing its
window at a boundary).

New behaviour needs a test that fails before your change.

## Development environment

Windows is the primary development target.

```bat
install.bat        :: venv + editable install + verify
run_tests.bat      :: extra pytest args pass through: run_tests.bat -k defence
shell.bat          :: a prompt with the venv active
clean.bat          :: start over
```

On Linux/macOS: `pip install -e ".[dev]" && pytest tests -q`.

Two Windows-specific rules for the `.bat` scripts, both covered by tests in
`tests/test_install.py`:

* **CRLF line endings.** A batch file with bare LF endings fails in confusing
  ways. `.gitattributes` enforces this; do not "fix" it.
* **Handle double-clicking.** A `.bat` launched from Explorer closes the instant
  it finishes, so an error message is visible for roughly a tenth of a second.
  Every script calls `_common.bat` with its own path and pauses when it detects
  it was double-clicked. Note that `%~f0` inside `_common.bat` resolves to
  `_common.bat`, not the caller — the caller's path must be passed in, or the
  check silently never fires.
