-- O.W.L. — Observation & Wisdom Ledger
-- Three layers: SUBSTRATE (immutable) / INDEX (decays) / DERIVED (rewritten).
-- Plus: information-flow partitions, bitemporal validity, event segmentation.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

-- ─────────────────────────────────────────────────────────────
-- PARTITIONS: information-flow lattice. Default is NO flow.
-- A sealed partition (no outflow) can never be read from elsewhere.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS partition (
    name        TEXT PRIMARY KEY,
    sealed      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS partition_flow (
    src   TEXT NOT NULL REFERENCES partition(name) ON DELETE CASCADE,
    dst   TEXT NOT NULL REFERENCES partition(name) ON DELETE CASCADE,
    -- 'full'    : raw observations cross the boundary
    -- 'summary' : only derived/abstracted content crosses (graded permeability)
    level TEXT NOT NULL DEFAULT 'full' CHECK (level IN ('full','summary')),
    PRIMARY KEY (src, dst)
);

-- ─────────────────────────────────────────────────────────────
-- LAYER 1 — SUBSTRATE. Append-only. Immutability enforced by trigger.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS observation (
    id           TEXT PRIMARY KEY,
    partition    TEXT NOT NULL REFERENCES partition(name),
    observed_at  REAL NOT NULL,          -- SYSTEM time: when we learned it
    valid_from   REAL,                   -- WORLD time: when the fact became true
    valid_to     REAL,                   -- WORLD time: when it stopped being true
    origin       TEXT NOT NULL CHECK (origin IN
                   ('user_utterance','document','tool_output')),
    source_ref   TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    context_env  TEXT NOT NULL DEFAULT '{}',
    episode_id   TEXT,
    period_id    TEXT,
    affect       REAL NOT NULL DEFAULT 0.0,  -- distress marker, 0..1 (companion mode)
    -- Epistemic half-life: retrievability and CREDIBILITY decay differently.
    -- "Route Alpha is open" stays perfectly retrievable and becomes worthless.
    claim_class  TEXT NOT NULL DEFAULT 'unknown'
                   CHECK (claim_class IN ('identity','capacity','status',
                                          'position','verbatim','unknown')),
    -- 'verbatim' is a PROTECTION, not a category. Some content is worthless
    -- unless exact: a grid reference, a serial, a dosage, a frequency, an
    -- account number. It must never be summarised, paraphrased, fused or
    -- compressed -- and a system that gets this wrong is dangerous rather
    -- than merely unhelpful. (The idea is MIRIX's Knowledge Vault; the
    -- enforcement is OWL's.)
    -- Admiralty scale (STANAG 2511): source reliability A-F, info credibility 1-6.
    -- Two axes, because a reliable source can report something implausible.
    reliability  TEXT NOT NULL DEFAULT 'F' CHECK (reliability IN
                   ('A','B','C','D','E','F')),
    -- Trust tier. A memory system is a persistence layer for BELIEFS:
    -- anything that writes to it is writing to the agent's mind, permanently.
    -- Prompt injection is transient; memory poisoning is not.
    -- Quarantined content is retrievable but never corroborates, never fuses,
    -- never supersedes, and never presents as fact.
    trust        TEXT NOT NULL DEFAULT 'trusted' CHECK (trust IN
                   ('trusted','untrusted','quarantined')),
    -- Which model formed this, if any. Given that graph-RAG pipelines invert
    -- below ~7B, knowing which beliefs a weak model formed is a SAFETY
    -- property -- and it is impossible to backfill.
    producer_model TEXT,
    -- What this cost to obtain. Everyone models memory as STORAGE; it is an
    -- investment. Some facts cost a three-day trip or a canvass of six
    -- vendors; others cost a glance at a filename. The right question for
    -- forgetting is not "how often was this used" but "what would it cost me
    -- to get it back".
    acquisition_cost REAL NOT NULL DEFAULT 0.0,
    credibility  INTEGER NOT NULL DEFAULT 6 CHECK (credibility BETWEEN 1 AND 6)
);

CREATE TRIGGER IF NOT EXISTS observation_is_immutable
BEFORE UPDATE ON observation
BEGIN
    SELECT RAISE(ABORT, 'observation is append-only: use supersede()');
END;

CREATE TRIGGER IF NOT EXISTS observation_no_delete
BEFORE DELETE ON observation
WHEN (SELECT COUNT(*) FROM redaction WHERE observation_id = OLD.id) = 0
BEGIN
    SELECT RAISE(ABORT, 'observation delete requires an explicit redaction record');
END;

-- Explicit user-commanded erasure. Deletion propagates via derivation edges.
CREATE TABLE IF NOT EXISTS redaction (
    observation_id TEXT PRIMARY KEY,
    redacted_at    REAL NOT NULL,
    reason         TEXT NOT NULL
);

-- ─────────────────────────────────────────────────────────────
-- LAYER 2 — INDEX. This is where forgetting lives. Freely mutable.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mem_index (
    node_id      TEXT PRIMARY KEY,
    partition    TEXT NOT NULL REFERENCES partition(name),
    stability    REAL NOT NULL,      -- FSRS S: days until R falls to 0.9 (storage strength)
    difficulty   REAL NOT NULL,      -- FSRS D: 1..10
    last_review  REAL NOT NULL,
    review_count INTEGER NOT NULL DEFAULT 0,
    access_log   TEXT NOT NULL DEFAULT '[]',   -- every access ts -> spacing effect
    surprise     REAL NOT NULL DEFAULT 0.5,    -- prediction error at encode
    open_loop    INTEGER NOT NULL DEFAULT 0,   -- Zeigarnik bonus
    -- "stop bringing this up" is a different request from "delete this".
    -- It is a forgetting operation, so it lives in the INDEX, not the record.
    -- (The append-only trigger caught this: these columns were originally on
    -- `observation`, which is exactly the layering mistake the trigger exists
    -- to prevent.)
    suppressed_at   REAL,
    suppress_reason TEXT,
    tier         TEXT NOT NULL DEFAULT 'hot'
                   CHECK (tier IN ('hot','warm','cold','pruned'))
);
CREATE INDEX IF NOT EXISTS idx_index_tier ON mem_index(tier, partition);

-- ─────────────────────────────────────────────────────────────
-- LAYER 3 — DERIVED. Rewritten freely, always traceable.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS derived (
    id            TEXT PRIMARY KEY,
    partition     TEXT NOT NULL REFERENCES partition(name),
    created_at    REAL NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN
                    ('summary','abstraction','graft','hypothesis','conflict',
                     'correction','reflection','community','decontext')),
    epistemic_tag TEXT NOT NULL CHECK (epistemic_tag IN
                    ('observed','reported','inferred','hypothesized')),
    producer      TEXT NOT NULL,
    producer_model TEXT,
    -- What this cost to obtain. Everyone models memory as STORAGE; it is an
    -- investment. Some facts cost a three-day trip or a canvass of six
    -- vendors; others cost a glance at a filename. The right question for
    -- forgetting is not "how often was this used" but "what would it cost me
    -- to get it back".
    acquisition_cost REAL NOT NULL DEFAULT 0.0,
    content       TEXT NOT NULL,
    confidence    REAL NOT NULL,
    supersedes    TEXT REFERENCES derived(id),
    locked_at     REAL,
    falsifier     TEXT,
    CHECK (kind <> 'hypothesis' OR falsifier IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS derivation_edge (
    child_id  TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    role      TEXT NOT NULL CHECK (role IN
                ('evidence','contradicts','graft','context')),
    PRIMARY KEY (child_id, parent_id, role)
);
CREATE INDEX IF NOT EXISTS idx_edge_parent ON derivation_edge(parent_id);

-- ─────────────────────────────────────────────────────────────
-- EPISODES — event segmentation at surprise boundaries (EM-LLM style)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episode (
    id         TEXT PRIMARY KEY,
    partition  TEXT NOT NULL REFERENCES partition(name),
    period_id  TEXT,
    started_at REAL NOT NULL,
    ended_at   REAL,
    boundary_surprise REAL NOT NULL DEFAULT 0.0,
    label      TEXT
);

-- Self-Memory System hierarchy: period -> episode -> observation
CREATE TABLE IF NOT EXISTS period (
    id         TEXT PRIMARY KEY,
    partition  TEXT NOT NULL REFERENCES partition(name),
    label      TEXT NOT NULL,
    opened_at  REAL NOT NULL,
    closed_at  REAL,
    summary_id TEXT REFERENCES derived(id)
);

-- ─────────────────────────────────────────────────────────────
-- ASSOCIATIVE GRAPH — for Personalized-PageRank retrieval
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS assoc_edge (
    src    TEXT NOT NULL,
    dst    TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    kind   TEXT NOT NULL DEFAULT 'cooccur',
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_assoc_src ON assoc_edge(src);

-- Successor representation: what tends to be needed next
CREATE TABLE IF NOT EXISTS succession (
    src   TEXT NOT NULL,
    dst   TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (src, dst)
);

-- ─────────────────────────────────────────────────────────────
-- PROSPECTIVE MEMORY — intentions, not the past
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS intention (
    id           TEXT PRIMARY KEY,
    partition    TEXT NOT NULL REFERENCES partition(name),
    created_at   REAL NOT NULL,
    trigger_kind TEXT NOT NULL CHECK (trigger_kind IN ('event','time')),
    trigger_spec TEXT NOT NULL,
    action       TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','fired','completed','expired')),
    origin_ref   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intent_status ON intention(status, trigger_kind);

-- Inverted index for Tier-0 (no-embedder) lexical recall + FOK density
CREATE TABLE IF NOT EXISTS lexeme (
    term    TEXT NOT NULL,
    node_id TEXT NOT NULL,
    tf      REAL NOT NULL,
    -- G5: denormalised from mem_index. A posting list is per-PARTITION or
    -- it is not a shard: with the partition only on mem_index, SQLite must
    -- visit every posting for a term and join each one before it can
    -- discard the ones this query may not see -- the filter running after
    -- the scan it exists to prevent. Safe to duplicate because a node's
    -- partition is immutable (nothing in OWL moves a memory between
    -- partitions, deliberately), and checked anyway by shards.verify().
    partition TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (term, node_id)
);
CREATE INDEX IF NOT EXISTS idx_lexeme_term ON lexeme(term);
-- The (partition, term, node_id, tf) covering index is created by
-- shards.migrate(), NOT here. It cannot live in this file: on a store
-- written before G5 the CREATE TABLE above is a no-op, the column does not
-- exist yet, and an index naming it would fail the whole script before the
-- ALTER TABLE that adds it has had a chance to run.

-- Calibration ledger: stated confidence vs outcome (Brier)
CREATE TABLE IF NOT EXISTS calibration (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    producer   TEXT NOT NULL,
    claim_kind TEXT NOT NULL,
    confidence REAL NOT NULL,
    outcome    INTEGER,
    recorded_at REAL NOT NULL
);

-- ─────────────────────────────────────────────────────────────
-- THEORY OF MIND — the EPISTEMIC plane only.
-- Transactive memory (Wegner): each party models what the other knows and
-- who is responsible for remembering what. Every memory system in the field
-- models what the MACHINE knows; none models what the PERSON knows.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS exposure (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    who       TEXT NOT NULL,          -- which person, not which model
    node_id   TEXT NOT NULL,
    at        REAL NOT NULL,
    channel   TEXT NOT NULL,          -- briefing | conversation | document | recall
    depth     REAL NOT NULL DEFAULT 1.0   -- levels-of-processing weight
);
CREATE INDEX IF NOT EXISTS idx_exposure_who ON exposure(who, node_id);

-- What the person is believed to hold, derived from exposures. Distinct from
-- what is TRUE (the ledger) -- the gap between them is a false-belief state.
CREATE TABLE IF NOT EXISTS belief_divergence (
    id          TEXT PRIMARY KEY,
    who         TEXT NOT NULL,
    held_node   TEXT NOT NULL,        -- what they were last exposed to
    truth_node  TEXT NOT NULL,        -- what the ledger now holds
    detected_at REAL NOT NULL,
    severity    REAL NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('user_stale','ledger_stale')),
    resolved_at REAL
);

-- Supersession events are the training signal for per-class half-life.
CREATE TABLE IF NOT EXISTS supersession (
    old_node    TEXT NOT NULL,
    new_node    TEXT NOT NULL,
    claim_class TEXT NOT NULL,
    survived    REAL NOT NULL,        -- seconds the old claim held
    at          REAL NOT NULL,
    PRIMARY KEY (old_node, new_node)
);

-- Negative memory: absence is expensive to establish and free to store.
CREATE TABLE IF NOT EXISTS absence (
    id        TEXT PRIMARY KEY,
    partition TEXT NOT NULL REFERENCES partition(name),
    query     TEXT NOT NULL,
    scope     TEXT NOT NULL,
    searched_at REAL NOT NULL,
    reason    TEXT NOT NULL DEFAULT 'searched, not found',
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_absence_part ON absence(partition);

-- ─────────────────────────────────────────────────────────────
-- VECTORS — two spaces, deliberately.
--
-- Standard RAG embeds everything in ONE space, where similar items collapse
-- together -- which CAUSES the interference problem it then works around.
-- Complementary Learning Systems says the write path should do the opposite
-- of the read path:
--
--   WRITE (pattern separation): embed content PLUS its distinguishing
--     context, so two similar weekly meetings land in different places.
--     The dentate gyrus orthogonalises on the way in; so do we.
--
--   READ (pattern completion): embed the query semantically, retrieve a
--     NEIGHBOURHOOD, then discriminate. Retrieve broadly, then narrow --
--     rather than retrieve narrowly and hope the top-1 is right.
-- ─────────────────────────────────────────────────────────────
-- F1: capture must never block. observe() writes the substrate row and queues
-- here; embedding happens on idle. Measured on Qwen3-Embedding-8B, an inline
-- embed costs ~330 ms -- six minutes to ingest a thousand notes, with the
-- session frozen throughout.
--
-- The queue carries the CONTEXT the write vector needs, not just the id,
-- because the pattern-separation signature is built from where a memory sat
-- when it arrived. Reconstructing that later from current state would embed
-- the wrong context for anything captured before a period or episode ended.
CREATE TABLE IF NOT EXISTS embed_queue (
    node_id    TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    partition  TEXT NOT NULL,
    episode_id TEXT,
    period_id  TEXT,
    source_ref TEXT NOT NULL DEFAULT '',
    when_ts    REAL NOT NULL DEFAULT 0.0,
    queued_at  REAL NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_embed_queue_order
    ON embed_queue(attempts, queued_at);

-- C2: communities whose IDENTITY persists across splits and merges.
-- Recomputing clusters every cycle churns their ids, and every composite
-- derived from an old id is then pointing at something that no longer
-- exists -- provenance chains break silently and the store fills with
-- orphans. `lineage` records what a community descends from, so a split is
-- traceable rather than looking like creation.
CREATE TABLE IF NOT EXISTS community (
    id         TEXT PRIMARY KEY,
    partition  TEXT NOT NULL,
    members    TEXT NOT NULL,          -- json array of node ids
    generation INTEGER NOT NULL DEFAULT 0,
    lineage    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_community_part ON community(partition);

CREATE TABLE IF NOT EXISTS vector (
    node_id  TEXT NOT NULL,
    space    TEXT NOT NULL CHECK (space IN ('write','read')),
    dim      INTEGER NOT NULL,
    model    TEXT NOT NULL,
    data     BLOB NOT NULL,          -- float32 little-endian, L2-normalised
    -- Mean similarity to a sample of other documents. In high-dimensional
    -- embedding spaces a few vectors become the nearest neighbour of very
    -- many queries regardless of meaning -- "hubness", a well-studied
    -- pathology of dense retrieval. Measured here so retrieval can discount
    -- documents that are close to everything.
    hubness  REAL NOT NULL DEFAULT 0.0,
    -- G5: see the note on lexeme.partition. This one buys more, because
    -- brute-force search reads `data` -- so an unscoped scan pulls every
    -- BLOB in the store off disk to compute similarities it will then
    -- discard on a partition check. The index on (space, partition) is
    -- what stops a private-partition query paying for the work
    -- partition's vectors.
    partition TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (node_id, space)
);
CREATE INDEX IF NOT EXISTS idx_vector_space ON vector(space);
-- idx_vector_shard is created by shards.migrate(); see lexeme above.

-- ─────────────────────────────────────────────────────────────
-- HETEROGENEOUS GRAPH — entities and observations in ONE structure.
--
-- The idea is MiniRAG's (ACL 2026): put named entities and text chunks in a
-- single graph, so retrieval can walk topology instead of relying on the
-- model to understand things. OWL adds one thing MiniRAG does not have --
-- EVERY EDGE CARRIES ITS EVIDENCE. A relation is not merely asserted; it
-- points at the observation that justifies it, so a retrieved path is
-- self-documenting and `why()` traverses it like anything else.
--
-- OWL does not extract entities. Extraction needs a model, belongs to the
-- host, and hosts usually already do it (ATK builds a link chart already).
-- The library accepts links and does the topology.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entity (
    id        TEXT PRIMARY KEY,
    partition TEXT NOT NULL REFERENCES partition(name),
    name      TEXT NOT NULL,
    kind      TEXT NOT NULL DEFAULT 'other' CHECK (kind IN
                ('person','org','place','artifact','event','quantity',
                 'identifier','time','other')),
    canonical TEXT NOT NULL,
    first_seen REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_canon
    ON entity(partition, canonical);
CREATE INDEX IF NOT EXISTS idx_entity_kind ON entity(kind);

CREATE TABLE IF NOT EXISTS mention (
    entity_id TEXT NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    node_id   TEXT NOT NULL,
    role      TEXT NOT NULL DEFAULT 'mentions',
    PRIMARY KEY (entity_id, node_id, role)
);
CREATE INDEX IF NOT EXISTS idx_mention_node ON mention(node_id);

CREATE TABLE IF NOT EXISTS relation (
    src           TEXT NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    dst           TEXT NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    evidence_node TEXT NOT NULL,    -- <- the observation that justifies it
    weight        REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, kind, evidence_node)
);
CREATE INDEX IF NOT EXISTS idx_relation_src ON relation(src);
CREATE INDEX IF NOT EXISTS idx_relation_dst ON relation(dst);

-- ─────────────────────────────────────────────────────────────
-- COMPOSITES — hierarchical fusion, built with ZERO model calls.
--
-- LycheeMem's Record Fusion Engine: dedupe by cosine, cluster the survivors
-- with union-find, promote each component to a composite, then run the same
-- pass over composites to grow a tree upward. Pure arithmetic.
--
-- OWL's addition: a composite is a `derived` node like any other, so the
-- monotonicity invariant applies -- a composite is never more certain than
-- its least certain member, and `why()` walks into it normally.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS composite_member (
    composite_id TEXT NOT NULL,
    member_id    TEXT NOT NULL,
    level        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (composite_id, member_id)
);
CREATE INDEX IF NOT EXISTS idx_comp_member ON composite_member(member_id);

-- ─────────────────────────────────────────────────────────────
-- THE FORWARD DIRECTION
--
-- Everything above points BACKWARD: why() answers "how do I know this?"
-- Nothing points FORWARD: "what do I believe because of this, and what did
-- I DO about it?"
--
-- An analyst reads "Route Alpha is open", routes a convoy, moves on. Three
-- days later a sitrep supersedes it. Every memory system in the field
-- updates the fact and stops. Nothing tells anyone the convoy decision is
-- now standing on a false premise.
--
-- That is the difference between a memory that answers questions and one
-- that prevents harm.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decision (
    id               TEXT PRIMARY KEY,
    partition        TEXT NOT NULL REFERENCES partition(name),
    statement        TEXT NOT NULL,
    decided_at       REAL NOT NULL,
    decided_by       TEXT NOT NULL DEFAULT 'user',
    -- NULL means irreversible. A decision you can still change is worth
    -- interrupting someone about; one you cannot is worth logging, not
    -- alarming over.
    reversible_until REAL,
    status           TEXT NOT NULL DEFAULT 'standing' CHECK (status IN
                       ('standing','revisit','reaffirmed','reversed','executed')),
    outcome          TEXT,
    resolved_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_decision_status ON decision(status, partition);

CREATE TABLE IF NOT EXISTS decision_basis (
    decision_id TEXT NOT NULL REFERENCES decision(id) ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (decision_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_basis_node ON decision_basis(node_id);

-- Raised when a decision's basis moves under it. Persisted rather than
-- computed on the fly so an alert can be acknowledged once and stay
-- acknowledged -- a system that re-raises the same warning every session
-- trains people to ignore it.
CREATE TABLE IF NOT EXISTS decision_impact (
    id           TEXT PRIMARY KEY,
    decision_id  TEXT NOT NULL REFERENCES decision(id) ON DELETE CASCADE,
    basis_node   TEXT NOT NULL,
    cause        TEXT NOT NULL CHECK (cause IN
                   ('superseded','discredited','stale','retracted')),
    detected_at  REAL NOT NULL,
    severity     REAL NOT NULL,
    acknowledged_at REAL
);
CREATE INDEX IF NOT EXISTS idx_impact_dec ON decision_impact(decision_id);

-- Load-bearing criticality: which memories, if wrong, invalidate the most?
-- Tells you where to spend verification effort, and is a far better
-- retention signal than access count -- the right question for forgetting
-- is not "how often was this used" but "what is resting on it".
CREATE TABLE IF NOT EXISTS criticality (
    node_id     TEXT PRIMARY KEY,
    score       REAL NOT NULL,
    dependents  INTEGER NOT NULL DEFAULT 0,
    decisions   INTEGER NOT NULL DEFAULT 0,
    computed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crit_score ON criticality(score DESC);

-- Immutable record of every recall: what was returned, why, what fired, and
-- what was considered and rejected. Downstream errors become traceable to a
-- retrieval decision; currently that evaporates the moment the call returns.
CREATE TABLE IF NOT EXISTS receipt (
    id          TEXT PRIMARY KEY,
    at          REAL NOT NULL,
    partition   TEXT NOT NULL,
    query       TEXT NOT NULL,
    state       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    returned    TEXT NOT NULL,        -- JSON [{node_id, score, ...}]
    rejected    TEXT NOT NULL,        -- JSON [{node_id, score}] near-misses
    tier        INTEGER NOT NULL,
    latency_ms  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_receipt_at ON receipt(at DESC);

-- ─────────────────────────────────────────────────────────────
-- SOURCE ASSESSMENT — trust is a judgement, and judgements get revised.
--
-- `observation.reliability/credibility` record what we thought AT INGEST.
-- They live on an append-only table and are therefore immutable, which is
-- correct: they are history.
--
-- What we think NOW is a different question and belongs in a mutable layer.
-- The append-only trigger caught an attempt to update reliability in place
-- during `discredit()` -- the same layering mistake as putting suppression
-- on the observation table. Third time the invariant has caught its author.
--
-- Keyed by SOURCE, not by node: learning a survey was out of date should
-- revalue every observation drawn from it, in one write.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS source_assessment (
    source_ref   TEXT PRIMARY KEY,
    reliability  TEXT NOT NULL CHECK (reliability IN ('A','B','C','D','E','F')),
    credibility  INTEGER NOT NULL CHECK (credibility BETWEEN 1 AND 6),
    reason       TEXT NOT NULL,
    assessed_at  REAL NOT NULL,
    superseded_count INTEGER NOT NULL DEFAULT 0,
    confirmed_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS write_screen (
    node_id    TEXT PRIMARY KEY,
    at         REAL NOT NULL,
    verdict    TEXT NOT NULL CHECK (verdict IN ('clean','suspect','blocked')),
    signals    TEXT NOT NULL,
    score      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS supersession_attempt (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         REAL NOT NULL,
    source_ref TEXT NOT NULL,
    old_node   TEXT NOT NULL,
    allowed    INTEGER NOT NULL,
    reason     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supatt_src ON supersession_attempt(source_ref, at);

-- ─────────────────────────────────────────────────────────────
-- SOURCE INDEPENDENCE
--
-- Corroboration currently counts DOCUMENTS. It should count independent
-- ORIGINS. If forty documents all trace to one upstream source, that is one
-- source, not forty -- and treating it as forty is exactly the source-flooding
-- attack: publish the same falsehood in bulk and manufacture consensus.
--
-- Nothing in the field does this. The same arithmetic also improves ordinary
-- corroboration quality, because "three sources agree" is only meaningful if
-- the three are actually three.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS source_origin (
    source_ref     TEXT PRIMARY KEY,
    origin_cluster TEXT NOT NULL,
    domain         TEXT,
    ingest_batch   TEXT,
    first_seen     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_origin_cluster ON source_origin(origin_cluster);

-- ─────────────────────────────────────────────────────────────
-- ATTRIBUTED BELIEF  (de dicto / de re)
--
-- "Ahmed said the parts arrive Thursday" is stored as a string, and the
-- Admiralty grade attaches to the DOCUMENT, not to Ahmed. So you cannot ask
-- "what does Ahmed believe?", "who else claims this?", or "Ahmed has been
-- wrong four times -- what else did he tell me?"
--
-- Intelligence tradecraft has treated this distinction as basic for a
-- century. No LLM memory system implements it.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claimant (
    id          TEXT PRIMARY KEY,
    partition   TEXT NOT NULL REFERENCES partition(name),
    name        TEXT NOT NULL,
    canonical   TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'person',
    first_seen  REAL NOT NULL,
    -- Learned from outcomes, not assigned by hand.
    claims_made      INTEGER NOT NULL DEFAULT 0,
    claims_confirmed INTEGER NOT NULL DEFAULT 0,
    claims_refuted   INTEGER NOT NULL DEFAULT 0,
    kept_count       INTEGER NOT NULL DEFAULT 0,
    broken_count     INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_claimant_canon
    ON claimant(partition, canonical);

CREATE TABLE IF NOT EXISTS claim (
    id          TEXT PRIMARY KEY,
    claimant_id TEXT NOT NULL REFERENCES claimant(id) ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    proposition TEXT NOT NULL,
    prop_hash   TEXT NOT NULL,
    asserted_at REAL NOT NULL,
    outcome     TEXT NOT NULL DEFAULT 'unresolved'
                  CHECK (outcome IN ('unresolved','confirmed','refuted'))
);
CREATE INDEX IF NOT EXISTS idx_claim_prop ON claim(prop_hash);
CREATE INDEX IF NOT EXISTS idx_claim_who ON claim(claimant_id);

-- ─────────────────────────────────────────────────────────────
-- COMMITMENTS — promises are not facts.
--
-- "I'll bring fuel Thursday" has a lifecycle: made -> due -> kept or broken.
-- Memanto has `commitment` as a memory TYPE; nobody tracks the lifecycle,
-- and the lifecycle is where the value is: a broken promise degrades the
-- claimant's reliability, which revalues everything they ever told you.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS commitment (
    id          TEXT PRIMARY KEY,
    partition   TEXT NOT NULL REFERENCES partition(name),
    claimant_id TEXT NOT NULL REFERENCES claimant(id) ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    statement   TEXT NOT NULL,
    made_at     REAL NOT NULL,
    due_at      REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN
                  ('open','due','kept','broken','waived')),
    resolved_at REAL,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_commit_due ON commitment(status, due_at);

-- ─────────────────────────────────────────────────────────────
-- FAILURE PATTERNS — tried, and it did not work.
--
-- OWL already records ABSENCE ("I looked, it is not there"). It did not
-- record FAILURE ("we tried this, here is why it did not work"), which for
-- an analyst toolkit is arguably the more valuable of the two -- and it is
-- what stops the same rejected option being re-proposed every week, which
-- is the specific behaviour that reads as not listening.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS failure (
    id         TEXT PRIMARY KEY,
    partition  TEXT NOT NULL REFERENCES partition(name),
    approach   TEXT NOT NULL,
    reason     TEXT NOT NULL,
    context    TEXT NOT NULL DEFAULT '',
    failed_at  REAL NOT NULL,
    node_id    TEXT,
    recurrence INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_failure_part ON failure(partition);
