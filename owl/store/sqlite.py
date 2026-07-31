"""SQLite store with a single writer.

WAL gives concurrent readers and ONE writer. A memory engine has several
background workers that all want to write (compression, grafting, interference
sweeps), so contention is guaranteed unless writes are serialized. Athena's
current answer is a 600-second busy timeout, which absorbs the contention
rather than removing it and turns a deadlock into a ten-minute hang.

Every mutation in OWL goes through `SqliteStore.write()`. No exceptions.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .. import shards
from ..protocols import PartitionError, ReadOnlyError

_SCHEMA = (Path(__file__).parent.parent / "schema.sql").read_text(encoding="utf-8")

_SENTINEL = object()


class SqliteStore:
    """Thread-safe store. Reads go direct; writes go through one queue."""

    def __init__(self, path: str | Path, *, readonly: bool = False):
        self.path = str(path)
        self.readonly = readonly
        self.immutable = False
        self._local = threading.local()
        self._wq: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._flow_cache: dict[str, frozenset[str]] | None = None
        self._on_write: list = []
        # G5. `sharded` is asked of the file rather than assumed, because a
        # store opened read-only cannot be migrated and must still answer
        # -- F2 does not get suspended for a speedup. The read paths carry
        # a legacy predicate for that case and nothing else differs.
        self.sharded = False
        self.migration: dict | None = None
        if readonly:
            # No schema creation, no writer thread, no lock taken. A store
            # on read-only media, a forensic copy, or a file another process
            # is actively writing must still be READABLE -- the machinery is
            # not allowed to be a gatekeeper on remembering.
            with self._connect() as conn:
                if not conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                        " AND name='observation'").fetchone()[0]:
                    raise ValueError(
                        f"{self.path} is not an OWL store (no substrate "
                        "table), and read-only mode cannot create one")
                self.sharded = shards.is_sharded(conn)
            self._writer = None
            return
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # G5: bring an older store up to the sharded layout, once, on
            # open. Reported rather than silent -- a migration that ran is
            # a fact about the store, and this engine does not make
            # invisible changes to files it was handed.
            self.migration = shards.migrate(conn)
            self.sharded = True
        self._writer = threading.Thread(
            target=self._writer_loop, name="owl-writer", daemon=True
        )
        self._writer.start()

    # ── connections ──────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        if self.readonly:
            uri = f"file:{Path(self.path).as_posix()}?mode=ro"
            # A WAL database needs to WRITE the -shm shared-memory file even
            # to read it, so `mode=ro` alone fails on genuinely read-only
            # media -- an archive, a mounted image, a file the OS marked
            # unwritable. That is precisely the case F2 exists for.
            #
            # So: try the live path first, because it sees concurrent writes
            # and gives a consistent snapshot of a store another process is
            # using. Fall back to `immutable=1`, which tells SQLite the file
            # cannot change so it skips WAL entirely.
            #
            # The fallback is NOT equivalent, and the difference is recorded
            # rather than hidden: an immutable reader will not see anything
            # written after it opened.
            for attempt, extra in ((0, ""), (1, "&immutable=1")):
                try:
                    conn = sqlite3.connect(uri + extra, uri=True, timeout=30.0,
                                           isolation_level=None)
                    conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
                    conn.row_factory = sqlite3.Row
                    self.immutable = bool(attempt)
                    return conn
                except sqlite3.OperationalError:
                    if attempt:
                        raise
            raise AssertionError("unreachable")
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @property
    def _reader(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    # ── the single writer ────────────────────────────────────────────
    def _writer_loop(self) -> None:
        conn = self._connect()
        while True:
            item = self._wq.get()
            if item is _SENTINEL:
                conn.close()
                self._wq.task_done()
                return
            fn, result_box, done = item
            try:
                conn.execute("BEGIN IMMEDIATE")
                result_box.append(fn(conn))
                conn.execute("COMMIT")
            except BaseException as exc:      # noqa: BLE001 - re-raised on caller
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                result_box.append(exc)
            finally:
                done.set()
                self._wq.task_done()

    def write(self, fn: Callable[[sqlite3.Connection], Any] = None, *,
              affects_answers: bool = True) -> Any:
        """Run `fn` inside the writer thread, in one transaction.

        `affects_answers=False` marks a write that cannot change what a
        query returns -- recall's own reinforcement and receipts. Without
        it a write-invalidated cache invalidates itself on every read,
        because recall WRITES: retrieval reinforces what it returned. The
        cache would then never hit once, which is a subtle way to ship a
        feature that does nothing.
        """
        if self.readonly:
            raise ReadOnlyError(
                "this store was opened read-only. Reading works; anything "
                "that would change the record does not.")
        if self._closed.is_set():
            raise RuntimeError("store is closed")
        # G3: any write invalidates every cached answer. Total and O(1) --
        # working out WHICH cached queries a write could have affected is
        # where cache bugs live, and a stale answer looks exactly like a
        # fresh one.
        if affects_answers:
            for hook in self._on_write:
                hook()
        box: list[Any] = []
        done = threading.Event()
        self._wq.put((fn, box, done))
        done.wait()
        out = box[0] if box else None
        if isinstance(out, BaseException):
            raise out
        return out

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._reader.execute(sql, params))

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._reader.execute(sql, params).fetchone()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._writer is None:            # read-only: nothing to drain
            return
        self._wq.put(_SENTINEL)
        self._writer.join(timeout=5.0)

    # ── partitions: information-flow control ─────────────────────────
    def ensure_partition(self, name: str, *, sealed: bool, now: float,
                         flows_to: Iterable[tuple[str, str]] = (),
                         reads_from: Iterable[tuple[str, str]] = ()) -> None:
        """`sealed` blocks OUTflow only. Inflow is a separate decision.

        A sealed partition can still read others -- that is a one-way membrane,
        not an island, and it is almost always what a confidential context
        actually wants: full awareness, zero leakage.
        """
        out = list(flows_to)
        inn = list(reads_from)
        if sealed and out:
            raise PartitionError(
                f"partition '{name}' is sealed; it cannot declare outflow"
            )

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR IGNORE INTO partition(name,sealed,created_at) "
                "VALUES(?,?,?)", (name, int(sealed), now),
            )
            for other, level in list(out) + list(inn):
                conn.execute(
                    "INSERT OR IGNORE INTO partition(name,sealed,created_at) "
                    "VALUES(?,0,?)", (other, now),
                )
            for dst, level in out:
                conn.execute(
                    "INSERT OR REPLACE INTO partition_flow(src,dst,level) "
                    "VALUES(?,?,?)", (name, dst, level),
                )
            for src, level in inn:
                if conn.execute("SELECT sealed FROM partition WHERE name=?",
                                (src,)).fetchone()[0]:
                    raise PartitionError(
                        f"'{name}' cannot read from sealed partition '{src}'")
                conn.execute(
                    "INSERT OR REPLACE INTO partition_flow(src,dst,level) "
                    "VALUES(?,?,?)", (src, name, level),
                )

        self.write(_w)
        self._flow_cache = None

    def readable_from(self, partition: str) -> dict[str, str]:
        """Partitions whose content may be surfaced while working in `partition`.

        Flow is DENIED by default and is NOT symmetric. `A flows_to B` means
        A's content may appear in B, not the reverse. A sealed partition has no
        outflow, so nothing but itself can ever read it -- which is what makes
        Athena's confidentiality boundary a property of the persistence layer
        rather than a rule application code has to remember.
        """
        if self._flow_cache is None:
            self._flow_cache = self._build_flow()
        return self._flow_cache.get(partition, {partition: "full"})

    def _build_flow(self) -> dict[str, dict[str, str]]:
        """Inbound transitive closure, with the WEAKEST level along any path.

        Graded permeability: a 'summary' edge admits only derived, abstracted
        content -- period and episode summaries -- not raw observations. So a
        companion context can know *that today was bad at the clinic* without
        holding the specific images. Level degrades along a path and never
        upgrades: full + summary = summary.
        """
        names = [r["name"] for r in self.query("SELECT name FROM partition")]
        inbound: dict[str, dict[str, str]] = {n: {n: "full"} for n in names}
        for r in self.query("SELECT src,dst,level FROM partition_flow"):
            inbound.setdefault(r["dst"], {r["dst"]: "full"})
            inbound[r["dst"]][r["src"]] = r["level"]
        changed = True
        while changed:
            changed = False
            for dst, srcs in inbound.items():
                for mid, lvl in list(srcs.items()):
                    for far, lvl2 in inbound.get(mid, {}).items():
                        weakest = "summary" if "summary" in (lvl, lvl2) else "full"
                        if far == dst:
                            continue
                        if far not in srcs:
                            srcs[far] = weakest
                            changed = True
                        elif srcs[far] == "full" and weakest == "summary":
                            srcs[far] = weakest
                            changed = True
        return inbound

    def assert_readable(self, viewer: str, owner: str) -> None:
        if owner not in self.readable_from(viewer):
            raise PartitionError(
                f"information-flow violation: partition '{viewer}' may not read "
                f"'{owner}'"
            )

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def jload(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return default
