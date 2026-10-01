"""Schema changes an existing store needs, applied once, on open, and reported.

SQLite can add a column in place but cannot change a CHECK constraint. The
`derived.kind` CHECK is the closed list of things a derivation may be, and a
store written before a new kind existed refuses that kind forever -- the
first `translation()` against an older ATK ledger failed with an
IntegrityError rather than anything a person could act on.

So a store whose list is short is rebuilt the way the SQLite documentation
prescribes for constraint changes: create the new table, copy every row,
drop the old, rename, re-create its indexes and triggers -- with foreign keys
off for the duration, inside one transaction, so a failure leaves the store
exactly as it was. The new DDL is made from the store's OWN stored DDL, so
every column it has (including any an older version added) is kept.
"""
from __future__ import annotations

import re
import sqlite3

#: Every kind a derived node may be. `schema.sql` carries the same list; a
#: test asserts the two agree.
DERIVED_KINDS = ("summary", "abstraction", "graft", "hypothesis", "conflict",
                 "correction", "reflection", "community", "decontext",
                 "translation")

_KIND_CHECK = re.compile(
    r"kind\s+TEXT\s+NOT\s+NULL\s+CHECK\s*\(\s*kind\s+IN\s*\((?P<list>[^)]*)\)",
    re.I | re.S)


def derived_kinds(conn: sqlite3.Connection) -> list[str] | None:
    """The kinds this store's `derived` table accepts, or None if unknown."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                       "AND name='derived'").fetchone()
    if row is None or not row[0]:
        return None
    m = _KIND_CHECK.search(row[0])
    return re.findall(r"'([a-z_]+)'", m.group("list")) if m else None


def widen_derived_kinds(conn: sqlite3.Connection) -> list[str]:
    """Rebuild `derived` if its kind list is missing any of DERIVED_KINDS.

    Returns the kinds added (empty when nothing was needed). Idempotent.
    """
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                       "AND name='derived'").fetchone()
    if row is None or not row[0]:
        return []
    sql = row[0]
    m = _KIND_CHECK.search(sql)
    if not m:
        # An unrecognised shape is left alone rather than guessed at. A kind
        # it does not accept will still fail loudly at the write.
        return []
    present = re.findall(r"'([a-z_]+)'", m.group("list"))
    missing = [k for k in DERIVED_KINDS if k not in present]
    if not missing:
        return []
    kinds = ",".join(f"'{k}'" for k in present + missing)
    new_sql = sql[:m.start("list")] + kinds + sql[m.end("list"):]
    new_sql, n = re.subn(r'^\s*CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?'
                         r'["`\[]?derived["`\]]?',
                         "CREATE TABLE derived__widen", new_sql, count=1,
                         flags=re.I)
    if not n:
        return []
    extras = [r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name='derived' "
        "AND type IN ('index','trigger') AND sql IS NOT NULL")]
    cols = ",".join(f'"{r[1]}"' for r in conn.execute(
        "PRAGMA table_info(derived)"))
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(new_sql)
            conn.execute(f"INSERT INTO derived__widen({cols}) "
                         f"SELECT {cols} FROM derived")
            conn.execute("DROP TABLE derived")
            conn.execute("ALTER TABLE derived__widen RENAME TO derived")
            for ddl in extras:
                conn.execute(ddl)
            broken = [r for r in conn.execute("PRAGMA foreign_key_check")
                      if r[0] in ("derived", "period")]
            if broken:
                raise sqlite3.IntegrityError(
                    f"rebuilding derived broke {len(broken)} reference(s)")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return missing
