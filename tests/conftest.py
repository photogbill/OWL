import os
import shutil
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from owl import Owl

DAY = 86400.0


class FakeClock:
    """The highest-leverage twelve lines in the codebase.

    Nothing in OWL calls time.time(). With an injectable clock a 90-day
    forgetting curve tests in three milliseconds; without one, decay is
    untestable and ships broken.
    """
    def __init__(self, start: float = 1_700_000_000.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, *, seconds=0.0, days=0.0):
        self._t += seconds + days * DAY
        return self._t


class FakeReasoner:
    def __init__(self, canned="ok"):
        self.canned = canned
        self.calls = []

    def complete(self, prompt, *, grammar=None, max_tokens=512, temperature=0.7):
        self.calls.append(prompt)
        return self.canned


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture(scope="session")
def _schema_template(tmp_path_factory):
    """Build the schema ONCE, then copy the file per test.

    Executing ~500 lines of DDL costs ~22 ms, which across the suite was
    2.6 s of pure setup -- more than everything else combined. Copying a
    prebuilt empty store is under a millisecond and is byte-identical to
    what `Owl.open` would have produced.
    """
    path = tmp_path_factory.mktemp("tmpl") / "template.owl"
    Owl.open(path).close()
    return path


@pytest.fixture
def mind(clock, tmp_path, _schema_template):
    target = tmp_path / "test.owl"
    shutil.copyfile(_schema_template, target)
    m = Owl.open(target, clock=clock)
    yield m
    m.close()
