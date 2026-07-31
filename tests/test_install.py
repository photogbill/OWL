"""The installer surface. A bad first-run experience is a real bug."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# These spawn a fresh interpreter each, which costs ~0.8 s apiece. They are
# smoke tests of the TOOLING, not of the engine, so they carry a marker and
# the fast development loop skips them:
#
#     pytest tests -q -m "not slow"     engine correctness, ~1 s
#     pytest tests -q                   everything, ~10 s
pytestmark = pytest.mark.slow


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=90)


def test_detect_backend_always_offers_a_fallback():
    """Whatever the hardware, there is always something to try."""
    r = _run("scripts/detect_backend.py")
    assert r.returncode == 0
    urls = r.stdout.split()
    assert urls and urls[-1] == "PYPI", "plain PyPI must be the last resort"
    assert any("/cpu" in u for u in urls), "CPU wheels must always be offered"


def test_detect_backend_explains_itself():
    r = _run("scripts/detect_backend.py", "--explain")
    assert r.returncode == 0 and "trying, in order" in r.stdout


def test_preflight_reports_tier_and_venv_state():
    r = _run("-m", "owl", "check")
    assert r.returncode == 0
    assert "MAX TIER" in r.stdout
    assert "owl-engine" in r.stdout
    assert "SYSTEM PYTHON" in r.stdout or "[venv]" in r.stdout


def test_validator_blocks_helpfully_without_the_backend():
    """The failure the user actually hit: a raw ImportError traceback.

    Missing backend and failed model load have completely different fixes;
    collapsing them into one exception is what produced the bad experience.
    """
    from owl.adapters import gguf_embed
    if gguf_embed.available():
        return                                    # nothing to assert here
    r = _run("bench/validate_embedder.py", "embedding model/bge-m3-Q6_K.gguf")
    assert r.returncode == 3, "missing backend needs its own exit code"
    assert "BLOCKED" in r.stdout
    assert "install" in r.stdout.lower()
    assert "Traceback" not in r.stderr, "must not surface a raw traceback"


def test_validator_rejects_a_missing_file_distinctly():
    r = _run("bench/validate_embedder.py", "definitely_not_here.gguf")
    assert r.returncode == 2, "missing file is a different failure from "\
                              "missing backend"
    assert "no such file" in r.stdout


BATCH = ("install.bat", "run_tests.bat", "validate.bat", "demo.bat",
         "bench.bat", "shell.bat", "clean.bat", "_common.bat")


def test_all_batch_scripts_exist():
    for name in BATCH:
        assert (ROOT / name).exists(), f"missing {name}"


def test_batch_scripts_use_the_venv_not_system_python():
    """The whole point of the installer is isolation. A script that reaches
    for whatever `python` is on PATH silently defeats it."""
    for name in BATCH:
        if name in ("_common.bat",):
            continue
        text = (ROOT / name).read_text()
        assert ".venv" in text, f"{name} must use the isolated venv"


def test_batch_scripts_pause_when_double_clicked():
    """On Windows a .bat launched from Explorer closes the instant it ends,
    so an error message is visible for about a tenth of a second. Every
    script must detect that and hold the window open."""
    for name in BATCH:
        if name == "shell.bat":
            continue                      # cmd /k already holds the window
        text = (ROOT / name).read_text()
        if name == "_common.bat":
            assert "cmdcmdline" in text
            continue
        assert "DBLCLICK" in text, f"{name} must handle being double-clicked"


def test_common_prelude_receives_the_callers_path():
    """`%~f0` inside a called .bat resolves to the callee, not the caller,
    so the double-click check would silently never fire."""
    common = (ROOT / "_common.bat").read_text()
    assert "%~1" in common, "_common.bat must use the passed-in caller path"
    assert "%~f0" not in common.split("REM ===")[-1], (
        "_common.bat must not test its own path")
    for name in ("run_tests.bat", "validate.bat", "demo.bat", "bench.bat"):
        text = (ROOT / name).read_text()
        assert 'call "%~dp0_common.bat" "%~f0"' in text, (
            f"{name} must pass its own path to the prelude")


def test_batch_scripts_are_crlf_safe():
    """A .bat with bare LF endings can fail in confusing ways on Windows."""
    for name in BATCH:
        raw = (ROOT / name).read_bytes()
        assert b"\r\n" in raw or b"\n" not in raw, (
            f"{name} should use CRLF line endings")


def test_install_never_builds_from_source():
    """Compiling llama-cpp-python needs CMake + VS Build Tools and usually
    fails. Prebuilt wheels only."""
    text = (ROOT / "scripts" / "install_backend.py").read_text()
    assert "--only-binary=:all:" in text
    assert (ROOT / "install.bat").read_text().count("install_backend.py") == 1


def test_backend_install_is_python_not_batch():
    """cmd's `for /f` with backquotes mangles quoted paths containing spaces.
    A folder named "owl-engine - Test" broke the loop AND silently reported
    nothing even though the wheel had installed."""
    bat = (ROOT / "install.bat").read_text()
    assert "for %%I" not in bat, (
        "wheel selection must not loop in batch - path quoting is not "
        "something to hand-roll in cmd")
    assert "usebackq" not in bat


def test_install_backend_reports_without_installing():
    r = _run("scripts/install_backend.py", "--check")
    assert r.returncode in (0, 1)
    assert "llama-cpp-python" in r.stdout


def test_backend_installer_does_not_leak_pip_errors():
    """`pip --quiet` does not suppress ERROR lines. A working fallback chain
    printed two alarming errors before succeeding and looked like a failure."""
    src = (ROOT / "scripts" / "install_backend.py").read_text()
    assert "capture_output=True" in src, (
        "pip output must be captured so a successful fallback is quiet")
    assert "errors.append" in src, (
        "captured errors must still be reported if EVERY candidate fails")
