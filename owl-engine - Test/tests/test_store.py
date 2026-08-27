"""Single writer, concurrent readers."""
import threading


def test_concurrent_writes_serialize_cleanly(mind):
    errors = []

    def worker(n):
        try:
            for i in range(20):
                mind.observe(f"worker {n} entry {i} routine check")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert mind._scalar("SELECT COUNT(*) FROM observation") == 120


def test_doctor_reports_healthy(mind):
    mind.observe("baseline entry for the health check")
    d = mind.doctor()
    assert d["healthy"], d["problems"]
    assert d["tier"] == 0
