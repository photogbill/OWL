"""Episodes are cut at surprise boundaries, not at fixed sizes."""


def test_topic_change_starts_a_new_episode(mind):
    ids = []
    for line in [
        "The north well pump failed again this morning.",
        "The pump gasket is worn and the well is the only clean source.",
        "Well water sampling showed contamination after the pump failure.",
        "Ahmed says pump parts for the well arrive Thursday.",
        "Separate matter entirely: vaccine cold chain refrigerator logging.",
        "Refrigerator logs show a cold chain excursion on Tuesday night.",
    ]:
        ids.append(mind.observe(line, source_ref="field-notes"))
    eps = [mind._s.one("SELECT episode_id FROM observation WHERE id=?", (i,))[0]
           for i in ids]
    assert len(set(eps)) >= 2, "no event boundary detected across a topic change"
    assert eps[0] == eps[1], "coherent material was split"
    assert eps[-1] != eps[0], "unrelated material was merged into one episode"
