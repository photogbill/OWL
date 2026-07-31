"""Decay lives in the index. The record is never touched."""
from owl import State
from owl.salience import retrievability, initial_state, review, DAY


def test_retrievability_decays_but_substrate_survives(mind, clock):
    nid = mind.observe("The generator serial is GX-4419.")
    assert mind.recall("generator serial").state is State.KNOW

    clock.advance(days=400)
    mind.tend()

    row = mind._s.one("SELECT tier FROM mem_index WHERE node_id=?", (nid,))
    assert row["tier"] in ("warm", "cold", "pruned"), "index failed to decay"
    # ...and the evidence is untouched:
    assert mind._node_row(nid)["content"] == "The generator serial is GX-4419."


def test_spacing_effect_is_representable():
    s, d = initial_state(3)
    s_massed, _ = review(s, d, 0.5 * DAY, 3)
    s_spaced, _ = review(s, d, 3.0 * DAY, 3)
    assert s_spaced > s_massed, (
        "a model that only knows last_accessed cannot express the spacing "
        "effect -- this is the defect in the v1 formula")


def test_forgetting_is_power_law_not_exponential():
    """The tail is the whole point.

    Wixted & Ebbesen (1991) found power (and hyperbolic) functions beat
    exponentials across paradigms. An exponential forgets old material far too
    fast: matched at a short horizon, it is orders of magnitude below the power
    law at a long one -- which in a memory engine means silently discarding
    material a person would still recall perfectly well.
    """
    import math
    s, _ = initial_state(3)
    short = 7 * DAY
    r_short = retrievability(short, s)
    lam = -math.log(r_short) / short          # exponential matched at 7 days
    for horizon in (90 * DAY, 365 * DAY):
        r_power = retrievability(horizon, s)
        r_exp = math.exp(-lam * horizon)
        assert r_power > r_exp * 5, (
            f"at {horizon/DAY:.0f}d power={r_power:.4f} exp={r_exp:.6f}")

    # Scale-free signature: at long horizons the ratio over a fixed multiple
    # approaches a constant, which an exponential never does.
    ratios = [retrievability(2 * t, s) / retrievability(t, s)
              for t in (200 * DAY, 400 * DAY, 800 * DAY)]
    assert max(ratios) - min(ratios) < 0.02


def test_use_restores_retrievability(mind, clock):
    mind.observe("Radio check frequency is 145.500 simplex.")
    clock.advance(days=30)
    r = mind.recall("radio check frequency")
    assert r.state is not State.DONT_KNOW
    clock.advance(days=30)
    assert mind.recall("radio check frequency").state is not State.DONT_KNOW
