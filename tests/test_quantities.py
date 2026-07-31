"""Dimensional integrity. A dropped unit is a dangerous sentence."""
import pytest
from owl import OwlError
from owl.quantities import (Quantity, conflicts, dropped, parse,
                            same_magnitude)


def test_parses_quantities_from_prose():
    assert [str(q) for q in parse("Depot holds 4,000 litres of diesel.")] == ["4000l"]
    assert [str(q) for q in parse("Give 250 mg every 6 hours.")] == ["250mg", "6h"]
    assert [str(q) for q in parse("A 5 kW generator, 12 beds.")] == ["5kw", "12beds"]


def test_units_convert_not_just_compare():
    a, b = parse("400 litres")[0], parse("105.669 gallons")[0]
    assert same_magnitude(a, b, tol=1e-4), "400 L IS 105.669 gal"
    assert not same_magnitude(a, parse("400 gallons")[0]), "400 L is NOT 400 gal"
    # 4 m3 is 4000 L, not 400 -- the code caught an error in this test.
    assert not same_magnitude(a, parse("4 m3")[0])
    assert same_magnitude(parse("4000 litres")[0], parse("4 m3")[0])


def test_the_dangerous_rewrites_are_rejected():
    assert conflicts("Depot holds 4000 litres.", "Depot holds 4000 gallons.")
    assert conflicts("Depot holds 4000 litres.", "The depot holds 4000.")
    assert conflicts("Give 250 mg every 6 hours.", "Give 250 every 6 hours.")


def test_abstraction_is_not_a_conflict():
    """An earlier version rejected this. Omitting detail is what abstraction
    IS -- the danger is keeping the NUMBER and losing the unit."""
    assert not conflicts("Depot holds 4000 litres.",
                         "Fuel is not a constraint this month.")
    assert not conflicts("Depot holds 4000 litres.", "Depot holds 4 m3.")
    assert [q.raw for q in dropped("Depot holds 4000 litres.",
                                   "Fuel is fine.")] == ["4000 litres"]


def test_derivation_rejects_a_stripped_unit(mind):
    n = mind.observe("Depot holds 4000 litres of diesel.", source_ref="audit")
    with pytest.raises(OwlError, match="dimensional integrity"):
        mind.derive("Depot holds 4000.", parents=[n], kind="summary",
                    producer="fusion")


def test_derivation_rejects_a_changed_magnitude(mind):
    n = mind.observe("Give 250 mg every six hours.", source_ref="protocol")
    with pytest.raises(OwlError, match="value changed"):
        mind.derive("Give 250 g every six hours.", parents=[n],
                    kind="abstraction", producer="analyst")


def test_derivation_allows_honest_abstraction(mind):
    n = mind.observe("Depot holds 4000 litres of diesel.", source_ref="audit")
    d = mind.derive("Fuel is not a constraint this month.", parents=[n],
                    kind="abstraction", producer="analyst")
    assert mind._node_row(d) is not None
