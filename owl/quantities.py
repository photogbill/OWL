"""Dimensional integrity — quantities are values with units, not substrings.

"400L" and "400 gallons" are not the same number, and a summariser that drops
the unit has produced a *dangerous* sentence rather than a shorter one. For
fuel, dosage, distance and time in the field this is a safety property, not a
nicety.

Nothing in the reviewed field does this. It is entirely deterministic, costs
nothing, and prevents a real class of error:

  * two quantities in different dimensions never fuse, whatever the cosine
  * a derivation that drops or changes a unit is rejected
  * a supersession that silently changes the dimension is flagged
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# canonical unit -> (dimension, factor to the SI-ish base of that dimension)
UNITS: dict[str, tuple[str, float]] = {
    # volume -> litres
    "l": ("volume", 1.0), "ml": ("volume", 0.001), "cl": ("volume", 0.01),
    "m3": ("volume", 1000.0), "gal": ("volume", 3.785411784),
    "impgal": ("volume", 4.54609), "qt": ("volume", 0.946352946),
    "pt": ("volume", 0.473176473), "floz": ("volume", 0.0295735296),
    "bbl": ("volume", 158.987294928),
    # mass -> grams
    "g": ("mass", 1.0), "mg": ("mass", 0.001), "ug": ("mass", 1e-6),
    "kg": ("mass", 1000.0), "t": ("mass", 1e6), "lb": ("mass", 453.59237),
    "oz": ("mass", 28.349523125),
    # length -> metres
    "m": ("length", 1.0), "mm": ("length", 0.001), "cm": ("length", 0.01),
    "km": ("length", 1000.0), "mi": ("length", 1609.344),
    "ft": ("length", 0.3048), "in": ("length", 0.0254),
    "nmi": ("length", 1852.0), "yd": ("length", 0.9144),
    # time -> seconds
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "day": ("time", 86400.0), "wk": ("time", 604800.0),
    # power / energy
    "w": ("power", 1.0), "kw": ("power", 1000.0), "mw": ("power", 1e6),
    "kwh": ("energy", 1.0), "wh": ("energy", 0.001),
    # frequency
    "hz": ("frequency", 1.0), "khz": ("frequency", 1e3),
    "mhz": ("frequency", 1e6), "ghz": ("frequency", 1e9),
    # temperature is NOT a scale factor -- handled separately
    "c": ("temperature", 1.0), "f": ("temperature", 1.0),
    # dimensionless counts
    "beds": ("count", 1.0), "people": ("count", 1.0), "units": ("count", 1.0),
    "cases": ("count", 1.0), "vials": ("count", 1.0), "doses": ("count", 1.0),
    "%": ("ratio", 1.0),
}

# surface form -> canonical
ALIASES: dict[str, str] = {
    "litre": "l", "litres": "l", "liter": "l", "liters": "l", "ltr": "l",
    "millilitre": "ml", "millilitres": "ml", "milliliter": "ml",
    "cubic metre": "m3", "cubic meter": "m3", "m³": "m3",
    "gallon": "gal", "gallons": "gal", "us gal": "gal",
    "imperial gallon": "impgal", "imp gal": "impgal",
    "quart": "qt", "quarts": "qt", "pint": "pt", "pints": "pt",
    "fluid ounce": "floz", "fl oz": "floz", "barrel": "bbl", "barrels": "bbl",
    "gram": "g", "grams": "g", "gramme": "g",
    "milligram": "mg", "milligrams": "mg", "mcg": "ug", "microgram": "ug",
    "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t",
    "pound": "lb", "pounds": "lb", "lbs": "lb",
    "ounce": "oz", "ounces": "oz",
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "millimetre": "mm", "millimetres": "mm", "millimeter": "mm",
    "centimetre": "cm", "centimetres": "cm", "centimeter": "cm",
    "kilometre": "km", "kilometres": "km", "kilometer": "km",
    "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft",
    "inch": "in", "inches": "in", "yard": "yd", "yards": "yd",
    "nautical mile": "nmi", "nautical miles": "nmi",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s",
    "minute": "min", "minutes": "min", "mins": "min",
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "days": "day", "week": "wk", "weeks": "wk",
    "watt": "w", "watts": "w", "kilowatt": "kw", "kilowatts": "kw",
    "megawatt": "mw", "kilowatt-hour": "kwh", "kilowatt hours": "kwh",
    "hertz": "hz", "kilohertz": "khz", "megahertz": "mhz",
    "gigahertz": "ghz",
    "bed": "beds", "person": "people", "persons": "people",
    "unit": "units", "case": "cases", "vial": "vials", "dose": "doses",
    "percent": "%", "pct": "%",
    "°c": "c", "celsius": "c", "centigrade": "c",
    "°f": "f", "fahrenheit": "f",
}

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_UNIT_WORDS = sorted(set(list(UNITS) + list(ALIASES)), key=len, reverse=True)
_UNIT_ALT = "|".join(re.escape(u) for u in _UNIT_WORDS)
_QTY = re.compile(rf"{_NUM}\s*({_UNIT_ALT})\b", re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str                 # canonical
    dimension: str
    raw: str

    @property
    def base(self) -> float:
        """Value in the dimension's base unit. Temperature has no scale
        factor, so it is left alone rather than silently mangled."""
        if self.dimension == "temperature":
            return self.value
        return self.value * UNITS[self.unit][1]

    def __str__(self) -> str:
        return f"{self.value:g}{self.unit}"


def canonical_unit(text: str) -> str | None:
    t = text.strip().lower()
    t = ALIASES.get(t, t)
    return t if t in UNITS else None


def parse(text: str) -> list[Quantity]:
    """Every quantity in a piece of text. Deterministic, no model."""
    out: list[Quantity] = []
    for m in _QTY.finditer(text or ""):
        unit = canonical_unit(m.group(2))
        if unit is None:
            continue
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append(Quantity(val, unit, UNITS[unit][0], m.group(0)))
    return out


def compatible(a: Quantity, b: Quantity) -> bool:
    return a.dimension == b.dimension


def same_magnitude(a: Quantity, b: Quantity, tol: float = 1e-6) -> bool:
    """Are these the same physical quantity, whatever the unit?

    400 litres and 105.7 gallons are the same thing. 400 litres and 400
    gallons are not, and a system that treats them as near-duplicates because
    the strings look alike is dangerous.
    """
    if not compatible(a, b):
        return False
    if a.base == 0 and b.base == 0:
        return True
    denom = max(abs(a.base), abs(b.base)) or 1.0
    return abs(a.base - b.base) / denom <= tol


_BARE_NUM = re.compile(_NUM)


def bare_numbers(text: str) -> set[float]:
    """Numbers in the text that are NOT attached to a unit."""
    attached = {q.value for q in parse(text)}
    out: set[float] = set()
    for m in _BARE_NUM.finditer(text or ""):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if v not in attached:
            out.add(v)
    return out


def conflicts(before: str, after: str) -> list[str]:
    """Unit problems introduced by rewriting `before` into `after`.

    Two things are dangerous and are rejected:

      * a magnitude changed  -- "4000 litres" became "4000 gallons"
      * a unit was STRIPPED  -- "4000 litres" became "holds 4000"

    Dropping the quantity entirely is NOT a conflict. That is what
    abstraction IS: "depot holds 4000 litres" summarising to "fuel is not a
    constraint" is a legitimate and useful derivation, and an earlier version
    of this function rejected exactly that. Use `dropped()` if you want the
    informational case.
    """
    a, b = parse(before), parse(after)
    problems: list[str] = []
    by_dim_b: dict[str, list[Quantity]] = {}
    for q in b:
        by_dim_b.setdefault(q.dimension, []).append(q)
    loose = bare_numbers(after)

    for q in a:
        same_dim = by_dim_b.get(q.dimension)
        if same_dim:
            if not any(same_magnitude(q, r) for r in same_dim):
                got = ", ".join(r.raw for r in same_dim)
                problems.append(f"{q.raw} became {got} - value changed")
        elif q.value in loose:
            problems.append(
                f"{q.raw} lost its unit - the number {q.value:g} survived "
                "without it")
    return problems


def dropped(before: str, after: str) -> list[Quantity]:
    """Quantities present before and absent after. Informational, not an
    error -- abstraction legitimately omits detail."""
    b_dims = {q.dimension for q in parse(after)}
    return [q for q in parse(before) if q.dimension not in b_dims]
