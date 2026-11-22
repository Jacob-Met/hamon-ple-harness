from itertools import combinations, product

import pytest

from hamon_scale import _independent_word_spans


def disjoint(spans):
    return all(a[1] <= b[0] or b[1] <= a[0] for a, b in combinations(spans, 2))


def test_hall_deficient_subset_and_printed_alternative():
    domains = {
        "SPAN CORE 100 mm": [(0, 4), (7, 11)],
        "CORE 100 mm": [(1, 4), (8, 11)],
        "100 mm TYP": [(2, 5), (9, 12)],
        "AUX 100 mm": [(14, 17), (20, 23)],
        "OTHER 100 mm": [(17, 20), (23, 26)],
    }
    with pytest.raises(ValueError, match="independent"):
        _independent_word_spans(domains)
    domains["CORE 100 mm"].append((27, 30))
    witness = _independent_word_spans(domains)
    assert witness["CORE 100 mm"] == (27, 30)
    assert all(span in domains[label] for label, span in witness.items())
    assert disjoint(witness.values())


def test_interval_matching_against_small_exhaustive_oracle():
    universe = [(0, 1), (0, 2), (1, 3), (2, 4)]
    choices = [[span for i, span in enumerate(universe) if mask & (1 << i)]
               for mask in range(1, 1 << len(universe))]
    for rows in product(choices, repeat=3):
        domains = dict(enumerate(rows))
        expected = any(disjoint(candidate) for candidate in product(*rows))
        if not expected:
            with pytest.raises(ValueError, match="independent"):
                _independent_word_spans(domains)
        else:
            witness = _independent_word_spans(domains)
            assert set(witness) == set(domains)
            assert all(witness[k] in domains[k] for k in domains)
            assert disjoint(witness.values())
