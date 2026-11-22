import base64

import pytest

from hamon_conservation_cad_verifier import verify_circuit_conservation


def test_one_deck_supplies_distinct_dc_and_ac_excitations():
    deck = "Bias and small-signal voltage\nV1 in 0 DC 12 AC 4 90\nR1 in out 1k\nR2 out 0 1k\n.ac lin 1 1000 1000\n.end"
    for entry in ({"netlist": deck}, {"file_b64": base64.b64encode(deck.encode()).decode()}):
        ac = verify_circuit_conservation({**entry, "proposed_node_voltages": {"out": [0, 2]}})
        dc = verify_circuit_conservation({**entry, "freq_hz": 0, "proposed_node_voltages": {"out": 6}})
        assert ac["conservation_verified"] and dc["conservation_verified"]
        assert ac["node_voltages"]["out"] == pytest.approx([0, 2])
        assert dc["node_voltages"]["out"] == pytest.approx(6)


@pytest.mark.parametrize("clause, ac, dc", [
    ("DC 7", 0, 7), ("AC 2", 2, 0), ("5 AC", 1, 5),
    ("AC 0 90 DC 7", 0, 7), ("", 0, 0),
])
def test_current_source_defaults_and_order(clause, ac, dc):
    net = f"Current source\nI1 0 n {clause}\nR1 n 0 1\n.end"
    for freq, value in ((0, dc), (1000, ac)):
        result = verify_circuit_conservation({"netlist": net, "freq_hz": freq, "proposed_node_voltages": {"n": value}})
        assert result["conservation_verified"]


@pytest.mark.parametrize("clause", ["DC 7 DC 9", "AC 2 90 AC 3", "DC AC 2", "DC 3.2.1", "DC 7 AC 2 90 44", "DC 7 SIN(0 1 60)"])
def test_ambiguous_or_unsupported_excitations_are_diagnosed(clause):
    net = f"Invalid source\nV1 n 0 {clause}\nR1 n 0 1\n.end"
    with pytest.raises(ValueError, match="independent source"):
        verify_circuit_conservation({"netlist": net})
