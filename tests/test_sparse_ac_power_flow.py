"""Sparse current-balance boundary cases and comparison-source isolation."""
import cmath
import math
import sys
from types import ModuleType

import numpy as np
import pytest

from hamon_ac_power_flow import verify_ac_power_flow
from hamon_ac_sparse import dc_angle_seed
from ship.verify_ac_regulation import load


@pytest.mark.parametrize(
    ("bus_type", "resistance", "reactance"),
    [("PQ", 0.1, 0.0), ("PQ", 0.02, 0.09), ("PV", 0.02, 0.09)],
    ids=["resistive_pq", "no_pv_buses", "no_pq_buses"],
)
def test_sparse_star_preserves_every_terminal(bus_type, resistance, reactance):
    slack_angle = 31.0
    buses = [{"id": "source", "type": "slack", "vm_pu": 1.0, "va_deg": slack_angle}]
    lines = []
    for index in range(128):
        name = str(index)
        bus = {"id": name, "type": bus_type, "p_injection_pu": -0.01}
        if bus_type == "PQ":
            bus["q_injection_pu"] = -0.005
        else:
            bus["vm_pu"] = 1.0
        buses.append(bus)
        lines.append({"id": name, "from": "source", "to": name,
                      "r_pu": resistance, "x_pu": reactance})

    answer = verify_ac_power_flow({"buses": buses, "lines": lines})
    assert answer["conservation_verified"], answer.get("reason")
    source = cmath.rect(1.0, math.radians(slack_angle))
    assert len(answer["voltages"]) == 129
    assert len(answer["lines"]) == 128
    for bus in buses[1:]:
        name = bus["id"]
        row = answer["voltages"][name]
        voltage = cmath.rect(row["vm_pu"], math.radians(row["va_deg"]))
        current = (voltage - source) / complex(resistance, reactance)
        injection = voltage * current.conjugate()
        assert injection.real == pytest.approx(bus["p_injection_pu"], abs=1e-9)
        if bus_type == "PQ":
            assert injection.imag == pytest.approx(bus["q_injection_pu"], abs=1e-9)
        else:
            assert abs(voltage) == pytest.approx(bus["vm_pu"], abs=1e-10)
        line = answer["lines"][name]
        assert complex(line["p_to_pu"], line["q_to_pu"]) == pytest.approx(injection, abs=1e-9)
        assert complex(line["p_from_pu"], line["q_from_pu"]) == pytest.approx(
            -source * current.conjugate(), abs=1e-9
        )


@pytest.mark.parametrize(
    "impedances", [[0.1 + 0j], [0.25 + 1j, 0.25 - 1j]],
    ids=["zero_reactance", "singular_signed_laplacian"],
)
def test_unavailable_dc_estimate_returns_no_seed(impedances):
    # Opposing reactances cancel exactly in the DC model. Positive resistance
    # keeps their parallel AC admittance nonzero.
    count = len(impedances)
    seed = dc_angle_seed(
        initial=np.array([1.0, 0.0]),
        unpack=lambda x: np.array([1.0 + 0j, x[0] + 1j * x[1]]),
        nonslack=np.array([1]), p=np.array([0.0, -0.01]), shunt=np.zeros(2, complex),
        f=np.zeros(count, int), t=np.ones(count, int), y=1 / np.array(impedances),
        tap=np.ones(count, complex),
    )
    assert seed is None


def test_comparison_loader_uses_each_sources_own_helper(tmp_path, monkeypatch):
    helper_name = "hamon_ac_sparse"
    cached = ModuleType(helper_name)
    cached.revision = "unrelated"
    monkeypatch.setitem(sys.modules, helper_name, cached)
    # The loader registers its target module. Let pytest restore the test name.
    module_name = "_sparse_ac_test_candidate"
    monkeypatch.setitem(sys.modules, module_name, None)
    loaded = []
    for revision in ("historic", "current", "missing"):
        directory = tmp_path / revision
        directory.mkdir()
        source = directory / "solver.py"
        source.write_text("from hamon_ac_sparse import revision\n")
        if revision == "missing":
            with pytest.raises(ImportError, match="hamon_ac_sparse"):
                load(module_name, source)
        else:
            (directory / "hamon_ac_sparse.py").write_text(f"revision = {revision!r}\n")
            loaded.append(load(module_name, source))
        assert sys.modules[helper_name] is cached
    assert [module.revision for module in loaded] == ["historic", "current"]
