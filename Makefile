PYTHON ?= python3

.PHONY: verify-cd-nozzle-shock
verify-cd-nozzle-shock:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m unittest discover -s tests -p 'test_cd_nozzle_shock.py' -v

.PHONY: verify-multispan-contact
verify-multispan-contact:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m unittest discover -s tests -p 'test_multispan_contact.py' -v

.PHONY: verify-advanced-ifc
verify-advanced-ifc:
	$(PYTHON) ship/verify_advanced_ifc.py

.PHONY: verify-provider-wire
verify-provider-wire:
	$(PYTHON) ship/replay_engineering_provider_wire.py --reference

.PHONY: verify-wave8-history
verify-wave8-history:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_drained_layer_history.py tests/test_wave8_certified_budget.py tests/test_wave8_certified_envelopes.py tests/test_wave8_participant_handoff.py

.PHONY: verify-indexed-profiles
verify-indexed-profiles:
	$(PYTHON) -m unittest discover -s tests -p 'test_ifc_indexed_curves.py' -v

.PHONY: verify-distillation
verify-distillation:
	$(PYTHON) -m unittest discover -s tests -p 'test_distillation*.py' -v
.PHONY: verify-witness-equipment
verify-witness-equipment:
	$(PYTHON) -m unittest discover -s tests -p 'test_witness_equipment.py' -v

.PHONY: verify-pipe-meshes
verify-pipe-meshes:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_pipe_network.py tests/test_pipe_sparse_loops.py tests/test_pipe_sparse_reservoirs.py tests/test_fixed_head_exchange.py tests/test_pipe_minor_exchange.py

.PHONY: verify-drainage-front
verify-drainage-front:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_consolidation_front.py tests/test_layered_consolidation_transient.py

.PHONY: verify-workbook-units
verify-workbook-units:
	$(PYTHON) -m unittest discover -s tests -p 'test_xlsx*.py' -v

.PHONY: verify-xlsx-unit-graph
verify-xlsx-unit-graph:
	@set -eu; output=$$(mktemp -d); trap 'rm -rf -- "$$output"' EXIT; \
	$(PYTHON) ship/verify_xlsx_unit_graph.py --out "$$output" --min-memory-gib 4

.PHONY: verify-sequence-faults
verify-sequence-faults:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_symmetrical_components.py
