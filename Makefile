PYTHON ?= python3

.PHONY: verify-multispan-contact
verify-multispan-contact:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m unittest discover -s tests -p 'test_multispan_contact.py' -v

.PHONY: verify-provider-wire
verify-provider-wire:
	$(PYTHON) ship/replay_engineering_provider_wire.py --reference

.PHONY: verify-wave8-history
verify-wave8-history:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_drained_layer_history.py tests/test_wave8_certified_budget.py tests/test_wave8_certified_envelopes.py tests/test_wave8_participant_handoff.py

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
