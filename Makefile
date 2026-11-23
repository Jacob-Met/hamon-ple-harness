PYTHON ?= python3

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
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_pipe_network.py tests/test_pipe_sparse_loops.py
