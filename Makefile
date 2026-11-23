PYTHON ?= python3

.PHONY: verify-distillation verify-pfr
verify-distillation:
	$(PYTHON) -m unittest discover -s tests -p 'test_distillation*.py' -v

verify-pfr:
	$(PYTHON) -m unittest discover -s tests -p 'test_pfr*.py' -v
.PHONY: verify-witness-equipment
verify-witness-equipment:
	$(PYTHON) -m unittest discover -s tests -p 'test_witness_equipment.py' -v
