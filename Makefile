PYTHON ?= python3

.PHONY: verify-distillation
verify-distillation:
	$(PYTHON) -m unittest discover -s tests -p 'test_distillation*.py' -v
.PHONY: verify-witness-equipment
verify-witness-equipment:
	$(PYTHON) -m unittest discover -s tests -p 'test_witness_equipment.py' -v
