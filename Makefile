PYTHON ?= python3

.PHONY: verify-distillation
verify-distillation:
	$(PYTHON) -m unittest discover -s tests -p 'test_distillation*.py' -v
