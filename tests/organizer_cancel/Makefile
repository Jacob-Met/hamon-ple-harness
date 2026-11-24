# Run on a dedicated local Docker host with uv installed.
KIT_COMMIT := 8320f0402fdca9f42476607f72bcfd5271cd50c9
HARNESS := $(abspath ../..)

.PHONY: verify-organizer-cancel
verify-organizer-cancel:
	@set -eu; \
	command -v uv >/dev/null; \
	docker info --format '{{.OSType}} {{.NCPU}}'; \
	work=$$(mktemp -d); \
	trap 'rm -rf "$$work"' EXIT; \
	git clone -q --filter=blob:none https://github.com/google-research/ple.git "$$work/organizer"; \
	git -C "$$work/organizer" checkout -q --detach $(KIT_COMMIT); \
	cd "$$work/organizer"; \
	uv run python "$(HARNESS)/tests/mock_sse_proxy.py" "$$work/organizer"
