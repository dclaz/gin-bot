# gin-bot Makefile — Phase 0 bootstraps this tooling (IMPLEMENTATION_PLAN.md).
# Gates for phases not yet reached exit non-zero with "not implemented", never 0.

export PYTORCH_ENABLE_MPS_FALLBACK=1

UV := uv run
PROBE := scripts/probe_env.py

.PHONY: setup probe facts-check test lint status board elo
.PHONY: gate-p0 gate-p1 gate-p2 gate-p3 gate-p4 gate-p5 gate-p6 gate-p7 gate-all

setup:
	uv sync
	$(MAKE) probe

probe:
	$(UV) python $(PROBE)

facts-check:
	$(UV) python $(PROBE) --check

test:
	$(UV) pytest -q

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

gate-p0: lint test
	$(UV) python scripts/gate_p0.py

gate-p1: lint test
	$(UV) python scripts/gate_p1.py

gate-p2: lint test
	$(UV) python scripts/gate_p2.py

gate-p3: lint test
	$(UV) python scripts/gate_p3.py

gate-p4:
	@echo "not implemented (Phase 4 — reduced gin rummy)"; exit 1

gate-p5:
	@echo "not implemented (Phase 5 — full game)"; exit 1

gate-p6:
	@echo "not implemented (Phase 6 — style study)"; exit 1

gate-p7:
	@echo "not implemented (Phase 7 — rules sweep and write-up)"; exit 1

gate-all: gate-p0 gate-p1 gate-p2 gate-p3 gate-p4 gate-p5 gate-p6 gate-p7

status:
	$(UV) python scripts/status.py

board:
	$(UV) trackio show --project ginrl

elo:
	$(UV) python scripts/elo.py
