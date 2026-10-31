# One-command verification: `make all`.
#
# The raw parquet log is gitignored (~14 MB) and produced deterministically by
# `liquidity.generate` from SEED = 42 (generation takes ~1 minute). `make data`
# generates it only when a raw table is missing, so `make all` never overwrites a
# log whose numbers the tests were pinned against.

UV ?= uv

RAW_DIR := data/raw
RAW_TABLES := categories regions sellers listings searches impressions views contacts
RAW_FILES := $(addprefix $(RAW_DIR)/,$(addsuffix .parquet,$(RAW_TABLES)))

.PHONY: all data test lint

all: data test lint

data:
	@missing=""; \
	for f in $(RAW_FILES); do [ -f "$$f" ] || missing="$$missing $$f"; done; \
	if [ -n "$$missing" ]; then \
		echo "generating raw data (missing:$$missing)"; \
		$(UV) run python -m liquidity.generate; \
	else \
		echo "raw data present - skipping generator"; \
	fi

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
