# One-command verification: `make all`.
#
# The raw parquet log is gitignored (~10 MB) and produced deterministically by
# `liquidity.generate` from SEED = 42 (generation takes seconds). `make data`
# regenerates when a raw table is missing *or* when the generator's own source has
# changed since the last run: existence alone used to be the whole guard, so editing
# the DGP left `make all` testing stale parquet.

UV ?= uv

RAW_DIR := data/raw
GEN_STAMP := $(RAW_DIR)/.generator-stamp
RAW_TABLES := categories regions sellers listings searches impressions views contacts
RAW_FILES := $(addprefix $(RAW_DIR)/,$(addsuffix .parquet,$(RAW_TABLES)))
GENERATOR_SOURCES := liquidity/generate.py liquidity/config.py

.PHONY: all data test lint clean

all: data test lint

data:
	@stamp="$(GEN_STAMP)"; \
	if command -v shasum >/dev/null 2>&1; then hash=shasum; else hash=sha256sum; fi; \
	want="$$($$hash $(GENERATOR_SOURCES) | $$hash | cut -d' ' -f1)"; \
	have="$$(cat "$$stamp" 2>/dev/null || echo none)"; \
	missing=""; \
	for f in $(RAW_FILES); do [ -f "$$f" ] || missing="$$missing $$f"; done; \
	if [ -n "$$missing" ] || [ "$$want" != "$$have" ]; then \
		echo "generating raw data (missing:$${missing:- none}; generator changed: $$([ "$$want" != "$$have" ] && echo yes || echo no))"; \
		$(UV) run python -m liquidity.generate; \
		mkdir -p "$$(dirname "$$stamp")"; \
		echo "$$want" > "$$stamp"; \
	else \
		echo "raw data present, generator unchanged - skipping"; \
	fi

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

clean:
	rm -rf $(RAW_DIR) data/staging data/marts .pytest_cache .ruff_cache
