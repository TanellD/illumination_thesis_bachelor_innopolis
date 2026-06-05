##############################################################################
# Makefile — thesis pipeline
#
# Prerequisites: Python 3.9+, PyTorch 2.2.2+cu12x installed separately,
#                then: pip install -e ".[dev]"
#
# Override output root:   export THESIS_OUTPUT_ROOT=/path/to/outputs
# Override data roots:    export THESIS_FFPP_ROOT=/data/faceforensics  etc.
# All other paths in:     paths.yaml
##############################################################################

PYTHON   := python
PYTEST   := pytest
CONFIGS  := configs
OUTPUTS  := outputs

# Pass FORCE=1 to re-run a stage even if its outputs already exist.
# Example:  make b1 FORCE=1
#           make train FORCE=1
_FORCE   := $(if $(FORCE),--force,)

.DEFAULT_GOAL := help

.PHONY: help tiny full test test-all dry-run lint format clean clean-tiny \
        corpus-fixture \
        download-ffpp download-ffpp-full \
        crop noise-pre \
        sbi tsbi manifests \
        b1 b2 b3 b4 \
        train dl-quartile shortcuts \
        eval robustness taxonomy report

##############################################################################
# Help
##############################################################################

help:
	@echo ""
	@echo "  thesis pipeline — available targets"
	@echo ""
	@echo "  Quickstart:"
	@echo "    make corpus-fixture   Build 16-crop tiny corpus (needed once)"
	@echo "    make test             Unit tests only  (~10 s, CPU)"
	@echo "    make test-all         Unit + integration tests"
	@echo "    make tiny             Smoke-test manifests + report on tiny corpus"
	@echo "    make dry-run          Show full pipeline DAG without running"
	@echo "    make full             Run everything end-to-end (~270 GPU-hours)"
	@echo ""
	@echo "  Data download (run once, requires FF++ access):"
	@echo "    make download-ffpp    Download FF++ sample (NUM_VIDEOS=100, COMPRESSION=c23)"
	@echo "    make download-ffpp-full  Download full FF++ dataset"
	@echo "    Customise: make download-ffpp DATA_DIR=/my/path NUM_VIDEOS=500 SERVER=EU2"
	@echo ""
	@echo "  Data preparation (A):"
	@echo "    make crop             A.2  MTCNN extraction — all 4 datasets"
	@echo "    make noise-pre        A.5  Noiseprint++ noise-map precomputation"
	@echo ""
	@echo "  Stage 1 — noise-residual channel (B):"
	@echo "    make b1               B.1  Three-model ablation (5 seeds)"
	@echo "    make b2               B.2  Seven-level bottleneck diagnostic"
	@echo "    make b3               B.3  Context-crop experiment (1.3x vs 2.7x)"
	@echo "    make b4               B.4  StatNoise-Fusion + ResAware-Fusion (5 seeds)"
	@echo ""
	@echo "  Stage 2 — T-SBI (C):"
	@echo "    make sbi              C.1  Classic SBI generation"
	@echo "    make tsbi             C.1  T-SBI generation"
	@echo "    make manifests        C.2+C.3  Regime manifest assembly"
	@echo "    make train            C.3  Five-regime training (A B_pure B_mix C_pure C_mix)"
	@echo "    make dl-quartile      C.4  HIGHDL / LOWDL multi-seed diagnostic"
	@echo "    make shortcuts        C.5  Shortcut ablation (N0-N6, P0-P5)"
	@echo ""
	@echo "  Shared evaluation (D-G):"
	@echo "    make eval             D    Evaluation — all (model, dataset) pairs"
	@echo "    make robustness       E    Perturbation grid"
	@echo "    make taxonomy         F    Failure taxonomy (attributes + stats)"
	@echo "    make report           G    Tables T1-T10, figures, RESULTS.md"
	@echo ""
	@echo "  Force re-run (skip output-exists check):"
	@echo "    make <target> FORCE=1"
	@echo "    Examples:"
	@echo "      make b1 FORCE=1              Re-run Stage 1 ablation from scratch"
	@echo "      make train FORCE=1           Re-run all five regimes"
	@echo "      make train-regime REGIME=C_mix FORCE=1   Re-run one regime only"
	@echo "      make noise-pre FORCE=1       Re-extract all noise maps"
	@echo ""
	@echo "  Dev:"
	@echo "    make lint             ruff + mypy"
	@echo "    make format           ruff format"
	@echo "    make clean            Remove outputs/ and __pycache__"
	@echo "    make clean-tiny       Remove outputs/tiny/ only"
	@echo ""

##############################################################################
# Tiny-corpus smoke test  (make tiny)
# Builds the fixture if needed, runs a subset of stages that work without GPU
# (manifest assembly + report layer).  Confirms the plumbing is wired.
##############################################################################

corpus-fixture:
	@echo ">>> Building 16-crop tiny corpus fixture"
	$(PYTHON) tests/fixtures/build_tiny_corpus.py \
		--out tests/fixtures/tiny_corpus/

tests/fixtures/tiny_corpus/manifest.csv: corpus-fixture

tiny: tests/fixtures/tiny_corpus/manifest.csv
	@echo ">>> Tiny pipeline: manifest assembly + report skeleton"
	$(PYTHON) experiments/a5_noise_precompute.py \
		--config $(CONFIGS)/data/ffpp.yaml \
		--manifest tests/fixtures/tiny_corpus/manifest.csv \
		--output-dir $(OUTPUTS)/tiny/noise \
		--tiny
	$(PYTHON) experiments/c2_build_manifests.py \
		--ff-csv  tests/fixtures/tiny_corpus/manifest.csv \
		--out-dir $(OUTPUTS)/tiny/manifests
	$(PYTHON) experiments/g_report.py \
		--config $(CONFIGS)/eval/report.yaml
	@echo ""
	@echo ">>> Tiny complete.  No GPU needed — GPU steps skipped."
	@echo "    For full GPU smoke test: run b1/train with --tiny flag manually."

##############################################################################
# Full pipeline  (make full)
# Runs all experiments in dependency order.
##############################################################################

full: crop noise-pre sbi tsbi manifests b1 b2 b3 b4 train dl-quartile shortcuts eval robustness taxonomy report
	@echo ""
	@echo ">>> Full pipeline complete.  See $(OUTPUTS)/RESULTS.md"

##############################################################################
# Data download (run once before anything else)
##############################################################################

# Download FF++ dataset (requires accepting TOS interactively).
# Adjust DATA_DIR, NUM_VIDEOS, and COMPRESSION as needed.
# After download, set THESIS_FFPP_ROOT to the output directory.
DATA_DIR   ?= faceforensics
NUM_VIDEOS ?= 50
COMPRESSION ?= c23
SERVER     ?= EU2

download-ffpp:
	@echo ">>> Downloading FF++ (-n $(NUM_VIDEOS) videos, $(COMPRESSION), server=$(SERVER))"
	@echo "    Output: $(DATA_DIR)"
	$(PYTHON) src/data/download_ffpp.py $(DATA_DIR) \
		-d all \
		-c $(COMPRESSION) \
		-t videos \
		-n $(NUM_VIDEOS) \
		--server $(SERVER)
	@echo ""
	@echo ">>> Done.  Set THESIS_FFPP_ROOT=$(DATA_DIR) before running make crop."

download-ffpp-full:
	@echo ">>> Downloading FULL FF++ dataset ($(COMPRESSION), server=$(SERVER))"
	$(PYTHON) src/data/download_ffpp.py $(DATA_DIR) \
		-d all \
		-c $(COMPRESSION) \
		-t videos \
		--server $(SERVER)

##############################################################################
# A — Data preparation
##############################################################################

crop:
	@echo ">>> A.2  MTCNN face extraction — FF++"
	$(PYTHON) experiments/a2_crop.py --config $(CONFIGS)/data/ffpp.yaml $(_FORCE)
	@echo ">>> A.2  MTCNN face extraction — Celeb-DF"
	$(PYTHON) experiments/a2_crop.py --config $(CONFIGS)/data/celebdf.yaml $(_FORCE)
	@echo ">>> A.2  MTCNN face extraction — DFDC"
	$(PYTHON) experiments/a2_crop.py --config $(CONFIGS)/data/dfdc.yaml $(_FORCE)
	@echo ">>> A.2  MTCNN face extraction — DFF"
	$(PYTHON) experiments/a2_crop.py --config $(CONFIGS)/data/dff.yaml $(_FORCE)

noise-pre:
	@echo ">>> A.5  Noiseprint++ noise-map precomputation — FF++"
	$(PYTHON) experiments/a5_noise_precompute.py \
		--config $(CONFIGS)/data/ffpp.yaml $(_FORCE)
	@echo ">>> A.5  Noiseprint++ noise-map precomputation — Celeb-DF"
	$(PYTHON) experiments/a5_noise_precompute.py \
		--config $(CONFIGS)/data/celebdf.yaml $(_FORCE)
	@echo ">>> A.5  Noiseprint++ noise-map precomputation — DFDC"
	$(PYTHON) experiments/a5_noise_precompute.py \
		--config $(CONFIGS)/data/dfdc.yaml $(_FORCE)
	@echo ">>> A.5  Noiseprint++ noise-map precomputation — DFF"
	$(PYTHON) experiments/a5_noise_precompute.py \
		--config $(CONFIGS)/data/dff.yaml $(_FORCE)

##############################################################################
# C.1 — SBI / T-SBI generation
##############################################################################

sbi:
	@echo ">>> C.1  Classic SBI generation"
	$(PYTHON) experiments/c1_generate_sbi.py \
		--config  $(CONFIGS)/stage2/tsbi.yaml \
		--out-dir $(OUTPUTS)/sbi \
		--out-csv $(OUTPUTS)/sbi_labels.csv $(_FORCE)

tsbi:
	@echo ">>> C.1  T-SBI generation"
	$(PYTHON) experiments/c1_generate_tsbi.py \
		--config  $(CONFIGS)/stage2/tsbi.yaml \
		--out-dir $(OUTPUTS)/tsbi \
		--out-csv $(OUTPUTS)/tsbi_labels.csv $(_FORCE)

##############################################################################
# C.2+C.3 — Manifest assembly
##############################################################################

manifests:
	@echo ">>> C.2+C.3  Regime manifest assembly"
	$(PYTHON) experiments/c2_build_manifests.py \
		--ff-csv   $(OUTPUTS)/crops/ff++/manifest.csv \
		--sbi-csv  $(OUTPUTS)/sbi_labels.csv \
		--tsbi-csv $(OUTPUTS)/tsbi_labels.csv \
		--out-dir  $(OUTPUTS)/manifests

##############################################################################
# B — Stage 1 experiments
##############################################################################

b1:
	@echo ">>> B.1  Three-model ablation (5 seeds)"
	$(PYTHON) experiments/b1_ablation.py \
		--config $(CONFIGS)/stage1/b1_ablation.yaml $(_FORCE)

b2:
	@echo ">>> B.2  Seven-level bottleneck diagnostic"
	$(PYTHON) experiments/b2_bottleneck.py \
		--config $(CONFIGS)/stage1/b2_bottleneck.yaml $(_FORCE)

b3:
	@echo ">>> B.3  Context-crop experiment (1.3x tight vs 2.7x context)"
	$(PYTHON) experiments/b3_context_crop.py \
		--config $(CONFIGS)/stage1/b3_context_crop.yaml $(_FORCE)

b4:
	@echo ">>> B.4  Fixed-fusion variants — StatNoise + ResAware (5 seeds each)"
	$(PYTHON) experiments/b4_fixed_fusion.py \
		--config $(CONFIGS)/stage1/b4_fixed_fusion.yaml $(_FORCE)

##############################################################################
# C.3 — Stage 2 training
##############################################################################

train:
	@echo ">>> C.3  Five-regime training (A B_pure B_mix C_pure C_mix)"
	$(PYTHON) experiments/c3_train_regimes.py \
		--config $(CONFIGS)/stage2/c3_train_regimes.yaml $(_FORCE)

# Run a single regime (e.g. make train-regime REGIME=C_mix)
train-regime:
	$(PYTHON) experiments/c3_train_regimes.py \
		--config $(CONFIGS)/stage2/c3_train_regimes.yaml \
		--regime $(REGIME) $(_FORCE)

##############################################################################
# C.4 — dL-quartile multi-seed diagnostic
##############################################################################

dl-quartile:
	@echo ">>> C.4  dL-quartile diagnostic (HIGHDL / LOWDL, 3 seeds)"
	$(PYTHON) experiments/c4_dl_quartile.py \
		--config $(CONFIGS)/stage2/c4_dl_quartile.yaml $(_FORCE)

##############################################################################
# C.5 — Shortcut ablation
##############################################################################

shortcuts:
	@echo ">>> C.5  Shortcut ablation (N0-N6 T-SBI, P0-P5 SBI)"
	$(PYTHON) experiments/c5_shortcuts.py \
		--config $(CONFIGS)/stage2/c5_shortcuts.yaml $(_FORCE)

##############################################################################
# D — Evaluation
##############################################################################

eval:
	@echo ">>> D  Evaluation — all (model, dataset) pairs"
	$(PYTHON) experiments/d_eval.py \
		--config $(CONFIGS)/eval/d_eval.yaml $(_FORCE)

# Evaluate a single checkpoint (e.g. make eval-one CONFIG=configs/eval/d_eval.yaml)
eval-one:
	$(PYTHON) experiments/d_eval.py \
		--config $(CONFIG) --force

##############################################################################
# E — Robustness perturbation grid
##############################################################################

robustness:
	@echo ">>> E  Robustness perturbation grid"
	$(PYTHON) experiments/e_robustness.py \
		--config $(CONFIGS)/eval/e_robustness.yaml $(_FORCE)

##############################################################################
# F — Failure taxonomy
##############################################################################

taxonomy:
	@echo ">>> F  Failure taxonomy (attributes + chi-square + Jaccard)"
	$(PYTHON) experiments/f_taxonomy.py \
		--config $(CONFIGS)/eval/f_taxonomy.yaml $(_FORCE)

# Attribute extraction only (CPU, no GPU, no inference cache needed)
taxonomy-attrs:
	@echo ">>> F.1  Per-frame attribute computation only"
	$(PYTHON) experiments/f_taxonomy.py \
		--config $(CONFIGS)/eval/f_taxonomy.yaml \
		--attrs-only $(_FORCE)

##############################################################################
# G — Report
##############################################################################

report:
	@echo ">>> G  Tables T1-T10, figures, RESULTS.md"
	$(PYTHON) experiments/g_report.py \
		--config $(CONFIGS)/eval/report.yaml $(_FORCE)

# Report from a specific prior run (e.g. to regenerate figures without rerunning eval)
report-force:
	$(PYTHON) experiments/g_report.py \
		--config $(CONFIGS)/eval/report.yaml \
		--force

##############################################################################
# Dry run — show pipeline DAG without executing
##############################################################################

dry-run:
	@echo ""
	@echo ">>> Full pipeline DAG (make full)"
	@echo ""
	@echo "  Step 1   make crop          A.2  MTCNN extraction — 4 datasets      ~6 h GPU"
	@echo "  Step 2   make noise-pre     A.5  Noiseprint++ precompute             ~8 h GPU"
	@echo "  Step 3   make sbi           C.1  Classic SBI generation              ~3 h GPU"
	@echo "  Step 4   make tsbi          C.1  T-SBI generation                    ~5 h GPU"
	@echo "  Step 5   make manifests     C.2  Regime manifest assembly            <5 min CPU"
	@echo "  ┌───────────── can run in parallel on separate GPUs ─────────────────┐"
	@echo "  │ Step 6   make b1          B.1  3-model ablation  5 seeds           ~40 h GPU"
	@echo "  │ Step 7   make b2          B.2  Bottleneck diagnostic                ~1 h CPU"
	@echo "  │ Step 8   make b3          B.3  Context-crop                         ~2 h GPU"
	@echo "  │ Step 9   make b4          B.4  StatNoise+ResAware  5 seeds         ~30 h GPU"
	@echo "  │ Step 10  make train       C.3  5-regime training                  ~120 h GPU"
	@echo "  └────────────────────────────────────────────────────────────────────┘"
	@echo "  Step 11  make dl-quartile   C.4  HIGHDL/LOWDL  3 seeds              ~15 h GPU"
	@echo "  Step 12  make shortcuts     C.5  Shortcut ablation                   ~3 h GPU"
	@echo "  Step 13  make eval          D    All (model × dataset) pairs          ~6 h GPU"
	@echo "  Step 14  make robustness    E    Perturbation grid                   ~20 h GPU"
	@echo "  Step 15  make taxonomy      F    Failure taxonomy                     ~2 h CPU"
	@echo "  Step 16  make report        G    Tables, figures, RESULTS.md         <10 min CPU"
	@echo ""
	@echo "  Total estimate:  ~260 GPU-hours + ~20 CPU-hours"
	@echo "  Steps 6-10 are independent — run them in parallel on separate GPUs to"
	@echo "  reduce wall-clock time to ~120 h."
	@echo ""
	@echo "  Config files used:"
	@echo "    A.2   configs/data/{ffpp,celebdf,dfdc,dff}.yaml"
	@echo "    A.5   configs/data/{ffpp,celebdf}.yaml"
	@echo "    B.1   configs/stage1/b1_ablation.yaml"
	@echo "    B.2   configs/stage1/b2_bottleneck.yaml"
	@echo "    B.3   configs/stage1/b3_context_crop.yaml"
	@echo "    B.4   configs/stage1/b4_fixed_fusion.yaml"
	@echo "    C.1   configs/stage2/tsbi.yaml"
	@echo "    C.3   configs/stage2/c3_train_regimes.yaml"
	@echo "    C.4   configs/stage2/c4_dl_quartile.yaml"
	@echo "    C.5   configs/stage2/c5_shortcuts.yaml"
	@echo "    D     configs/eval/d_eval.yaml"
	@echo "    E     configs/eval/e_robustness.yaml"
	@echo "    F     configs/eval/f_taxonomy.yaml"
	@echo "    G     configs/eval/report.yaml"
	@echo ""

##############################################################################
# Tests
##############################################################################

test:
	$(PYTEST) tests/unit/ -v

test-all: tests/fixtures/tiny_corpus/manifest.csv
	$(PYTEST) tests/ -v

##############################################################################
# Code quality
##############################################################################

lint:
	ruff check src/ experiments/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ experiments/ tests/

##############################################################################
# Cleanup
##############################################################################

clean:
	@echo ">>> Removing outputs/ and __pycache__"
	rm -rf $(OUTPUTS)/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-tiny:
	rm -rf $(OUTPUTS)/tiny/
