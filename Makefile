# Reproduce the fraud-detection stacking pipeline end to end.
# Override the interpreter if you are not in an activated venv:
#   make all PYTHON=.venv/bin/python
PYTHON ?= python

.PHONY: all reproduce install data prepare base select stack meta-mlp eval test clean

all: prepare base select stack ## prepare data -> train base models -> select (val) -> stack + eval (test once)

reproduce: install data all meta-mlp ## full clean-clone reproduction

install: ## editable install of the `fd` package + dev tools
	$(PYTHON) -m pip install -e ".[dev]"

data: ## download creditcard.csv via kagglehub (no-op if present)
	$(PYTHON) scripts/download_data.py

prepare: ## build time-ordered train/val/test splits
	$(PYTHON) scripts/prepare_data.py

base: ## train every layer-0 base model into models/<name>/model.joblib
	$(PYTHON) scripts/ensamble_models.py
	$(PYTHON) scripts/train_layer0_mlp.py
	$(PYTHON) scripts/train_lightgbm.py
	$(PYTHON) scripts/isolation_forest.py

select: ## rank base learners by VALIDATION AUPRC -> reports/layer0.txt
	$(PYTHON) scripts/select_layer0.py

stack: ## train meta-LR on validation, then score test exactly once
	$(PYTHON) scripts/train_stack.py
	$(PYTHON) scripts/eval_stack.py

meta-mlp: ## optional compact-MLP meta-model
	$(PYTHON) scripts/train_stack_mlp.py --option B_wLGBM

eval: ## re-score the frozen stack on test (test-once evaluator)
	$(PYTHON) scripts/eval_stack.py

test: ## run the test suite
	$(PYTHON) -m pytest -q

clean: ## remove trained models and generated outputs (keeps data + manifest)
	rm -rf models outputs/stack_* outputs/*/preds
