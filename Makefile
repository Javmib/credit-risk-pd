.PHONY: install run fast test
install:
	pip install -r requirements.txt
run:
	python scripts/run_pipeline.py
fast:
	python scripts/run_pipeline.py --fast
test:
	python -m pytest -q
