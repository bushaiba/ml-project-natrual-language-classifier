venv:
	python3 -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

ingest:
	python ingest.py --dataset dair-ai/emotion --format csv

test:
	pytest -q
