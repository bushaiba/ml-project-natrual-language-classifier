venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

ingest:
	. .venv/bin/activate && python ingest.py --dataset dair-ai/emotion --format csv

test:
	. .venv/bin/activate && pytest -q
