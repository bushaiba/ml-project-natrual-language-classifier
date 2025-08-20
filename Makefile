venv:
	python3 -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

ingest:
	python ingest.py --dataset dair-ai/emotion --format csv

test:
	pytest -q
	
train:
	python train_model.py --input data/processed/emotion_clean.csv --use_split_column --model_out models/emotion_logreg_tfidf.pkl
