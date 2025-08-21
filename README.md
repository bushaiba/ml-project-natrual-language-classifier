# Machine Learning Project – Natural Language Classifier  

## Overview  
This project ingests, processes, and trains machine learning models on text datasets (starting with Hugging Face’s **[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)**).  

The standard schema is:  
```
text | label | split
```

The ingestion pipeline:  
1. Loads dataset splits (train/validation/test) from Hugging Face.  
2. Cleans text (removes extra spaces, trims).  
3. Normalises labels (lowercased, consistent spacing).  
4. Drops empty rows and duplicates.  
5. Combines all splits into one DataFrame.  
6. Saves the processed dataset to CSV or JSON.  

The training pipeline:  
1. Loads processed dataset.  
2. Splits into train/validation sets (either random or by `split` column).  
3. Builds a pipeline: **TF-IDF Vectorizer + Logistic Regression**.  
4. Trains the model.  
5. Evaluates on validation set (accuracy, precision, recall, F1, confusion matrix).  
6. Saves trained model as `.pkl`.  

---

## Project Structure  
```
NLC_EMOTION/
│── data/
│   └── processed/emotion_clean.csv
│── models/
│   └── emotion_model.pkl
│── nlc_ingest/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── config.py
│   └── io.py
│── src/
│   ├── emotion_classifier.py
|   ├── ingest.py
│   ├── simple_interface.py
│   └── train_model.py
│── logs/
│   ├── train.log
│   ├── interface.log
│   └── ingest.log
│── reports/
│   └── figures/confusion_matrix.png
│── tests/
│── venv/
│── Makefile
│── README.md
│── requirements.txt
│── .gitignore

```

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/bushaiba/ml-project-natural-language-classifier.git
   cd ml-project-natural-language-classifier
   ```

2. Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  

### Ingest Data
Run the ingestion script from the command line:
```bash
python ingest.py --dataset (name of dataset) --format (csv or json) --out (path to output file)
```

Arguments:  
- `--dataset` → Hugging Face dataset ID (default: `dair-ai/emotion`)  
- `--format`  → Output file format (`csv` or `json`, default: `csv`)  
- `--out`     → Output file path (default: from `config.py`)  

Examples:  
```bash
python ingest.py
python ingest.py --format json --out data/processed/emotion.json
python ingest.py --dataset imdb --format csv --out data/processed/imdb_clean.csv
```

### Train Model
After ingestion, train the classifier with:  
```bash
python train_model.py --input data/processed/emotion_clean.csv --use_split_column --model_out models/emotion_logreg_tfidf.pkl
```

Arguments:  
- `--input` → Path to processed dataset  
- `--model_out` → Where to save the trained `.pkl` model  
- `--use_split_column` → Use dataset’s `split` column if available  
- `--val_size` → Validation fraction for random split (default: 0.2)  
- `--random_state` → Random seed (default: 42)  

Alternatively, you can use the **Makefile** shortcut:  
```bash
make ingest     # run ingestion with defaults
make train      # train model and save to models/
make interface  # start CLI interface
```

### 3. Run Classifier Interface  
```bash
python simple_interface.py
```
Instructions shown in console:  
- Type a sentence → get label (`joy`, `anger`, `fear`, etc.)  
- Type `clear` → clear screen  
- Type `exit`/`quit`/`q`/`x` → exit  

---

---

## Logging  
- `ingest.log` → ingestion process  
- `train.log` → training + evaluation  
- `interface.log` → interface runtime 

Training log example:  
```
2025-08-20 19:01:22 INFO Loading dataset from data/processed/emotion_clean.csv
2025-08-20 19:01:31 INFO Training model...
2025-08-20 19:01:51 INFO Accuracy: 0.8280
2025-08-20 19:01:53 INFO Confusion Matrix:
[[206  5  28   1  35   0]
 [  4 139  36   0  9   3]
 ...]
2025-08-20 19:01:53 INFO Model saved to: models/emotion_logreg_tfidf.pkl
2025-08-20 19:01:53 INFO Done.
```

---

## Processed Data  
Processed files are saved in `data/processed/`.  
Schema:  

| text                                        | label    | split   |
|---------------------------------------------|----------|---------|
| "i didnt feel humiliated"                   | sadness  | train   |
| "i can go from hopeless to hopeful quickly" | joy      | train   |

---



### Emotion Classifier (Interactive Predictions)
After training, you can use the provided `emotion_classifier.py` to interact with the saved model.  

Example usage:  
```bash
python emotion_classifier.py
```

This script:  
- Loads the trained model from `models/`.  
- Cleans and preprocesses input text.  
- Provides a key method:  
  - **classify(text)** → returns the single best predicted label.   

Example output:  
```
Single prediction: joy

```


## Citation  

If you use the **Emotion dataset**, cite:  

```bibtex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697"
}
```
