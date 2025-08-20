# Machine Learning Project – Natural Language Classifier  

## Overview  
This project ingests and processes text datasets (starting with Hugging Face’s **[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)**) into a **standard schema** for machine learning:  

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

---

## Project Structure  
```
nlc_emotion/
│
├── ingest.py              # Main entry script (command-line interface)
├── ingest.log             # Log file (created when you run ingest.py)
├── nlc_ingest/
│   ├── __init__.py
│   ├── cleaning.py        # clean_text, normalise_label, standardise
│   ├── config.py          # default paths, allowed formats
│   ├── io.py              # load_data, save_dataframe
│   └── ...
└── data/
    └── processed/         # processed dataset output
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
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  

Run the ingestion script from the command line:

```bash
python ingest.py [--dataset DATASET] [--format {csv,json}] [--out OUTPUT_PATH]
```

### Arguments
- `--dataset` → Hugging Face dataset ID (default: `dair-ai/emotion`)  
- `--format`  → Output file format (`csv` or `json`, default: `csv`)  
- `--out`     → Output file path (default: from `config.py`)  

### Examples  

#### Use defaults (Emotion dataset, CSV, default path):  
```bash
python ingest.py
```

#### Save as JSON instead of CSV:  
```bash
python ingest.py --format json --out data/processed/emotion.json
```

#### Process a completely different dataset (e.g., `imdb` sentiment data):  
```bash
python ingest.py --dataset imdb --format csv --out data/processed/imdb_clean.csv
```

---

## Logging  
When you run ingestion, logs are written to `ingest.log`.  
Example log flow:  
```
2025-08-20 14:12:01 INFO Loading dataset: dair-ai/emotion
2025-08-20 14:12:02 INFO Cleaning and standardising splits...
2025-08-20 14:12:03 INFO Combining 3 cleaned splits
2025-08-20 14:12:03 INFO Saving 20000 rows to data/processed/emotion_clean.csv (csv)
2025-08-20 14:12:04 INFO Ingest complete.
```

This helps track exactly what was processed and saved.

---

## Processed Data  
Processed files are saved in `data/processed/`.  
Schema:  

| text                                        | label    | split   |
|---------------------------------------------|----------|---------|
| "i didnt feel humiliated"                   | sadness  | train   |
| "i can go from hopeless to hopeful quickly" | joy      | train   |

---

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
