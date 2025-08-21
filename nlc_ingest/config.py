from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = PROCESSED_DIR / "emotion_clean.csv"

ALLOWED_FORMATS = {"csv", "json"}

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "emotion_model.pkl"
