import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def setup_logging_for_model():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="train.log",
        filemode="w",
    )


def parse_args_for_model():
    parser = argparse.ArgumentParser(description="Train a baseline text classifier.")
    parser.add_argument("--input", default="data/processed/emotion_clean.csv",
                        help="Path to cleaned CSV (Columns: text, label, split).")
    parser.add_argument("--model_out", default="models/emotion_model.pkl",
                        help="Where to save the trained model as pickle file (.pkl).")
    parser.add_argument("--use_split_column", action="store_true",
                        help="Use the 'split' column (train/validation/test) if present.")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Validation fraction if random split is used.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def load_dataset_csv(path_str: str) -> pd.DataFrame:
    logging.info("Loading dataset from %s", path_str)
    df = pd.read_csv(path_str)

    # ensure dataset has 'text' and 'label'
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns.")
    return df

def split_data(df: pd.DataFrame, use_split: bool, val_size: float, seed:int):
    """ 
    Returns X_train, y_train, X_val, y_val
    If use_split=True and 'split' column exists, use it (train vs validation/test)
    Otherwise, do a random stratified split [stratified split makes sure the proportions stay the same]
    """
    if use_split and "split" in df.columns:
        has_validation = (df["split"]) == "validation".any()
        # it sets has_validation to True if at least one row is "validation", otherwise False

        if has_validation:
            train_df = df[df["split"] == "train"]
            val_df = df[df["split"] == "validation"]
        else:
            train_df = df[df["split"] == "train"]
            val_df = df[df["split"] == "test"]
        if  len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("Split column present but train/validation/test are empty.")

        # .tolist() converts pandas Series/columns into a plain Python list
        X_train = train_df["text"].tolist()
        y_train = train_df["label"].tolist()
        
        X_val = val_df["text"].tolist()
        y_val = val_df["label"].tolist()
        logging.info("Using provided split column: train=%d, val=%d", len(X_train), len(X_val))
        
        return X_train, y_train, X_val, y_val
    
    # random split as a fallback mechanism
    X = df["text"].tolist()
    y = df["label"].tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    logging.info("Using random split: train=%d, val=%d (val_side=%.2f)", len(X_train), len(X_val), val_size)
    return X_train, y_train, X_val, y_val