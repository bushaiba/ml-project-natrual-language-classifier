import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="train.log",
        filemode="w",
    )


def parse_args():
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


def save_model(model, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved to: %s", path)


def load_dataset(path_str: str) -> pd.DataFrame:
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
        has_validation = (df["split"] == "validation").any()
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
    logging.info("Using random split: train=%d, val=%d (val_size=%.2f)", len(X_train), len(X_val), val_size)
    return X_train, y_train, X_val, y_val


def build_pipeline():
    """
    TF-IDF bag-of-words + Logistic Regression
    - lowercase handled by vectorizer
    - ngram_range=(1,2) often helps a bit for short texts
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1, 
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        random_state=42,
    )
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("logreg", clf),
    ])
    return pipe


def evaluate(y_true, y_pred, label_names=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    logging.info("Accuracy: %.4f", acc)
    logging.info("Precision (weighted): %.4f", precision)
    logging.info("Recall (weighted): %.4f", recall)
    logging.info("F1 (weighted): %.4f", f1)

    report = classification_report(y_true, y_pred, zero_division=0)
    logging.info("Classification Report:\n%s", report)

    if label_names:
        cm = confusion_matrix(y_true, y_pred, labels=label_names)
    else:
        cm = confusion_matrix(y_true, y_pred)

    try:
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        logging.info("Confusion Matrix (counts):\n%s", cm_df.to_string())

        # row_pct = (cm_df.div(cm_df.sum(axis=1), axis=0) * 100).round(1)
        # logging.info("Confusion Matrix (row %%):\n%s", row_pct.to_string())

    except Exception:
        logging.info("Confusion Matrix (raw):\n%s", cm)



def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    df = load_dataset(args.input)
    X_train, y_train, X_val, y_val = split_data(
    df, use_split=args.use_split_column, val_size=args.val_size, seed=args.random_state
    )

    # Label names for consistent confusion matrix ordering (sorted for readability)
    label_names = sorted(list(set(df["label"].tolist())))

    model = build_pipeline()

    logger.info("Training model...")
    model.fit(X_train, y_train)

    logger.info("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    evaluate(y_val, y_pred, label_names=label_names)

    save_model(model, args.model_out)
    logger.info("Done.")
    
    return model


if __name__ == "__main__":
    model = main()

    # string = "i feel energetic"
    # print(model.predict([string]))

