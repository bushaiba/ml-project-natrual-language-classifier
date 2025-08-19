import pandas as pd
import re, logging
from ingest import logger
from nlc_ingest.io import load_data, save_dataframe

logger = logging.getLogger(__name__)

HANDLE_WHITESPACE = re.compile(r"\s+")

def clean_text(text: str):
    text = str(text)
    text = text.strip()
    text = HANDLE_WHITESPACE.sub(" ", text)
    return text

def normalise_label(label: str):
    label = str(label)
    label = " ".join(label.split())
    label = label.lower()
    return label

def standardise(df: pd.DataFrame, split_name: str):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("standardise expected a pandas DataFrame")
    
    remapped_columns = {}

    for original in df.columns:
        lower = original.lower()
        remapped_columns[lower] = original

    if "text" in remapped_columns:
        text_column = remapped_columns["text"]
    else:
        text_column = df.columns[0]

    if "label_text" in remapped_columns:
        label_column = df[remapped_columns["label_text"]]
    else:
        if "label" in remapped_columns and df[remapped_columns["label"]].dtype == "O":
            label_column = df[remapped_columns["label"]]
        else:
            if "label" in remapped_columns:
                fallback_column = remapped_columns["label"]
            else:
                if len(df.columns) < 2:
                    raise ValueError("Cannot get label column from DataFrame")
                fallback_column = df.columns[1]
            label_column = df[fallback_column].astype(str)

    # build the dataframe
    dataframe = pd.DataFrame()
    dataframe["text"] = df[text_column].astype(str).map(clean_text) 
    dataframe["label"] = label_column.astype(str).map(normalise_label)
    dataframe["split"] = split_name

    # drop empty texts
    before = len(dataframe)
    mask = dataframe["text"].str.len() > 0
    dataframe = dataframe[mask]
    after = len(dataframe)
    if before != after:
        logger.info("Dropped %d empty-text rows", before - after)

    # drop duplicate text/label pairs
    before = len(dataframe)
    de_duplicated = dataframe.drop_duplicates(subset=["text", "label"])
    after = len(de_duplicated)
    if before != after:
        logger.info("Dropped %d duplicate (text, label) rows", before - after)
    
    # ensure column order
    final = de_duplicated[["text", "label", "split"]]

    return final
