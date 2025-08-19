import pandas as pd
from datasets import load_dataset
from pathlib import Path

def load_data(dataset_name: str = "dair-ai/emotion", split = "train"):
    dataset = load_dataset(dataset_name)
    output = {}

    for split in dataset.keys(): 
        df = dataset[split].to_pandas()
        output[split] = df

    return output

def save_dataframe(df: pd.DataFrame, path: Path, format: str = "csv"):

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(path, index=False) # index=False to maintin true indexing [otherwise pandas by default creates a new column of indices]
    elif format == "json":
        df.to_json(path, orient="records", lines=True) # "records" is a list of row dicts || lines=True adds new records to new lines
    else:
        raise ValueError(f"Unsupported format: {format}")
    
