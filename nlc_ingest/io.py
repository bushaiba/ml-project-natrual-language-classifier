import pandas as pd
from datasets import load_dataset
from pathlib import Path

def load_data(dataset_name: str = "dair-ai/emotion"):
    """
    Load HF dataset splits as pandas DataFrames.
    Also adds a human-readable 'label_text' column using the dataset's mapping.
    """
    dataset = load_dataset(dataset_name)
    output = {}

    for split in dataset.keys():
        # grab the int -> string mapper BEFORE converting to pandas
        int_to_str = dataset[split].features["label"].int2str

        # convert this split to pandas
        df = dataset[split].to_pandas()

        # add label_text using the mapper (keep it explicit and readable)
        if "label" in df.columns:
            df["label_text"] = df["label"].map(lambda x: int_to_str(int(x)))

        # store in result
        output[split] = df

    return output


def save_dataframe(df: pd.DataFrame, path: Path, format: str = "csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
