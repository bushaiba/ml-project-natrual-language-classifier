import argparse
import logging
from pathlib import Path
import pandas as pd

from nlc_ingest.io import load_data, save_dataframe
from nlc_ingest.cleaning import standardise
from nlc_ingest.config import PROCESSED_FILE, ALLOWED_FORMATS

# dynamic arguments, with defaults
def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Emotion Dataset")
    parser.add_argument("--dataset", default="dair-ai/emotion",
                        help="HuggingFace dataset ID (e.g., dair-ai/emotion)")
    parser.add_argument("--format", default="csv", choices=ALLOWED_FORMATS,
                        help="Output format")
    parser.add_argument("--out", default=str(PROCESSED_FILE),
                        help="Output file path")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="logs/ingest.log",
        filemode="w"
    )


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset: %s", args.dataset)
    # load_data returns a dict of split_name -> DataFrame (e.g., train/validation/test)
    splits = load_data(args.dataset)

    logger.info("Cleaning and standardising splits...")
    frames = []
    for split_name in splits.keys():
        # loop over each split (train, validation, test)
        split_df = splits[split_name]
        # grab its DataFrame
        cleaned_df = standardise(split_df, split_name)
        # clean it with standardise
        frames.append(cleaned_df)
    # each split is consistent and ready to combine

    logger.info("Combining %d cleaned splits", len(frames))
    combined = pd.concat(frames, ignore_index=True)

    out_path = Path(args.out)
    logger.info("Saving %d rows to %s as %s", len(combined), out_path, args.format)
    save_dataframe(combined, out_path, args.format)

    logger.info("Ingest complete.")


if __name__ == "__main__":
    main()
