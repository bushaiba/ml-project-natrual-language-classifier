import logging, json
import pandas as pd


def logger(message):
    logging.basicConfig(
        level=logging.INFO,
        filename="ingest.log",
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w"
    )
    logging.info(message)

