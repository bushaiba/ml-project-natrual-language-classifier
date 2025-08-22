import logging
import sys
import os

from emotion_classifier import EmotionClassifier
from nlc_ingest.config import MODEL_FILE

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="logs/interface.log",
        filemode="w",
    )

def simple_interface():
    setup_logging()
    logger = logging.getLogger(__name__)

    model_path = MODEL_FILE

    try:
        classifer = EmotionClassifier(model_path)
    except Exception:
        print(f"[ERROR] Failed to load model: {Exception}")
        logger.exception("Failed to load model")
        sys.exit(1)

    print("\n---Emotion Classifier---\n")

    print("-------------------------------------------\n")
    print("[INSTRUCTIONS]\n")
    print("- Type a sentence and press Enter to classify.\n")
    print("- Type 'exit', 'quit', 'q' or 'x' to EXIT.\n")
    print("- Type 'clear' to CLEAR the history.")
    print("\n")

    while True:
        try: 
            user_text = input("> ").strip()
        except(EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() in {"exit", "quit", "q", "x"}:
            print("\nBye!")
            break

        if user_text == "":
            print("[Info] Can not classify empty input, please enter text.")
            continue

        if user_text.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")  
            continue

        try:
            label = classifer.classify(user_text)
            print(f"Result: {label}\n")
        except Exception:
            logger.exception("Classification error")
            print(f"[ERROR] Could not classify input: {Exception}\n")


if __name__ == "__main__":
    simple_interface()


