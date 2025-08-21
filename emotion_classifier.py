import pickle
import logging
from pathlib import Path

from nlc_ingest.cleaning import clean_text

class EmotionClassifier:
    def __init__(self, model_path:str):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.load(model_path)

    def load(self, model_path: str):
        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file path not found at: {model_path}")
        
        with open(path_obj, "rb") as model_file:
            self.model = pickle.load(model_file)

        try:
            step_names = list(self.model.named_steps.keys())
            self.logger.info("Loaded pipeline steps : %s", step_names)
        except Exception:
            pass

    def classify(self, text: str):
        cleaned = clean_text(str(text))
        prediction = self.model.predict([cleaned])
        return prediction[0]
    
    