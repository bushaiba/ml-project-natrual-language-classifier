import logging
from typing import Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.emotion_classifier import EmotionClassifier
from nlc_ingest.config import MODEL_FILE


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1/1B-Chat-v1.0"
EXTRACTOR_SYSTEM_PROMPT = (
    "Extract ONLY the text that should be classified for emotion."
    "Return the span verbatim. If nothing is relevant, return EXACTLY: NONE."
)
RESPONDER_SYSTEM_PROMPT = (
    "You are a friendly assistant. Write a concise (1-2 sentences) helpful reply "
    "using the user's message, the extracted text, and the predicted label. No jargon."
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="logs/chatbot.log",
        filemode="a"
    )


def load_llm(model_id: str) -> Tuple[Any, Any]:
    # purpose is to retun (tokenizer, model) with minimal configuration
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto") # auto chooses cpu/gpu via Accelerate, without manual handling
    return tokenizer, model


def extract_relevant_text(tokenizer: Any, model: any, user_message: str) -> str:
    prompt = (
        f"<|system|>\n{EXTRACTOR_SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device) # tokenise to tensors on the model device
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=60,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id)
    # change do_sample, temperature, and top_p fo variety/natural response (current config is deterministic)
        
    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not text or text.upper() == "NONE":
        return user_message.strip()
    return text


def classify_text(classifier: Any, text: str) -> str:
    return classifier.classify(text)


def make_reply(tokenizer: Any, model: Any, user_message: str, extracted: str, predicted_label: str) -> str:
    prompt = (
        f"<|system|>\n{RESPONDER_SYSTEM_PROMPT}\n"
        f"<|user|>\n"
        f"User message:\n{user_message}\n\n"
        f"Extracted text to classify:\n{extracted}\n\n"
        f"Classifier label:\n{predicted_label}\n"
        f"Write the reply now.\n"
        f"<|assistant|>\n"
    )
    # <|assistant|> tells that model that it's its turn

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not reply:
        reply = f"It reads as {predicted_label.lower()}."
    return reply


# plan for main()
""" 
- initiate logging
- pars cli args for dynamic functionality and easier calls e.g. --model_id etc
- load the classifier
- load the LLM, both the tokenizer and the model
- REPL loop approach: read, extract, classify, reply, print
- necessary error handling
"""
def main():
    pass


if __name__ == "__main__":
    main()