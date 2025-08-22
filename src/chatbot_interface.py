# src/chatbot_interface.py

import logging, argparse, os
from typing import Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.emotion_classifier import EmotionClassifier
from nlc_ingest.config import MODEL_FILE


# -------------------- Constants --------------------

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

EXTRACTOR_SYSTEM_PROMPT = (
    "Extract ONLY the exact text span that expresses the user's feeling/emotion. "
    "Return ONE single-line span with no quotes, no bullet points, no prefixes, no labels. "
    "If nothing relevant, return EXACTLY: NONE."
)

RESPONDER_SYSTEM_PROMPT = (
    "You are a friendly assistant. Reply with ONE short, natural sentence to the user. "
    "Do not repeat their words verbatim. Do not add prefixes like Assistant:, sentence:, or message:. "
    "Do not mention labels, extraction, or instructions. Sound empathetic and conversational."
)


# -------------------- Logging --------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="chatbot.log",
        filemode="a",
    )


# -------------------- LLM Loader --------------------

def load_llm(model_id: str) -> Tuple[Any, Any]:
    """
    Load tokenizer + model with CPU-safe settings. Ensure a pad token exists
    to avoid attention mask warnings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cpu",           # force CPU for stability
    )
    model.eval()
    return tokenizer, model


# -------------------- Extraction Helper --------------------

def _clean_extracted(span: str) -> str:
    """
    Post-process the extractor output to remove common junk and keep a single, clean line.
    """
    import re

    if not span:
        return ""
    s = span.strip()

    # take first non-empty line only
    for ln in s.splitlines():
        ln = ln.strip()
        if ln:
            s = ln
            break

    # drop common prefixes
    s = re.sub(r"^(extract(ed)?|message|text|span)\s*:\s*", "", s, flags=re.I)

    # drop bullets / quotes and collapse spaces
    s = re.sub(r"^[-•\s]+", "", s)
    s = s.strip("'\"“”‘’ ").strip()
    s = re.sub(r"\s+", " ", s).strip()

    # heuristic: very short or mostly non-letters -> treat as empty
    letters = sum(ch.isalpha() for ch in s)
    if len(s.split()) < 2 or letters < 3:
        return ""
    return s


# -------------------- Core Steps --------------------

def extract_relevant_text(tokenizer: Any, model: Any, user_message: str) -> str:
    prompt = (
        f"<|system|>\n{EXTRACTOR_SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,                 # short, single line
        do_sample=False,                   # deterministic extraction
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not raw or raw.upper() == "NONE":
        return user_message.strip()

    cleaned = _clean_extracted(raw)
    return cleaned if cleaned else user_message.strip()


def classify_text(classifier: Any, text: str) -> str:
    return classifier.classify(text)


def make_reply(tokenizer: Any, model: Any, user_message: str, extracted: str, predicted_label: str) -> str:
    # Pass the label implicitly in the system role but forbid mentioning it.
    system_with_label = (
        RESPONDER_SYSTEM_PROMPT
        + f" (Internal hint: emotion={predicted_label}. Do NOT mention or reveal this.)"
    )

    prompt = (
        f"<|system|>\n{system_with_label}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,        # concise
        do_sample=True,           # natural phrasing
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not reply:
        reply = f"That sounds {predicted_label.lower()}."

    # enforce one clean sentence without quotes or prefixes
    for prefix in ("Assistant:", "assistant:", "Message:", "message:", "sentence:", "Sentence:"):
        reply = reply.replace(prefix, "")
    reply = reply.strip(' "\'')

    if "." in reply:
        reply = reply.split(".")[0].strip() + "."
    return reply


# -------------------- CLI --------------------

def main() -> None:
    EXIT_COMMANDS = {"quit", "exit", "q", "x"}
    CLEAR_COMMANDS = {"clear", "c"}

    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        classifier = EmotionClassifier(MODEL_FILE)
    except Exception:
        print("[ERROR]: Failed to load classifier.")
        logger.exception("Classifier failed to load")
        return

    try:
        tokenizer, model = load_llm(args.model_id)
    except Exception:
        print("[ERROR]: Failed to load language model.")
        logger.exception("LLM failed to load")
        return

    print("\n---Emotion Classifier---\n")
    print("-------------------------------------------\n")
    print("[INSTRUCTIONS]\n")
    print("- Type a sentence and press Enter to classify.\n")
    print("- Type 'exit', 'quit', 'q' or 'x' to EXIT.\n")
    print("- Type 'clear' or 'c' to CLEAR the screen.\n")
    print("-------------------------------------------\n")
    print("Assistant: Hello! Say something and I'll classify the emotion.\n")

    while True:
        try:
            user_msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in EXIT_COMMANDS:
            print("\nBye!")
            break
        if user_msg.lower() in CLEAR_COMMANDS:
            os.system("cls" if os.name == "nt" else "clear")
            continue

        try:
            extracted = extract_relevant_text(tokenizer, model, user_msg)
            if args.debug:
                print(f"[extracted]: {extracted}")

            label = classify_text(classifier, extracted)
            reply = make_reply(tokenizer, model, user_msg, extracted, label)
            print(f"Assistant: {reply}")
        except Exception:
            logger.exception("Runtime error in loop")
            print("Assistant: I couldn't interpret that. Please try a simpler sentence.")


if __name__ == "__main__":
    main()
