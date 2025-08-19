import pandas as pd
import pytest

from nlc_ingest.cleaning import (
    clean_text,
    normalise_label,
    standardise,
)

# ---------- clean_text ----------

def test_clean_text_trims_and_collapses_spaces():
    s = "  Hello   world  \n"
    out = clean_text(s)
    assert out == "Hello world"


def test_clean_text_handles_tabs_and_newlines():
    s = "\tHi\tthere\n\nfriend"
    out = clean_text(s)
    assert out == "Hi there friend"


def test_clean_text_casts_non_string():
    s = 42
    out = clean_text(s)
    assert out == "42"


# ---------- normalise_label ----------

def test_normalise_label_basic():
    lbl = "  Joy "
    out = normalise_label(lbl)
    assert out == "joy"


def test_normalise_label_collapses_internal_spaces():
    lbl = "Very    Happy"
    out = normalise_label(lbl)
    assert out == "very happy"


# ---------- standardise ----------

def test_standardise_prefers_label_text_column():
    raw = pd.DataFrame({
        "Text": ["  Hi  ", "Hi  "],
        "Label_Text": ["Joy", "Joy"]
    })
    out = standardise(raw, "train")
    assert list(out.columns) == ["text", "label", "split"]
    assert out.iloc[0]["text"] == "Hi"
    assert out.iloc[0]["label"] == "joy"
    assert out.iloc[0]["split"] == "train"
    # dedup happened
    assert len(out) == 1


def test_standardise_uses_object_label_if_present():
    raw = pd.DataFrame({
        "text": [" A ", "B"],
        "label": ["Anger", "Fear"]   # dtype object (strings)
    })
    out = standardise(raw, "validation")
    assert set(out["label"]) == {"anger", "fear"}
    assert out["split"].nunique() == 1 and out["split"].iloc[0] == "validation"


def test_standardise_fallback_from_numeric_label_to_string():
    raw = pd.DataFrame({
        "text": ["good", "bad"],
        "label": [1, 0]  # numeric; your code casts to str via fallback
    })
    out = standardise(raw, "test")
    # labels are strings now
    assert set(out["label"]) == {"1", "0"}
    assert out["split"].iloc[0] == "test"


def test_standardise_fallback_to_second_column_when_no_label_cols():
    raw = pd.DataFrame({
        "TEXT": ["hello", "hello"],
        "SOMETHING": ["Joy", "Joy"]
    })
    out = standardise(raw, "train")
    assert list(out.columns) == ["text", "label", "split"]
    assert out.iloc[0]["text"] == "hello"
    assert out.iloc[0]["label"] == "joy"
    # deduplicate happened (2 identical rows -> 1)
    assert len(out) == 1


def test_standardise_drops_empty_text_rows_and_logs_counts(caplog):
    raw = pd.DataFrame({
        "text": ["", "   ", "ok"],
        "label_text": ["joy", "joy", "joy"]
    })
    with caplog.at_level("INFO"):
        out = standardise(raw, "train")
    assert len(out) == 1
    assert out.iloc[0]["text"] == "ok"
    # ensure a log line was emitted
    found = False
    for rec in caplog.records:
        if "empty-text" in rec.getMessage():
            found = True
            break
    assert found is True


def test_standardise_raises_type_error_for_non_dataframe():
    with pytest.raises(TypeError):
        standardise(["not", "a", "df"], "train")


def test_standardise_raises_when_cannot_infer_label_column():
    # Only one column and it's the text; no label available
    raw = pd.DataFrame({"only_text": ["hi"]})
    with pytest.raises(ValueError):
        standardise(raw, "train")
