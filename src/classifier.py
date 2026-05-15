"""
Bias classifier module - loads the fine-tuned RoBERTa checkpoint and runs inference.

Other modules (the Streamlit app, framing pipeline, etc.) import from here.
They never load the model themselves - they call predict() or predict_batch().
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# derived empirically from the confidence error-rate table in 02_classifier_eval.ipynb
# below this threshold, error rate stays above 29% - not trustworthy enough for a hard label
CONFIDENCE_THRESHOLD = 0.85

MODEL_PATH = "models/bias_classifier"

LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# module-level cache so the model is only loaded once per process
_tokenizer = None
_model     = None
_device    = None


def _load_model():
    """Load tokenizer and model into the module cache. Called automatically on first use."""
    global _tokenizer, _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model     = _model.to(_device)
    _model.eval()  # disables dropout - required before any inference


def _build_input(headline: str, body: str) -> str:
    """
    Build the RoBERTa input string from headline and body text.

    Uses the same format as training: headline + </s> separator + first 400 chars of body.
    </s> is RoBERTa's separator token - not [SEP], which is BERT-specific.
    """
    return f"{headline} </s> {body[:400]}"


def predict(headline: str, body: str) -> dict:
    """
    Classify a single article. Returns a dict with label, confidence, and trusted flag.

    Args:
        headline: article headline
        body:     full article body text (truncated internally to 400 chars)

    Returns:
        {
            "label":      "left" | "center" | "right",
            "confidence": float (0-1),
            "trusted":    bool  (True if confidence >= CONFIDENCE_THRESHOLD)
        }
    """
    if _model is None:
        _load_model()

    text    = _build_input(headline, body)
    encoded = _tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        logits = _model(**encoded).logits

    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id    = int(probs.argmax())
    confidence = float(probs.max())

    return {
        "label":      ID2LABEL[pred_id],
        "confidence": confidence,
        "trusted":    confidence >= CONFIDENCE_THRESHOLD,
    }


def predict_batch(articles: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Classify a list of articles in batches. More efficient than calling predict() in a loop.

    Args:
        articles:   list of dicts, each with "headline" and "body" keys
        batch_size: how many articles to tokenize and score at once

    Returns:
        list of result dicts matching the predict() return format, in the same order
    """
    if _model is None:
        _load_model()

    texts   = [_build_input(a["headline"], a["body"]) for a in articles]
    results = []

    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        encoded = _tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,       # pad to the longest sequence in this batch
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            logits = _model(**encoded).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for prob_row in probs:
            pred_id    = int(prob_row.argmax())
            confidence = float(prob_row.max())
            results.append({
                "label":      ID2LABEL[pred_id],
                "confidence": confidence,
                "trusted":    confidence >= CONFIDENCE_THRESHOLD,
            })

    return results
