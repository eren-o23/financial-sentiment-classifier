"""Inference wrapper around the fine-tuned DistilBERT checkpoint.

Reuses the exact load + eval contract from the training notebook: rebuild the
3-class architecture from `distilbert-base-uncased`, load the saved state_dict,
then run sentences through in eval mode. Confidence is the softmax max-probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from .config import CHECKPOINT_PATH, ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_NAME


@dataclass
class Prediction:
    label: str
    confidence: float


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FinancialSentimentClassifier:
    """Loads the checkpoint once and scores batches of text."""

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device: torch.device | None = None):
        self.device = device or _select_device()
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        """Return a Prediction(label, confidence) for each input text."""
        predictions: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            for i, c in zip(idx.cpu().tolist(), conf.cpu().tolist()):
                predictions.append(Prediction(label=ID2LABEL[i], confidence=round(c, 4)))
        return predictions

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]
