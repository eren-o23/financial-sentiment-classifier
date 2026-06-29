# Financial Sentiment Classifier

Fine-tuned DistilBERT for three-class sentiment classification on financial analyst reports.

---

## The Problem

A sentence like *"The company issued a profit warning ahead of Q3 results"* is unambiguously negative to anyone who reads earnings releases. A general-purpose sentiment model, trained on product reviews, tweets, and movie ratings, sees the word "profit" and hedges toward positive or neutral.

This is the core failure mode of applying off-the-shelf NLP to finance: the domain has its own vocabulary, idioms, and framing conventions that general models haven't seen. Phrases like *"headwinds in the core segment"*, *"revenues declined in line with guidance"*, or *"the board remains cautious"* carry clear sentiment signals to a trained analyst that are invisible to a general model.

This project fine-tunes DistilBERT on the **Financial PhraseBank** (Malo et al., 2013), 3,453 sentences from analyst reports hand-labelled as **positive**, **negative**, or **neutral** by finance researchers, and proves the domain adaptation works by comparing against a VADER baseline.

---

## Architecture Decision: DistilBERT over Decoder Models

The model choice is deliberate. Qwen3, GPT-style models, and most frontier LLMs are **decoder-only** (autoregressive): they generate tokens one at a time, predicting each token from everything to its left. That's the right architecture for generation. It's wasteful for classification.

**DistilBERT is encoder-only.** Its self-attention layers are bidirectional, every token attends to every other token simultaneously. This produces a rich, context-aware representation of the whole sentence captured in the `[CLS]` token, which a linear head maps directly to the three sentiment classes.

DistilBERT is also 40% smaller and 60% faster than BERT while retaining 97% of its language understanding (Sanh et al., 2019). For a ~3,400 sentence dataset, that size is a feature — it reduces overfitting risk.

```
Input sentence
     ↓
DistilBERT tokenizer (WordPiece, max_length=128)
     ↓
6 transformer encoder layers (bidirectional attention)
     ↓
[CLS] token embedding (768-dim)
     ↓
Linear classification head (768 → 3)
     ↓
{positive, neutral, negative}
```

---

## Dataset

**Financial PhraseBank v1.0** (Malo et al., 2013)

- 5 annotators (finance researchers) labelled each sentence
- We use `Sentences_75Agree.txt` — 3,453 sentences with ≥75% annotator agreement
- Label distribution: ~60% neutral, ~28% positive, ~12% negative

> Malo, P., Sinha, A., Takala, P., Korhonen, P. & Wallenius, J. (2013). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology.

---

## Results

| Model | Positive F1 | Neutral F1 | Negative F1 | **Macro F1** |
|-------|------------|-----------|------------|-------------|
| VADER (baseline) | 0.515 | 0.645 | 0.273 | 0.478 |
| DistilBERT fine-tuned | 0.819 | 0.933 | 0.864 | **0.872** |
| Δ improvement | +0.304 | +0.288 | **+0.591** | **+0.394** |

90% accuracy on the test set. Best checkpoint saved at epoch 2 (val macro F1: 0.917).

---

## Key Findings

- **Domain adaptation works**: +39.4 points macro F1 over VADER on the same test set — not a marginal improvement, a qualitative one.
- **Negative class is the biggest win** (+0.591 F1): VADER correctly identifies only 28% of negative sentences; the fine-tuned model identifies 90%. Financial downside language ("tacked lower", "in stoppage", "non-responsive") has no signal in a general-domain lexicon.
- **Hardest boundary is positive ↔ neutral** (24 of 36 errors): the model is reading factual sentences, turnarounds stated as number comparisons, partnerships described in neutral future tense, that the annotators marked positive from surrounding article context. The model only sees the sentence. This is an information problem, not a model problem.
- **Neutral→negative errors** reveal domain mismatch in the opposite direction: sentences about physical events ("collisions started", "high winds toppled semi-trailers") are labelled neutral by finance annotators but the model correctly identifies them as negative in general register. The label reflects investor-perspective neutrality; the model learned general-register negativity.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter lab financial_sentiment_classifier.ipynb
```

Run all cells top-to-bottom. Training takes ~5–10 minutes on MPS (Apple Silicon) or CPU.

---

## Project Structure

```
financial-sentiment-classifier/
├── financial_sentiment_classifier.ipynb   # Full pipeline: EDA → baseline → training → evaluation → error analysis
├── requirements.txt
├── README.md
└── FinancialPhraseBank-v1.0/
    ├── Sentences_75Agree.txt              # Used for training (3,453 sentences)
    ├── Sentences_AllAgree.txt
    ├── Sentences_66Agree.txt
    ├── Sentences_50Agree.txt
    └── README.txt
```

---

## Connection to AMD/HuggingFace Coursework

This project extends the tokenization pipeline, embedding layers, and attention mechanisms covered in the AMD/HuggingFace course. The difference: instead of using pre-trained weights for inference, we fine-tune them, updating all 66M parameters of DistilBERT on domain-specific labelled data and plugging in a classification head at the `[CLS]` output.
