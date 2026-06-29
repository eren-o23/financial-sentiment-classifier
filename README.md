# Financial Sentiment Classifier & Monitor

A two-part project:

1. **The classifier**, a DistilBERT model fine-tuned on Financial PhraseBank for three-class sentiment on financial text, with a full evaluation against a general-purpose baseline.
2. **The monitor**, a serving layer that pulls live news for a ticker, scores each headline with the classifier, and plots a daily sentiment index against price movement to **surface divergences between market language and price**. It is a monitoring tool, not a prediction tool: price is context for interpreting sentiment, not a target to forecast.

---

## Part 1 — The Classifier

### The Problem

A sentence like *"The company issued a profit warning ahead of Q3 results"* is unambiguously negative to anyone who reads earnings releases. A general-purpose sentiment model, trained on product reviews, tweets, and movie ratings, sees the word "profit" and hedges toward positive or neutral.

This is the core failure mode of applying off-the-shelf NLP to finance: the domain has its own vocabulary, idioms, and framing conventions that general models haven't seen. Phrases like *"headwinds in the core segment"*, *"revenues declined in line with guidance"*, or *"the board remains cautious"* carry clear sentiment signals to a trained analyst that are invisible to a general model.

This project fine-tunes DistilBERT on the **Financial PhraseBank** (Malo et al., 2013), 3,453 sentences from analyst reports hand-labelled as **positive**, **negative**, or **neutral** by finance researchers, and proves the domain adaptation works by comparing against a VADER baseline.

---

### Architecture Decision: DistilBERT over Decoder Models

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

### Dataset

**Financial PhraseBank v1.0** (Malo et al., 2013)

- 5 annotators (finance researchers) labelled each sentence
- We use `Sentences_75Agree.txt` — 3,453 sentences with ≥75% annotator agreement
- Label distribution: ~60% neutral, ~28% positive, ~12% negative

> Malo, P., Sinha, A., Takala, P., Korhonen, P. & Wallenius, J. (2013). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology.

---

### Results

| Model | Positive F1 | Neutral F1 | Negative F1 | **Macro F1** |
|-------|------------|-----------|------------|-------------|
| VADER (baseline) | 0.515 | 0.645 | 0.273 | 0.478 |
| DistilBERT fine-tuned | 0.819 | 0.933 | 0.864 | **0.872** |
| Δ improvement | +0.304 | +0.288 | **+0.591** | **+0.394** |

90% accuracy on the test set. Best checkpoint saved at epoch 2 (val macro F1: 0.917).

---

### Key Findings

- **Domain adaptation works**: +39.4 points macro F1 over VADER on the same test set — not a marginal improvement, a qualitative one.
- **Negative class is the biggest win** (+0.591 F1): VADER correctly identifies only 28% of negative sentences; the fine-tuned model identifies 90%. Financial downside language ("tacked lower", "in stoppage", "non-responsive") has no signal in a general-domain lexicon.
- **Hardest boundary is positive ↔ neutral** (24 of 36 errors): the model is reading factual sentences, turnarounds stated as number comparisons, partnerships described in neutral future tense, that the annotators marked positive from surrounding article context. The model only sees the sentence. This is an information problem, not a model problem.
- **Neutral→negative errors** reveal domain mismatch in the opposite direction: sentences about physical events ("collisions started", "high winds toppled semi-trailers") are labelled neutral by finance annotators but the model correctly identifies them as negative in general register. The label reflects investor-perspective neutrality; the model learned general-register negativity.

---

## Part 2 — The Sentiment Monitor

### Why monitoring, not prediction

Claiming a headline-sentiment model *predicts* price invites the obvious rebuttal: if it worked, everyone would already be doing it. So this tool doesn't predict. It **monitors**, it tracks how the language around a company shifts over time and uses price movement as *context* to interpret what it sees.

The interesting output is the **divergences**: days where sentiment and price move in opposite directions. Negative language while the price rises often means the market had already priced in worse, or that positioning contradicts the news flow. Those mismatches are where the insight lives, the tool surfaces them and lets a human ask *why*, rather than emitting a prediction and asking you to trust it.

### How it works

```
Alpha Vantage news API  ──▶  dedup (SQLite)  ──▶  DistilBERT classifier
        │                                                   │
        ▼                                                   ▼
   ticker headlines                              label + confidence per headline
                                                            │
                                                            ▼
                                          daily sentiment index  (mean of sign×confidence)
                                                            │
   yfinance daily close ──────────────────────────────────┤
                                                            ▼
                              align on trading days  ──▶  z-score divergence flagging
                                                            │
                                                            ▼
                                  FastAPI  ──▶  interactive Plotly dual-axis chart
```

- **Daily sentiment index**: each headline scores `sign(label) × confidence` (positive +1, neutral 0, negative −1); the day's index is the mean, a confidence-weighted net sentiment in roughly [−1, +1].
- **Divergence detection**: sentiment and price-return series are z-scored over the window; a day is flagged when the two signs oppose and both magnitudes exceed 1σ (requires ≥5 paired observations so the z-scores are meaningful).
- **Caching**: scored headlines are stored in SQLite keyed by URL, so re-runs never re-classify or waste the Alpha Vantage free-tier budget (25 requests/day).
- **Serving**: the 255 MB model loads once at FastAPI startup and is reused across requests.

### Running the monitor

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your free Alpha Vantage key

# Ingest + score headlines for a ticker (cron-ready)
python -m monitor.run_daily --ticker AAPL --start 2024-05-01 --end 2024-05-31

# Serve the API + interactive chart
uvicorn monitor.api:app --reload
#  POST /monitor              -> JSON: sentiment_series, price_series, divergences
#  GET  /monitor/view?ticker=AAPL&start=2024-05-01&end=2024-05-31&refresh=true
```

### Honest limitation: train/inference domain shift

The classifier was trained on **analyst-report sentences** ("Operating profit rose to EUR 11.2 mn from EUR 9.8 mn"), but news **headlines** are a different register, terser, present-tense, proper-noun heavy. The model is measurably less reliable on them: it scores *"Apple unveils record quarterly revenue beating estimates"* as negative, and *"Record quarterly revenue beats analyst estimates"* as neutral, while still nailing explicitly evaluative phrasing like *"Analysts upgrade Apple on strong demand"*.

This is exactly the domain-adaptation lesson from Part 1, observed in reverse: a model is only reliable on text that looks like its training distribution. The monitoring framing is robust to this (it reads a *trend* in sentiment, not a single label), but the right next step would be to fine-tune on a headline-labelled set or calibrate confidence thresholds before trusting day-level scores.

---

## Part 3 — The Eval Harness

Part 2 *asserts* a domain gap with anecdotes. Part 3 **measures** it, and makes that measurement repeatable so any future change to the model or inference pipeline can be regression-tested against a stable reference.

### What it does

Two checks against a committed **golden dataset** of 66 manually-labelled headlines:

- **Consistency check** — scores each headline multiple times and verifies the label and confidence are stable across repeated runs *and* across batch sizes (single vs batched). The model is deterministic in `eval()` mode, so this is a guard: it catches accidental nondeterminism (dropout/sampling left on) or batch/padding bugs. Forcing the model into `train()` mode, for example, makes it fail loudly.
- **Accuracy check** — compares predictions to ground truth and reports per-class precision/recall/F1 and macro F1, using the **same `sklearn.classification_report`** computation as the Part 1 notebook, so the numbers are directly comparable.

A **regression runner** sits on top: every run is stored in SQLite (`eval_runs` table, alongside the headline cache), and each run is compared against the previous one. A macro-F1 drop beyond a configurable threshold (default 0.05) is flagged.

### The headline domain gap, measured

| Test set | Macro F1 |
|----------|---------|
| Part 1 — analyst-report sentences (notebook test split) | **0.872** |
| Part 3 — news headlines (this golden set) | **0.848** |

The classifier holds up better on headlines than the anecdotes suggest, but the ~2-point macro-F1 drop, and the specific misclassifications the harness surfaces, confirm the domain shift is real and now quantified rather than asserted.

### How the golden dataset was built

The 776 cached headlines (4 tickers) were exported to JSONL and run through an **error-analysis workflow** (clustering on TF-IDF + structured features, then a diversity-first stratified sample) to pick 66 headlines spanning all three classes, all four tickers, and the full confidence range, deliberately oversampling low-confidence and minority-class cases where ground truth is most likely to diverge from the model. Each was then **manually labelled** with ground truth in a lightweight review UI. The result is committed at `eval_harness/golden/golden_headlines.jsonl`.

### Running the eval

```bash
# One-shot eval from the CLI (prints accuracy, macro F1, consistency, regression delta)
python -m eval_harness.run_eval

# Or via the API (same FastAPI app as the monitor)
uvicorn monitor.api:app --reload
#  POST /eval            -> full report: accuracy, per-class F1, consistency, misclassifications, regression delta
#  GET  /eval/history    -> past runs, newest first (regression trend)
```

---

## How to Run

```bash
pip install -r requirements.txt
```

**Part 1 — train/evaluate the classifier** (produces `best_model.pt`):
```bash
jupyter lab financial_sentiment_classifier.ipynb   # run all cells; ~5–10 min on MPS/CPU
```

**Part 2 — run the monitor** (needs `best_model.pt` + an Alpha Vantage key in `.env`):
```bash
uvicorn monitor.api:app --reload
```

---

## Project Structure

```
financial-sentiment-classifier/
├── financial_sentiment_classifier.ipynb   # Part 1: EDA → baseline → training → evaluation → error analysis
├── monitor/                                # Part 2: sentiment monitoring system
│   ├── classifier.py                       #   load checkpoint, score text (label + confidence)
│   ├── news.py                             #   Alpha Vantage NEWS_SENTIMENT ingestion
│   ├── prices.py                           #   yfinance daily prices
│   ├── storage.py                          #   SQLite cache + dedup
│   ├── aggregate.py                        #   daily sentiment index
│   ├── divergence.py                       #   z-score alignment + divergence flagging
│   ├── pipeline.py                         #   orchestration (cache-first)
│   ├── viz.py                              #   Plotly dual-axis chart
│   ├── api.py                              #   FastAPI service (monitor + eval endpoints)
│   └── run_daily.py                        #   cron-ready CLI
├── eval_harness/                           # Part 3: evaluation & regression testing
│   ├── golden.py                           #   load golden dataset (JSONL)
│   ├── consistency.py                      #   repeat-run + batch-invariance checks
│   ├── accuracy.py                         #   per-class P/R/F1 vs ground truth (sklearn)
│   ├── regression.py                       #   compare macro F1 vs previous run
│   ├── storage.py                          #   eval_runs table (regression history)
│   ├── runner.py                           #   orchestration -> EvalReport
│   ├── run_eval.py                         #   CLI entry point
│   ├── export_headlines.py                 #   dump headline_scores -> JSONL
│   └── golden/
│       └── golden_headlines.jsonl          #   66 manually-labelled headlines (committed)
├── requirements.txt
├── .env.example                            # ALPHA_VANTAGE_API_KEY
├── README.md
└── FinancialPhraseBank-v1.0/               # training data (not redistributed)
    └── Sentences_75Agree.txt               # used for training (3,453 sentences)
```

---

## Connection to AMD/HuggingFace Coursework

This project extends the tokenization pipeline, embedding layers, and attention mechanisms covered in the AMD/HuggingFace course. The difference: instead of using pre-trained weights for inference, we fine-tune them, updating all 66M parameters of DistilBERT on domain-specific labelled data and plugging in a classification head at the `[CLS]` output.
