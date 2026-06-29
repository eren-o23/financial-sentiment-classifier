# Handoff — Financial Sentiment Classifier & Monitor

_Last updated: 2026-06-29_

---

## Goal

Extend a fine-tuned DistilBERT sentiment classifier (trained on Financial PhraseBank, 0.872 macro F1) into a full sentiment monitoring system: pull live news for a ticker via Alpha Vantage, score headlines with the model, aggregate a daily sentiment index, plot it against price movement (yfinance), and surface divergences (negative language + rising price, etc.) via a FastAPI-served Plotly page. Framing is monitoring, not prediction — divergences are the output, price is context not target.

---

## Current State

All code is written, committed to `git@github-personal:eren-o23/financial-sentiment-classifier.git` (main, 4 commits), and pushed. The repo is clean. The plan from plan mode is fully implemented.

What is confirmed working (written and committed, but not end-to-end live-run yet — the user just obtained their Alpha Vantage key):
- `monitor/` package: all 10 modules implemented
- `monitor/classifier.py` — loads `best_model.pt`, batch predicts (label + confidence)
- `monitor/news.py` — Alpha Vantage `NEWS_SENTIMENT` ingestion with relevance threshold
- `monitor/storage.py` — SQLite dedup via URL or sha1(title), `init_db` guard in `existing_keys`
- `monitor/aggregate.py` — daily index as mean of `sign(label) × confidence`
- `monitor/divergence.py` — z-score flagging with `DIVERGENCE_MIN_OBS=5` guard
- `monitor/pipeline.py` — cache-first orchestration
- `monitor/viz.py` — Plotly dual-axis HTML
- `monitor/api.py` — FastAPI with lifespan (model loads once), POST `/monitor` + GET `/monitor/view`
- `monitor/run_daily.py` — cron-ready CLI
- README updated with Part 2 section, monitoring framing, honest domain-shift limitation

**First live run completed 2026-06-29.** System confirmed working end-to-end: 57 AAPL headlines scored for May 2024, chart serving at `/monitor/view`, POST `/monitor` endpoint working (fields are `start_date`/`end_date`, not `start`/`end`).

**Part 3 — Eval Harness added 2026-06-29 (session 3).** A new `eval_harness/` package layered on top of `monitor/`:
- `eval_harness/golden/golden_headlines.jsonl` — 66 manually-labelled headlines (committed golden set). Ground-truth distribution 27 neutral / 21 positive / 18 negative.
- `consistency.py` — repeat-run determinism + batch-invariance checks. PASSES on current model; FAILS if model is in `train()` mode (proven).
- `accuracy.py` — per-class P/R/F1 + macro F1 via `sklearn.classification_report` (same as notebook). **Headlines macro F1 = 0.848** vs 0.872 on analyst sentences → domain gap quantified.
- `storage.py` — new `eval_runs` table on the **same** `monitor.db` (headline_scores untouched).
- `regression.py` — flags macro-F1 drop > 0.05 vs previous run.
- `runner.py` / `run_eval.py` (CLI) / `export_headlines.py`.
- Two endpoints added to `monitor/api.py`: POST `/eval`, GET `/eval/history`. All validated end-to-end (CLI + API + 3 stored runs).
- Pool was broadened first: backfilled TSLA/JPM/NVDA (now 776 headlines, 4 tickers).

---

## Key Invariants

- **Python environment**: ALL project commands must use `/opt/anaconda3/bin/python`, NOT `python3` or `pip3`. The system Python 3.9 at `/usr/bin/python3` does not have torch, transformers, fastapi, yfinance, plotly, etc. Using the wrong interpreter causes silent import errors.

- **Model load contract**: `best_model.pt` is a state_dict only, not a full model. Must do:
  ```python
  model = DistilBertForSequenceClassification.from_pretrained(
      'distilbert-base-uncased', num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
  model.load_state_dict(torch.load('best_model.pt', map_location=device))
  ```
  `LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}` — order matters for logit→label mapping.

- **Alpha Vantage budget**: Free tier is 25 requests/day. The SQLite dedup (`UNIQUE(ticker, dedup_key)`) prevents re-classifying already-seen headlines. Always use `refresh=false` for the view endpoint when data is already cached; only use `refresh=true` when fetching new date ranges.

- **Divergence minimum obs**: `DIVERGENCE_MIN_OBS=5` in `config.py` and enforced in `divergence.py`. With fewer than 5 paired (sentiment + price) observations, every point is trivially ±1σ, so flagging is suppressed entirely. Don't lower this.

- **SSH remote**: The GitHub remote uses `github-personal` host alias (not `github.com`). Push with `git push origin main` — the alias in `~/.ssh/config` routes it correctly. The key is `~/.ssh/id_ed25519_personal`. Do NOT use `git push git@github.com:...` directly.

- **API key**: Must be in `.env` (not committed). File not yet created by the user — they need `cp .env.example .env` then paste their key.

---

## What We Tried That Failed

| Approach | Why it failed |
|----------|--------------|
| `pip3 install` for project deps | `pip3` routes to system Python 3.9, not Anaconda 3.12. Packages install to the wrong env and imports fail silently. |
| `DIVERGENCE_MIN_OBS=2` | With n=2 paired observations, z-scores are exactly ±1 for every point — meaningless divergence flags everywhere. Raised to 5. |
| `existing_keys()` without `init_db()` | On a fresh DB, `headline_scores` table doesn't exist yet. `SELECT` raises `sqlite3.OperationalError`. Fixed by calling `init_db()` at the top of `existing_keys()`. |
| `git push` before rebase | Remote had a "Fix punctuation" commit from GitHub web UI. Push rejected. Resolved with `git rebase origin/main` (no conflicts). |

---

## Don't Touch

- `best_model.pt` — trained weights, gitignored, 255 MB. Do not retrain unless the notebook is re-run in full. The checkpoint is from epoch 2 (best val macro F1: 0.917).
- `monitor/storage.py` dedup logic — the `UNIQUE(ticker, dedup_key)` constraint and the `init_db()` guard in `existing_keys` are both load-bearing for budget preservation.
- `eval_harness/golden/golden_headlines.jsonl` — the committed golden set is hand-labelled ground truth. Don't regenerate it programmatically; re-labelling requires the error-discovery UI (`error_discovery_data/`, gitignored). Eval numbers are only comparable across runs if the golden set is stable.

---

## Next Step

Parts 1-3 are complete and validated. Nothing committed/pushed yet for Part 3 (see below). Candidate follow-ons:

```bash
# Re-run the eval anytime (creates a new eval_runs row, compares vs previous):
/opt/anaconda3/bin/python -m eval_harness.run_eval

# Serve monitor + eval together:
/opt/anaconda3/bin/python -m uvicorn monitor.api:app --reload
#  POST /eval  ·  GET /eval/history  ·  POST /monitor  ·  GET /monitor/view
```

- **To commit Part 3**: `eval_harness/` (package + golden set), `monitor/api.py`, `README.md`, `.gitignore`, `handoff.md` are changed/new and uncommitted. `error_discovery_data/` and `headlines_export.jsonl` are gitignored.
- **Possible next work**: expand the golden set beyond 66; add more tickers; calibrate confidence thresholds on headlines; or fine-tune on a headline-labelled set to close the 0.848→0.872 gap (then regression-test the new model against this same golden set).

**Note on POST /monitor fields**: the Pydantic model uses `start_date` and `end_date` (not `start`/`end`). The GET `/monitor/view` endpoint uses `start`/`end` query params.

---

## Open Questions / Blockers

- No blockers. Parts 1-3 are working and validated end-to-end.
- 1-month AAPL window (May 2024) yields 0 divergent days because most days have 0.0 sentiment (sparse coverage), compressing z-scores. Extend to 3 months to surface divergences.
- Part 3 changes are not yet committed/pushed — awaiting user go-ahead.

---

## Session History

_Append-only. One line per session — never overwrite previous entries._

- 2026-06-29: Built full project from scratch — Part 1 (DistilBERT fine-tune on Financial PhraseBank, 0.872 macro F1 vs 0.48 VADER) + Part 2 (sentiment monitoring system: Alpha Vantage → SQLite → classifier → daily index → Plotly/FastAPI). Fixed storage init bug, z-score minimum obs guard, SSH push config, em-dash style. All code committed and pushed. User obtained Alpha Vantage key; live run is next.
- 2026-06-29 (session 2): Confirmed first live run. 57 AAPL headlines scored for May 2024, chart rendering, 0 divergences (expected for 1-month sparse window). API and SQLite cache both verified working. Lowered `DIVERGENCE_Z_THRESHOLD` 1.0→0.75 (committed/pushed) to surface near-miss divergences.
- 2026-06-29 (session 3): Built Part 3 eval harness. Broadened pool (backfilled TSLA/JPM/NVDA → 776 headlines). Built `eval_harness/` package (consistency + accuracy + regression + SQLite history + CLI + 2 API endpoints). Constructed 66-headline golden set via error-discovery clustering + manual labelling UI. Validated end-to-end: headlines macro F1 0.848 (vs 0.872 analyst), consistency PASS (fails under train() mode), regression two-run demo correct, headline_scores untouched. README Part 3 + handoff updated. Not yet committed.
