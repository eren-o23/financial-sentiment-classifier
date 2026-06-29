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

---

## Next Step

The system is running. To get divergence flags, try a 3-month window (costs 1 Alpha Vantage request):

```bash
/opt/anaconda3/bin/python -m monitor.run_daily --ticker AAPL --start 2024-03-01 --end 2024-05-31
# Then view: http://127.0.0.1:8000/monitor/view?ticker=AAPL&start=2024-03-01&end=2024-05-31&refresh=false
```

**Note on POST /monitor fields**: the Pydantic model uses `start_date` and `end_date` (not `start`/`end`). The GET `/monitor/view` endpoint uses `start`/`end` query params, which is why the chart worked immediately.

---

## Open Questions / Blockers

- No blockers. System is live and confirmed working.
- 1-month AAPL window (May 2024) yields 0 divergent days because most days have 0.0 sentiment (sparse coverage), compressing z-scores. Extend to 3 months to surface divergences.

---

## Session History

_Append-only. One line per session — never overwrite previous entries._

- 2026-06-29: Built full project from scratch — Part 1 (DistilBERT fine-tune on Financial PhraseBank, 0.872 macro F1 vs 0.48 VADER) + Part 2 (sentiment monitoring system: Alpha Vantage → SQLite → classifier → daily index → Plotly/FastAPI). Fixed storage init bug, z-score minimum obs guard, SSH push config, em-dash style. All code committed and pushed. User obtained Alpha Vantage key; live run is next.
- 2026-06-29 (session 2): Confirmed first live run. 57 AAPL headlines scored for May 2024, chart rendering, 0 divergences (expected for 1-month sparse window). API and SQLite cache both verified working.
