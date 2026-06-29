"""FastAPI service for the sentiment monitor.

The 255 MB model is loaded once at startup (lifespan) and reused across requests.

Endpoints:
  POST /monitor          -> JSON: sentiment_series, price_series, divergences, stats
  GET  /monitor/view     -> interactive Plotly HTML page
  POST /eval             -> JSON: full eval report (accuracy, consistency, regression)
  GET  /eval/history     -> JSON: past eval runs (regression trend)
  GET  /health           -> liveness check
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from eval_harness.config import GOLDEN_PATH
from eval_harness.runner import run_eval
from eval_harness.storage import fetch_history

from .classifier import FinancialSentimentClassifier
from .pipeline import run_monitor
from .viz import figure_to_html

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the checkpoint once for the process lifetime.
    _state["classifier"] = FinancialSentimentClassifier()
    yield
    _state.clear()


app = FastAPI(title="Financial Sentiment Monitor", version="0.1.0", lifespan=lifespan)


def _default_start() -> str:
    return (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


class MonitorRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start_date: str = Field(default_factory=_default_start, examples=["2024-05-01"])
    end_date: str = Field(default_factory=_today, examples=["2024-05-31"])
    refresh: bool = Field(
        default=False,
        description="If true, fetch+classify new headlines from Alpha Vantage. "
        "If false, serve only what is already cached in SQLite.",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "classifier" in _state}


@app.post("/monitor")
def monitor(req: MonitorRequest) -> dict:
    try:
        result = run_monitor(
            req.ticker, req.start_date, req.end_date,
            classifier=_state["classifier"], refresh=req.refresh,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ticker": result.ticker,
        "start": result.start,
        "end": result.end,
        "stats": result.stats,
        "sentiment_series": result.sentiment_series,
        "price_series": result.price_series,
        "divergences": result.divergences,
    }


@app.get("/monitor/view", response_class=HTMLResponse)
def monitor_view(
    ticker: str = Query(..., examples=["AAPL"]),
    start: str = Query(default_factory=_default_start),
    end: str = Query(default_factory=_today),
    refresh: bool = Query(default=False),
) -> HTMLResponse:
    try:
        result = run_monitor(
            ticker, start, end, classifier=_state["classifier"], refresh=refresh
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return HTMLResponse(content=figure_to_html(result))


class EvalRequest(BaseModel):
    golden_path: str = Field(
        default=str(GOLDEN_PATH),
        description="Path to the golden JSONL dataset. Defaults to the committed set.",
    )


@app.post("/eval")
def eval_endpoint(req: EvalRequest = EvalRequest()) -> dict:
    """Run the full eval (consistency + accuracy + regression) and persist it."""
    try:
        report = run_eval(_state["classifier"], golden_path=req.golden_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return report.to_dict()


@app.get("/eval/history")
def eval_history(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    """Return past eval runs (newest first) to track performance over time."""
    return {"runs": fetch_history(limit=limit)}
