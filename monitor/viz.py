"""Plotly dual-axis visualisation: sentiment index vs price, divergences flagged."""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pipeline import MonitorResult


def build_figure(result: MonitorResult) -> go.Figure:
    """Dual-axis figure: daily sentiment index (left) and close price (right)."""
    price = result.price_series
    dates = [r["date"] for r in price]
    closes = [r["close"] for r in price]
    sentiment = [r.get("sentiment_index") for r in price]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price line (right axis)
    fig.add_trace(
        go.Scatter(
            x=dates, y=closes, name="Close price",
            line=dict(color="#1f2937", width=2),
            hovertemplate="%{x}<br>Close: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Sentiment index (left axis) — gaps for days with no news
    fig.add_trace(
        go.Scatter(
            x=dates, y=sentiment, name="Sentiment index",
            mode="lines+markers", connectgaps=False,
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5),
            hovertemplate="%{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Neutral reference line at sentiment = 0
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1, dash="dot"), secondary_y=False)

    # Divergence markers (red diamonds on the price line)
    if result.divergences:
        close_by_date = {r["date"]: r["close"] for r in price}
        div_dates = [d["date"] for d in result.divergences]
        div_y = [close_by_date.get(d) for d in div_dates]
        div_text = [
            f"{d['note']}<br>sentiment z={d['sentiment_z']}, price z={d['price_z']}"
            for d in result.divergences
        ]
        fig.add_trace(
            go.Scatter(
                x=div_dates, y=div_y, name="Divergence",
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#dc2626",
                            line=dict(width=1, color="#7f1d1d")),
                text=div_text,
                hovertemplate="%{x}<br>%{text}<extra></extra>",
            ),
            secondary_y=True,
        )

    n_div = len(result.divergences)
    fig.update_layout(
        title=(
            f"{result.ticker} — Sentiment vs Price ({result.start} to {result.end})"
            f"<br><sub>Monitoring tool: {n_div} divergence day(s) flagged where "
            "language and price move opposite ways</sub>"
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Net sentiment index (−1 to +1)", secondary_y=False)
    fig.update_yaxes(title_text="Close price", secondary_y=True)
    fig.update_xaxes(title_text="Date")
    return fig


def figure_to_html(result: MonitorResult) -> str:
    """Standalone HTML document for the served view endpoint."""
    return build_figure(result).to_html(include_plotlyjs="cdn", full_html=True)
