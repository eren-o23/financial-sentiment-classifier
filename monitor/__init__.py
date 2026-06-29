"""Financial Sentiment Monitor.

Extends the fine-tuned DistilBERT classifier into a monitoring tool: pulls live
financial news for a ticker, scores each headline, aggregates a daily sentiment
index, and surfaces divergences between market language and price movement.

This is a monitoring tool, not a prediction tool — price is used as context to
interpret sentiment, not as a target to forecast.
"""

__version__ = "0.1.0"
