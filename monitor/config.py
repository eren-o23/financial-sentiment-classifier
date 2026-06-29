"""Configuration: paths, label maps, model constants, and environment.

Label mapping mirrors the training notebook exactly (cells 12/14/18) so the saved
state_dict loads into an identically-shaped head.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "best_model.pt"
DB_PATH = PROJECT_ROOT / "monitor.db"

# --- Model (must match training notebook) ---
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
MODEL_VERSION = "distilbert-finphrasebank-75agree-v1"

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Signed orientation for the sentiment index: positive +1, neutral 0, negative -1
LABEL_SIGN = {"positive": 1, "neutral": 0, "negative": -1}

# --- News source ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
# Keep only headlines whose ticker relevance clears this bar (Alpha Vantage score 0-1)
RELEVANCE_THRESHOLD = 0.1

# --- Divergence detection ---
# A day is flagged when sentiment and price z-scores oppose in sign and both exceed
# this magnitude. Tunable.
DIVERGENCE_Z_THRESHOLD = 0.75
# Z-scores are meaningless on tiny samples (with n=2 every point is exactly ±1σ), so
# only compute divergences once we have at least this many paired observations.
DIVERGENCE_MIN_OBS = 5
