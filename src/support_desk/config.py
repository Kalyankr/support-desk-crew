"""Single source of truth for paths, models, budgets and policy thresholds.

Policy thresholds live here rather than in prompts so that Phase 5's Policy Guard and the
golden set in `data/tickets/` can never drift apart.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
SEED_SQL = DATA_DIR / "seed.sql"
GOLDEN_TICKETS = DATA_DIR / "tickets" / "golden.json"

LOCAL_DIR = ROOT / ".local"
DB_PATH = LOCAL_DIR / "support_desk.db"
CHROMA_PATH = LOCAL_DIR / "chroma"
CHECKPOINT_PATH = LOCAL_DIR / "checkpoints.db"
TRACE_DIR = LOCAL_DIR / "traces"

# --- Fixed clock -------------------------------------------------------------

# Every policy window is measured against this date, never date.today(). A moving clock
# would silently invalidate the golden set and make eval failures impossible to attribute.
AS_OF = date(2026, 10, 1)

# --- Models ------------------------------------------------------------------

# OpenRouter is an OpenAI-compatible endpoint: langchain-openai's ChatOpenAI works unchanged
# once base_url and api_key point here. Model ids are OpenRouter's "provider/model" form.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "openai/gpt-4o-mini")
KNOWLEDGE_MODEL = os.getenv("KNOWLEDGE_MODEL", "openai/gpt-4o-mini")
ACCOUNT_MODEL = os.getenv("ACCOUNT_MODEL", "openai/gpt-4o-mini")
RESOLVER_MODEL = os.getenv("RESOLVER_MODEL", "openai/gpt-4o")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "openai/gpt-4o-mini")

# USD per 1M tokens, (prompt, completion). OpenRouter passes through provider pricing;
# check https://openrouter.ai/models for the current rate of whichever model you pick.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "google/gemma-4-26b-a4b-it:free": (0.0, 0.0),
}
FALLBACK_PRICING = (1.00, 3.00)

# --- Budgets -----------------------------------------------------------------

MAX_COST_PER_TICKET_USD = float(os.getenv("MAX_COST_PER_TICKET_USD", "0.05"))
MAX_TOKENS_PER_TICKET = int(os.getenv("MAX_TOKENS_PER_TICKET", "40000"))
MAX_REVISIONS = 2

# --- Policy thresholds -------------------------------------------------------

STANDARD_REFUND_WINDOW_DAYS = 30
PLUS_REFUND_WINDOW_DAYS = 60
APPROVAL_THRESHOLD_USD = 100.00
EXPEDITED_FEE_USD = 25.00
WARRANTY_MONTHS = 24
TRACKING_STALE_DAYS = 7
TRACKING_LOST_DAYS = 14

CATEGORIES = ("billing", "product", "shipping", "unknown")
ACTIONS = ("reply_only", "resend_tracking", "replace_unit", "refund", "escalate_to_human")
