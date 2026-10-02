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

TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "gpt-4o-mini")
KNOWLEDGE_MODEL = os.getenv("KNOWLEDGE_MODEL", "gpt-4o-mini")
ACCOUNT_MODEL = os.getenv("ACCOUNT_MODEL", "gpt-4o-mini")
RESOLVER_MODEL = os.getenv("RESOLVER_MODEL", "gpt-4o")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gpt-4o-mini")

# USD per 1M tokens, (prompt, completion). Update to match your provider's current pricing.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
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
WARRANTY_MONTHS = 24
TRACKING_STALE_DAYS = 7
TRACKING_LOST_DAYS = 14

CATEGORIES = ("billing", "product", "shipping", "unknown")
ACTIONS = ("reply_only", "resend_tracking", "replace_unit", "refund", "escalate_to_human")
