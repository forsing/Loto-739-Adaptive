"""
Production model - imports from core.models
Kept for backward compatibility
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.models import (
    generate_adaptive_portfolio,
    portfolio_statistics,
    generate_ticket_safe,
    frequency_probability
)

__all__ = [
    'generate_adaptive_portfolio',
    'portfolio_statistics',
    'generate_ticket_safe',
    'frequency_probability'
]