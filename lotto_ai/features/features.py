import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    CSV_DRAWS_PATH,
    MAX_NUMBER,
    MIN_NUMBER,
    NUMBERS_PER_DRAW,
)

NUMBERS = range(MIN_NUMBER, MAX_NUMBER + 1)


def load_draws():
    df = pd.read_csv(CSV_DRAWS_PATH)

    num_cols_upper = [f"NUM{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]
    upper_map = {c.upper(): c for c in df.columns}
    if all(c in upper_map for c in num_cols_upper):
        numbers = df[[upper_map[c] for c in num_cols_upper]].copy()
    else:
        numbers = df.iloc[:, :NUMBERS_PER_DRAW].copy()

    numbers = numbers.apply(pd.to_numeric, errors="coerce").dropna().astype(int)

    date_col = None
    for candidate in ("draw_date", "date", "datum", "Date", "Datum"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        draw_dates = pd.Series(
            [f"draw_{i+1:05d}" for i in range(len(numbers))],
            index=numbers.index,
        )
    else:
        draw_dates = df.loc[numbers.index, date_col].astype(str)

    out = pd.DataFrame({"draw_date": draw_dates.values}, index=numbers.index)
    for i in range(1, NUMBERS_PER_DRAW + 1):
        out[f"n{i}"] = numbers.iloc[:, i - 1].values

    return out.reset_index(drop=True)


def build_feature_matrix(window=10):
    df = load_draws()

    records = []

    for number in NUMBERS:
        appeared = df[
            (df[[f"n{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]] == number).any(axis=1)
        ]

        hits = appeared.index.tolist()

        for i in range(1, len(df)):
            past_hits = [h for h in hits if h < i]

            records.append(
                {
                    "number": number,
                    "draw_index": i,
                    "freq": len(past_hits) / i,
                    "gap": i - past_hits[-1] if past_hits else i,
                    "rolling_freq": sum(h >= i - window for h in past_hits) / window,
                    "hit": int(i in hits),
                }
            )

    return pd.DataFrame(records)
