"""
Feature engineering: from daily traces to one behavioural profile per person.

The daily signals are aggregated into interpretable indicators grouped in three
families that map onto the burnout literature:

    OVERLOAD    — mean work hours, evening screen time, weekend work ratio;
    RECOVERY    — mean sleep, mean steps, mean social messages;
    REGULARITY  — variability of sleep duration and sleep onset (circadian
                  instability), and a "routine break" score = how much the last
                  two weeks deviate from the first two (rising instability).

`latent_stress` and the probability are kept ONLY for diagnostics/plots and are
dropped before any modelling to avoid leakage.

Run:
    python src/features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from generate import load_daily

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURES_CSV = DATA_DIR / "processed" / "person_features.csv"

FEATURE_COLS = [
    "work_hours_mean", "screen_evening_mean", "weekend_work_ratio",
    "sleep_mean", "steps_mean", "social_mean",
    "sleep_var", "onset_var", "routine_break",
]


def _routine_break(g: pd.DataFrame, col: str) -> float:
    """Absolute shift in a signal's mean between the first and last fortnight."""
    early = g[g.day < 14][col].mean()
    late = g[g.day >= g.day.max() - 13][col].mean()
    return float(abs(late - early))


def build_features(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_daily() if daily is None else daily
    out = []
    for pid, g in df.groupby("person_id"):
        weekday = g[~g.weekend.astype(bool)]
        weekend = g[g.weekend.astype(bool)]
        rec = {
            "person_id": pid,
            "work_hours_mean": g.work_hours.mean(),
            "screen_evening_mean": g.screen_evening_h.mean(),
            "weekend_work_ratio": (weekend.work_hours.mean()
                                   / max(weekday.work_hours.mean(), 1e-6)),
            "sleep_mean": g.sleep_hours.mean(),
            "steps_mean": g.steps.mean(),
            "social_mean": g.social_msgs.mean(),
            "sleep_var": g.sleep_hours.std(),
            "onset_var": g.sleep_onset_h.std(),
            "routine_break": (_routine_break(g, "sleep_hours")
                              + _routine_break(g, "work_hours")
                              + _routine_break(g, "steps") / 1000.0),
            # diagnostics only (dropped before modelling)
            "latent_stress": g.latent_stress.iloc[0],
            "burnout": int(g.burnout.iloc[0]),
            "burnout_prob_true": g.burnout_prob_true.iloc[0],
        }
        out.append(rec)
    feats = pd.DataFrame(out)
    return feats


def save(df: pd.DataFrame) -> None:
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[features] saved -> {FEATURES_CSV.relative_to(DATA_DIR.parent)}")


def load_features() -> pd.DataFrame:
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV)
    out = build_features()
    save(out)
    return out


if __name__ == "__main__":
    frame = build_features()
    save(frame)
    print(f"{len(frame)} people | {len(FEATURE_COLS)} features")
    print(frame[FEATURE_COLS].describe().round(2).T)
    # quick correlation with the outcome (diagnostic)
    corr = frame[FEATURE_COLS + ["burnout"]].corr()["burnout"].drop("burnout")
    print("\ncorrelation with burnout:\n",
          corr.sort_values(key=abs, ascending=False).round(3))
