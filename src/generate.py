"""
Synthetic behavioural-trace generator for burnout-risk modelling.

There is no public, ethically-usable dataset of raw personal behavioural signals
(screen time, app usage, mobility, sleep) labelled with burnout outcomes. We
therefore SIMULATE one, with an explicit, documented causal structure so the
downstream models learn a recoverable — not circular — signal:

    1. each person has a hidden `latent_stress` in [0, 1] (chronic workload);
    2. daily behaviours are generated as functions of that latent state, day of
       week, and noise:
           work_hours     ↑ with stress
           screen_evening ↑ with stress (doomscrolling after work)
           sleep_hours    ↓ with stress, and more variable
           sleep_onset    later & more irregular with stress
           steps          ↓ with stress (less activity)
           social_msgs    ↓ with stress (withdrawal)
           weekend_work   ↑ with stress (no recovery)
    3. a burnout OUTCOME is drawn from a logistic function of the *behaviours
       actually realised over the observation window* (not directly from the
       latent), plus noise — so the label is earned from the observable traces,
       exactly what a real early-warning system would have to work with.

The generator emits daily rows; feature engineering (features.py) aggregates
them to one row per person.

Run:
    python src/generate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DAILY_CSV = DATA_DIR / "raw" / "behavioral_daily.csv"

N_PEOPLE = 1200
DAYS = 56  # 8-week observation window


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(save: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    # Hidden chronic-stress level per person (mixture: mostly ok, a tail high).
    latent = np.clip(rng.beta(2.2, 3.0, N_PEOPLE), 0, 1)

    rows = []
    for pid in range(N_PEOPLE):
        s = latent[pid]
        # stable personal offsets
        base_sleep = rng.normal(7.5 - 1.8 * s, 0.3)
        base_steps = rng.normal(8500 - 3500 * s, 800)
        base_work = rng.normal(7.5 + 3.0 * s, 0.4)
        for d in range(DAYS):
            dow = d % 7
            weekend = dow >= 5

            work = max(0.0, rng.normal(
                base_work * (0.3 if weekend else 1.0)
                + (2.5 * s if weekend else 0.0), 1.0))
            screen_evening = max(0.0, rng.normal(1.5 + 3.0 * s, 0.6))
            sleep = float(np.clip(rng.normal(
                base_sleep - (0.6 if not weekend else 0.0), 0.6 + 0.8 * s),
                3.0, 11.0))
            # sleep onset hour (24h+), later & noisier with stress
            onset = rng.normal(23.0 + 2.2 * s, 0.4 + 1.0 * s)
            steps = max(0.0, rng.normal(
                base_steps * (1.1 if weekend else 1.0), 1500))
            social = max(0, rng.normal(40 - 25 * s, 10))
            rows.append(dict(
                person_id=pid, day=d, dow=dow, weekend=weekend,
                work_hours=round(work, 2),
                screen_evening_h=round(screen_evening, 2),
                sleep_hours=round(sleep, 2),
                sleep_onset_h=round(onset, 2),
                steps=int(steps),
                social_msgs=int(social),
                latent_stress=round(s, 3)))

    df = pd.DataFrame(rows)

    # ---- burnout outcome from realised behaviours over the window --------
    agg = df.groupby("person_id").agg(
        work=("work_hours", "mean"),
        screen=("screen_evening_h", "mean"),
        sleep=("sleep_hours", "mean"),
        sleep_var=("sleep_hours", "std"),
        onset_var=("sleep_onset_h", "std"),
        steps=("steps", "mean"),
        social=("social_msgs", "mean"),
        weekend_work=("work_hours",
                      lambda x: x[df.loc[x.index, "weekend"]].mean()),
    )
    z = (0.9 * _std(agg.work) + 0.8 * _std(agg.screen)
         - 0.7 * _std(agg.sleep) + 0.6 * _std(agg.sleep_var)
         + 0.5 * _std(agg.onset_var) - 0.5 * _std(agg.steps)
         - 0.4 * _std(agg.social) + 0.6 * _std(agg.weekend_work))
    prob = _sigmoid(1.0 * z - 2.7)  # intercept/slope tuned to a realistic base rate (~22%)
    outcome = rng.binomial(1, np.clip(prob, 0.01, 0.99))
    labels = pd.DataFrame({
        "person_id": agg.index.to_numpy(), "burnout": outcome,
        "burnout_prob_true": prob.round(3)}).reset_index(drop=True)
    df = df.merge(labels, on="person_id")

    if save:
        DAILY_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DAILY_CSV, index=False)
        print(f"[generate] {df.person_id.nunique()} people x {DAYS} days "
              f"= {len(df):,} daily rows")
        print(f"[generate] burnout prevalence: "
              f"{labels.burnout.mean()*100:.1f}%")
        print(f"[generate] saved -> {DAILY_CSV.relative_to(DATA_DIR.parent)}")
    return df


def _std(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def load_daily() -> pd.DataFrame:
    if DAILY_CSV.exists():
        return pd.read_csv(DAILY_CSV)
    return generate()


if __name__ == "__main__":
    frame = generate()
    print(frame.head())
