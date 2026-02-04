from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PLAYER_ARCHETYPE_FEATURES = [
    "sca_per_100_frames",
    "cut_rate",
    "drag_rate",
    "screen_rate",
    "decoy_rate",
    "shot_in_5s_rate",
]

POSSESSION_FEATURES = [
    "possession_sca_total",
    "possession_sca_mean",
    "possession_cut_rate",
    "possession_drag_rate",
    "possession_screen_rate",
    "possession_decoy_rate",
]

POSSESSION_BASELINE = ["duration_seconds"]


def run_player_archetypes(player_summary: pd.DataFrame, output_dir: Path, n_clusters: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if player_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    usable = player_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=PLAYER_ARCHETYPE_FEATURES).copy()
    if usable.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_clusters = max(2, min(n_clusters, len(usable)))
    X = StandardScaler().fit_transform(usable[PLAYER_ARCHETYPE_FEATURES])
    km = KMeans(n_clusters=n_clusters, random_state=11, n_init=20)
    usable["archetype_id"] = km.fit_predict(X)

    archetypes = (
        usable.groupby("archetype_id")[PLAYER_ARCHETYPE_FEATURES + ["sca_total", "sca_mean", "n_frames", "possessions"]]
        .mean()
        .reset_index()
    )
    archetypes["n_players"] = usable.groupby("archetype_id")["player_id"].size().values

    usable.to_csv(output_dir / "player_archetypes.csv", index=False)
    archetypes.to_csv(output_dir / "archetype_summary.csv", index=False)
    return usable, archetypes


def _cv_auc(X: pd.DataFrame, y: pd.Series, model=None) -> tuple[float, float]:
    if model is None:
        model = LogisticRegression(max_iter=2000)
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return np.nan, np.nan
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        return np.nan, np.nan

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=11)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def run_outcome_validation(possession_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if possession_summary.empty:
        return pd.DataFrame()

    rows = []
    candidates = [
        ("sca_signature", POSSESSION_FEATURES),
        ("duration_baseline", POSSESSION_BASELINE),
    ]

    for name, feat_cols in candidates:
        model_df = possession_summary[feat_cols + ["shot_or_goal_in_possession"]].replace([np.inf, -np.inf], np.nan).dropna()
        if model_df.empty or model_df["shot_or_goal_in_possession"].nunique() < 2:
            auc_mean, auc_std = np.nan, np.nan
            n_obs = len(model_df)
        else:
            X = model_df[feat_cols]
            y = model_df["shot_or_goal_in_possession"].astype(int)
            auc_mean, auc_std = _cv_auc(X, y)
            n_obs = len(X)
        rows.append(
            {
                "model": name,
                "n_obs": int(n_obs),
                "features": ",".join(feat_cols),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "outcome_validation.csv", index=False)
    payload = {"models": out.to_dict(orient="records")}
    with (output_dir / "outcome_validation.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out

