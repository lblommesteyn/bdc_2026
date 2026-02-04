from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PRESSURE_FEATURES = [
    "mean_pressure_at_carrier",
    "peak_pressure_at_carrier",
    "mean_gradient_mag",
    "mean_funnel_to_boards",
    "mean_open_corridor_cost",
    "mean_compactness_area",
    "mean_defenders_near_carrier",
    "mean_local_pressure",
    "mean_pinchers",
    "share_time_high_pressure",
    "corridor_closure_rate_3s",
]

BASELINE_DISTANCE_FEATURES = ["mean_nearest_defender_distance"]
BASELINE_COUNT_FEATURES = ["mean_defenders_near_carrier", "mean_pinchers"]


def _clean_feature_frame(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    keep = feature_cols + [target_col]
    model_df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)
    return X, y


def _cv_auc(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> tuple[float, float]:
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return np.nan, np.nan
    folds = min(n_splits, int(class_counts.min()))
    if folds < 2:
        return np.nan, np.nan

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def run_predictive_validation(features_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []

    candidate_models = [
        (
            "pressure_logreg",
            PRESSURE_FEATURES,
            LogisticRegression(max_iter=2000),
        ),
        (
            "pressure_gb",
            PRESSURE_FEATURES,
            GradientBoostingClassifier(random_state=7),
        ),
        (
            "baseline_nearest_distance",
            BASELINE_DISTANCE_FEATURES,
            LogisticRegression(max_iter=1000),
        ),
        (
            "baseline_counts",
            BASELINE_COUNT_FEATURES,
            LogisticRegression(max_iter=1000),
        ),
    ]

    for model_name, feature_cols, model in candidate_models:
        X, y = _clean_feature_frame(features_df, feature_cols, "turnover_in_dzone")
        if len(X) < 20 or y.nunique() < 2:
            auc_mean = np.nan
            auc_std = np.nan
            n_obs = len(X)
        else:
            auc_mean, auc_std = _cv_auc(model, X, y)
            n_obs = len(X)
        records.append(
            {
                "model": model_name,
                "n_obs": int(n_obs),
                "features": ",".join(feature_cols),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            }
        )

    out = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictive_validation.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "n_episodes": int(features_df["episode_id"].nunique()) if "episode_id" in features_df.columns else int(len(features_df)),
        "target_rate_turnover_in_dzone": float(features_df["turnover_in_dzone"].mean())
        if "turnover_in_dzone" in features_df.columns
        else np.nan,
        "models": out.to_dict(orient="records"),
    }
    with (output_dir / "predictive_validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def run_archetype_clustering(
    features_df: pd.DataFrame,
    output_dir: Path,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = features_df.copy()
    usable = usable.replace([np.inf, -np.inf], np.nan)
    usable = usable.dropna(subset=PRESSURE_FEATURES)
    if usable.empty:
        return pd.DataFrame(), pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(usable[PRESSURE_FEATURES])

    n_clusters = max(2, min(n_clusters, len(usable)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init=20)
    labels = kmeans.fit_predict(X)
    usable["cluster_id"] = labels

    summary = (
        usable.groupby("cluster_id")[PRESSURE_FEATURES + ["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]]
        .mean()
        .reset_index()
    )
    summary["n_episodes"] = usable.groupby("cluster_id")["episode_id"].size().values

    output_dir.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output_dir / "episode_clusters.csv", index=False)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    return usable, summary

