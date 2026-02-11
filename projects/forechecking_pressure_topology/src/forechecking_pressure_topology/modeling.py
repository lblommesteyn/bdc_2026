from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


CORE_PRESSURE_FEATURES = [
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

ROLE_AWARE_FEATURES = [
    "mean_d_pinch_support",
    "mean_f1_distance",
    "mean_f2_distance",
    "mean_f3_distance",
    "mean_f1_pressure_contrib",
    "mean_f2_pressure_contrib",
    "mean_f3_pressure_contrib",
    "mean_f1_share_pressure",
    "mean_f2_share_pressure",
    "mean_f3_share_pressure",
]

PRESSURE_ROLE_FEATURES = CORE_PRESSURE_FEATURES + ROLE_AWARE_FEATURES
SCORE_CONTEXT_NUMERIC_FEATURES = ["score_diff_forechecking_start"]
SCORE_CONTEXT_CATEGORICAL_FEATURES = ["forechecking_score_state"]

BASELINE_DISTANCE_FEATURES = ["mean_nearest_defender_distance"]
BASELINE_COUNT_FEATURES = ["mean_defenders_near_carrier", "mean_pinchers"]


def _available_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [col for col in feature_cols if col in df.columns]


def _prepare_design_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    keep = list(dict.fromkeys(numeric_features + categorical_features + [target_col]))
    model_df = df[keep].replace([np.inf, -np.inf], np.nan)
    required = numeric_features + [target_col]
    model_df = model_df.dropna(subset=required)
    if model_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    if categorical_features:
        for col in categorical_features:
            model_df[col] = model_df[col].fillna("unknown").astype(str)
        X = pd.get_dummies(
            model_df[numeric_features + categorical_features],
            columns=categorical_features,
            drop_first=True,
            dtype=float,
        )
    else:
        X = model_df[numeric_features].astype(float)

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
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        proba = fold_model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def _add_score_state_adjustments(
    features_df: pd.DataFrame,
    cols_to_adjust: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    adjusted = features_df.copy()
    required_cols = {"forechecking_team", "forechecking_score_state"}
    if not required_cols.issubset(adjusted.columns):
        return adjusted, []

    group_keys = ["forechecking_team", "forechecking_score_state"]
    adjusted_cols: list[str] = []
    for col in cols_to_adjust:
        if col not in adjusted.columns:
            continue
        group_mean = adjusted.groupby(group_keys)[col].transform("mean")
        adj_col = f"adj_{col}"
        adjusted[adj_col] = adjusted[col] - group_mean
        adjusted_cols.append(adj_col)
    return adjusted, adjusted_cols


def _feature_string(numeric_features: list[str], categorical_features: list[str]) -> str:
    features = list(numeric_features)
    features.extend([f"{c} (one-hot)" for c in categorical_features])
    return ",".join(features)


def run_predictive_validation(features_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []
    features_with_adjustments, adjusted_features = _add_score_state_adjustments(features_df, PRESSURE_ROLE_FEATURES)

    candidate_models = [
        {
            "model_name": "pressure_logreg",
            "numeric_features": CORE_PRESSURE_FEATURES,
            "categorical_features": [],
            "model": LogisticRegression(max_iter=2500),
        },
        {
            "model_name": "pressure_gb",
            "numeric_features": CORE_PRESSURE_FEATURES,
            "categorical_features": [],
            "model": GradientBoostingClassifier(random_state=7),
        },
        {
            "model_name": "pressure_role_logreg",
            "numeric_features": PRESSURE_ROLE_FEATURES,
            "categorical_features": [],
            "model": LogisticRegression(max_iter=3000),
        },
        {
            "model_name": "pressure_role_score_logreg",
            "numeric_features": PRESSURE_ROLE_FEATURES + SCORE_CONTEXT_NUMERIC_FEATURES,
            "categorical_features": SCORE_CONTEXT_CATEGORICAL_FEATURES,
            "model": LogisticRegression(max_iter=3000),
        },
        {
            "model_name": "baseline_nearest_distance",
            "numeric_features": BASELINE_DISTANCE_FEATURES,
            "categorical_features": [],
            "model": LogisticRegression(max_iter=1000),
        },
        {
            "model_name": "baseline_counts",
            "numeric_features": BASELINE_COUNT_FEATURES,
            "categorical_features": [],
            "model": LogisticRegression(max_iter=1000),
        },
    ]
    if adjusted_features:
        candidate_models.append(
            {
                "model_name": "pressure_role_score_adjusted_logreg",
                "numeric_features": adjusted_features + SCORE_CONTEXT_NUMERIC_FEATURES,
                "categorical_features": SCORE_CONTEXT_CATEGORICAL_FEATURES,
                "model": LogisticRegression(max_iter=3500),
            }
        )

    for spec in candidate_models:
        model_name = spec["model_name"]
        numeric_features = _available_features(features_with_adjustments, spec["numeric_features"])
        categorical_features = _available_features(features_with_adjustments, spec["categorical_features"])
        if not numeric_features and not categorical_features:
            auc_mean = np.nan
            auc_std = np.nan
            n_obs = 0
            feature_str = ""
        else:
            X, y = _prepare_design_matrix(
                features_with_adjustments,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_col="turnover_in_dzone",
            )
            n_obs = len(X)
            feature_str = _feature_string(numeric_features, categorical_features)
            if len(X) < 20 or y.nunique() < 2 or X.shape[1] == 0:
                auc_mean = np.nan
                auc_std = np.nan
            else:
                auc_mean, auc_std = _cv_auc(spec["model"], X, y)
        records.append(
            {
                "model": model_name,
                "n_obs": int(n_obs),
                "features": feature_str,
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
    clustering_features = _available_features(usable, PRESSURE_ROLE_FEATURES)
    if not clustering_features:
        return pd.DataFrame(), pd.DataFrame()

    usable = usable.dropna(subset=clustering_features)
    if usable.empty:
        return pd.DataFrame(), pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(usable[clustering_features])

    n_clusters = max(2, min(n_clusters, len(usable)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init=20)
    labels = kmeans.fit_predict(X)
    usable["cluster_id"] = labels

    summary = (
        usable.groupby("cluster_id")[clustering_features + ["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]]
        .mean()
        .reset_index()
    )
    summary["n_episodes"] = usable.groupby("cluster_id")["episode_id"].size().values

    output_dir.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output_dir / "episode_clusters.csv", index=False)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    return usable, summary
