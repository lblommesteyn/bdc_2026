from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _plot_possession_quintiles(pos: pd.DataFrame, out_dir: Path) -> None:
    q_df = pos.copy()
    q_df = q_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["possession_sca_total", "shot_or_goal_in_possession"])
    q_df["sca_quintile"] = pd.qcut(q_df["possession_sca_total"], 5, labels=["Q1 Low", "Q2", "Q3", "Q4", "Q5 High"])
    summary = (
        q_df.groupby("sca_quintile", observed=False)["shot_or_goal_in_possession"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "shot_rate", "size": "n"})
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(summary["sca_quintile"], 100 * summary["shot_rate"], marker="o", color="#1f78b4", linewidth=2.2)
    for _, r in summary.iterrows():
        ax.text(r["sca_quintile"], 100 * r["shot_rate"] + 0.6, f"{100*r['shot_rate']:.1f}%", ha="center")
    ax.set_ylim(0, max(100 * summary["shot_rate"]) * 1.2)
    ax.set_ylabel("Possessions ending with shot/goal (%)")
    ax.set_title("Shot Conversion Improves with Higher Possession SCA")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_possession_quintiles.png")
    plt.close(fig)


def _plot_player_leaderboard(player: pd.DataFrame, out_dir: Path) -> None:
    p = player[player["n_frames"] >= 120].copy()
    p = p.sort_values("sca_per_100_frames", ascending=False)
    top = p.head(12)
    bot = p.tail(12).sort_values("sca_per_100_frames", ascending=True)
    board = pd.concat([top, bot], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = ["#1b9e77"] * len(top) + ["#d95f02"] * len(bot)
    ax.barh(board["player_id"], board["sca_per_100_frames"], color=colors)
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_xlabel("SCA per 100 off-puck frames")
    ax.set_title("Player Space Creation Leaderboard (min 120 sampled frames)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_player_leaderboard.png")
    plt.close(fig)


def _plot_archetype_heatmap(archetypes: pd.DataFrame, out_dir: Path) -> None:
    feat_cols = ["sca_per_100_frames", "cut_rate", "drag_rate", "screen_rate", "decoy_rate", "shot_in_5s_rate"]
    a = archetypes.copy()
    a = a[a["n_players"] >= 5].sort_values("archetype_id")
    X = a[feat_cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-1.7, vmax=1.7)
    ax.set_yticks(np.arange(len(a)))
    ax.set_yticklabels([f"A{int(x)} (n={int(n)})" for x, n in zip(a["archetype_id"], a["n_players"])])
    ax.set_xticks(np.arange(len(feat_cols)))
    ax.set_xticklabels(["SCA/100", "Cut", "Drag", "Screen", "Decoy", "Shot+5s"])
    ax.set_title("Off-Puck Archetypes (z-scored feature profile)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("z-score")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_archetype_heatmap.png")
    plt.close(fig)


def _cv_auc(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[float, float]:
    model_df = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        model = LogisticRegression(max_iter=2000)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def _plot_model_auc(pos: pd.DataFrame, out_dir: Path) -> None:
    models = [
        ("Duration only", ["duration_seconds"], "#a6bddb"),
        (
            "SCA signature",
            [
                "possession_sca_total",
                "possession_sca_mean",
                "possession_cut_rate",
                "possession_drag_rate",
                "possession_screen_rate",
                "possession_decoy_rate",
            ],
            "#67a9cf",
        ),
        (
            "Duration + SCA",
            [
                "duration_seconds",
                "possession_sca_total",
                "possession_sca_mean",
                "possession_cut_rate",
                "possession_drag_rate",
                "possession_screen_rate",
                "possession_decoy_rate",
            ],
            "#02818a",
        ),
    ]
    rows = []
    for name, cols, color in models:
        mean_auc, std_auc = _cv_auc(pos, cols, "shot_or_goal_in_possession")
        rows.append({"name": name, "auc": mean_auc, "std": std_auc, "color": color})
    m = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(m["name"], m["auc"], color=m["color"])
    ax.errorbar(m["name"], m["auc"], yerr=m["std"], fmt="none", ecolor="#333", capsize=3)
    ax.set_ylim(0.55, 0.78)
    ax.set_ylabel("Cross-validated AUC")
    ax.set_title("Possession Shot Prediction: Incremental Value of SCA")
    for b, v in zip(bars, m["auc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_auc.png")
    plt.close(fig)


def _plot_action_effects(frame: pd.DataFrame, out_dir: Path) -> None:
    f = frame.copy()
    metrics = {
        "cut": "is_cut",
        "drag": "is_drag",
        "screen": "is_screen",
        "decoy": "is_decoy",
    }
    rows = []
    for label, col in metrics.items():
        on = f[f[col] == 1]["sca_delta_opportunity"]
        off = f[f[col] == 0]["sca_delta_opportunity"]
        rows.append(
            {
                "action": label.title(),
                "delta_when_on": float(on.mean()),
                "delta_when_off": float(off.mean()),
                "lift": float(on.mean() - off.mean()),
                "support": int(len(on)),
            }
        )
    m = pd.DataFrame(rows).sort_values("lift", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(m["action"], m["lift"], color="#7570b3")
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_ylabel("Mean SCA lift when action is present")
    ax.set_title("Estimated Action-Level Lift on Opportunity")
    for b, n in zip(bars, m["support"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + np.sign(b.get_height()) * 0.0003, f"n={n:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_action_lift.png")
    plt.close(fig)


def _plot_spatial_sca(frame: pd.DataFrame, out_dir: Path) -> None:
    f = frame.copy()
    f = f[f["sca_delta_opportunity"].notna() & f["player_x"].notna() & f["player_y"].notna()]
    if f.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    hb = ax.hexbin(
        f["player_x"],
        f["player_y"],
        C=f["sca_delta_opportunity"],
        reduce_C_function=np.mean,
        gridsize=26,
        extent=(-100, 100, -42.5, 42.5),
        cmap="coolwarm",
        mincnt=20,
        vmin=-0.015,
        vmax=0.015,
    )
    ax.axvline(-25, color="#555", lw=0.6, ls="--")
    ax.axvline(25, color="#555", lw=0.6, ls="--")
    ax.axhline(0, color="#555", lw=0.5, ls=":")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_xlabel("X (ft)")
    ax.set_ylabel("Y (ft)")
    ax.set_title("Where Off-Puck Movement Adds Opportunity (mean SCA by location)")
    cbar = fig.colorbar(hb, ax=ax, shrink=0.9)
    cbar.set_label("Mean SCA delta")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_spatial_sca.png")
    plt.close(fig)


def main() -> None:
    _style()
    out_dir = Path("projects/off_puck_space_creation_value/report_v1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path("projects/off_puck_space_creation_value/outputs")
    frame = pd.read_csv(base / "frame_contributions.csv")
    player = pd.read_csv(base / "player_sca_summary.csv")
    pos = pd.read_csv(base / "possession_sca_summary.csv")
    archetypes = pd.read_csv(base / "archetypes/archetype_summary.csv")

    _plot_possession_quintiles(pos, out_dir)
    _plot_player_leaderboard(player, out_dir)
    _plot_archetype_heatmap(archetypes, out_dir)
    _plot_action_effects(frame, out_dir)
    _plot_spatial_sca(frame, out_dir)
    _plot_model_auc(pos, out_dir)


if __name__ == "__main__":
    main()
