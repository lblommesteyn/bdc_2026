from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _plot_outcome_distribution(episodes: pd.DataFrame, out_dir: Path) -> None:
    order = ["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed", "stoppage", "timeout"]
    labels = {
        "turnover_in_dzone": "Turnover in D-zone",
        "forced_dump_out": "Forced dump-out",
        "controlled_exit_allowed": "Controlled exit allowed",
        "stoppage": "Stoppage",
        "timeout": "Timeout",
    }
    counts = episodes["outcome"].value_counts().reindex(order, fill_value=0)
    rates = 100.0 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.barh([labels[k] for k in order], rates.values, color=["#1b9e77", "#66a61e", "#d95f02", "#7570b3", "#bdbdbd"])
    ax.set_xlabel("Episode share (%)")
    ax.set_title("Forecheck Episode Outcomes (n = {:,})".format(len(episodes)))
    for bar, rate, count in zip(bars, rates.values, counts.values):
        ax.text(rate + 0.4, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}% ({count})", va="center")
    ax.set_xlim(0, max(rates.values) * 1.22)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_outcome_distribution.png")
    plt.close(fig)


def _plot_start_maps(episodes: pd.DataFrame, out_dir: Path) -> None:
    x = episodes["start_x"].to_numpy()
    y = episodes["start_y"].to_numpy()
    z = episodes["turnover_in_dzone"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)

    hb1 = axes[0].hexbin(x, y, gridsize=24, extent=(-100, 100, -42.5, 42.5), cmap="Blues", mincnt=1)
    axes[0].set_title("Episode Start Density")
    axes[0].set_xlabel("X (ft)")
    axes[0].set_ylabel("Y (ft)")
    axes[0].set_xlim(-100, 100)
    axes[0].set_ylim(-42.5, 42.5)
    cbar1 = fig.colorbar(hb1, ax=axes[0], shrink=0.9)
    cbar1.set_label("Episodes per hex")

    hb2 = axes[1].hexbin(
        x,
        y,
        C=z,
        reduce_C_function=np.mean,
        gridsize=24,
        extent=(-100, 100, -42.5, 42.5),
        cmap="RdYlGn",
        mincnt=8,
        vmin=0.1,
        vmax=0.6,
    )
    axes[1].set_title("Turnover Rate by Start Location")
    axes[1].set_xlabel("X (ft)")
    axes[1].set_ylabel("Y (ft)")
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(-42.5, 42.5)
    cbar2 = fig.colorbar(hb2, ax=axes[1], shrink=0.9)
    cbar2.set_label("P(turnover in D-zone)")

    for ax in axes:
        ax.axvline(-25, color="#555", lw=0.6, ls="--")
        ax.axvline(25, color="#555", lw=0.6, ls="--")
        ax.axhline(0, color="#555", lw=0.5, ls=":")

    fig.savefig(out_dir / "fig_start_maps.png")
    plt.close(fig)


def _plot_cluster_outcomes(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    rates = (
        clustered_episodes.groupby("cluster_id")[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]]
        .mean()
        .sort_index()
    )
    n = clustered_episodes.groupby("cluster_id").size().sort_index()
    clusters = rates.index.to_list()

    x = np.arange(len(clusters))
    w = 0.26

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - w, 100 * rates["turnover_in_dzone"].values, width=w, label="Turnover in D-zone", color="#1b9e77")
    ax.bar(x, 100 * rates["forced_dump_out"].values, width=w, label="Forced dump-out", color="#66a61e")
    ax.bar(x + w, 100 * rates["controlled_exit_allowed"].values, width=w, label="Controlled exit allowed", color="#d95f02")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_ylabel("Outcome rate (%)")
    ax.set_title("Forecheck Archetypes: Outcome Signature by Cluster")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.set_ylim(0, 72)

    for i, c in enumerate(clusters):
        ax.text(i, 69.5, f"n={int(n.loc[c])}", ha="center", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_cluster_outcomes.png")
    plt.close(fig)


def _plot_model_auc(model_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = model_df.copy()
    plot_df = plot_df.sort_values("auc_mean", ascending=False).reset_index(drop=True)
    labels = {
        "pressure_logreg": "Pressure LR",
        "pressure_gb": "Pressure GB",
        "baseline_nearest_distance": "Nearest Defender",
        "baseline_counts": "Defender Count",
    }
    plot_df["label"] = plot_df["model"].map(labels).fillna(plot_df["model"])

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.bar(plot_df["label"], plot_df["auc_mean"], color=["#2c7fb8", "#41b6c4", "#a1dab4", "#fdae61"])
    ax.errorbar(plot_df["label"], plot_df["auc_mean"], yerr=plot_df["auc_std"], fmt="none", ecolor="#333333", capsize=3)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Cross-validated AUC")
    ax.set_title("Turnover Prediction: Pressure Features vs Baselines")
    for b, v in zip(bars, plot_df["auc_mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_auc.png")
    plt.close(fig)


def _plot_team_pressure_index(features: pd.DataFrame, out_dir: Path) -> None:
    team = (
        features.groupby("forechecking_team")[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed", "mean_pressure_at_carrier"]]
        .mean()
        .rename(
            columns={
                "turnover_in_dzone": "turnover_rate",
                "forced_dump_out": "forced_dump_rate",
                "controlled_exit_allowed": "controlled_exit_rate",
                "mean_pressure_at_carrier": "pressure_level",
            }
        )
        .reset_index()
    )
    team["tfpi"] = 100 * (team["turnover_rate"] + 0.5 * team["forced_dump_rate"] - team["controlled_exit_rate"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sc = ax.scatter(team["pressure_level"], team["tfpi"], s=90, c=team["tfpi"], cmap="RdYlGn", edgecolor="black", linewidth=0.4)
    for _, r in team.iterrows():
        ax.text(r["pressure_level"] + 0.005, r["tfpi"] + 0.2, r["forechecking_team"], fontsize=8)
    ax.set_xlabel("Mean pressure at carrier")
    ax.set_ylabel("TFPI (higher is better)")
    ax.set_title("Team Forecheck Pressure Index (TFPI)")
    fig.colorbar(sc, ax=ax, label="TFPI")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_team_tfpi.png")
    plt.close(fig)


def main() -> None:
    _style()
    base = Path("projects/forechecking_pressure_topology/outputs")
    out_dir = Path("projects/forechecking_pressure_topology/report_v1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(base / "forecheck_episodes.csv")
    features = pd.read_csv(base / "forecheck_episode_features.csv")
    clustered_episodes = pd.read_csv(base / "clustering/episode_clusters.csv")
    model_df = pd.read_csv(base / "modeling/predictive_validation.csv")

    _plot_outcome_distribution(episodes, out_dir)
    _plot_start_maps(episodes, out_dir)
    if "cluster_id" in clustered_episodes.columns:
        _plot_cluster_outcomes(clustered_episodes, out_dir)
    _plot_model_auc(model_df, out_dir)
    _plot_team_pressure_index(features, out_dir)


if __name__ == "__main__":
    main()
