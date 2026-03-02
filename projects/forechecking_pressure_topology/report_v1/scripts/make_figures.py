from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

try:
    from mplhockey import NHLRink

    HAS_MPLHOCKEY = True
except Exception:
    HAS_MPLHOCKEY = False


OUTCOME_ORDER = ["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed", "stoppage", "timeout"]
OUTCOME_LABELS = {
    "turnover_in_dzone": "Turnover in D-zone",
    "forced_dump_out": "Forced dump-out",
    "controlled_exit_allowed": "Controlled exit allowed",
    "stoppage": "Stoppage",
    "timeout": "Timeout",
}
OUTCOME_COLORS = {
    "turnover_in_dzone": "#1f9d55",
    "forced_dump_out": "#7a9a01",
    "controlled_exit_allowed": "#d97706",
    "stoppage": "#5b6c8f",
    "timeout": "#909090",
}

OUTCOME_MIX_ORDER = ["controlled_exit_allowed", "forced_dump_out", "turnover_in_dzone", "other"]
OUTCOME_MIX_LABELS = {
    "controlled_exit_allowed": "Controlled exit",
    "forced_dump_out": "Forced dump",
    "turnover_in_dzone": "Turnover",
    "other": "Other (stoppage/timeout)",
}
OUTCOME_MIX_COLORS = {
    "controlled_exit_allowed": OUTCOME_COLORS["controlled_exit_allowed"],
    "forced_dump_out": OUTCOME_COLORS["forced_dump_out"],
    "turnover_in_dzone": OUTCOME_COLORS["turnover_in_dzone"],
    "other": "#8A9199",
}

CLUSTER_NAME_MAP = {
    0: "Tight Squeeze",
    1: "Light Pressure",
    2: "Spread Support",
    3: "Heavy Disrupt",
}

LANE_SHORT = {
    "middle_lane": "M",
    "strong_boards": "SB",
    "weak_boards": "WB",
    "behind_net": "BN",
}

HEX_GRIDSIZE = 24
HEX_MINCNT_RATE = 10
# Blue-to-red density colormap: avoids pale/white colours that blend with ice
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_density",
    ["#93C5FD", "#2563EB", "#F97316", "#7F1D1D"],
)
PROBABILITY_CMAP = "YlOrRd"
PROBABILITY_VMIN = 0.05
PROBABILITY_VMAX = 0.60
MIN_TFPI_CELL_N = 5


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "font.family": "DejaVu Sans",
            "font.size": 11.3,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.3,
            "axes.labelweight": "medium",
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.0,
            "axes.grid": True,
            "grid.color": "#D6D8DB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.42,
            "grid.alpha": 0.42,
            "axes.edgecolor": "#2F3133",
            "axes.linewidth": 0.8,
        }
    )


def _clean_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label_bars_h(ax: plt.Axes, bars, rates: np.ndarray, counts: np.ndarray) -> None:
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            rate + 0.35,
            bar.get_y() + bar.get_height() / 2.0,
            f"{rate:.1f}% ({count})",
            va="center",
            ha="left",
            fontsize=9,
            color="#232527",
        )


def _cluster_name(cluster_id: int, include_id: bool = False) -> str:
    cid = int(cluster_id)
    base = CLUSTER_NAME_MAP.get(cid, f"Cluster {cid}")
    if include_id and cid in CLUSTER_NAME_MAP:
        return f"{base} (C{cid})"
    return base


def _wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    denom = 1.0 + (z**2) / np.clip(n, 1.0, None)
    center = (p + (z**2) / (2.0 * np.clip(n, 1.0, None))) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / np.clip(n, 1.0, None)) + (z**2) / (4.0 * (np.clip(n, 1.0, None) ** 2)))) / denom
    lower = np.clip(center - margin, 0.0, 1.0)
    upper = np.clip(center + margin, 0.0, 1.0)
    return lower, upper


def _add_rink_guides(ax: plt.Axes) -> None:
    # Subtle rink context for orientation without overpowering the data layer.
    ax.add_patch(Rectangle((-100, -42.5), 200, 85, fill=False, ec="#8A8F96", lw=0.7, alpha=0.65, zorder=1))
    ax.axvline(-25, color="#72777E", lw=0.55, ls="--", alpha=0.55, zorder=1)
    ax.axvline(25, color="#72777E", lw=0.55, ls="--", alpha=0.55, zorder=1)
    ax.axhline(0, color="#72777E", lw=0.5, ls=":", alpha=0.55, zorder=1)
    for x0 in (-69, -22, 22, 69):
        ax.add_patch(Circle((x0, 0), radius=1.2, fill=False, ec="#9CA3AF", lw=0.45, alpha=0.65, zorder=1))


def _add_rink_lines_overlay(ax: plt.Axes, alpha: float = 0.52) -> None:
    """Draw NHL rink lines on top of existing data without any background fill."""
    z = 6
    # Boards outline (no fill)
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle="round,pad=0,rounding_size=14",
        fill=False, edgecolor="#333333", lw=1.4, alpha=alpha * 0.8, zorder=z,
    ))
    # Center red line
    ax.plot([0, 0], [-42.5, 42.5], color="#C8102E", lw=2.2, alpha=alpha, zorder=z, solid_capstyle="butt")
    # Blue lines
    for x in (-25, 25):
        ax.plot([x, x], [-42.5, 42.5], color="#0033A0", lw=2.2, alpha=alpha, zorder=z, solid_capstyle="butt")
    # Goal lines
    for x in (-89, 89):
        ax.plot([x, x], [-42.5, 42.5], color="#C8102E", lw=1.1, alpha=alpha * 0.8, zorder=z, solid_capstyle="butt")
    # End-zone face-off circles + dots
    for fx, fy in [(-69, 22), (-69, -22), (69, 22), (69, -22)]:
        ax.add_patch(Circle((fx, fy), radius=15, fill=False, edgecolor="#C8102E", lw=0.9, alpha=alpha * 0.7, zorder=z))
        ax.add_patch(Circle((fx, fy), radius=1.2, facecolor="#C8102E", edgecolor="none", alpha=alpha, zorder=z))
    # Center circle + dot
    ax.add_patch(Circle((0, 0), radius=15, fill=False, edgecolor="#0033A0", lw=0.9, alpha=alpha * 0.65, zorder=z))
    ax.add_patch(Circle((0, 0), radius=1.2, facecolor="#0033A0", edgecolor="none", alpha=alpha, zorder=z))


def _apply_hexbin_alpha_by_sample_size(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    hb_rate,
    gridsize: int,
    extent: tuple[float, float, float, float],
    alpha_min: float = 0.22,
    alpha_max: float = 0.97,
) -> None:
    hb_count = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        extent=extent,
        mincnt=1,
        linewidths=0.0,
        edgecolors="none",
        visible=False,
    )
    count_offsets = hb_count.get_offsets()
    count_values = np.asarray(hb_count.get_array(), dtype=float)
    hb_count.remove()
    if count_values.size == 0:
        return
    rate_offsets = hb_rate.get_offsets()
    if len(rate_offsets) == 0:
        return

    count_map: dict[tuple[float, float], float] = {}
    for (ox, oy), c in zip(count_offsets, count_values):
        count_map[(round(float(ox), 4), round(float(oy), 4))] = float(c)
    lookup = np.array([count_map.get((round(float(ox), 4), round(float(oy), 4)), 1.0) for ox, oy in rate_offsets], dtype=float)
    cmin, cmax = float(np.min(lookup)), float(np.max(lookup))
    if cmax <= cmin + 1e-9:
        alpha = np.full_like(lookup, alpha_max, dtype=float)
    else:
        # Use sqrt scaling so small-sample bins remain visible but clearly deemphasized.
        s = (np.sqrt(lookup) - np.sqrt(cmin)) / (np.sqrt(cmax) - np.sqrt(cmin))
        alpha = alpha_min + s * (alpha_max - alpha_min)

    fig = ax.figure
    fig.canvas.draw()
    face = hb_rate.get_facecolors()
    if len(face) == len(alpha):
        face[:, 3] = alpha
        hb_rate.set_facecolors(face)


def _plot_outcome_distribution(episodes: pd.DataFrame, out_dir: Path) -> None:
    total = len(episodes)
    if total == 0:
        return

    if all(col in episodes.columns for col in ("controlled_exit_allowed", "forced_dump_out", "turnover_in_dzone")):
        counts = {
            "controlled_exit_allowed": int(episodes["controlled_exit_allowed"].sum()),
            "forced_dump_out": int(episodes["forced_dump_out"].sum()),
            "turnover_in_dzone": int(episodes["turnover_in_dzone"].sum()),
        }
        counts["other"] = max(total - sum(counts.values()), 0)
    else:
        base = episodes["outcome"].value_counts()
        counts = {
            "controlled_exit_allowed": int(base.get("controlled_exit_allowed", 0)),
            "forced_dump_out": int(base.get("forced_dump_out", 0)),
            "turnover_in_dzone": int(base.get("turnover_in_dzone", 0)),
            "other": int(base.get("stoppage", 0) + base.get("timeout", 0)),
        }

    count_vec = np.array([counts[k] for k in OUTCOME_MIX_ORDER], dtype=float)
    rates = 100.0 * count_vec / max(count_vec.sum(), 1.0)
    labels = [OUTCOME_MIX_LABELS[k] for k in OUTCOME_MIX_ORDER]
    colors = [OUTCOME_MIX_COLORS[k] for k in OUTCOME_MIX_ORDER]

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    bars = ax.barh(labels, rates, color=colors, edgecolor="#1f1f1f", linewidth=0.3)
    _label_bars_h(ax, bars, rates, count_vec.astype(int))
    ax.invert_yaxis()
    ax.set_xlabel("Episode share (%)")
    ax.set_title(f"Controlled exits remain the largest retrieval outcome (n = {total:,})")
    ax.set_xlim(0, max(rates) * 1.27)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_outcome_distribution.png")
    plt.close(fig)


def _plot_rdt_pipeline_schematic(out_dir: Path) -> None:
    stages = [
        (
            "1", "Pressure field",
            "Each defender assigned\na pressure contribution\nbased on ETA to carrier\nand heading alignment",
        ),
        (
            "2", "Corridor costs",
            "Pressure integrated\nalong 4 exit routes:\nMiddle, Strong boards,\nWeak boards, Behind net",
        ),
        (
            "3", "Cheapest corridor",
            "Minimum-cost route\n= carrier's best\nescape option at\nany given frame",
        ),
        (
            "4", "3s closure rate",
            "How fast cheapest\nlane cost rises from\nt=0 to t=3s —\nmeasures lane removal",
        ),
        (
            "5", "Denial outcome",
            "Fast closure →\nlane collapse →\nturnover in D-zone\nor forced dump-out",
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x0 = 0.022
    w = 0.172
    gap = 0.025
    y = 0.15
    h = 0.76
    box_color = "#F8FAFC"
    edge_color = "#475569"
    accent = "#1F9D55"
    last_accent = "#B42318"

    for idx, (num, title, subtitle) in enumerate(stages):
        x = x0 + idx * (w + gap)
        is_last = idx == len(stages) - 1
        fc = "#FEF2F2" if is_last else box_color
        ec = last_accent if is_last else edge_color
        lw = 1.6 if is_last else 1.0
        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw)
        ax.add_patch(rect)
        badge_color = last_accent if is_last else accent
        ax.text(x + 0.013, y + h - 0.085, f"Step {num}", fontsize=7.8, color=badge_color, weight="bold", ha="left", va="center")
        ax.text(x + 0.013, y + h - 0.23, title, fontsize=9.0, color="#111827", weight="semibold", ha="left", va="center")
        ax.plot([x + 0.01, x + w - 0.01], [y + h - 0.315, y + h - 0.315], color="#CBD5E1", lw=0.7)
        ax.text(x + 0.013, y + 0.27, subtitle, fontsize=7.9, color="#334155", ha="left", va="center", linespacing=1.38)
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap - 0.006, y + 0.5 * h),
                xytext=(x + w + 0.007, y + 0.5 * h),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#64748B"},
            )
    ax.text(
        0.022,
        0.055,
        "Key insight: denial quality depends on how quickly the best escape lane closes, not just how many defenders are nearby.",
        fontsize=8.3,
        color="#334155",
        ha="left",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rdt_pipeline.png")
    plt.close(fig)


def _draw_rink_background(ax: plt.Axes) -> None:
    if HAS_MPLHOCKEY:
        rink = NHLRink(theme="over-light", linewidth=0.7)
        rink.draw(ax=ax, display_range="full")
    else:
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.axvline(-25, color="#545454", lw=0.5, ls="--")
        ax.axvline(25, color="#545454", lw=0.5, ls="--")
        ax.axhline(0, color="#545454", lw=0.45, ls=":")


def _binned_surface(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray | None = None,
    bins_x: int = 52,
    bins_y: int = 24,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(-100.0, 100.0, bins_x + 1)
    y_edges = np.linspace(-42.5, 42.5, bins_y + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    counts = counts.T
    if values is None:
        surface = counts.copy()
    else:
        weighted, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
        weighted = weighted.T
        with np.errstate(divide="ignore", invalid="ignore"):
            surface = weighted / counts
        surface[counts < max(1, min_count)] = np.nan
    return x_edges, y_edges, surface, counts


def _plot_rink_surface(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray | None,
    title: str,
    cmap: str,
    cbar_label: str,
    min_count: int = 1,
    vmin: float | None = None,
    vmax: float | None = None,
    apply_log: bool = False,
) -> None:
    x_edges, y_edges, surface, counts = _binned_surface(
        x=x,
        y=y,
        values=values,
        bins_x=52,
        bins_y=24,
        min_count=min_count,
    )
    if apply_log:
        surface_to_plot = np.log1p(surface)
    else:
        surface_to_plot = surface

    _draw_rink_background(ax)
    mappable = ax.pcolormesh(
        x_edges,
        y_edges,
        surface_to_plot,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
    )
    if HAS_MPLHOCKEY:
        rink = NHLRink(theme="over-light", linewidth=0.7)
        rink.draw(ax=ax, display_range="full")

    if np.isfinite(surface).any():
        iy, ix = np.unravel_index(np.nanargmax(surface), surface.shape)
        x0 = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        y0 = 0.5 * (y_edges[iy] + y_edges[iy + 1])
        ax.scatter([x0], [y0], marker="x", s=42, color="#111111", linewidth=1.2, zorder=5)
        val = surface[iy, ix]
        if values is None:
            txt = f"Peak density: {int(round(val))}"
        else:
            txt = f"Peak turnover bin: {100 * val:.1f}%"
        ax.text(
            x0 + 4.5,
            y0 + 2.0,
            txt,
            fontsize=8.3,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "#555555", "boxstyle": "round,pad=0.25", "alpha": 0.9},
            zorder=6,
        )
    ax.grid(False)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_title(title)
    ax.set_xlabel("Rink X (ft)")
    ax.set_ylabel("Rink Y (ft)")
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.9, pad=0.01)
    cbar.set_label(cbar_label)

    if values is not None:
        if np.any(counts >= min_count):
            yc = 0.5 * (y_edges[:-1] + y_edges[1:])
            xc = 0.5 * (x_edges[:-1] + x_edges[1:])
            masked = np.where(counts >= min_count, surface, np.nan)
            finite_vals = masked[np.isfinite(masked)]
            if finite_vals.size >= 10:
                q_levels = np.quantile(finite_vals, [0.6, 0.75, 0.9])
                q_levels = np.unique(np.round(q_levels, 4))
                if q_levels.size > 0:
                    ax.contour(
                        xc,
                        yc,
                        np.nan_to_num(masked, nan=np.nanmin(finite_vals)),
                        levels=q_levels,
                        colors="#1f1f1f",
                        linewidths=0.55,
                        alpha=0.5,
                    )


def _plot_start_maps(episodes: pd.DataFrame, out_dir: Path) -> None:
    x = episodes["start_x"].to_numpy()
    y = episodes["start_y"].to_numpy()
    z = episodes["turnover_in_dzone"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.45), constrained_layout=False)

    hb1 = axes[0].hexbin(
        x,
        y,
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
        cmap=DENSITY_CMAP,
        mincnt=1,
    )
    axes[0].set_title("Retrieval starts cluster in\ncorners and half walls", fontsize=11.4)
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
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
        cmap=PROBABILITY_CMAP,
        mincnt=1,
        vmin=PROBABILITY_VMIN,
        vmax=PROBABILITY_VMAX,
    )
    _apply_hexbin_alpha_by_sample_size(
        axes[1],
        x=x,
        y=y,
        hb_rate=hb2,
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
    )
    axes[1].set_title("Turnover risk peaks on deep-corner\nand half-wall starts", fontsize=11.4)
    axes[1].set_xlabel("X (ft)")
    axes[1].set_ylabel("Y (ft)")
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(-42.5, 42.5)
    cbar2 = fig.colorbar(hb2, ax=axes[1], shrink=0.9)
    cbar2.set_label("P(turnover in D-zone); alpha scales with sample size")
    axes[1].annotate(
        "Deep-corner starts show\nhigher turnover risk pockets",
        xy=(-78, -26),
        xytext=(-22, 30),
        arrowprops={"arrowstyle": "->", "color": "#7F1D1D", "lw": 1.05},
        fontsize=8.3,
        color="#7F1D1D",
        bbox={"facecolor": "white", "edgecolor": "#B91C1C", "alpha": 0.84, "boxstyle": "round,pad=0.24"},
        ha="left",
    )

    for ax in axes:
        _add_rink_lines_overlay(ax)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        _clean_spines(ax)
    fig.text(
        0.01,
        0.015,
        "Coordinate convention: X goal line to goal line (ft), Y boards to boards (ft).",
        fontsize=8.4,
        color="#4B5563",
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    fig.savefig(out_dir / "fig_start_maps.png")
    plt.close(fig)


def _plot_turnover_end_maps(episodes: pd.DataFrame, out_dir: Path) -> None:
    valid = episodes["end_x"].notna() & episodes["end_y"].notna()
    if not valid.any():
        return

    end_all = episodes.loc[valid].copy()
    end_turnovers = end_all[end_all["turnover_in_dzone"] == 1]
    if end_turnovers.empty:
        return

    x_all = end_all["end_x"].to_numpy()
    y_all = end_all["end_y"].to_numpy()
    z_all = end_all["turnover_in_dzone"].to_numpy(dtype=float)

    x_tov = end_turnovers["end_x"].to_numpy()
    y_tov = end_turnovers["end_y"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), constrained_layout=True)

    hb1 = axes[0].hexbin(
        x_tov,
        y_tov,
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
        cmap=DENSITY_CMAP,
        mincnt=1,
    )
    axes[0].set_title(f"Turnover endings concentrate in\ncompressed pockets (n={len(end_turnovers):,})", fontsize=11.2)
    axes[0].set_xlabel("X (ft)")
    axes[0].set_ylabel("Y (ft)")
    axes[0].set_xlim(-100, 100)
    axes[0].set_ylim(-42.5, 42.5)
    cbar1 = fig.colorbar(hb1, ax=axes[0], shrink=0.9)
    cbar1.set_label("Turnovers per hex")

    hb2 = axes[1].hexbin(
        x_all,
        y_all,
        C=z_all,
        reduce_C_function=np.mean,
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
        cmap=PROBABILITY_CMAP,
        mincnt=1,
        vmin=PROBABILITY_VMIN,
        vmax=PROBABILITY_VMAX,
    )
    _apply_hexbin_alpha_by_sample_size(
        axes[1],
        x=x_all,
        y=y_all,
        hb_rate=hb2,
        gridsize=HEX_GRIDSIZE,
        extent=(-100, 100, -42.5, 42.5),
    )
    axes[1].set_title("Middle-lane endings still carry turnover risk", fontsize=11.3)
    axes[1].set_xlabel("X (ft)")
    axes[1].set_ylabel("Y (ft)")
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(-42.5, 42.5)
    cbar2 = fig.colorbar(hb2, ax=axes[1], shrink=0.9)
    cbar2.set_label("P(turnover in D-zone); alpha scales with sample size")
    axes[1].annotate(
        "Middle lane still carries\nnon-zero turnover risk",
        xy=(35, 0),
        xytext=(4, 27),
        arrowprops={"arrowstyle": "->", "color": "#7F1D1D", "lw": 1.05},
        fontsize=8.4,
        color="#7F1D1D",
        bbox={"facecolor": "white", "edgecolor": "#B91C1C", "alpha": 0.84, "boxstyle": "round,pad=0.24"},
        ha="left",
    )

    for ax in axes:
        _add_rink_lines_overlay(ax)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42.5, 42.5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        _clean_spines(ax)

    fig.savefig(out_dir / "fig_turnover_end_maps.png")
    plt.close(fig)


def _plot_micro_play_topology(frame_metrics: pd.DataFrame, episodes: pd.DataFrame, out_dir: Path) -> None:
    needed = [
        "episode_id",
        "elapsed_seconds",
        "carrier_x",
        "carrier_y",
        "corridor_middle_lane",
        "corridor_strong_boards",
        "corridor_weak_boards",
        "corridor_behind_net",
        "most_open_corridor",
        "most_open_corridor_cost",
    ]
    if any(col not in frame_metrics.columns for col in needed):
        return

    role_cols = [
        "f1_distance",
        "f2_distance",
        "f3_distance",
        "f1_share_pressure",
        "f2_share_pressure",
        "f3_share_pressure",
    ]
    present_role_cols = [col for col in role_cols if col in frame_metrics.columns]
    cols = needed + ["pressure_at_carrier"] + present_role_cols
    frames = frame_metrics[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    fd_col = "forced_dump_out" if "forced_dump_out" in episodes.columns else None
    meta_col_list = ["episode_id", "outcome", "turnover_in_dzone", "forechecking_team", "possessing_team"]
    if fd_col:
        meta_col_list.append(fd_col)
    meta_cols = meta_col_list
    meta = episodes[meta_cols].drop_duplicates("episode_id")
    frames = frames.merge(meta, on="episode_id", how="left")
    if frames.empty:
        return

    picks: list[dict[str, float | str | int]] = []
    for episode_id, group in frames.groupby("episode_id"):
        group = group.sort_values("elapsed_seconds")
        t0 = float(group["elapsed_seconds"].iloc[0])
        first3 = group[group["elapsed_seconds"] <= (t0 + 3.0)]
        if len(first3) < 4:
            continue
        start = float(first3["most_open_corridor_cost"].iloc[0])
        end = float(first3["most_open_corridor_cost"].iloc[-1])
        closure_drop = start - end
        step = first3[["carrier_x", "carrier_y"]].diff().pow(2).sum(axis=1).pow(0.5).fillna(0.0)
        picks.append(
            {
                "episode_id": episode_id,
                "n_frames": int(len(first3)),
                "closure_drop": closure_drop,
                "path_len": float(step.sum()),
                "turnover": int(first3["turnover_in_dzone"].iloc[0]) if "turnover_in_dzone" in first3.columns else 0,
                "forced_dump": int(first3[fd_col].iloc[0]) if fd_col and fd_col in first3.columns else 0,
            }
        )

    if not picks:
        return
    pick_df = pd.DataFrame(picks)

    # Prefer forced-dump-out episodes: carrier was clearly denied but this avoids
    # selecting a catastrophic turnover-into-goal play as the illustrative example.
    base_filter = (pick_df["closure_drop"] > 0.0) & (pick_df["path_len"].between(8, 55))
    candidates = pick_df[base_filter & (pick_df["forced_dump"] == 1)].copy()
    if len(candidates) < 3:
        # Fall back to turnovers if not enough forced-dump examples
        candidates = pick_df[base_filter & (pick_df["turnover"] == 1)].copy()
    if candidates.empty:
        candidates = pick_df[base_filter].copy()
    if candidates.empty:
        candidates = pick_df[pick_df["closure_drop"] > 0.0].copy()
    if candidates.empty:
        return

    # Pick from the upper-middle range of closure drop (60th-85th pct) with most frames.
    # This gives a clear, representative denial rather than the single most extreme play.
    if len(candidates) >= 5:
        q60 = candidates["closure_drop"].quantile(0.60)
        q85 = candidates["closure_drop"].quantile(0.85)
        mid = candidates[(candidates["closure_drop"] >= q60) & (candidates["closure_drop"] <= q85)]
        if not mid.empty:
            candidates = mid

    best_episode = (
        candidates.sort_values(["n_frames", "closure_drop", "path_len"], ascending=[False, False, True])
        .iloc[0]["episode_id"]
    )

    seq = frames[frames["episode_id"] == best_episode].sort_values("elapsed_seconds").copy()
    t0 = float(seq["elapsed_seconds"].iloc[0])
    seq = seq[seq["elapsed_seconds"] <= (t0 + 3.0)].copy()
    if len(seq) < 4:
        return
    seq["t"] = seq["elapsed_seconds"] - t0

    fig, (ax_rink, ax_cost) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
        constrained_layout=True,
    )

    # --- LEFT PANEL: carrier path on rink section with best-exit guide ---
    # Compute tight zoom bounds
    px_min = float(seq["carrier_x"].min())
    px_max = float(seq["carrier_x"].max())
    py_min = float(seq["carrier_y"].min())
    py_max = float(seq["carrier_y"].max())
    pad_x = max(10, (px_max - px_min) * 0.45)
    pad_y = max(7, (py_max - py_min) * 0.50)
    xl0 = max(-100.0, px_min - pad_x)
    xl1 = min(100.0, px_max + pad_x)
    yl0 = max(-42.5, py_min - pad_y)
    yl1 = min(42.5, py_max + pad_y)
    ax_rink.set_xlim(xl0, xl1)
    ax_rink.set_ylim(yl0, yl1)
    ax_rink.set_facecolor("#E8F4FB")  # ice blue background
    ax_rink.grid(False)

    # --- Draw rink markings visible in this zoom window ---
    _rz = 2
    # Boards outline (full rink rectangle, matplotlib clips to axes bounds)
    ax_rink.add_patch(Rectangle((-100, -42.5), 200, 85,
                                fill=False, edgecolor="#444444", lw=2.0, zorder=_rz))
    # Goal lines
    for gx in (-89, 89):
        if xl0 - 2 <= gx <= xl1 + 2:
            ax_rink.plot([gx, gx], [-42.5, 42.5], color="#C8102E", lw=2.2,
                         alpha=0.88, zorder=_rz, solid_capstyle="butt")
    # Blue lines
    for bx in (-25, 25):
        if xl0 - 2 <= bx <= xl1 + 2:
            ax_rink.plot([bx, bx], [-42.5, 42.5], color="#0033A0", lw=2.4,
                         alpha=0.88, zorder=_rz, solid_capstyle="butt")
    # Center red line
    if xl0 - 2 <= 0 <= xl1 + 2:
        ax_rink.plot([0, 0], [-42.5, 42.5], color="#C8102E", lw=2.2,
                     alpha=0.88, zorder=_rz, solid_capstyle="butt")
    # Face-off circles (draw if within 18 ft of the visible window)
    for (fx, fy) in [(-69, 22), (-69, -22), (69, 22), (69, -22)]:
        if (xl0 - 18 <= fx <= xl1 + 18) and (yl0 - 18 <= fy <= yl1 + 18):
            ax_rink.add_patch(Circle((fx, fy), radius=15, fill=False,
                                     edgecolor="#C8102E", lw=1.4, alpha=0.60, zorder=_rz))
            ax_rink.add_patch(Circle((fx, fy), radius=1.5, facecolor="#C8102E",
                                     edgecolor="none", alpha=0.85, zorder=_rz))
    # Goal creases (D-shaped arc in front of each goal)
    for gx in (-89, 89):
        if xl0 - 8 <= gx <= xl1 + 8:
            inward = -np.sign(gx)
            thetas = np.linspace(-np.pi / 2, np.pi / 2, 40)
            crease_x = gx + inward * 4.5 + np.cos(thetas) * 6
            crease_y = np.sin(thetas) * 6
            ax_rink.fill(crease_x, crease_y, color="#C8102E", alpha=0.10, zorder=_rz)
            ax_rink.plot(crease_x, crease_y, color="#C8102E", lw=1.0, alpha=0.55, zorder=_rz)
    # Which end zone the carrier is in (for labelling)
    end_zone_x = 89 if 0.5 * (px_min + px_max) > 0 else -89
    zone_lbl_x = 0.97 if end_zone_x > 0 else 0.03
    zone_lbl_ha = "right" if end_zone_x > 0 else "left"
    ax_rink.text(zone_lbl_x, 0.03, "Defensive Zone",
                 transform=ax_rink.transAxes, fontsize=7.2, color="#4B5563",
                 ha=zone_lbl_ha, va="bottom", alpha=0.75, zorder=_rz)

    # --- Carrier path and scatter ---
    ax_rink.plot(seq["carrier_x"], seq["carrier_y"],
                 color="#1f2937", lw=1.6, alpha=0.30, zorder=4, solid_capstyle="round")
    scatter = ax_rink.scatter(
        seq["carrier_x"], seq["carrier_y"],
        c=seq["t"], cmap="plasma", s=72,
        edgecolor="#ffffff", linewidth=0.7,
        zorder=5, vmin=0, vmax=3,
    )

    # START marker
    sx, sy = float(seq["carrier_x"].iloc[0]), float(seq["carrier_y"].iloc[0])
    ax_rink.scatter([sx], [sy], s=160, marker="o", color="#111111",
                    edgecolor="#ffffff", linewidth=1.4, zorder=7)
    ax_rink.text(sx, sy + 1.6, "t=0", fontsize=7.6, color="#111111",
                 weight="semibold", ha="center", va="bottom", zorder=8)

    # END marker
    end_x, end_y = float(seq["carrier_x"].iloc[-1]), float(seq["carrier_y"].iloc[-1])
    ax_rink.scatter([end_x], [end_y], s=180, marker="X", color="#B42318",
                    edgecolor="#ffffff", linewidth=1.4, zorder=7)
    ax_rink.text(end_x, end_y - 1.6, "t=3s", fontsize=7.6, color="#B42318",
                 weight="semibold", ha="center", va="top", zorder=8)

    # --- Best exit path: green dashed arrow from START toward optimal corridor ---
    LANE_FULL = {
        "middle_lane": "Middle lane",
        "strong_boards": "Strong boards",
        "weak_boards": "Weak boards",
        "behind_net": "Behind net",
    }
    row0 = seq.iloc[0]
    row1 = seq.iloc[-1]
    corridor0 = str(row0["most_open_corridor"])
    s_cost = float(row0["most_open_corridor_cost"])
    e_cost = float(row1["most_open_corridor_cost"])
    closure_shown = e_cost - s_cost

    # Direction toward the exit based on which end zone the carrier is in
    end_sign = 1.0 if 0.5 * (px_min + px_max) > 0 else -1.0  # +1: defending right goal
    neutral_dir = -end_sign  # direction toward neutral zone (exit direction)

    if corridor0 == "middle_lane":
        tgt_x = sx + neutral_dir * 18
        tgt_y = sy + (0.0 - sy) * 0.35
    elif corridor0 == "strong_boards":
        board_y = np.sign(sy) * 40.0 if abs(sy) > 3 else 35.0
        tgt_x = sx + neutral_dir * 13
        tgt_y = sy + (board_y - sy) * 0.55
    elif corridor0 == "weak_boards":
        board_y = -np.sign(sy) * 40.0 if abs(sy) > 3 else -35.0
        tgt_x = sx + neutral_dir * 13
        tgt_y = sy + (board_y - sy) * 0.55
    elif corridor0 == "behind_net":
        tgt_x = sx + end_sign * 10
        tgt_y = sy + (np.sign(sy) if abs(sy) > 3 else 1.0) * 8
    else:
        tgt_x = sx + neutral_dir * 16
        tgt_y = sy

    tgt_x = float(np.clip(tgt_x, xl0 + 2, xl1 - 2))
    tgt_y = float(np.clip(tgt_y, yl0 + 2, yl1 - 2))

    arr_dx = tgt_x - sx
    arr_dy = tgt_y - sy
    arr_len = float(np.sqrt(arr_dx ** 2 + arr_dy ** 2))

    # Dashed line body
    ax_rink.plot([sx, tgt_x], [sy, tgt_y],
                 color="#15803D", lw=2.2, linestyle="--", alpha=0.88, zorder=8)
    # Solid arrowhead at target
    if arr_len > 0.5:
        shaft = min(arr_len * 0.15, 2.5)
        xt = tgt_x - shaft * arr_dx / arr_len
        yt = tgt_y - shaft * arr_dy / arr_len
        ax_rink.annotate("", xy=(tgt_x, tgt_y), xytext=(xt, yt),
                         arrowprops={"arrowstyle": "-|>", "color": "#15803D",
                                     "lw": 2.0, "mutation_scale": 20},
                         zorder=9)

    # Label the arrow at its midpoint, offset perpendicular so it doesn't sit on the line
    if arr_len > 0.5:
        perp_x = -arr_dy / arr_len  # unit perpendicular
        perp_y = arr_dx / arr_len
        lbl_x = 0.5 * (sx + tgt_x) + perp_x * 2.5
        lbl_y = 0.5 * (sy + tgt_y) + perp_y * 2.5
        ax_rink.text(lbl_x, lbl_y, f"Best exit:\n{LANE_FULL.get(corridor0, corridor0)}\n(cost {s_cost:.2f})",
                     fontsize=7.4, color="#14532D", ha="center", va="center", zorder=10,
                     bbox={"facecolor": "#F0FDF4", "edgecolor": "#15803D",
                           "alpha": 0.93, "boxstyle": "round,pad=0.25"})

    # Closure summary — bottom-right corner (references right panel)
    ax_rink.text(0.98, 0.04,
                 f"3s closure = {closure_shown:+.2f}",
                 transform=ax_rink.transAxes,
                 fontsize=8.0, color="#7A1D1D", ha="right", va="bottom",
                 bbox={"facecolor": "#FEF2F2", "edgecolor": "#B42318",
                       "alpha": 0.92, "boxstyle": "round,pad=0.3"},
                 zorder=9)

    cbar = fig.colorbar(scatter, ax=ax_rink, pad=0.01, shrink=0.86)
    cbar.set_label("Elapsed time (s)")
    ax_rink.set_xlabel("Rink X (ft)")
    ax_rink.set_ylabel("Rink Y (ft)")
    ax_rink.set_title("Carrier path + best exit route (first 3s)\n"
                      "(colour = time elapsed; green = cheapest corridor at t=0)",
                      fontsize=10.2)
    _clean_spines(ax_rink)

    # --- RIGHT PANEL: cheapest corridor over time, lanes shown as range band ---
    lane_cols = [
        "corridor_middle_lane",
        "corridor_strong_boards",
        "corridor_weak_boards",
        "corridor_behind_net",
    ]

    # Grey band = full range of all four lane costs (context without clutter)
    all_lanes = np.stack([seq[col].to_numpy() for col in lane_cols])
    lane_lo = all_lanes.min(axis=0)
    lane_hi = all_lanes.max(axis=0)
    ax_cost.fill_between(
        seq["t"], lane_lo, lane_hi,
        alpha=0.14, color="#6B7280",
        label="Range of all exit lanes",
    )

    # Bold cheapest corridor — the key signal
    ax_cost.plot(
        seq["t"], seq["most_open_corridor_cost"],
        marker="D", markersize=5.5, lw=3.0,
        color="#111111", zorder=5,
        label="Cheapest corridor (carrier's best exit)",
    )

    # Closure annotation
    start_cost = float(seq["most_open_corridor_cost"].iloc[0])
    end_cost = float(seq["most_open_corridor_cost"].iloc[-1])
    drop = start_cost - end_cost
    t_end = float(seq["t"].iloc[-1])
    x_bracket = min(t_end + 0.14, 2.95)
    ax_cost.annotate(
        "",
        xy=(x_bracket, end_cost),
        xytext=(x_bracket, start_cost),
        arrowprops={"arrowstyle": "<->", "lw": 1.6, "color": "#B42318"},
    )
    ax_cost.text(
        x_bracket - 0.06, 0.5 * (start_cost + end_cost),
        f"3s closure\n= {drop:.2f}",
        ha="right", va="center", fontsize=8.8, color="#7A1D1D", weight="semibold",
        bbox={"facecolor": "#FEF2F2", "edgecolor": "#B42318", "alpha": 0.9, "boxstyle": "round,pad=0.24"},
    )

    ax_cost.axvline(0, color="#374151", lw=0.8, ls="--", alpha=0.55)
    ax_cost.axvline(t_end, color="#B42318", lw=0.8, ls="--", alpha=0.55)
    ax_cost.set_xlabel("Seconds from retrieval start")
    ax_cost.set_ylabel("Corridor cost\n(higher = harder to escape via this lane)")
    ax_cost.set_title("Cheapest exit lane cost over first 3s\n(grey band = full range of all four lanes)", fontsize=10.5)
    ax_cost.set_xlim(-0.08, 3.12)
    ax_cost.grid(False)
    ax_cost.legend(loc="upper right", frameon=True, fontsize=8.0, framealpha=0.9)
    _clean_spines(ax_cost)
    fig.savefig(out_dir / "fig_micro_play_topology.png")
    plt.close(fig)


def _plot_cluster_outcomes(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    cols = ["controlled_exit_allowed", "forced_dump_out", "turnover_in_dzone", "stoppage", "timeout"]
    rates = clustered_episodes.groupby("cluster_id")[cols].mean().sort_index()
    rates["other"] = rates["stoppage"] + rates["timeout"]
    counts = clustered_episodes.groupby("cluster_id").size().sort_index()
    clusters = rates.index.to_list()
    x = np.arange(len(clusters))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    stack_order = ["controlled_exit_allowed", "forced_dump_out", "turnover_in_dzone", "other"]
    bottom = np.zeros(len(rates))
    for key in stack_order:
        vals = 100 * rates[key].to_numpy()
        ax.bar(
            x,
            vals,
            width=0.62,
            bottom=bottom,
            label=OUTCOME_MIX_LABELS[key],
            color=OUTCOME_MIX_COLORS[key],
            edgecolor="#1f1f1f",
            linewidth=0.25,
        )
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([_cluster_name(c, include_id=False) for c in clusters], rotation=14, ha="right")
    ax.set_ylabel("Outcome composition (%)")
    ax.set_title("Denial mix differs by archetype (100% stacked)", pad=10)
    ax.set_ylim(0, 112)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, borderaxespad=0.0)
    for i, c in enumerate(clusters):
        denial_pct = 100 * float(rates.loc[c, "turnover_in_dzone"] + rates.loc[c, "forced_dump_out"])
        ax.text(i, 106.4, f"Denial {denial_pct:.1f}%", ha="center", va="bottom", fontsize=8.3, color="#111827", weight="semibold")
        ax.text(i, 102.9, f"n={int(counts.loc[c])}", ha="center", va="bottom", fontsize=8.3, color="#374151")
    _clean_spines(ax)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "fig_cluster_outcomes.png")
    plt.close(fig)


def _cluster_topology_summary(clustered_episodes: pd.DataFrame) -> pd.DataFrame:
    required = [
        "cluster_id",
        "turnover_in_dzone",
        "forced_dump_out",
        "controlled_exit_allowed",
        "mean_pressure_at_carrier",
        "mean_open_corridor_cost",
        "corridor_closure_rate_3s",
        "mean_compactness_area",
        "mean_f1_distance",
        "mean_f2_distance",
        "mean_f3_distance",
        "mean_d_pinch_support",
        "mean_corridor_middle",
        "mean_corridor_strong_boards",
        "mean_corridor_weak_boards",
        "mean_corridor_behind_net",
        "mean_gradient_mag",
    ]
    df = clustered_episodes[required].replace([np.inf, -np.inf], np.nan).dropna()
    grouped = df.groupby("cluster_id").mean().sort_index()
    grouped["denial_rate"] = grouped["turnover_in_dzone"] + grouped["forced_dump_out"]
    grouped["n_episodes"] = clustered_episodes.groupby("cluster_id").size().reindex(grouped.index).values
    return grouped


def _plot_best_pattern_relative_lift(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    grouped = _cluster_topology_summary(clustered_episodes)
    if grouped.empty:
        return

    best_cluster = int(grouped["denial_rate"].idxmax())
    best = grouped.loc[best_cluster]
    field = grouped.mean(axis=0)

    feature_map = [
        ("mean_pressure_at_carrier", "Pressure at carrier", +1.0),
        ("corridor_closure_rate_3s", "3s lane closure", +1.0),
        ("mean_open_corridor_cost", "Cheapest lane cost", +1.0),
        ("mean_compactness_area", "Support compactness", +1.0),
        ("mean_d_pinch_support", "D pinch support", +1.0),
        ("mean_f1_distance", "F1 distance", -1.0),
        ("mean_f2_distance", "F2 distance", -1.0),
        ("mean_f3_distance", "F3 distance", -1.0),
    ]

    rows = []
    for col, label, direction in feature_map:
        vals = grouped[col]
        std = float(vals.std(ddof=0))
        if std < 1e-9:
            z = 0.0
        else:
            z = direction * float((best[col] - field[col]) / std)
        rows.append((label, z))
    lift = pd.DataFrame(rows, columns=["feature", "relative_z"]).sort_values("relative_z", ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = np.where(lift["relative_z"] >= 0, "#1f9d55", "#B42318")
    bars = ax.barh(lift["feature"], lift["relative_z"], color=colors, edgecolor="#1f1f1f", linewidth=0.35)
    ax.axvline(0.0, color="#2f2f2f", lw=0.9)
    ax.set_xlabel("Relative lift vs field (z-score; positive = better for denial)")
    best_name = _cluster_name(best_cluster, include_id=True)
    ax.set_title(f"Best-denial archetype topology drivers ({best_name} vs field)")
    ax.text(0.98, 0.05, "More denial-like  ->", transform=ax.transAxes, fontsize=8.2, color="#14532D", ha="right", va="bottom")
    for bar, val in zip(bars, lift["relative_z"]):
        x = val + 0.05 if val >= 0 else val - 0.05
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2.0, f"{val:+.2f}", va="center", ha=ha, fontsize=8.2)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_best_pattern_relative_lift.png")
    plt.close(fig)


def _plot_topology_phase_space(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    grouped = _cluster_topology_summary(clustered_episodes)
    if grouped.empty:
        return

    openness = grouped["mean_open_corridor_cost"].max() - grouped["mean_open_corridor_cost"]
    x = openness.to_numpy()
    y = grouped["corridor_closure_rate_3s"].to_numpy()
    sizes = 40 + 0.7 * grouped["mean_compactness_area"].to_numpy()
    color = grouped["denial_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=color,
        cmap="RdYlGn",
        edgecolor="#111111",
        linewidth=0.5,
        alpha=0.95,
    )
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = 0.05 * max(x_max - x_min, 0.05)
    y_pad = 0.08 * max(y_max - y_min, 0.01)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for cid, row in grouped.iterrows():
        x_pt = float(openness.loc[cid])
        y_pt = float(row["corridor_closure_rate_3s"])
        dx = -0.01 if x_pt > (x_max - 0.18 * (x_max - x_min + 1e-6)) else 0.004
        ha = "right" if dx < 0 else "left"
        ax.text(
            x_pt + dx,
            y_pt + 0.0006,
            _cluster_name(int(cid), include_id=True),
            fontsize=9.0,
            weight="bold",
            color="#202020",
            ha=ha,
        )

    x_mid = float(np.median(x))
    y_mid = float(np.median(y))
    ax.axvline(x_mid, color="#6B7280", lw=0.9, ls="--", alpha=0.65)
    ax.axhline(y_mid, color="#6B7280", lw=0.9, ls="--", alpha=0.65)
    ax.text(x_min + 0.01 * (x_max - x_min), y_max - 0.08 * (y_max - y_min), "Trap\n(open + closing fast)", fontsize=8.0, color="#1F2937")
    ax.text(x_min + 0.01 * (x_max - x_min), y_min + 0.04 * (y_max - y_min), "Permissive\n(open + not closing)", fontsize=8.0, color="#4B5563")
    ax.text(x_mid + 0.02 * (x_max - x_min), y_max - 0.08 * (y_max - y_min), "Suffocating\n(less open + closing fast)", fontsize=8.0, color="#1F2937")
    ax.text(x_mid + 0.02 * (x_max - x_min), y_min + 0.04 * (y_max - y_min), "Reactive\n(less open + not closing)", fontsize=8.0, color="#4B5563")
    ax.set_xlabel("How open the best exit lane is (higher = more open)")
    ax.set_ylabel("How fast that lane closes in first 3s")
    ax.set_title("Forecheck Topology Phase Space")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Denial rate")
    ax.grid(False)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_topology_phase_space.png")
    plt.close(fig)


def _plot_corridor_lane_profile(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    grouped = _cluster_topology_summary(clustered_episodes)
    if grouped.empty:
        return

    lane_cols = ["mean_corridor_middle", "mean_corridor_strong_boards", "mean_corridor_weak_boards", "mean_corridor_behind_net"]
    lane_labels = ["Middle", "Strong boards", "Weak boards", "Behind net"]
    centered = grouped[lane_cols].sub(grouped[lane_cols].mean(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    mat = centered.to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-0.35, vmax=0.35)
    row_labels = [
        f"{_cluster_name(int(cid), include_id=True)}  ({100*grouped.loc[cid, 'denial_rate']:.1f}% denial)"
        for cid in centered.index
    ]
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(lane_labels)))
    ax.set_xticklabels(lane_labels, rotation=16, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8.2, color="#111111")
    ax.set_title("Lane Preference Profile by Archetype")
    ax.text(
        0.01,
        -0.14,
        "Note: values are centered within each archetype; positive = more expensive (closed), negative = cheaper (open).",
        transform=ax.transAxes,
        fontsize=8.1,
        color="#4B5563",
        ha="left",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Relative lane cost (+: closed, -: open)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_corridor_lane_profile.png")
    plt.close(fig)


def _plot_model_auc(model_df: pd.DataFrame, out_dir: Path) -> None:
    labels = {
        "pressure_role_logreg": "Pressure + Role LR",
        "pressure_gb": "Pressure GB",
        "pressure_role_score_logreg": "Pressure + Role + Score LR",
        "pressure_role_score_adjusted_logreg": "Adjusted Pressure + Role + Score LR",
        "pressure_logreg": "Pressure LR",
        "baseline_nearest_distance": "Nearest Defender Baseline",
        "baseline_counts": "Defender Count Baseline",
    }
    plot_df = model_df.copy()
    plot_df["label"] = plot_df["model"].map(labels).fillna(plot_df["model"])
    plot_df = plot_df.sort_values("auc_mean", ascending=False).reset_index(drop=True)

    baseline_nearest = float(
        plot_df.loc[plot_df["model"] == "baseline_nearest_distance", "auc_mean"].iloc[0]
        if (plot_df["model"] == "baseline_nearest_distance").any()
        else np.nan
    )
    baseline_counts = float(
        plot_df.loc[plot_df["model"] == "baseline_counts", "auc_mean"].iloc[0]
        if (plot_df["model"] == "baseline_counts").any()
        else np.nan
    )
    top_val = float(plot_df["auc_mean"].max())

    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    y = np.arange(len(plot_df))
    colors = np.array(["#9CA3AF"] * len(plot_df), dtype=object)
    top_mask = np.isclose(plot_df["auc_mean"].to_numpy(dtype=float), top_val, atol=1e-6)
    colors[top_mask] = "#1F9D55"
    bars = ax.barh(y, plot_df["auc_mean"], color=colors, edgecolor="#202224", linewidth=0.3)
    ax.errorbar(
        plot_df["auc_mean"],
        y,
        xerr=plot_df["auc_std"],
        fmt="none",
        ecolor="#2f2f2f",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Cross-validated AUC")
    ax.set_title("Topology models outperform distance/count baselines")
    x_left = max(0.45, float(plot_df["auc_mean"].min()) - 0.03)
    x_right = min(0.86, top_val + 0.04)
    ax.set_xlim(x_left, x_right)
    if np.isfinite(baseline_nearest):
        ax.axvline(baseline_nearest, color="#B42318", ls="--", lw=1.2, alpha=0.9)
        ax.text(baseline_nearest + 0.002, -0.62, "Nearest baseline", color="#B42318", fontsize=8.4, va="bottom")
    if np.isfinite(baseline_counts):
        ax.axvline(baseline_counts, color="#7F1D1D", ls=":", lw=1.2, alpha=0.9)
        ax.text(baseline_counts + 0.002, -0.24, "Count baseline", color="#7F1D1D", fontsize=8.2, va="bottom")
    for idx, row in plot_df.iterrows():
        ax.text(row["auc_mean"] + 0.004, idx, f"{row['auc_mean']:.3f}", va="center", ha="left", fontsize=8.5)
    for idx, tick in enumerate(ax.get_yticklabels()):
        if bool(top_mask[idx]):
            tick.set_fontweight("bold")
            bars[idx].set_linewidth(0.8)
    ax.invert_yaxis()
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_auc.png")
    plt.close(fig)


def _team_tfpi(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby("forechecking_team")
    team = (
        grouped[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed", "mean_pressure_at_carrier"]]
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
    if "episode_id" in features.columns:
        team["n_episodes"] = grouped["episode_id"].size().values
    else:
        team["n_episodes"] = 1
    team["tfpi"] = 100 * (team["turnover_rate"] + 0.5 * team["forced_dump_rate"] - team["controlled_exit_rate"])
    return team.sort_values("tfpi", ascending=False).reset_index(drop=True)


def _plot_team_pressure_index(features: pd.DataFrame, out_dir: Path) -> None:
    team = _team_tfpi(features)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.grid(False)
    size = 25 + 0.7 * team["n_episodes"].to_numpy()
    sc = ax.scatter(
        team["pressure_level"],
        team["tfpi"],
        s=size,
        c=team["tfpi"],
        cmap="RdYlGn",
        edgecolor="#111111",
        linewidth=0.45,
        alpha=0.95,
    )
    top_labels = team.nlargest(3, "tfpi")["forechecking_team"].tolist()
    bottom_labels = team.nsmallest(3, "tfpi")["forechecking_team"].tolist()
    label_set = set(top_labels + bottom_labels)
    x_min = float(team["pressure_level"].min())
    x_max = float(team["pressure_level"].max())
    x_pad = 0.05 * max(x_max - x_min, 0.05)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    for _, r in team.iterrows():
        if r["forechecking_team"] not in label_set:
            continue
        near_right = float(r["pressure_level"]) > (x_max - 0.2 * max(x_max - x_min, 0.05))
        ax.text(
            r["pressure_level"] - 0.0035 if near_right else r["pressure_level"] + 0.0035,
            r["tfpi"] + 0.15,
            r["forechecking_team"],
            fontsize=8.5,
            color="#1f1f1f",
            weight="semibold",
            ha="right" if near_right else "left",
        )
    if len(team) >= 2:
        coef = np.polyfit(team["pressure_level"], team["tfpi"], deg=1)
        x_line = np.linspace(team["pressure_level"].min(), team["pressure_level"].max(), 100)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, color="#2f2f2f", lw=1.4, ls="--", alpha=0.8)
    ax.set_xlabel("Mean pressure at carrier")
    ax.set_ylabel("TFPI (higher is better)")
    ax.set_title("Process quality separates teams beyond pressure volume")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("TFPI")
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_team_tfpi.png")
    plt.close(fig)


def _plot_pressure_decile_denial(features: pd.DataFrame, out_dir: Path) -> None:
    cols = ["mean_pressure_at_carrier", "turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]
    df = features[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df["pressure_decile"] = pd.qcut(df["mean_pressure_at_carrier"], q=10, labels=False, duplicates="drop") + 1
    grouped = df.groupby("pressure_decile").agg(
        turnover_sum=("turnover_in_dzone", "sum"),
        dump_sum=("forced_dump_out", "sum"),
        controlled_sum=("controlled_exit_allowed", "sum"),
        n=("turnover_in_dzone", "size"),
    )
    grouped["denial_sum"] = grouped["turnover_sum"] + grouped["dump_sum"]
    grouped["denial_rate"] = grouped["denial_sum"] / grouped["n"]
    grouped["controlled_exit_rate"] = grouped["controlled_sum"] / grouped["n"]
    grouped = grouped.reset_index()

    denial_lo, denial_hi = _wilson_interval(grouped["denial_sum"].to_numpy(), grouped["n"].to_numpy())
    ctrl_lo, ctrl_hi = _wilson_interval(grouped["controlled_sum"].to_numpy(), grouped["n"].to_numpy())
    x = grouped["pressure_decile"].to_numpy(dtype=float)
    denial = 100 * grouped["denial_rate"].to_numpy()
    ctrl = 100 * grouped["controlled_exit_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.errorbar(
        x,
        denial,
        yerr=np.vstack([denial - 100 * denial_lo, 100 * denial_hi - denial]),
        fmt="o",
        color=OUTCOME_COLORS["turnover_in_dzone"],
        ms=6.0,
        lw=1.7,
        capsize=3,
        label="Denial rate",
        zorder=4,
    )
    ax.errorbar(
        x,
        ctrl,
        yerr=np.vstack([ctrl - 100 * ctrl_lo, 100 * ctrl_hi - ctrl]),
        fmt="s",
        color=OUTCOME_COLORS["controlled_exit_allowed"],
        ms=5.2,
        lw=1.5,
        capsize=3,
        label="Controlled exit allowed",
        zorder=3,
    )
    ax.set_xlabel("Mean-pressure decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Rate (%) with 95% CI")
    ax.set_title("Pressure-response sanity check: higher pressure deciles show higher denial")
    ax.set_xticks(grouped["pressure_decile"])
    ax.set_ylim(18, 75)
    for _, row in grouped.iterrows():
        ax.text(
            row["pressure_decile"],
            19.2,
            f"n={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#4B5563",
        )
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(False)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pressure_decile_denial.png")
    plt.close(fig)


def _plot_score_state_outcomes(features: pd.DataFrame, out_dir: Path) -> None:
    order = ["leading", "tied", "trailing"]
    labels = ["Leading", "Tied", "Trailing"]
    grouped = (
        features.groupby("forechecking_score_state")[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]]
        .mean()
        .reindex(order)
    )
    counts = features.groupby("forechecking_score_state").size().reindex(order).fillna(0).astype(int)
    grouped["other"] = np.clip(1.0 - grouped[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]].sum(axis=1), 0.0, 1.0)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.2, 4.65))
    stack_order = ["controlled_exit_allowed", "forced_dump_out", "turnover_in_dzone", "other"]
    bottom = np.zeros(len(grouped))
    for key in stack_order:
        vals = 100 * grouped[key].to_numpy()
        ax.bar(
            x,
            vals,
            width=0.62,
            bottom=bottom,
            color=OUTCOME_MIX_COLORS[key],
            label=OUTCOME_MIX_LABELS[key],
            edgecolor="#1f1f1f",
            linewidth=0.25,
        )
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Outcome composition (%)")
    ax.set_title("Score-state changes the denial mix", pad=10)
    ax.set_ylim(0, 111)
    for i, key in enumerate(order):
        ax.text(i, 103.0, f"n={counts.loc[key]}", ha="center", va="bottom", fontsize=8.4, color="#374151")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, borderaxespad=0.0)
    ax.grid(False)
    _clean_spines(ax)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "fig_score_state_outcomes.png")
    plt.close(fig)


def _plot_role_pressure_shares(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    cols = ["mean_f1_pressure_contrib", "mean_f2_pressure_contrib", "mean_f3_pressure_contrib", "cluster_id"]
    df = clustered_episodes[cols].replace([np.inf, -np.inf], np.nan).dropna()
    global_means = df[["mean_f1_pressure_contrib", "mean_f2_pressure_contrib", "mean_f3_pressure_contrib"]].mean()
    global_shares = global_means / max(float(global_means.sum()), 1e-9)
    ref_f1 = 100 * float(global_shares["mean_f1_pressure_contrib"])
    ref_f2 = 100 * float(global_shares["mean_f2_pressure_contrib"])
    ref_f3 = 100 * float(global_shares["mean_f3_pressure_contrib"])
    grouped = (
        df.groupby("cluster_id")[["mean_f1_pressure_contrib", "mean_f2_pressure_contrib", "mean_f3_pressure_contrib"]]
        .mean()
        .sort_index()
    )
    grouped = grouped.div(grouped.sum(axis=1), axis=0).fillna(0.0)

    x = np.arange(len(grouped))
    f1 = 100 * grouped["mean_f1_pressure_contrib"].to_numpy()
    f2 = 100 * grouped["mean_f2_pressure_contrib"].to_numpy()
    f3 = 100 * grouped["mean_f3_pressure_contrib"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(x, f1, color="#1f77b4", label="F1")
    ax.bar(x, f2, bottom=f1, color="#ff7f0e", label="F2")
    ax.bar(x, f3, bottom=f1 + f2, color="#2ca02c", label="F3")
    ax.set_xticks(x)
    ax.set_xticklabels([_cluster_name(int(c), include_id=False) for c in grouped.index], rotation=14, ha="right")
    ax.set_ylabel("Normalized role-pressure share (%)")
    ax.set_title("Role-Pressure Composition by Archetype")
    ax.set_ylim(0, 103)
    ax.axhline(ref_f1, color="#1E3A8A", lw=0.9, ls="--", alpha=0.36)
    ax.axhline(ref_f2, color="#9A3412", lw=0.9, ls="--", alpha=0.36)
    ax.axhline(ref_f3, color="#14532D", lw=0.9, ls="--", alpha=0.36)
    ax.text(len(grouped) - 0.1, ref_f1 + 0.8, f"Global F1 {ref_f1:.1f}%", color="#1E3A8A", fontsize=7.8, ha="right")
    ax.text(len(grouped) - 0.1, ref_f2 + 0.8, f"Global F2 {ref_f2:.1f}%", color="#9A3412", fontsize=7.8, ha="right")
    ax.text(len(grouped) - 0.1, ref_f3 + 0.8, f"Global F3 {ref_f3:.1f}%", color="#14532D", fontsize=7.8, ha="right")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.grid(False)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_role_pressure_shares.png")
    plt.close(fig)


def _plot_cluster_feature_heatmap(clustered_episodes: pd.DataFrame, out_dir: Path) -> None:
    selected = [
        "mean_pressure_at_carrier",
        "mean_open_corridor_cost",
        "corridor_closure_rate_3s",
        "mean_compactness_area",
        "mean_f1_distance",
        "mean_f2_distance",
        "mean_f3_distance",
        "mean_d_pinch_support",
    ]
    pretty_cols = [
        "Pressure at carrier",
        "Cheapest lane cost",
        "3s lane closure",
        "Support compactness",
        "Role geometry F1",
        "Role geometry F2",
        "Role geometry F3",
        "D-pinch support",
    ]
    df = clustered_episodes[["cluster_id"] + selected].replace([np.inf, -np.inf], np.nan).dropna()
    grouped = df.groupby("cluster_id")[selected].mean().sort_index()
    outcome_rates = (
        clustered_episodes.groupby("cluster_id")[["turnover_in_dzone", "forced_dump_out"]]
        .mean()
        .sum(axis=1)
        .reindex(grouped.index)
    )
    z = (grouped - grouped.mean(axis=0)) / grouped.std(axis=0, ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.grid(False)
    mat = z.to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-2.0, vmax=2.0)
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(
        [
            f"{_cluster_name(int(c), include_id=True)} ({100*float(outcome_rates.loc[c]):.1f}% denial)"
            for c in z.index
        ]
    )
    ax.set_xticks(np.arange(len(pretty_cols)))
    ax.set_xticklabels(pretty_cols, rotation=24, ha="right")
    ax.set_title("Archetype Feature Profiles (z-score within feature)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text_color = "white" if abs(val) >= 0.9 else "#111111"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8.1, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("z-score")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cluster_feature_heatmap.png")
    plt.close(fig)


def _plot_closure_tradeoff(features: pd.DataFrame, out_dir: Path) -> None:
    cols = ["corridor_closure_rate_3s", "turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]
    df = features[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        return

    df["denial"] = (df["turnover_in_dzone"] + df["forced_dump_out"]).astype(int)
    df["closure_decile"] = pd.qcut(df["corridor_closure_rate_3s"], q=10, labels=False, duplicates="drop") + 1
    summ = (
        df.groupby("closure_decile", observed=False)
        .agg(
            mean_closure=("corridor_closure_rate_3s", "mean"),
            n=("denial", "size"),
            denial_sum=("denial", "sum"),
            controlled_exit_rate=("controlled_exit_allowed", "mean"),
        )
        .dropna()
        .reset_index()
    )
    if summ.empty:
        return

    p = summ["denial_sum"] / summ["n"]
    z = 1.96
    denom = 1.0 + (z**2) / summ["n"]
    center = (p + (z**2) / (2.0 * summ["n"])) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / summ["n"]) + (z**2) / (4.0 * (summ["n"] ** 2)))) / denom
    lower = np.clip(center - margin, 0.0, 1.0)
    upper = np.clip(center + margin, 0.0, 1.0)

    x = summ["closure_decile"].to_numpy(dtype=float)
    y = 100.0 * p.to_numpy()
    lo = 100.0 * lower.to_numpy()
    hi = 100.0 * upper.to_numpy()
    yerr = np.vstack([y - lo, hi - y])
    ctrl = 100.0 * summ["controlled_exit_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="#B42318",
        lw=2.2,
        ms=5.0,
        capsize=3,
        label="Denial rate (95% CI)",
        zorder=4,
    )
    ax.plot(
        x,
        ctrl,
        marker="s",
        color=OUTCOME_COLORS["controlled_exit_allowed"],
        lw=2.0,
        ms=4.6,
        alpha=0.95,
        label="Controlled exit allowed",
        zorder=3,
    )
    for xi, n in zip(x, summ["n"].to_numpy()):
        ax.text(xi, 36.2, f"n={int(n)}", ha="center", va="bottom", fontsize=7.4, color="#4B5563")

    ax.set_xlabel("3s lane-closure decile (1 = slowest, 10 = fastest)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Faster 3s lane closure tracks higher denial conversion")
    ax.set_xticks(np.arange(1, len(summ) + 1))
    ax.set_ylim(35, 72)
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_closure_tradeoff.png")
    plt.close(fig)


def _plot_team_scorestate_heatmap(features: pd.DataFrame, out_dir: Path) -> None:
    required = ["forechecking_team", "forechecking_score_state", "turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]
    df = features[required].replace([np.inf, -np.inf], np.nan).dropna()
    grouped = (
        df.groupby(["forechecking_team", "forechecking_score_state"])[["turnover_in_dzone", "forced_dump_out", "controlled_exit_allowed"]]
        .mean()
        .reset_index()
    )
    counts = (
        df.groupby(["forechecking_team", "forechecking_score_state"])
        .size()
        .rename("n")
        .reset_index()
    )
    grouped = grouped.merge(counts, on=["forechecking_team", "forechecking_score_state"], how="left")
    grouped["tfpi"] = 100 * (
        grouped["turnover_in_dzone"] + 0.5 * grouped["forced_dump_out"] - grouped["controlled_exit_allowed"]
    )
    order_cols = ["leading", "tied", "trailing"]
    heat = grouped.pivot(index="forechecking_team", columns="forechecking_score_state", values="tfpi")
    heat_n = grouped.pivot(index="forechecking_team", columns="forechecking_score_state", values="n")
    heat = heat.reindex(columns=order_cols)
    heat_n = heat_n.reindex(columns=order_cols)
    team_order = heat.mean(axis=1).sort_values(ascending=False).index
    heat = heat.reindex(team_order)
    heat_n = heat_n.reindex(team_order)
    heat = heat.where(heat_n >= MIN_TFPI_CELL_N)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.grid(False)
    mat = heat.to_numpy()
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        plt.close(fig)
        return
    vmin, vmax = np.quantile(finite, [0.05, 0.95])
    if np.isclose(vmin, vmax):
        span = max(1.0, abs(vmin) * 0.2)
        vmin, vmax = vmin - span, vmax + span
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#E5E7EB")
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(["Leading", "Tied", "Trailing"])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Team TFPI by Score-State Context\n(clipped 5th-95th pct; n<5 masked)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("TFPI")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.8, color="#1b1b1b")
            else:
                ax.text(j, i, f"n<{MIN_TFPI_CELL_N}", ha="center", va="center", fontsize=7.2, color="#6B7280")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_team_scorestate_tfpi_heatmap.png")
    plt.close(fig)


def main() -> None:
    _style()
    base = Path("projects/forechecking_pressure_topology/outputs")
    out_dir = Path("projects/forechecking_pressure_topology/report_v1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(base / "forecheck_episodes.csv")
    features = pd.read_csv(base / "forecheck_episode_features.csv")
    frame_metrics = pd.read_csv(base / "forecheck_frame_metrics.csv")
    clustered_episodes = pd.read_csv(base / "clustering/episode_clusters.csv")
    model_df = pd.read_csv(base / "modeling/predictive_validation.csv")

    _plot_rdt_pipeline_schematic(out_dir)
    _plot_outcome_distribution(episodes, out_dir)
    _plot_start_maps(episodes, out_dir)
    _plot_turnover_end_maps(episodes, out_dir)
    _plot_micro_play_topology(frame_metrics, episodes, out_dir)
    _plot_cluster_outcomes(clustered_episodes, out_dir)
    _plot_best_pattern_relative_lift(clustered_episodes, out_dir)
    _plot_topology_phase_space(clustered_episodes, out_dir)
    _plot_corridor_lane_profile(clustered_episodes, out_dir)
    _plot_model_auc(model_df, out_dir)
    _plot_team_pressure_index(features, out_dir)
    _plot_pressure_decile_denial(features, out_dir)
    _plot_score_state_outcomes(features, out_dir)
    _plot_role_pressure_shares(clustered_episodes, out_dir)
    _plot_cluster_feature_heatmap(clustered_episodes, out_dir)
    _plot_closure_tradeoff(features, out_dir)
    _plot_team_scorestate_heatmap(features, out_dir)


if __name__ == "__main__":
    main()
