from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data_loading import load_camera_orientations, load_events_for_game, load_tracking_for_game
from .modeling import run_archetype_clustering, run_predictive_validation
from .pressure_metrics import PressureConfig, aggregate_episode_features, build_frame_index, compute_episode_frame_metrics
from .segmentation import segment_forecheck_episodes
from .utils import ensure_dir, game_meta_from_game_id, iter_game_ids


@dataclass
class ForecheckPipelineConfig:
    raw_data_dir: str = "data/raw"
    output_dir: str = "projects/forechecking_pressure_topology/outputs"
    max_games: int | None = None
    max_episode_seconds: float = 25.0
    frame_stride: int = 15
    n_clusters: int = 4


def load_config(path: Path | None = None) -> ForecheckPipelineConfig:
    if path is None:
        return ForecheckPipelineConfig()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ForecheckPipelineConfig(**data)


def run_pipeline(config: ForecheckPipelineConfig) -> dict[str, str]:
    raw_dir = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)

    camera_lookup = load_camera_orientations(raw_dir)
    game_ids = iter_game_ids(raw_dir)
    if config.max_games is not None:
        game_ids = game_ids[: config.max_games]

    all_episodes: list[pd.DataFrame] = []
    all_frame_metrics: list[pd.DataFrame] = []

    for game_id in game_ids:
        game_meta = game_meta_from_game_id(game_id)
        events = load_events_for_game(raw_dir, game_id)
        episodes = segment_forecheck_episodes(
            events_df=events,
            game_meta=game_meta,
            camera_lookup=camera_lookup,
            max_episode_seconds=config.max_episode_seconds,
        )
        if episodes.empty:
            continue

        episodes = episodes.copy()
        episodes["episode_id"] = [f"{game_id}__{i:04d}" for i in range(len(episodes))]
        all_episodes.append(episodes)

        tracking = load_tracking_for_game(raw_dir, game_id)
        frame_index, frame_to_indices = build_frame_index(tracking)
        pressure_config = PressureConfig()

        frame_parts: list[pd.DataFrame] = []
        for episode_row in episodes.itertuples(index=False):
            ep_df = compute_episode_frame_metrics(
                episode_row=pd.Series(episode_row._asdict()),
                tracking_df=tracking,
                frame_index_df=frame_index,
                frame_to_indices=frame_to_indices,
                game_meta=game_meta,
                camera_lookup=camera_lookup,
                config=pressure_config,
                frame_stride=config.frame_stride,
            )
            if not ep_df.empty:
                frame_parts.append(ep_df)

        if frame_parts:
            all_frame_metrics.append(pd.concat(frame_parts, ignore_index=True))

    episodes_df = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    frame_metrics_df = pd.concat(all_frame_metrics, ignore_index=True) if all_frame_metrics else pd.DataFrame()
    feature_df = aggregate_episode_features(frame_metrics_df, episodes_df) if not episodes_df.empty else pd.DataFrame()

    episodes_path = out_dir / "forecheck_episodes.csv"
    frames_path = out_dir / "forecheck_frame_metrics.csv"
    features_path = out_dir / "forecheck_episode_features.csv"

    if not episodes_df.empty:
        episodes_df.to_csv(episodes_path, index=False)
    if not frame_metrics_df.empty:
        frame_metrics_df.to_csv(frames_path, index=False)
    if not feature_df.empty:
        feature_df.to_csv(features_path, index=False)

    cluster_dir = out_dir / "clustering"
    model_dir = out_dir / "modeling"
    if not feature_df.empty and "turnover_in_dzone" in feature_df.columns:
        run_archetype_clustering(feature_df, cluster_dir, n_clusters=config.n_clusters)
        run_predictive_validation(feature_df, model_dir)

    summary = {
        "config": asdict(config),
        "games_processed": len(game_ids),
        "episodes_rows": int(len(episodes_df)),
        "frame_metric_rows": int(len(frame_metrics_df)),
        "feature_rows": int(len(feature_df)),
        "outputs": {
            "episodes": str(episodes_path),
            "frame_metrics": str(frames_path),
            "episode_features": str(features_path),
            "cluster_dir": str(cluster_dir),
            "model_dir": str(model_dir),
        },
    }

    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

