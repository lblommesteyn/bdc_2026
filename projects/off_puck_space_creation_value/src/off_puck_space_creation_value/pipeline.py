from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import load_camera_orientations, load_events_for_game, load_tracking_for_game
from .modeling import run_outcome_validation, run_player_archetypes
from .opportunity import OpportunityConfig, aggregate_sca, build_frame_index, compute_possession_frame_contributions
from .possessions import segment_offensive_possessions
from .utils import ensure_dir, game_meta_from_game_id, iter_game_ids


@dataclass
class OffPuckPipelineConfig:
    raw_data_dir: str = "data/raw"
    output_dir: str = "projects/off_puck_space_creation_value/outputs"
    max_games: int | None = None
    max_possession_seconds: float = 30.0
    frame_stride: int = 20
    n_archetypes: int = 5


def load_config(path: Path | None = None) -> OffPuckPipelineConfig:
    if path is None:
        return OffPuckPipelineConfig()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return OffPuckPipelineConfig(**data)


def _shot_times_by_team(events_df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    shot_df = events_df[events_df["Event"].isin(["Shot", "Goal"]) & events_df["Team"].notna()]
    for team, group in shot_df.groupby("Team"):
        out[str(team)] = np.sort(group["elapsed_seconds"].to_numpy(dtype=float))
    return out


def run_pipeline(config: OffPuckPipelineConfig) -> dict[str, str]:
    raw_dir = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)

    camera_lookup = load_camera_orientations(raw_dir)
    game_ids = iter_game_ids(raw_dir)
    if config.max_games is not None:
        game_ids = game_ids[: config.max_games]

    all_possessions: list[pd.DataFrame] = []
    all_frame_contribs: list[pd.DataFrame] = []
    opp_cfg = OpportunityConfig(frame_stride=config.frame_stride)

    for game_id in game_ids:
        meta = game_meta_from_game_id(game_id)
        events = load_events_for_game(raw_dir, game_id)
        possessions = segment_offensive_possessions(
            events_df=events,
            game_meta=meta,
            camera_lookup=camera_lookup,
            max_possession_seconds=config.max_possession_seconds,
        )
        if possessions.empty:
            continue
        all_possessions.append(possessions)

        shot_map = _shot_times_by_team(events)
        tracking = load_tracking_for_game(raw_dir, game_id)
        frame_index, frame_to_indices = build_frame_index(tracking)

        game_contribs: list[pd.DataFrame] = []
        for possession_row in possessions.itertuples(index=False):
            possession_series = pd.Series(possession_row._asdict())
            shot_times = shot_map.get(str(possession_series["attacking_team"]), np.array([], dtype=float))
            contrib = compute_possession_frame_contributions(
                possession_row=possession_series,
                tracking_df=tracking,
                frame_index_df=frame_index,
                frame_to_indices=frame_to_indices,
                camera_lookup=camera_lookup,
                shot_times=shot_times,
                config=opp_cfg,
            )
            if not contrib.empty:
                game_contribs.append(contrib)

        if game_contribs:
            all_frame_contribs.append(pd.concat(game_contribs, ignore_index=True))

    possessions_df = pd.concat(all_possessions, ignore_index=True) if all_possessions else pd.DataFrame()
    frame_contrib_df = pd.concat(all_frame_contribs, ignore_index=True) if all_frame_contribs else pd.DataFrame()
    player_summary, possession_summary = aggregate_sca(frame_contrib_df, possessions_df)

    possessions_path = out_dir / "offensive_possessions.csv"
    frame_path = out_dir / "frame_contributions.csv"
    player_path = out_dir / "player_sca_summary.csv"
    possession_path = out_dir / "possession_sca_summary.csv"

    if not possessions_df.empty:
        possessions_df.to_csv(possessions_path, index=False)
    if not frame_contrib_df.empty:
        frame_contrib_df.to_csv(frame_path, index=False)
    if not player_summary.empty:
        player_summary.to_csv(player_path, index=False)
    if not possession_summary.empty:
        possession_summary.to_csv(possession_path, index=False)

    model_dir = out_dir / "modeling"
    archetype_dir = out_dir / "archetypes"
    if not player_summary.empty:
        run_player_archetypes(player_summary, archetype_dir, n_clusters=config.n_archetypes)
    if not possession_summary.empty:
        run_outcome_validation(possession_summary, model_dir)

    summary = {
        "config": asdict(config),
        "games_processed": len(game_ids),
        "possessions_rows": int(len(possessions_df)),
        "frame_contribution_rows": int(len(frame_contrib_df)),
        "player_summary_rows": int(len(player_summary)),
        "possession_summary_rows": int(len(possession_summary)),
        "outputs": {
            "possessions": str(possessions_path),
            "frame_contributions": str(frame_path),
            "player_summary": str(player_path),
            "possession_summary": str(possession_path),
            "archetype_dir": str(archetype_dir),
            "model_dir": str(model_dir),
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

