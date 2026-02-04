from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loading import team_attacks_right


@dataclass
class OpportunityConfig:
    frame_stride: int = 20
    lane_sigma_ft: float = 7.0
    slot_distance_scale_ft: float = 12.0
    release_distance_scale_ft: float = 8.0
    seam_weight: float = 0.35
    slot_weight: float = 0.45
    release_weight: float = 0.20


def build_frame_index(tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame_index = tracking_df[["image_id", "elapsed_seconds", "period_label"]].drop_duplicates()
    frame_index = frame_index.sort_values("elapsed_seconds").reset_index(drop=True)
    frame_to_indices = tracking_df.groupby("image_id").indices
    return frame_index, frame_to_indices


def _min_dist_to_defenders(point_xy: np.ndarray, defenders_xy: np.ndarray) -> float:
    if defenders_xy.size == 0:
        return np.nan
    d = np.linalg.norm(defenders_xy - point_xy.reshape(1, 2), axis=1)
    return float(np.min(d))


def _distance_point_to_segment(point_xy: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    seg = seg_b - seg_a
    seg_norm = np.dot(seg, seg)
    if seg_norm == 0:
        return float(np.linalg.norm(point_xy - seg_a))
    t = np.dot(point_xy - seg_a, seg) / seg_norm
    t = min(1.0, max(0.0, t))
    proj = seg_a + t * seg
    return float(np.linalg.norm(point_xy - proj))


def _carrier_index(offense_xy: np.ndarray, puck_xy: np.ndarray) -> int:
    d = np.linalg.norm(offense_xy - puck_xy.reshape(1, 2), axis=1)
    return int(np.argmin(d))


def _opportunity_components(
    offense_xy: np.ndarray,
    defense_xy: np.ndarray,
    puck_xy: np.ndarray,
    carrier_idx: int,
    net_x: float,
    attack_dir: float,
    config: OpportunityConfig,
) -> tuple[float, float, float, float]:
    slot_points = np.array(
        [
            [net_x - attack_dir * 14.0, 0.0],
            [net_x - attack_dir * 20.0, 10.0],
            [net_x - attack_dir * 20.0, -10.0],
        ],
        dtype=float,
    )
    slot_openness_vals = []
    for pt in slot_points:
        d_min = _min_dist_to_defenders(pt, defense_xy)
        if np.isnan(d_min):
            slot_openness_vals.append(0.5)
        else:
            slot_openness_vals.append(1.0 - np.exp(-d_min / config.slot_distance_scale_ft))
    slot_openness = float(np.mean(slot_openness_vals))

    seam_scores = []
    for idx, r_xy in enumerate(offense_xy):
        if idx == carrier_idx:
            continue
        lane_risk = 0.0
        for d_xy in defense_xy:
            lane_d = _distance_point_to_segment(d_xy, puck_xy, r_xy)
            risk = np.exp(-0.5 * (lane_d / config.lane_sigma_ft) ** 2)
            lane_risk = max(lane_risk, float(risk))
        lane_open = 1.0 - min(1.0, lane_risk)
        receiver_to_net = float(np.linalg.norm(r_xy - np.array([net_x, 0.0])))
        xg_prior = np.exp(-receiver_to_net / 35.0)
        seam_scores.append(lane_open * xg_prior)
    seam_openness = float(max(seam_scores)) if seam_scores else 0.0

    shooter_idx = int(np.argmin(np.linalg.norm(offense_xy - slot_points[0].reshape(1, 2), axis=1)))
    d_shooter = _min_dist_to_defenders(offense_xy[shooter_idx], defense_xy)
    if np.isnan(d_shooter):
        release_space = 0.5
    else:
        release_space = float(1.0 - np.exp(-d_shooter / config.release_distance_scale_ft))

    opportunity = (
        config.slot_weight * slot_openness
        + config.seam_weight * seam_openness
        + config.release_weight * release_space
    )
    return opportunity, slot_openness, seam_openness, release_space


def _shot_in_horizon(shot_times: np.ndarray, t: float, horizon_s: float = 5.0) -> int:
    if shot_times.size == 0:
        return 0
    left = np.searchsorted(shot_times, t, side="left")
    right = np.searchsorted(shot_times, t + horizon_s, side="right")
    return int(right > left)


def compute_possession_frame_contributions(
    possession_row: pd.Series,
    tracking_df: pd.DataFrame,
    frame_index_df: pd.DataFrame,
    frame_to_indices: dict[str, np.ndarray],
    camera_lookup: dict[str, str],
    shot_times: np.ndarray,
    config: OpportunityConfig,
) -> pd.DataFrame:
    start_t = float(possession_row["start_elapsed_seconds"])
    end_t = float(possession_row["end_elapsed_seconds"])
    attacking_team = possession_row["attacking_team"]
    defending_team = possession_row["defending_team"]
    game_id = possession_row["game_id"]
    possession_id = possession_row["possession_id"]

    frames = frame_index_df[(frame_index_df["elapsed_seconds"] >= start_t) & (frame_index_df["elapsed_seconds"] <= end_t)]
    if frames.empty:
        return pd.DataFrame()
    frames = frames.iloc[:: max(1, config.frame_stride)]

    prev_state: dict[str, dict[str, float]] = {}
    records: list[dict[str, float | int | str]] = []

    for frame in frames.itertuples(index=False):
        frame_rows = tracking_df.iloc[frame_to_indices[frame.image_id]]
        puck_rows = frame_rows[(frame_rows["object_type"] == "Puck") & frame_rows["x"].notna() & frame_rows["y"].notna()]
        offense = frame_rows[
            (frame_rows["object_type"] == "Player")
            & (frame_rows["team_name"] == attacking_team)
            & frame_rows["x"].notna()
            & frame_rows["y"].notna()
        ]
        defense = frame_rows[
            (frame_rows["object_type"] == "Player")
            & (frame_rows["team_name"] == defending_team)
            & frame_rows["x"].notna()
            & frame_rows["y"].notna()
        ]
        if puck_rows.empty or offense.empty or defense.empty:
            continue

        puck_xy = puck_rows.iloc[0][["x", "y"]].to_numpy(dtype=float)
        offense_xy = offense[["x", "y"]].to_numpy(dtype=float)
        defense_xy = defense[["x", "y"]].to_numpy(dtype=float)
        if "player_key" in offense.columns:
            offense_ids = offense["player_key"].fillna(offense["player_id"].astype(str)).astype(str).to_list()
        else:
            offense_ids = offense["player_id"].astype(str).to_list()

        attacks_right = team_attacks_right(attacking_team, str(frame.period_label), game_id, camera_lookup)
        attack_dir = 1.0 if attacks_right else -1.0
        net_x = 89.0 if attacks_right else -89.0
        slot_center = np.array([net_x - attack_dir * 14.0, 0.0], dtype=float)

        carrier_idx = _carrier_index(offense_xy, puck_xy)
        carrier_player_id = offense_ids[carrier_idx]

        opp_all, slot_open, seam_open, release_space = _opportunity_components(
            offense_xy=offense_xy,
            defense_xy=defense_xy,
            puck_xy=puck_xy,
            carrier_idx=carrier_idx,
            net_x=net_x,
            attack_dir=attack_dir,
            config=config,
        )

        nearest_def_dists = np.min(
            np.linalg.norm(offense_xy[:, None, :] - defense_xy[None, :, :], axis=2),
            axis=1,
        )

        for idx, player_id in enumerate(offense_ids):
            if idx == carrier_idx:
                continue
            current_xy = offense_xy[idx]
            prev = prev_state.get(player_id)

            if prev is None:
                delta_opp = 0.0
                speed = np.nan
                cut = 0
                drag = 0
                decoy = 0
            else:
                dt = max(float(frame.elapsed_seconds) - prev["elapsed_seconds"], 1e-3)
                disp = current_xy - np.array([prev["x"], prev["y"]], dtype=float)
                speed = float(np.linalg.norm(disp) / dt)

                frozen_offense = offense_xy.copy()
                frozen_offense[idx, :] = np.array([prev["x"], prev["y"]], dtype=float)
                opp_minus, _, _, _ = _opportunity_components(
                    offense_xy=frozen_offense,
                    defense_xy=defense_xy,
                    puck_xy=puck_xy,
                    carrier_idx=carrier_idx,
                    net_x=net_x,
                    attack_dir=attack_dir,
                    config=config,
                )
                delta_opp = float(opp_all - opp_minus)

                v_norm = np.linalg.norm(disp)
                to_slot = slot_center - current_xy
                to_slot_norm = np.linalg.norm(to_slot)
                cos_to_slot = 0.0 if (v_norm == 0 or to_slot_norm == 0) else float(np.dot(disp, to_slot) / (v_norm * to_slot_norm))
                lateral_speed = abs(float(disp[1] / dt))
                cut = int(speed >= 10.0 and cos_to_slot >= 0.7 and abs(current_xy[1]) <= 25.0)
                drag = int(speed >= 8.0 and lateral_speed >= 7.0 and abs(current_xy[0] - slot_center[0]) <= 25.0)
                decoy = int(speed >= 12.0 and (prev["nearest_def_dist"] - float(nearest_def_dists[idx])) >= 2.0)

            dist_to_net = float(np.linalg.norm(current_xy - np.array([net_x, 0.0], dtype=float)))
            if attacks_right:
                between_puck_and_net = current_xy[0] > puck_xy[0] and current_xy[0] <= net_x
            else:
                between_puck_and_net = current_xy[0] < puck_xy[0] and current_xy[0] >= net_x
            screen = int(dist_to_net <= 10.0 and abs(current_xy[1]) <= 12.0 and between_puck_and_net)

            records.append(
                {
                    "possession_id": possession_id,
                    "game_id": game_id,
                    "attacking_team": attacking_team,
                    "defending_team": defending_team,
                    "elapsed_seconds": float(frame.elapsed_seconds),
                    "period_label": str(frame.period_label),
                    "image_id": frame.image_id,
                    "player_id": player_id,
                    "carrier_player_id": carrier_player_id,
                    "player_x": float(current_xy[0]),
                    "player_y": float(current_xy[1]),
                    "nearest_defender_distance": float(nearest_def_dists[idx]),
                    "opportunity_all": float(opp_all),
                    "slot_openness": float(slot_open),
                    "seam_openness": float(seam_open),
                    "release_space": float(release_space),
                    "sca_delta_opportunity": float(delta_opp),
                    "speed_ft_s": speed,
                    "is_cut": cut,
                    "is_drag": drag,
                    "is_screen": screen,
                    "is_decoy": decoy,
                    "shot_or_goal_in_next_5s": _shot_in_horizon(shot_times, float(frame.elapsed_seconds), 5.0),
                }
            )

        for idx, player_id in enumerate(offense_ids):
            prev_state[player_id] = {
                "x": float(offense_xy[idx, 0]),
                "y": float(offense_xy[idx, 1]),
                "elapsed_seconds": float(frame.elapsed_seconds),
                "nearest_def_dist": float(nearest_def_dists[idx]),
            }

    return pd.DataFrame(records)


def aggregate_sca(frame_contrib_df: pd.DataFrame, possessions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame_contrib_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    player_summary = (
        frame_contrib_df.groupby(["attacking_team", "player_id"])
        .agg(
            n_frames=("image_id", "size"),
            possessions=("possession_id", "nunique"),
            sca_total=("sca_delta_opportunity", "sum"),
            sca_mean=("sca_delta_opportunity", "mean"),
            sca_std=("sca_delta_opportunity", "std"),
            cut_rate=("is_cut", "mean"),
            drag_rate=("is_drag", "mean"),
            screen_rate=("is_screen", "mean"),
            decoy_rate=("is_decoy", "mean"),
            shot_in_5s_rate=("shot_or_goal_in_next_5s", "mean"),
        )
        .reset_index()
    )
    player_summary["sca_per_100_frames"] = 100.0 * player_summary["sca_total"] / player_summary["n_frames"].clip(lower=1)

    possession_summary = (
        frame_contrib_df.groupby("possession_id")
        .agg(
            n_frames=("image_id", "size"),
            n_players=("player_id", "nunique"),
            possession_sca_total=("sca_delta_opportunity", "sum"),
            possession_sca_mean=("sca_delta_opportunity", "mean"),
            possession_cut_rate=("is_cut", "mean"),
            possession_drag_rate=("is_drag", "mean"),
            possession_screen_rate=("is_screen", "mean"),
            possession_decoy_rate=("is_decoy", "mean"),
            possession_shot5_rate=("shot_or_goal_in_next_5s", "mean"),
        )
        .reset_index()
    )
    possession_summary = possession_summary.merge(
        possessions_df[
            [
                "possession_id",
                "attacking_team",
                "defending_team",
                "duration_seconds",
                "shot_or_goal_in_possession",
                "slot_shot_or_goal_in_possession",
            ]
        ],
        on="possession_id",
        how="left",
    )
    return player_summary, possession_summary
