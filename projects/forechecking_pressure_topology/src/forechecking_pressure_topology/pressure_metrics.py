from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    BLUE_LINE_X_ABS_FT,
    ETA_TAU_S,
    FORECHECKER_MAX_SPEED_FT_S,
    PRESSURE_SIGMA_FT,
    RINK_X_MAX_FT,
    RINK_X_MIN_FT,
    RINK_Y_MAX_FT,
    RINK_Y_MIN_FT,
)
from .data_loading import team_defends_right
from .utils import GameMeta, clamp_point


@dataclass
class PressureConfig:
    sigma_ft: float = PRESSURE_SIGMA_FT
    max_speed_ft_s: float = FORECHECKER_MAX_SPEED_FT_S
    eta_tau_s: float = ETA_TAU_S
    max_accel_ft_s2: float = 14.0
    min_reach_speed_ft_s: float = 4.0
    anisotropy_forward_bonus: float = 0.35
    anisotropy_backward_penalty: float = 0.25
    turn_penalty_s: float = 0.35
    pressure_threshold: float = 1.2
    compactness_extent_ft: float = 35.0
    compactness_step_ft: float = 7.0
    local_pressure_radius_ft: float = 15.0
    nearby_defender_radius_ft: float = 20.0


def _as_xy_array(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 2), dtype=float)
    return df[["x", "y"]].to_numpy(dtype=float)


def _as_velocity_array(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 2), dtype=float)
    if "vx" not in df.columns or "vy" not in df.columns:
        return np.zeros((len(df), 2), dtype=float)
    return (
        df[["vx", "vy"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _defender_pressure_contributions(
    defenders_xy: np.ndarray,
    defenders_vxy: np.ndarray,
    point_xy: tuple[float, float],
    config: PressureConfig,
) -> np.ndarray:
    if defenders_xy.size == 0:
        return np.empty((0,), dtype=float)

    if defenders_vxy.size == 0:
        defenders_vxy = np.zeros_like(defenders_xy)
    elif defenders_vxy.shape != defenders_xy.shape:
        defenders_vxy = np.zeros_like(defenders_xy)

    point = np.array(point_xy, dtype=float).reshape(1, 2)
    delta = point - defenders_xy
    distance = np.linalg.norm(delta, axis=1)
    safe_distance = np.maximum(distance, 1e-6)
    direction = delta / safe_distance[:, None]

    speed = np.linalg.norm(defenders_vxy, axis=1)
    speed_cap = max(config.max_speed_ft_s * 2.0, config.min_reach_speed_ft_s + 1.0)
    speed = np.clip(speed, 0.0, speed_cap)
    speed_safe = np.maximum(speed, 1e-6)
    velocity_dir = defenders_vxy / speed_safe[:, None]
    heading_alignment = np.sum(velocity_dir * direction, axis=1)
    heading_alignment = np.where(speed > 0.5, heading_alignment, 0.0)

    sigma_multiplier = (
        1.0
        + config.anisotropy_forward_bonus * np.clip(heading_alignment, 0.0, 1.0)
        - config.anisotropy_backward_penalty * np.clip(-heading_alignment, 0.0, 1.0)
    )
    sigma_multiplier = np.clip(sigma_multiplier, 0.65, 1.6)
    effective_sigma = config.sigma_ft * sigma_multiplier
    kernel = np.exp(-0.5 * (distance / effective_sigma) ** 2)

    v_toward = np.maximum(np.sum(defenders_vxy * direction, axis=1), 0.0)
    accel = max(config.max_accel_ft_s2, 1e-6)
    eta = (-v_toward + np.sqrt(np.maximum(v_toward * v_toward + 2.0 * accel * distance, 0.0))) / accel
    eta_no_accel = distance / np.maximum(v_toward, config.min_reach_speed_ft_s)
    eta = np.where(config.max_accel_ft_s2 > 0, eta, eta_no_accel)
    eta = eta + config.turn_penalty_s * np.clip(-heading_alignment, 0.0, 1.0)
    eta = np.nan_to_num(eta, nan=0.0, posinf=10.0, neginf=0.0)

    eta_weight = np.exp(-eta / config.eta_tau_s)
    return eta_weight * kernel


def pressure_at_point(
    defenders_xy: np.ndarray,
    defenders_vxy: np.ndarray,
    point_xy: tuple[float, float],
    config: PressureConfig,
) -> float:
    contributions = _defender_pressure_contributions(defenders_xy, defenders_vxy, point_xy, config)
    return float(np.sum(contributions))


def pressure_gradient(
    defenders_xy: np.ndarray,
    defenders_vxy: np.ndarray,
    point_xy: tuple[float, float],
    config: PressureConfig,
    delta_ft: float = 2.0,
) -> tuple[float, float]:
    x, y = point_xy
    p_x_plus = pressure_at_point(defenders_xy, defenders_vxy, (x + delta_ft, y), config)
    p_x_minus = pressure_at_point(defenders_xy, defenders_vxy, (x - delta_ft, y), config)
    p_y_plus = pressure_at_point(defenders_xy, defenders_vxy, (x, y + delta_ft), config)
    p_y_minus = pressure_at_point(defenders_xy, defenders_vxy, (x, y - delta_ft), config)
    gx = (p_x_plus - p_x_minus) / (2.0 * delta_ft)
    gy = (p_y_plus - p_y_minus) / (2.0 * delta_ft)
    return gx, gy


def corridor_targets(
    carrier_xy: tuple[float, float],
    defending_right: bool,
) -> dict[str, tuple[float, float]]:
    x, y = carrier_xy
    exit_dir = -1.0 if defending_right else 1.0
    own_net_x = 89.0 if defending_right else -89.0
    strong_sign = np.sign(y) if y != 0 else 1.0

    targets = {
        "middle_lane": (x + exit_dir * 45.0, 0.0),
        "strong_boards": (x + exit_dir * 35.0, strong_sign * 37.0),
        "weak_boards": (x + exit_dir * 35.0, -strong_sign * 37.0),
        "behind_net": (own_net_x + (5.0 if defending_right else -5.0), 0.0),
    }
    return {
        k: clamp_point(v[0], v[1], RINK_X_MIN_FT, RINK_X_MAX_FT, RINK_Y_MIN_FT, RINK_Y_MAX_FT)
        for k, v in targets.items()
    }


def integrated_corridor_pressure(
    defenders_xy: np.ndarray,
    defenders_vxy: np.ndarray,
    origin_xy: tuple[float, float],
    target_xy: tuple[float, float],
    config: PressureConfig,
    n_samples: int = 25,
) -> float:
    xs = np.linspace(origin_xy[0], target_xy[0], n_samples)
    ys = np.linspace(origin_xy[1], target_xy[1], n_samples)
    values = [pressure_at_point(defenders_xy, defenders_vxy, (x, y), config) for x, y in zip(xs, ys)]
    return float(np.mean(values))


def high_pressure_area(
    defenders_xy: np.ndarray,
    defenders_vxy: np.ndarray,
    center_xy: tuple[float, float],
    config: PressureConfig,
) -> float:
    extent = config.compactness_extent_ft
    step = config.compactness_step_ft
    xs = np.arange(center_xy[0] - extent, center_xy[0] + extent + step, step)
    ys = np.arange(center_xy[1] - extent, center_xy[1] + extent + step, step)
    high = 0
    for x in xs:
        for y in ys:
            if not (RINK_X_MIN_FT <= x <= RINK_X_MAX_FT and RINK_Y_MIN_FT <= y <= RINK_Y_MAX_FT):
                continue
            if pressure_at_point(defenders_xy, defenders_vxy, (x, y), config) >= config.pressure_threshold:
                high += 1
    return float(high * (step * step))


def _carrier_for_frame(
    frame_df: pd.DataFrame,
    possessing_team: str,
) -> tuple[float, float, str] | None:
    puck_rows = frame_df[(frame_df["object_type"] == "Puck") & frame_df["x"].notna() & frame_df["y"].notna()]
    offense = frame_df[
        (frame_df["object_type"] == "Player")
        & (frame_df["team_name"] == possessing_team)
        & frame_df["x"].notna()
        & frame_df["y"].notna()
    ]
    if puck_rows.empty or offense.empty:
        return None
    puck = puck_rows.iloc[0]
    px, py = float(puck["x"]), float(puck["y"])
    offense_xy = offense[["x", "y"]].to_numpy(dtype=float)
    d = np.linalg.norm(offense_xy - np.array([[px, py]]), axis=1)
    idx = int(np.argmin(d))
    carrier = offense.iloc[idx]
    if pd.isna(carrier["x"]) or pd.isna(carrier["y"]):
        return None
    return float(carrier["x"]), float(carrier["y"]), str(carrier["player_id"])


def _nearest_defender_distance(defenders_xy: np.ndarray, carrier_xy: tuple[float, float]) -> float:
    if defenders_xy.size == 0:
        return np.nan
    point = np.array(carrier_xy, dtype=float).reshape(1, 2)
    return float(np.min(np.linalg.norm(defenders_xy - point, axis=1)))


def _local_defender_count(defenders_xy: np.ndarray, carrier_xy: tuple[float, float], radius_ft: float) -> int:
    if defenders_xy.size == 0:
        return 0
    point = np.array(carrier_xy, dtype=float).reshape(1, 2)
    d = np.linalg.norm(defenders_xy - point, axis=1)
    return int(np.sum(d <= radius_ft))


def compute_episode_frame_metrics(
    episode_row: pd.Series,
    tracking_df: pd.DataFrame,
    frame_index_df: pd.DataFrame,
    frame_to_indices: dict[str, np.ndarray],
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
    config: PressureConfig,
    frame_stride: int,
) -> pd.DataFrame:
    start_t = episode_row["start_elapsed_seconds"]
    end_t = episode_row["end_elapsed_seconds"]
    candidate_frames = frame_index_df[
        (frame_index_df["elapsed_seconds"] >= start_t) & (frame_index_df["elapsed_seconds"] <= end_t)
    ]
    if candidate_frames.empty:
        return pd.DataFrame()

    sampled = candidate_frames.iloc[:: max(1, frame_stride)]
    possessing_team = episode_row["possessing_team"]
    forechecking_team = episode_row["forechecking_team"]
    episode_id = episode_row["episode_id"]

    records: list[dict[str, float | int | str]] = []

    for frame in sampled.itertuples(index=False):
        frame_rows = tracking_df.iloc[frame_to_indices[frame.image_id]]
        carrier = _carrier_for_frame(frame_rows, possessing_team)
        if carrier is None:
            continue

        carrier_x, carrier_y, carrier_player_id = carrier
        if not (np.isfinite(carrier_x) and np.isfinite(carrier_y)):
            continue
        defenders = frame_rows[(frame_rows["object_type"] == "Player") & (frame_rows["team_name"] == forechecking_team)]
        defenders_xy = _as_xy_array(defenders)
        defenders_vxy = _as_velocity_array(defenders)
        if defenders_xy.size == 0:
            continue

        p_carrier = pressure_at_point(defenders_xy, defenders_vxy, (carrier_x, carrier_y), config)
        gx, gy = pressure_gradient(defenders_xy, defenders_vxy, (carrier_x, carrier_y), config)
        grad_mag = float(np.hypot(gx, gy))

        defending_right = team_defends_right(possessing_team, str(frame.period_label), game_meta, camera_lookup)
        targets = corridor_targets((carrier_x, carrier_y), defending_right)
        corridor_costs = {
            name: integrated_corridor_pressure(defenders_xy, defenders_vxy, (carrier_x, carrier_y), target_xy, config)
            for name, target_xy in targets.items()
        }
        most_open_corridor = min(corridor_costs, key=corridor_costs.get)
        min_corridor_cost = corridor_costs[most_open_corridor]

        board_sign = 1.0 if carrier_y >= 0 else -1.0
        if grad_mag > 0:
            funnel_strength = float((gy * board_sign) / grad_mag)
        else:
            funnel_strength = 0.0

        compactness_area = high_pressure_area(defenders_xy, defenders_vxy, (carrier_x, carrier_y), config)
        nearest_defender_dist = _nearest_defender_distance(defenders_xy, (carrier_x, carrier_y))
        defenders_near = _local_defender_count(defenders_xy, (carrier_x, carrier_y), config.nearby_defender_radius_ft)

        d = np.linalg.norm(defenders_xy - np.array([[carrier_x, carrier_y]]), axis=1)
        contributions = _defender_pressure_contributions(defenders_xy, defenders_vxy, (carrier_x, carrier_y), config)
        role_order = np.argsort(d)
        f_indices = role_order[:3]
        role_mask = np.zeros(len(d), dtype=bool)
        role_mask[f_indices] = True

        sorted_d = d[f_indices]
        sorted_contrib = contributions[f_indices]
        f1_distance = float(sorted_d[0]) if len(sorted_d) > 0 else np.nan
        f2_distance = float(sorted_d[1]) if len(sorted_d) > 1 else np.nan
        f3_distance = float(sorted_d[2]) if len(sorted_d) > 2 else np.nan
        f1_pressure = float(sorted_contrib[0]) if len(sorted_contrib) > 0 else np.nan
        f2_pressure = float(sorted_contrib[1]) if len(sorted_contrib) > 1 else np.nan
        f3_pressure = float(sorted_contrib[2]) if len(sorted_contrib) > 2 else np.nan
        role_total = np.nansum([f1_pressure, f2_pressure, f3_pressure])
        if role_total > 0:
            f1_share = f1_pressure / role_total
            f2_share = f2_pressure / role_total
            f3_share = f3_pressure / role_total
        else:
            f1_share = np.nan
            f2_share = np.nan
            f3_share = np.nan

        local_pressure_points = d <= config.local_pressure_radius_ft
        if np.any(local_pressure_points):
            local_pressure_mean = float(np.mean(contributions[local_pressure_points]))
        else:
            local_pressure_mean = 0.0

        if defending_right:
            blue_line_x = BLUE_LINE_X_ABS_FT
        else:
            blue_line_x = -BLUE_LINE_X_ABS_FT
        pinchers = int(np.sum(np.abs(defenders_xy[:, 0] - blue_line_x) <= 7.0))
        d_pinch_support = int(np.sum((np.abs(defenders_xy[:, 0] - blue_line_x) <= 7.0) & (~role_mask)))

        records.append(
            {
                "episode_id": episode_id,
                "game_id": episode_row["game_id"],
                "elapsed_seconds": float(frame.elapsed_seconds),
                "period_label": str(frame.period_label),
                "image_id": frame.image_id,
                "carrier_player_id": carrier_player_id,
                "carrier_x": carrier_x,
                "carrier_y": carrier_y,
                "pressure_at_carrier": p_carrier,
                "gradient_x": gx,
                "gradient_y": gy,
                "gradient_magnitude": grad_mag,
                "funnel_to_boards": funnel_strength,
                "corridor_middle_lane": corridor_costs["middle_lane"],
                "corridor_strong_boards": corridor_costs["strong_boards"],
                "corridor_weak_boards": corridor_costs["weak_boards"],
                "corridor_behind_net": corridor_costs["behind_net"],
                "most_open_corridor": most_open_corridor,
                "most_open_corridor_cost": min_corridor_cost,
                "pressure_compactness_area": compactness_area,
                "nearest_defender_distance": nearest_defender_dist,
                "defenders_near_carrier": defenders_near,
                "local_pressure_mean": local_pressure_mean,
                "pinchers_near_blueline": pinchers,
                "d_pinch_support_count": d_pinch_support,
                "f1_distance": f1_distance,
                "f2_distance": f2_distance,
                "f3_distance": f3_distance,
                "f1_pressure_contrib": f1_pressure,
                "f2_pressure_contrib": f2_pressure,
                "f3_pressure_contrib": f3_pressure,
                "f1_share_pressure": f1_share,
                "f2_share_pressure": f2_share,
                "f3_share_pressure": f3_share,
                "n_forecheckers": int(defenders_xy.shape[0]),
            }
        )

    return pd.DataFrame(records)


def aggregate_episode_features(frame_metrics: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if frame_metrics.empty:
        return episodes.copy()

    by_ep = frame_metrics.groupby("episode_id")

    agg = by_ep.agg(
        n_frames=("image_id", "size"),
        mean_pressure_at_carrier=("pressure_at_carrier", "mean"),
        peak_pressure_at_carrier=("pressure_at_carrier", "max"),
        mean_gradient_mag=("gradient_magnitude", "mean"),
        mean_funnel_to_boards=("funnel_to_boards", "mean"),
        mean_corridor_middle=("corridor_middle_lane", "mean"),
        mean_corridor_strong_boards=("corridor_strong_boards", "mean"),
        mean_corridor_weak_boards=("corridor_weak_boards", "mean"),
        mean_corridor_behind_net=("corridor_behind_net", "mean"),
        mean_open_corridor_cost=("most_open_corridor_cost", "mean"),
        mean_compactness_area=("pressure_compactness_area", "mean"),
        mean_nearest_defender_distance=("nearest_defender_distance", "mean"),
        mean_defenders_near_carrier=("defenders_near_carrier", "mean"),
        mean_local_pressure=("local_pressure_mean", "mean"),
        mean_pinchers=("pinchers_near_blueline", "mean"),
        mean_d_pinch_support=("d_pinch_support_count", "mean"),
        mean_f1_distance=("f1_distance", "mean"),
        mean_f2_distance=("f2_distance", "mean"),
        mean_f3_distance=("f3_distance", "mean"),
        mean_f1_pressure_contrib=("f1_pressure_contrib", "mean"),
        mean_f2_pressure_contrib=("f2_pressure_contrib", "mean"),
        mean_f3_pressure_contrib=("f3_pressure_contrib", "mean"),
        mean_f1_share_pressure=("f1_share_pressure", "mean"),
        mean_f2_share_pressure=("f2_share_pressure", "mean"),
        mean_f3_share_pressure=("f3_share_pressure", "mean"),
    ).reset_index()

    threshold = frame_metrics["pressure_at_carrier"].quantile(0.75)
    sustained = (
        frame_metrics.assign(above_threshold=lambda x: x["pressure_at_carrier"] >= threshold)
        .groupby("episode_id")["above_threshold"]
        .mean()
        .reset_index(name="share_time_high_pressure")
    )

    closure_records = []
    for ep_id, group in by_ep:
        group = group.sort_values("elapsed_seconds")
        t0 = group["elapsed_seconds"].min()
        t3 = t0 + 3.0
        early = group[group["elapsed_seconds"] <= t3]
        if early.empty:
            closure_rate = np.nan
        else:
            first_val = float(early["most_open_corridor_cost"].iloc[0])
            last_val = float(early["most_open_corridor_cost"].iloc[-1])
            dt = max(float(early["elapsed_seconds"].iloc[-1] - t0), 0.5)
            closure_rate = (first_val - last_val) / dt
        closure_records.append({"episode_id": ep_id, "corridor_closure_rate_3s": closure_rate})
    closure = pd.DataFrame(closure_records)

    merged = episodes.merge(agg, on="episode_id", how="left")
    merged = merged.merge(sustained, on="episode_id", how="left")
    merged = merged.merge(closure, on="episode_id", how="left")
    return merged


def build_frame_index(tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame_index = tracking_df[["image_id", "elapsed_seconds", "period_label"]].drop_duplicates()
    frame_index = frame_index.sort_values("elapsed_seconds").reset_index(drop=True)
    frame_to_indices = tracking_df.groupby("image_id").indices
    return frame_index, frame_to_indices
