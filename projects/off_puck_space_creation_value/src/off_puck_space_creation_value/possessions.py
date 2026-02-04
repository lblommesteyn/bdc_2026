from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import team_attacks_right
from .utils import GameMeta


OFFENSIVE_ZONE_X_ABS_FT = 25.0
START_EVENTS = {"Zone Entry", "Play", "Puck Recovery", "Takeaway", "Faceoff Win"}
LOSS_EVENTS = {"Puck Recovery", "Takeaway", "Faceoff Win"}
STOP_EVENTS = {"Goal", "Penalty Taken"}


@dataclass
class OffensivePossession:
    possession_id: str
    game_id: str
    attacking_team: str
    defending_team: str
    start_elapsed_seconds: float
    end_elapsed_seconds: float
    duration_seconds: float
    start_period_label: str
    end_period_label: str
    start_event: str
    end_event: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    end_reason: str
    shot_or_goal_in_possession: int
    slot_shot_or_goal_in_possession: int
    n_events: int


def _is_5v5(row: dict[str, Any]) -> bool:
    return row.get("Home_Team_Skaters") == 5 and row.get("Away_Team_Skaters") == 5


def _is_offensive_zone(team_name: str, period_label: str, x_coord: float, game_id: str, camera_lookup: dict[str, str]) -> bool:
    if pd.isna(x_coord) or not isinstance(team_name, str):
        return False
    attacks_right = team_attacks_right(team_name, period_label, game_id, camera_lookup)
    if attacks_right:
        return x_coord >= OFFENSIVE_ZONE_X_ABS_FT
    return x_coord <= -OFFENSIVE_ZONE_X_ABS_FT


def _is_outside_offensive_zone(
    team_name: str,
    period_label: str,
    x_coord: float,
    game_id: str,
    camera_lookup: dict[str, str],
) -> bool:
    return not _is_offensive_zone(team_name, period_label, x_coord, game_id, camera_lookup)


def _slot_shot(team_name: str, period_label: str, x: float, y: float, game_id: str, camera_lookup: dict[str, str]) -> bool:
    if pd.isna(x) or pd.isna(y):
        return False
    attacks_right = team_attacks_right(team_name, period_label, game_id, camera_lookup)
    in_inner_lane = abs(y) <= 20
    if attacks_right:
        return in_inner_lane and x >= 69
    return in_inner_lane and x <= -69


def segment_offensive_possessions(
    events_df: pd.DataFrame,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
    max_possession_seconds: float = 30.0,
) -> pd.DataFrame:
    rows = events_df.to_dict("records")
    possessions: list[dict[str, Any]] = []
    counter = 0

    for i, row in enumerate(rows):
        team = row.get("Team")
        event = row.get("Event")
        period_label = row.get("period_label")
        x = row.get("X_Coordinate")

        if event not in START_EVENTS:
            continue
        if not isinstance(team, str) or team == "":
            continue
        if not _is_5v5(row):
            continue
        if not _is_offensive_zone(team, period_label, x, game_meta.game_id, camera_lookup):
            continue

        attacking_team = team
        defending_team = game_meta.home_team if team == game_meta.away_team else game_meta.away_team
        start_elapsed = float(row["elapsed_seconds"])
        end_row = row
        end_reason = "timeout"

        for j in range(i + 1, len(rows)):
            next_row = rows[j]
            dt = float(next_row["elapsed_seconds"]) - start_elapsed
            if dt > max_possession_seconds:
                end_row = next_row
                end_reason = "timeout"
                break

            next_team = next_row.get("Team")
            next_event = next_row.get("Event")
            next_period = next_row.get("period_label")
            next_x = next_row.get("X_Coordinate")

            if next_team == defending_team and next_event in LOSS_EVENTS:
                end_row = next_row
                end_reason = "turnover_or_recovery"
                break

            if (
                next_team == attacking_team
                and next_event in {"Dump In/Out", "Play", "Incomplete Play", "Takeaway"}
                and _is_outside_offensive_zone(attacking_team, next_period, next_x, game_meta.game_id, camera_lookup)
            ):
                end_row = next_row
                end_reason = "zone_exit"
                break

            if next_event in STOP_EVENTS:
                end_row = next_row
                end_reason = "stoppage"
                break
        else:
            end_row = rows[-1]
            end_reason = "timeout"

        in_window = events_df[
            (events_df["elapsed_seconds"] >= start_elapsed)
            & (events_df["elapsed_seconds"] <= float(end_row["elapsed_seconds"]))
            & (events_df["Team"] == attacking_team)
        ]
        shot_or_goal = in_window["Event"].isin(["Shot", "Goal"]).any()
        slot_shot_or_goal = in_window.apply(
            lambda r: r["Event"] in {"Shot", "Goal"}
            and _slot_shot(attacking_team, r["period_label"], r["X_Coordinate"], r["Y_Coordinate"], game_meta.game_id, camera_lookup),
            axis=1,
        ).any()

        possession = OffensivePossession(
            possession_id=f"{game_meta.game_id}__pos_{counter:04d}",
            game_id=game_meta.game_id,
            attacking_team=attacking_team,
            defending_team=defending_team,
            start_elapsed_seconds=start_elapsed,
            end_elapsed_seconds=float(end_row["elapsed_seconds"]),
            duration_seconds=float(end_row["elapsed_seconds"] - start_elapsed),
            start_period_label=str(row["period_label"]),
            end_period_label=str(end_row["period_label"]),
            start_event=str(row["Event"]),
            end_event=str(end_row["Event"]),
            start_x=float(row["X_Coordinate"]),
            start_y=float(row["Y_Coordinate"]),
            end_x=float(end_row.get("X_Coordinate", np.nan)),
            end_y=float(end_row.get("Y_Coordinate", np.nan)),
            end_reason=end_reason,
            shot_or_goal_in_possession=int(bool(shot_or_goal)),
            slot_shot_or_goal_in_possession=int(bool(slot_shot_or_goal)),
            n_events=int(len(in_window)),
        )
        possessions.append(asdict(possession))
        counter += 1

    out = pd.DataFrame(possessions)
    if not out.empty:
        out = out.sort_values(["game_id", "start_elapsed_seconds"]).reset_index(drop=True)
    return out

