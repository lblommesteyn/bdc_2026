from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import BLUE_LINE_X_ABS_FT, DZ_THRESHOLD_X_ABS_FT, EPISODE_END_STOPPAGES, FORECHECK_START_EVENTS
from .data_loading import team_defends_right
from .utils import GameMeta


@dataclass
class ForecheckEpisode:
    game_id: str
    date: str
    away_team: str
    home_team: str
    possessing_team: str
    forechecking_team: str
    start_event_idx: int
    end_event_idx: int
    start_elapsed_seconds: float
    end_elapsed_seconds: float
    duration_seconds: float
    start_period_label: str
    end_period_label: str
    start_clock: str
    end_clock: str
    start_event: str
    end_event: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    outcome: str
    turnover_in_dzone: int
    forced_dump_out: int
    controlled_exit_allowed: int
    stoppage: int
    timeout: int
    exit_lane: str
    start_is_5v5: int


def _is_5v5(row: pd.Series) -> bool:
    return row["Home_Team_Skaters"] == 5 and row["Away_Team_Skaters"] == 5


def _is_in_team_dzone(
    team_name: str,
    x_coord: float,
    period_label: str,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
) -> bool:
    if pd.isna(x_coord) or not isinstance(team_name, str) or team_name == "":
        return False
    defending_right = team_defends_right(team_name, period_label, game_meta, camera_lookup)
    if defending_right:
        return x_coord >= DZ_THRESHOLD_X_ABS_FT
    return x_coord <= -DZ_THRESHOLD_X_ABS_FT


def _outside_team_dzone(
    team_name: str,
    x_coord: float,
    period_label: str,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
) -> bool:
    return not _is_in_team_dzone(team_name, x_coord, period_label, game_meta, camera_lookup)


def _neutral_or_offensive_zone(
    team_name: str,
    x_coord: float,
    period_label: str,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
) -> bool:
    if pd.isna(x_coord):
        return False
    defending_right = team_defends_right(team_name, period_label, game_meta, camera_lookup)
    if defending_right:
        return x_coord < BLUE_LINE_X_ABS_FT
    return x_coord > -BLUE_LINE_X_ABS_FT


def _exit_lane(y_coord: float) -> str:
    if pd.isna(y_coord):
        return "unknown"
    if abs(y_coord) <= 12:
        return "middle"
    if y_coord > 12:
        return "boards_positive"
    return "boards_negative"


def _to_episode_record(episode: ForecheckEpisode) -> dict[str, Any]:
    return asdict(episode)


def segment_forecheck_episodes(
    events_df: pd.DataFrame,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
    max_episode_seconds: float = 25.0,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    rows = events_df.to_dict("records")
    episodes: list[dict[str, Any]] = []

    last_team = None
    last_elapsed = -np.inf

    for i, start_row in enumerate(rows):
        start_team = start_row.get("Team")
        start_event = start_row.get("Event")
        start_period_label = start_row.get("period_label")
        start_elapsed = start_row.get("elapsed_seconds")
        start_x = start_row.get("X_Coordinate")

        if start_event not in FORECHECK_START_EVENTS:
            last_team = start_team if isinstance(start_team, str) else last_team
            last_elapsed = start_elapsed if pd.notna(start_elapsed) else last_elapsed
            continue

        if not isinstance(start_team, str) or start_team == "":
            continue
        if not _is_5v5(pd.Series(start_row)):
            continue
        if not _is_in_team_dzone(start_team, start_x, start_period_label, game_meta, camera_lookup):
            continue
        if last_team == start_team and (start_elapsed - last_elapsed) <= 2 and start_event != "Faceoff Win":
            continue

        possessing_team = start_team
        forechecking_team = game_meta.home_team if possessing_team == game_meta.away_team else game_meta.away_team

        end_idx = i
        end_row = start_row
        outcome = "timeout"
        for j in range(i + 1, len(rows)):
            row = rows[j]
            row_elapsed = row.get("elapsed_seconds")
            if pd.isna(row_elapsed):
                continue
            if row_elapsed - start_elapsed > max_episode_seconds:
                end_idx = j
                end_row = row
                outcome = "timeout"
                break

            row_event = row.get("Event")
            row_team = row.get("Team")
            row_period_label = row.get("period_label")
            row_x = row.get("X_Coordinate")

            if row_team == possessing_team:
                if row_event == "Dump In/Out" and str(row.get("Detail_1", "")).strip().lower() == "lost":
                    end_idx = j
                    end_row = row
                    outcome = "forced_dump_out"
                    break

                if row_event in {"Zone Entry", "Play", "Incomplete Play"} and _neutral_or_offensive_zone(
                    possessing_team,
                    row_x,
                    row_period_label,
                    game_meta,
                    camera_lookup,
                ):
                    end_idx = j
                    end_row = row
                    outcome = "controlled_exit_allowed"
                    break

            if (
                row_team == forechecking_team
                and row_event in {"Puck Recovery", "Takeaway", "Faceoff Win"}
                and _is_in_team_dzone(possessing_team, row_x, row_period_label, game_meta, camera_lookup)
            ):
                end_idx = j
                end_row = row
                outcome = "turnover_in_dzone"
                break

            if row_event in EPISODE_END_STOPPAGES and row_elapsed > start_elapsed:
                end_idx = j
                end_row = row
                outcome = "stoppage"
                break
        else:
            end_idx = len(rows) - 1
            end_row = rows[-1]
            outcome = "timeout"

        episode = ForecheckEpisode(
            game_id=game_meta.game_id,
            date=game_meta.date,
            away_team=game_meta.away_team,
            home_team=game_meta.home_team,
            possessing_team=possessing_team,
            forechecking_team=forechecking_team,
            start_event_idx=int(start_row["event_idx"]),
            end_event_idx=int(end_row["event_idx"]),
            start_elapsed_seconds=float(start_row["elapsed_seconds"]),
            end_elapsed_seconds=float(end_row["elapsed_seconds"]),
            duration_seconds=float(end_row["elapsed_seconds"] - start_row["elapsed_seconds"]),
            start_period_label=str(start_row["period_label"]),
            end_period_label=str(end_row["period_label"]),
            start_clock=str(start_row["Clock"]),
            end_clock=str(end_row["Clock"]),
            start_event=str(start_row["Event"]),
            end_event=str(end_row["Event"]),
            start_x=float(start_row["X_Coordinate"]),
            start_y=float(start_row["Y_Coordinate"]),
            end_x=float(end_row.get("X_Coordinate", np.nan)),
            end_y=float(end_row.get("Y_Coordinate", np.nan)),
            outcome=outcome,
            turnover_in_dzone=int(outcome == "turnover_in_dzone"),
            forced_dump_out=int(outcome == "forced_dump_out"),
            controlled_exit_allowed=int(outcome == "controlled_exit_allowed"),
            stoppage=int(outcome == "stoppage"),
            timeout=int(outcome == "timeout"),
            exit_lane=_exit_lane(float(end_row.get("Y_Coordinate", np.nan))),
            start_is_5v5=1,
        )
        episodes.append(_to_episode_record(episode))

        last_team = start_team
        last_elapsed = start_elapsed

    out = pd.DataFrame(episodes)
    if not out.empty:
        out = out.sort_values(["game_id", "start_elapsed_seconds"]).reset_index(drop=True)
    return out

