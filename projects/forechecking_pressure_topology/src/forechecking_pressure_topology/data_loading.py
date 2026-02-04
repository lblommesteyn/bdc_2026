from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    GameMeta,
    TRACKING_SUFFIXES,
    elapsed_from_period_clock,
    game_meta_from_game_id,
    parse_clock_to_seconds,
    parse_period_label,
)


EVENT_COLUMNS = [
    "Date",
    "Home_Team",
    "Away_Team",
    "Period",
    "Clock",
    "Home_Team_Skaters",
    "Away_Team_Skaters",
    "Home_Team_Goals",
    "Away_Team_Goals",
    "Team",
    "Player_Id",
    "Event",
    "X_Coordinate",
    "Y_Coordinate",
    "Detail_1",
    "Detail_2",
    "Detail_3",
    "Detail_4",
    "Player_Id_2",
    "X_Coordinate_2",
    "Y_Coordinate_2",
]

TRACKING_COLUMNS = [
    "Image Id",
    "Period",
    "Game Clock",
    "Player or Puck",
    "Team",
    "Player Id",
    "Rink Location X (Feet)",
    "Rink Location Y (Feet)",
    "Rink Location Z (Feet)",
]


def load_camera_orientations(raw_dir: Path) -> dict[str, str]:
    path = raw_dir / "camera_orientations.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Game" not in df.columns or "GoalieTeamOnRightSideOfRink1stPeriod" not in df.columns:
        return {}
    return dict(
        zip(
            df["Game"].astype(str).str.strip(),
            df["GoalieTeamOnRightSideOfRink1stPeriod"].astype(str).str.strip(),
        )
    )


def team_defends_right(
    team_name: str,
    period_label: str,
    game_meta: GameMeta,
    camera_lookup: dict[str, str],
) -> bool:
    p1_goalie_side = camera_lookup.get(game_meta.pretty, "Home")
    if p1_goalie_side == "Home":
        right_team_p1 = game_meta.home_team
        left_team_p1 = game_meta.away_team
    else:
        right_team_p1 = game_meta.away_team
        left_team_p1 = game_meta.home_team

    if period_label == "OT":
        period_number = 4
    else:
        try:
            period_number = int(period_label)
        except ValueError:
            period_number = 1

    if period_number % 2 == 1:
        right_team = right_team_p1
    else:
        right_team = left_team_p1
    return team_name == right_team


def load_events_for_game(raw_dir: Path, game_id: str) -> pd.DataFrame:
    path = raw_dir / f"{game_id}.Events.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, usecols=EVENT_COLUMNS)
    df["game_id"] = game_id
    df["event_idx"] = np.arange(len(df), dtype=int)
    df["period_label"] = df["Period"].map(parse_period_label).replace({"4": "OT"})
    df["clock_seconds_remaining"] = df["Clock"].map(parse_clock_to_seconds)
    df["elapsed_seconds"] = [
        elapsed_from_period_clock(period_label, clock)
        for period_label, clock in zip(df["period_label"], df["clock_seconds_remaining"])
    ]
    df = df.sort_values(["elapsed_seconds", "event_idx"], kind="mergesort").reset_index(drop=True)
    return df


def _tracking_file_paths(raw_dir: Path, game_id: str) -> list[Path]:
    out: list[Path] = []
    for suffix in TRACKING_SUFFIXES:
        path = raw_dir / f"{game_id}.{suffix}.csv"
        if path.exists():
            out.append(path)
    return out


def _image_numeric(image_id: str) -> float:
    if not isinstance(image_id, str) or "_" not in image_id:
        return np.nan
    maybe_num = image_id.rsplit("_", 1)[-1]
    try:
        return float(int(maybe_num))
    except ValueError:
        return np.nan


def load_tracking_for_game(raw_dir: Path, game_id: str) -> pd.DataFrame:
    game_meta = game_meta_from_game_id(game_id)
    paths = _tracking_file_paths(raw_dir, game_id)
    if not paths:
        raise FileNotFoundError(f"No tracking files found for {game_id}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        chunk = pd.read_csv(path, usecols=TRACKING_COLUMNS, low_memory=False)
        chunk["source_file"] = path.name
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)

    rename_map = {
        "Image Id": "image_id",
        "Period": "period_raw",
        "Game Clock": "game_clock",
        "Player or Puck": "object_type",
        "Team": "team_side",
        "Player Id": "player_id",
        "Rink Location X (Feet)": "x",
        "Rink Location Y (Feet)": "y",
        "Rink Location Z (Feet)": "z",
    }
    df = df.rename(columns=rename_map)
    df["game_id"] = game_id
    df["period_label"] = df["period_raw"].map(parse_period_label)
    df["clock_seconds_remaining"] = df["game_clock"].map(parse_clock_to_seconds)
    df["elapsed_seconds_base"] = [
        elapsed_from_period_clock(period_label, clock)
        for period_label, clock in zip(df["period_label"], df["clock_seconds_remaining"])
    ]
    df["image_numeric"] = df["image_id"].astype(str).map(_image_numeric)

    period_order = {"1": 1, "2": 2, "3": 3, "OT": 4}
    frame_table = (
        df[["image_id", "period_label", "clock_seconds_remaining", "elapsed_seconds_base", "image_numeric"]]
        .drop_duplicates()
        .copy()
    )
    frame_table["period_order"] = frame_table["period_label"].map(period_order).fillna(99)
    frame_table = frame_table.sort_values(
        ["period_order", "clock_seconds_remaining", "image_numeric"], ascending=[True, False, True]
    )
    group_cols = ["period_label", "clock_seconds_remaining"]
    frame_table["frame_rank_in_second"] = frame_table.groupby(group_cols).cumcount()
    frame_table["frames_in_second"] = frame_table.groupby(group_cols)["image_id"].transform("size")
    frame_table["subsecond"] = (frame_table["frame_rank_in_second"] + 0.5) / frame_table["frames_in_second"]
    frame_table["elapsed_seconds"] = frame_table["elapsed_seconds_base"] + frame_table["subsecond"]
    df = df.merge(frame_table[["image_id", "elapsed_seconds"]], on="image_id", how="left")

    df["team_name"] = np.where(
        df["team_side"] == "Home",
        game_meta.home_team,
        np.where(df["team_side"] == "Away", game_meta.away_team, np.nan),
    )
    return df

