from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import elapsed_from_period_clock, game_meta_from_game_id, parse_clock_to_seconds, parse_period_label


TRACKING_SUFFIXES = ("Tracking_P1", "Tracking_P2", "Tracking_P3", "Tracking_POT")

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
    "Player Jersey Number",
    "Rink Location X (Feet)",
    "Rink Location Y (Feet)",
    "Rink Location Z (Feet)",
]


def load_camera_orientations(raw_dir: Path) -> dict[str, str]:
    path = raw_dir / "camera_orientations.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(
        zip(
            df["Game"].astype(str).str.strip(),
            df["GoalieTeamOnRightSideOfRink1stPeriod"].astype(str).str.strip(),
        )
    )


def team_defends_right(team_name: str, period_label: str, game_id: str, camera_lookup: dict[str, str]) -> bool:
    meta = game_meta_from_game_id(game_id)
    p1_side = camera_lookup.get(meta.pretty, "Home")
    if p1_side == "Home":
        right_team_p1, left_team_p1 = meta.home_team, meta.away_team
    else:
        right_team_p1, left_team_p1 = meta.away_team, meta.home_team

    period_num = 4 if period_label == "OT" else int(period_label)
    right_team = right_team_p1 if period_num % 2 == 1 else left_team_p1
    return team_name == right_team


def team_attacks_right(team_name: str, period_label: str, game_id: str, camera_lookup: dict[str, str]) -> bool:
    return not team_defends_right(team_name, period_label, game_id, camera_lookup)


def load_events_for_game(raw_dir: Path, game_id: str) -> pd.DataFrame:
    path = raw_dir / f"{game_id}.Events.csv"
    df = pd.read_csv(path, usecols=EVENT_COLUMNS)
    df["game_id"] = game_id
    df["period_label"] = df["Period"].map(parse_period_label).replace({"4": "OT"})
    df["clock_seconds_remaining"] = df["Clock"].map(parse_clock_to_seconds)
    df["elapsed_seconds"] = [
        elapsed_from_period_clock(period_label, clock)
        for period_label, clock in zip(df["period_label"], df["clock_seconds_remaining"])
    ]
    df["event_idx"] = np.arange(len(df), dtype=int)
    return df.sort_values(["elapsed_seconds", "event_idx"], kind="mergesort").reset_index(drop=True)


def _tracking_file_paths(raw_dir: Path, game_id: str) -> list[Path]:
    out: list[Path] = []
    for suffix in TRACKING_SUFFIXES:
        p = raw_dir / f"{game_id}.{suffix}.csv"
        if p.exists():
            out.append(p)
    return out


def _image_numeric(image_id: str) -> float:
    if not isinstance(image_id, str) or "_" not in image_id:
        return np.nan
    tail = image_id.rsplit("_", 1)[-1]
    try:
        return float(int(tail))
    except ValueError:
        return np.nan


def load_tracking_for_game(raw_dir: Path, game_id: str) -> pd.DataFrame:
    meta = game_meta_from_game_id(game_id)
    parts = []
    for path in _tracking_file_paths(raw_dir, game_id):
        parts.append(pd.read_csv(path, usecols=TRACKING_COLUMNS, low_memory=False))
    if not parts:
        raise FileNotFoundError(f"No tracking files for {game_id}")
    df = pd.concat(parts, ignore_index=True)

    df = df.rename(
        columns={
            "Image Id": "image_id",
            "Period": "period_raw",
            "Game Clock": "game_clock",
            "Player or Puck": "object_type",
            "Team": "team_side",
            "Player Id": "player_id",
            "Player Jersey Number": "jersey_number",
            "Rink Location X (Feet)": "x",
            "Rink Location Y (Feet)": "y",
            "Rink Location Z (Feet)": "z",
        }
    )

    df["game_id"] = game_id
    df["period_label"] = df["period_raw"].map(parse_period_label)
    df["clock_seconds_remaining"] = df["game_clock"].map(parse_clock_to_seconds)
    df["elapsed_seconds_base"] = [
        elapsed_from_period_clock(period_label, clock)
        for period_label, clock in zip(df["period_label"], df["clock_seconds_remaining"])
    ]
    df["image_numeric"] = df["image_id"].astype(str).map(_image_numeric)

    frame_table = (
        df[["image_id", "period_label", "clock_seconds_remaining", "elapsed_seconds_base", "image_numeric"]]
        .drop_duplicates()
        .copy()
    )
    frame_table["period_order"] = frame_table["period_label"].map({"1": 1, "2": 2, "3": 3, "OT": 4}).fillna(99)
    frame_table = frame_table.sort_values(
        ["period_order", "clock_seconds_remaining", "image_numeric"], ascending=[True, False, True]
    )
    frame_table["rank_in_second"] = frame_table.groupby(["period_label", "clock_seconds_remaining"]).cumcount()
    frame_table["n_in_second"] = frame_table.groupby(["period_label", "clock_seconds_remaining"])["image_id"].transform(
        "size"
    )
    frame_table["subsecond"] = (frame_table["rank_in_second"] + 0.5) / frame_table["n_in_second"]
    frame_table["elapsed_seconds"] = frame_table["elapsed_seconds_base"] + frame_table["subsecond"]
    df = df.merge(frame_table[["image_id", "elapsed_seconds"]], on="image_id", how="left")

    df["team_name"] = np.where(
        df["team_side"] == "Home",
        meta.home_team,
        np.where(df["team_side"] == "Away", meta.away_team, np.nan),
    )
    jersey = pd.to_numeric(df["jersey_number"], errors="coerce")
    jersey_key = jersey.map(lambda v: str(int(v)) if pd.notna(v) else "")
    player_id_key = pd.to_numeric(df["player_id"], errors="coerce").map(lambda v: str(int(v)) if pd.notna(v) else "")
    df["player_key"] = np.where(
        df["object_type"] == "Player",
        np.where(
            jersey_key != "",
            df["team_name"].astype(str) + "#" + jersey_key,
            df["team_name"].astype(str) + "@id" + player_id_key,
        ),
        np.nan,
    )
    return df
