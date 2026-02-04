from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GameMeta:
    game_id: str
    date: str
    away_team: str
    home_team: str

    @property
    def pretty(self) -> str:
        return f"{self.date} {self.away_team} @ {self.home_team}"


def game_meta_from_game_id(game_id: str) -> GameMeta:
    pretty = game_id.replace(".@.", " @ ").replace(".", " ")
    date, matchup = pretty.split(" ", 1)
    away_team, home_team = matchup.split(" @ ")
    return GameMeta(game_id=game_id, date=date, away_team=away_team, home_team=home_team)


def parse_clock_to_seconds(clock_value: object) -> float:
    if pd.isna(clock_value):
        return np.nan
    text = str(clock_value)
    if ":" not in text:
        return np.nan
    mm, ss = text.split(":", 1)
    try:
        return float(int(mm) * 60 + int(ss))
    except ValueError:
        return np.nan


def parse_period_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"OT", "SO"}:
        return text
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def period_duration_seconds(period_label: str) -> int:
    return 300 if period_label == "OT" else 1200


def period_start_elapsed(period_label: str) -> int:
    if period_label == "OT":
        return 3600
    return (int(period_label) - 1) * 1200


def elapsed_from_period_clock(period_label: str, clock_seconds_remaining: float) -> float:
    if np.isnan(clock_seconds_remaining):
        return np.nan
    return period_start_elapsed(period_label) + period_duration_seconds(period_label) - clock_seconds_remaining


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_game_ids(raw_dir: Path) -> list[str]:
    game_ids = {p.name.replace(".Events.csv", "") for p in raw_dir.glob("*.Events.csv")}
    return sorted(game_ids)

