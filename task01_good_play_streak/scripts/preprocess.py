import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
    "fixed_drive", "drive", "qtr", "down", "ydstogo", "goal_to_go", "yardline_100",
    "half_seconds_remaining", "game_seconds_remaining", "score_differential",
    "posteam_score", "defteam_score", "play_type", "yards_gained", "epa", "success",
    "pass", "rush", "qb_kneel", "qb_spike", "aborted_play", "two_point_attempt",
    "shotgun", "no_huddle", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "spread_line", "total_line", "desc", "first_down", "touchdown", "interception",
    "fumble_lost", "sack", "complete_pass", "air_yards", "yards_after_catch",
    "passer_player_id", "rusher_player_id", "receiver_player_id", "xpass", "pass_oe",
    "drive_play_count", "drive_start_yard_line", "drive_end_yard_line",
    "fixed_drive_result", "drive_ended_with_score", "play_deleted",
]

frames = []
for path in sorted(RAW.glob("play_by_play_*.csv.gz")):
    frame = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    frames.append(frame)
    print(path.name, len(frame), flush=True)

plays = pd.concat(frames, ignore_index=True)
print("raw rows", len(plays))

plays = plays[plays["season_type"].isin(["REG", "POST"])]
plays = plays[plays["play_type"].isin(["pass", "run"])]
plays = plays[plays["play_deleted"].fillna(0) == 0]
for column in ["qb_kneel", "qb_spike", "aborted_play", "two_point_attempt"]:
    plays = plays[plays[column].fillna(0) == 0]
plays = plays[plays["epa"].notna()]
plays = plays[plays["posteam"].notna()]
plays = plays[plays["down"].notna()]
plays = plays[plays["fixed_drive"].notna()]

plays["fixed_drive"] = plays["fixed_drive"].astype("int32")
plays["good"] = (plays["epa"] > 0).astype("int8")
plays = plays.sort_values(["game_id", "fixed_drive", "play_id"]).reset_index(drop=True)

prev = plays.groupby(["game_id", "fixed_drive"], sort=False)["good"].shift(1)
prev = prev.fillna(0).astype("int8")
block = (prev != 1).cumsum()
plays["streak"] = prev.groupby(block).cumsum().astype("int8")
plays["prev_good"] = prev

print("analysis rows", len(plays))
print("seasons", plays["season"].min(), plays["season"].max())
print("good rate", plays["good"].mean())
print(plays.groupby("season")["good"].agg(["size", "mean"]).tail(10).to_string())
print(plays["streak"].value_counts().sort_index().to_string())

plays.to_parquet(OUT / "plays.parquet", index=False)
print("saved", OUT / "plays.parquet")
