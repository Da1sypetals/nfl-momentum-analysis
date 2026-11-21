import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "processed" / "plays.parquet"
OUT = ROOT / "task03_behavior_change" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def task_next_pass():
    cols = [
        "game_id", "fixed_drive", "play_id", "season", "posteam", "defteam",
        "down", "ydstogo", "play_type", "pass", "complete_pass", "interception",
        "sack", "rush", "epa", "score_differential", "half_seconds_remaining", "qtr",
        "xpass", "pass_oe",
    ]
    plays = pd.read_parquet(BASE, columns=cols)
    plays = plays.sort_values(["game_id", "fixed_drive", "play_id"]).reset_index(drop=True)
    grouped = plays.groupby(["game_id", "fixed_drive"], sort=False)
    plays["next_play_type"] = grouped["play_type"].shift(-1)
    plays["next_down"] = grouped["down"].shift(-1)
    plays["next_ydstogo"] = grouped["ydstogo"].shift(-1)
    plays["next_pass"] = (plays["next_play_type"] == "pass").astype(float)
    plays.loc[plays["next_play_type"].isna(), "next_pass"] = np.nan
    plays["next_pass_oe"] = grouped["pass_oe"].shift(-1)

    first10 = plays[(plays["down"] == 1) & (plays["ydstogo"] == 10)].copy()
    first10["result"] = "other"
    first10.loc[first10["rush"].fillna(0) == 1, "result"] = "rush"
    first10.loc[first10["complete_pass"].fillna(0) == 1, "result"] = "complete"
    first10.loc[
        (first10["pass"].fillna(0) == 1)
        & (first10["complete_pass"].fillna(0) == 0)
        & (first10["interception"].fillna(0) == 0)
        & (first10["sack"].fillna(0) == 0),
        "result",
    ] = "incomplete"
    first10.loc[first10["sack"].fillna(0) == 1, "result"] = "sack"
    first10.loc[first10["interception"].fillna(0) == 1, "result"] = "interception"

    overall = (
        first10.groupby("result")
        .agg(
            size=("next_pass", "count"),
            mean=("next_pass", "mean"),
            mean_pass_oe=("next_pass_oe", "mean"),
        )
        .reset_index()
    )
    overall["level"] = "all"
    overall["next_down"] = np.nan

    by_down = (
        first10.groupby(["result", "next_down"])
        .agg(
            size=("next_pass", "count"),
            mean=("next_pass", "mean"),
            mean_pass_oe=("next_pass_oe", "mean"),
        )
        .reset_index()
    )
    by_down["level"] = "next_down"

    return pd.concat([overall, by_down], ignore_index=True)


def task_lead_pass():
    cols = [
        "game_id", "fixed_drive", "play_id", "posteam", "defteam",
        "score_differential", "drive_ended_with_score", "play_type",
        "qtr", "game_seconds_remaining",
    ]
    plays = pd.read_parquet(BASE, columns=cols)
    drives = (
        plays.groupby(["game_id", "fixed_drive"])
        .agg(
            posteam=("posteam", "first"),
            start_score=("score_differential", "first"),
            ended_score=("drive_ended_with_score", "max"),
            pass_rate=("play_type", lambda x: (x == "pass").mean()),
            n_plays=("play_id", "size"),
            start_time=("game_seconds_remaining", "first"),
            qtr_start=("qtr", "first"),
        )
        .reset_index()
    )
    drives = drives.sort_values(["game_id", "fixed_drive"])
    drives["prev_ended_score"] = drives.groupby("game_id")["ended_score"].shift(1)
    drives["prev_posteam"] = drives.groupby("game_id")["posteam"].shift(1)
    leading = drives[
        (drives["start_score"] > 0)
        & (drives["prev_posteam"].notna())
        & (drives["prev_posteam"] != drives["posteam"])
    ].copy()
    leading["score_bucket"] = pd.cut(
        leading["start_score"], [0, 3, 7, 14, 100],
        labels=["1-3", "4-7", "8-14", "15+"],
    )

    overall = leading.groupby("prev_ended_score")["pass_rate"].agg(["size", "mean"]).reset_index()
    overall["level"] = "all"
    overall["score_bucket"] = "all"

    by_score = (
        leading.groupby(["score_bucket", "prev_ended_score"], observed=True)["pass_rate"]
        .agg(["size", "mean"])
        .reset_index()
    )
    by_score["level"] = "score_bucket"

    return pd.concat([overall, by_score], ignore_index=True)


def task_rb_share():
    cols = ["game_id", "posteam", "play_id", "qtr", "rush", "yards_gained", "rusher_player_id"]
    plays = pd.read_parquet(BASE, columns=cols)
    rush = plays[plays["rush"].fillna(0) == 1].copy()
    rush = rush[rush["rusher_player_id"].notna()]
    rush["half"] = np.where(rush["qtr"] <= 2, "first", "second")

    first = (
        rush[rush["half"] == "first"]
        .groupby(["game_id", "posteam", "rusher_player_id"])
        .agg(carries=("play_id", "size"), yards=("yards_gained", "sum"))
        .reset_index()
    )

    second = rush[rush["half"] == "second"]
    team = second.groupby(["game_id", "posteam"]).size().rename("team_carries").reset_index()

    rows = []
    for min_first in [1, 2, 3, 5]:
        for min_team in [1, 3]:
            eligible = first[first["carries"] >= min_first]
            if eligible.empty:
                continue
            best = (
                eligible.sort_values(["game_id", "posteam", "yards"], ascending=[True, True, False])
                .drop_duplicates(["game_id", "posteam"])
            )
            best_second = second.merge(
                best[["game_id", "posteam", "rusher_player_id"]],
                on=["game_id", "posteam", "rusher_player_id"],
            )
            best_counts = best_second.groupby(["game_id", "posteam"]).size().rename("best_carries").reset_index()
            merged = team.merge(best_counts, on=["game_id", "posteam"], how="left")
            merged["best_carries"] = merged["best_carries"].fillna(0)
            merged = merged[merged["team_carries"] >= min_team]
            merged["share"] = merged["best_carries"] / merged["team_carries"]
            rows.append({
                "min_first_half_carries": min_first,
                "min_second_half_team_carries": min_team,
                "n_team_games": len(merged),
                "mean_share": merged["share"].mean(),
                "median_share": merged["share"].median(),
            })
    return pd.DataFrame(rows)


def task_long_run():
    cols = ["game_id", "posteam", "play_id", "qtr", "rush", "yards_gained", "epa", "rusher_player_id"]
    plays = pd.read_parquet(BASE, columns=cols)
    rush = plays[plays["rush"].fillna(0) == 1].copy()
    rush = rush[rush["rusher_player_id"].notna()]
    rush = rush.sort_values(["game_id", "rusher_player_id", "play_id"])
    rush["prev_yards"] = rush.groupby(["game_id", "rusher_player_id"])["yards_gained"].shift(1)
    rush = rush[rush["prev_yards"].notna()].copy()
    rush["prev_group"] = np.where(rush["prev_yards"] >= 10, "long", "short")

    return (
        rush.groupby("prev_group")
        .agg(
            n=("play_id", "size"),
            mean_epa=("epa", "mean"),
            mean_yards=("yards_gained", "mean"),
            success=("epa", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )


def task_sack():
    cols = ["game_id", "posteam", "play_id", "pass", "sack"]
    plays = pd.read_parquet(BASE, columns=cols)
    dropbacks = plays[plays["pass"].fillna(0) == 1].copy()
    dropbacks = dropbacks.sort_values(["game_id", "posteam", "play_id"])
    dropbacks["prev_sack"] = dropbacks.groupby(["game_id", "posteam"])["sack"].shift(1)
    dropbacks = dropbacks[dropbacks["prev_sack"].notna()]

    return (
        dropbacks.groupby("prev_sack")
        .agg(n=("play_id", "size"), sack_rate=("sack", "mean"))
        .reset_index()
    )


def main():
    jobs = {
        "next_pass": task_next_pass,
        "lead_pass": task_lead_pass,
        "rb_share": task_rb_share,
        "long_run": task_long_run,
        "sack": task_sack,
    }
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(func) for name, func in jobs.items()}
        results = {name: future.result() for name, future in futures.items()}

    for name, table in results.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(name)
        print(table.to_string(index=False))

    summary = {name: table.to_dict(orient="records") for name, table in results.items()}
    with open(OUT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("saved")


if __name__ == "__main__":
    main()
