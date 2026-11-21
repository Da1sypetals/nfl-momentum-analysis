import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "plays.parquet"
RESULTS = ROOT / "task01_good_play_streak" / "outputs"
RESULTS.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "play_type", "down", "ydstogo",
    "yardline_100", "goal_to_go", "score_differential", "half_seconds_remaining",
    "qtr", "shotgun", "no_huddle", "posteam", "defteam", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "spread_line", "total_line", "epa", "good",
    "streak", "fixed_drive",
]
plays = pd.read_parquet(PROCESSED, columns=COLUMNS)
plays["streak_bucket"] = plays["streak"].clip(upper=4)
TEAM_CODE = {team: index + 1 for index, team in enumerate(sorted(plays["posteam"].dropna().unique()))}


def build_features(frame, with_streak):
    features = pd.DataFrame(index=frame.index)
    features["down"] = frame["down"].fillna(0).astype("int16")
    features["qtr"] = frame["qtr"].fillna(0).astype("int16")
    features["play_type"] = (frame["play_type"] == "pass").astype("int16")
    features["posteam"] = frame["posteam"].map(TEAM_CODE).fillna(0).astype("int16")
    features["defteam"] = frame["defteam"].map(TEAM_CODE).fillna(0).astype("int16")
    features["ydstogo"] = frame["ydstogo"].astype("float32")
    features["yardline_100"] = frame["yardline_100"].astype("float32")
    features["goal_to_go"] = frame["goal_to_go"].fillna(0).astype("int16")
    features["score_differential"] = frame["score_differential"].astype("float32")
    features["half_seconds_remaining"] = frame["half_seconds_remaining"].astype("float32")
    features["shotgun"] = frame["shotgun"].fillna(0).astype("int16")
    features["no_huddle"] = frame["no_huddle"].fillna(0).astype("int16")
    features["posteam_timeouts_remaining"] = frame["posteam_timeouts_remaining"].astype("float32")
    features["defteam_timeouts_remaining"] = frame["defteam_timeouts_remaining"].astype("float32")
    features["spread_line"] = frame["spread_line"].astype("float32")
    features["total_line"] = frame["total_line"].astype("float32")
    features["season"] = frame["season"].astype("int16")
    if with_streak:
        features["streak"] = frame["streak"].astype("int16")
    return features


CATEGORICAL = [0, 1, 2, 3, 4]


def fit_oof(features, target, seed=0):
    prediction = np.zeros(len(features))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(folds.split(features)):
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
            categorical_features=CATEGORICAL,
        )
        model.fit(features.iloc[train_index], target.iloc[train_index])
        prediction[test_index] = model.predict_proba(features.iloc[test_index])[:, 1]
        print("fold", fold, "done", flush=True)
    return prediction


target = plays["good"].astype("int8")
features_base = build_features(plays, with_streak=False)
features_streak = build_features(plays, with_streak=True)

prediction_base = fit_oof(features_base, target)
prediction_streak = fit_oof(features_streak, target)
plays["pred_base"] = prediction_base
plays["pred_streak"] = prediction_streak

plays = plays.sort_values(["game_id", "posteam", "play_id"]).reset_index(drop=True)
prev_game = plays.groupby(["game_id", "posteam"], sort=False)["good"].shift(1)
prev_game = prev_game.fillna(0).astype("int8")
block_game = (prev_game != 1).cumsum()
plays["streak_game"] = prev_game.groupby(block_game).cumsum().astype("int8")
plays["streak_game_bucket"] = plays["streak_game"].clip(upper=4)


def summarize(frame, streak_column):
    rows = []
    for bucket, group in frame.groupby(streak_column):
        actual = group["good"].mean()
        expected = group["pred_base"].mean()
        rows.append({
            "streak": int(bucket),
            "n": int(len(group)),
            "actual": actual,
            "expected": expected,
            "diff": actual - expected,
        })
    return pd.DataFrame(rows)


def cluster_ci(frame, bucket, streak_column, n_boot=300, seed=1):
    group = frame[frame[streak_column] == bucket]
    per_game = group.groupby("game_id").agg(
        n=("good", "size"),
        good_sum=("good", "sum"),
        pred_sum=("pred_base", "sum"),
    )
    counts = per_game["n"].to_numpy(dtype=float)
    good_sums = per_game["good_sum"].to_numpy(dtype=float)
    pred_sums = per_game["pred_sum"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for draw in range(n_boot):
        index = rng.integers(0, len(counts), len(counts))
        draws[draw] = good_sums[index].sum() / counts[index].sum() - pred_sums[index].sum() / counts[index].sum()
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def add_cluster_ci(table, frame, streak_column):
    low = []
    high = []
    for bucket in table["streak"]:
        lower, upper = cluster_ci(frame, bucket, streak_column)
        low.append(lower)
        high.append(upper)
    table["ci_low"] = low
    table["ci_high"] = high
    return table


overall = summarize(plays, "streak_bucket")
overall = add_cluster_ci(overall, plays, "streak_bucket")
print(overall.to_string(index=False))
overall.to_csv(RESULTS / "streak_overall.csv", index=False)

overall_game = summarize(plays, "streak_game_bucket")
overall_game = add_cluster_ci(overall_game, plays, "streak_game_bucket")
print(overall_game.to_string(index=False))
overall_game.to_csv(RESULTS / "streak_game_overall.csv", index=False)

metrics = {
    "n_plays": int(len(plays)),
    "good_rate": float(plays["good"].mean()),
    "auc_base": float(roc_auc_score(target, prediction_base)),
    "auc_streak": float(roc_auc_score(target, prediction_streak)),
    "logloss_base": float(log_loss(target, prediction_base)),
    "logloss_streak": float(log_loss(target, prediction_streak)),
}
print(json.dumps(metrics, indent=2))


def subset_rows(label, frame):
    rows = []
    for bucket in [0, 1, 2, 3, 4]:
        group = frame[frame["streak_bucket"] == bucket]
        rows.append({
            "subset": label,
            "streak": bucket,
            "n": int(len(group)),
            "actual": group["good"].mean(),
            "expected": group["pred_base"].mean(),
            "diff": group["good"].mean() - group["pred_base"].mean(),
        })
    return rows


subset_table = pd.DataFrame(
    subset_rows("exclude_2026", plays[plays["season"] < 2026])
    + subset_rows("REG_only", plays[plays["season_type"] == "REG"])
)
print(subset_table.to_string(index=False))
subset_table.to_csv(RESULTS / "streak_subsets.csv", index=False)

season_rows = []
for (season, bucket), group in plays.groupby(["season", "streak_bucket"]):
    if bucket != 2:
        continue
    season_rows.append({
        "season": int(season),
        "n": int(len(group)),
        "actual": group["good"].mean(),
        "expected": group["pred_base"].mean(),
        "diff": group["good"].mean() - group["pred_base"].mean(),
    })
season_table = pd.DataFrame(season_rows)
print(season_table.to_string(index=False))
season_table.to_csv(RESULTS / "streak2_by_season.csv", index=False)

type_rows = []
for play_type, type_frame in plays.groupby("play_type"):
    for bucket, group in type_frame.groupby("streak_bucket"):
        type_rows.append({
            "play_type": play_type,
            "streak": int(bucket),
            "n": int(len(group)),
            "actual": group["good"].mean(),
            "expected": group["pred_base"].mean(),
            "diff": group["good"].mean() - group["pred_base"].mean(),
        })
type_table = pd.DataFrame(type_rows)
print(type_table.to_string(index=False))
type_table.to_csv(RESULTS / "streak_by_play_type.csv", index=False)

plays["score_bucket"] = pd.cut(
    plays["score_differential"], bins=[-100, -14, -7, -1, 0, 7, 14, 100],
    labels=["-14以下", "-14到-7", "-7到-1", "0", "1到7", "7到14", "14以上"],
)
score_rows = []
for (score_bucket, bucket), group in plays.groupby(["score_bucket", "streak_bucket"], observed=True):
    score_rows.append({
        "score_bucket": str(score_bucket),
        "streak": int(bucket),
        "n": int(len(group)),
        "actual": group["good"].mean(),
        "expected": group["pred_base"].mean(),
        "diff": group["good"].mean() - group["pred_base"].mean(),
    })
score_table = pd.DataFrame(score_rows)
print(score_table.to_string(index=False))
score_table.to_csv(RESULTS / "streak_by_score.csv", index=False)

with open(RESULTS / "model_metrics.json", "w") as handle:
    json.dump(metrics, handle, indent=2)

saved_columns = [
    "game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
    "fixed_drive", "play_type", "down", "ydstogo", "yardline_100",
    "score_differential", "half_seconds_remaining", "epa", "good", "streak",
    "streak_game", "pred_base", "pred_streak",
]
plays[saved_columns].to_parquet(RESULTS / "plays_with_pred.parquet", index=False)
print("saved predictions and results")
