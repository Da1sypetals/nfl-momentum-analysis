import json
import multiprocessing as mp
import pathlib

import numpy as np
import pandas as pd
from numba import njit

ROOT = pathlib.Path(__file__).resolve().parents[2]
PRED = ROOT / "task01_good_play_streak" / "outputs" / "plays_with_pred.parquet"
BASE = ROOT / "data" / "processed" / "plays.parquet"
OUT = ROOT / "task02_coin_model" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def prob_at_least_one_run(n, p, k):
    dp = np.zeros(k, dtype=np.float64)
    dp[0] = 1.0
    hit = 0.0
    for _ in range(n):
        new = np.zeros(k, dtype=np.float64)
        new[0] = dp.sum() * (1.0 - p)
        if k > 1:
            new[1:] = dp[:-1] * p
        hit += dp[-1] * p
        dp = new
    return hit


def coin_simulation(n_sim, n, p, ks, seed):
    rng = np.random.default_rng(seed)
    flips = rng.random((n_sim, n)) < p
    run = np.zeros(n_sim, dtype=np.int32)
    best = np.zeros(n_sim, dtype=np.int32)
    for column in range(n):
        run = (run + 1) * flips[:, column]
        best = np.maximum(best, run)
    return {k: float((best >= k).mean()) for k in ks}


@njit(cache=True)
def streak_hist(good, new_drive):
    n = good.shape[0]
    hist = np.zeros(5, dtype=np.int64)
    run = 0
    for i in range(n):
        if new_drive[i]:
            run = 0
        bucket = run if run < 4 else 4
        hist[bucket] += 1
        if good[i] == 1:
            run += 1
        else:
            run = 0
    return hist


def simulate_chunk(job):
    seed, n_sim, pred, new_drive = job
    rng = np.random.default_rng(seed)
    hists = np.zeros((n_sim, 5), dtype=np.int64)
    for draw in range(n_sim):
        good_sim = (rng.random(pred.shape[0]) < pred).astype(np.int8)
        hists[draw] = streak_hist(good_sim, new_drive)
    return hists


def run_simulation(pred, new_drive, n_sim, workers, seed_base):
    base = n_sim // workers
    extra = n_sim % workers
    jobs = []
    for index in range(workers):
        count = base + (1 if index < extra else 0)
        if count > 0:
            jobs.append((seed_base + index, count, pred, new_drive))
    context = mp.get_context("spawn")
    with context.Pool(len(jobs)) as pool:
        chunks = pool.map(simulate_chunk, jobs)
    return np.vstack(chunks)


def summarize_simulation(hists, real_hist):
    mean = hists.mean(axis=0)
    std = hists.std(axis=0)
    low = np.percentile(hists, 2.5, axis=0)
    high = np.percentile(hists, 97.5, axis=0)
    rows = []
    for bucket in range(5):
        rows.append({
            "streak_bucket": bucket,
            "real": int(real_hist[bucket]),
            "real_share": float(real_hist[bucket] / real_hist.sum()),
            "sim_mean": float(mean[bucket]),
            "sim_share": float(mean[bucket] / mean.sum()),
            "sim_low": float(low[bucket]),
            "sim_high": float(high[bucket]),
            "z": float((real_hist[bucket] - mean[bucket]) / std[bucket]),
        })
    return pd.DataFrame(rows), mean, std, low, high


def main():
    plays = pd.read_parquet(PRED)
    drive_result = (
        pd.read_parquet(BASE, columns=["game_id", "fixed_drive", "drive_ended_with_score"])
        .groupby(["game_id", "fixed_drive"], as_index=False)["drive_ended_with_score"]
        .max()
    )
    plays = plays.merge(drive_result, on=["game_id", "fixed_drive"], how="left")
    plays = plays.sort_values(["game_id", "fixed_drive", "play_id"]).reset_index(drop=True)

    pred = plays["pred_base"].to_numpy(dtype=np.float64)
    good = plays["good"].to_numpy(dtype=np.int8)
    game = plays["game_id"].to_numpy()
    drive = plays["fixed_drive"].to_numpy()
    new_drive = np.ones(len(plays), dtype=np.bool_)
    new_drive[1:] = (game[1:] != game[:-1]) | (drive[1:] != drive[:-1])

    real_hist = streak_hist(good, new_drive)
    check = plays["streak"].clip(upper=4).value_counts().reindex(range(5), fill_value=0).to_numpy()
    print("real", real_hist)
    print("check", check)

    coin_rows = []
    for k in range(1, 11):
        exact = prob_at_least_one_run(20, 0.5, k)
        coin_rows.append({
            "n": 20,
            "p": 0.5,
            "k": k,
            "at_least_one": exact,
            "p_power_k": 0.5 ** k,
            "ratio": exact / (0.5 ** k),
        })
    coin_table = pd.DataFrame(coin_rows)
    print(coin_table.to_string(index=False))
    coin_table.to_csv(OUT / "coin_exact.csv", index=False)

    ks = list(range(2, 9))
    simulated = coin_simulation(1_000_000, 20, 0.5, ks, seed=7)
    coin_sim_table = pd.DataFrame([
        {
            "k": k,
            "exact": prob_at_least_one_run(20, 0.5, k),
            "simulated": simulated[k],
        }
        for k in ks
    ])
    print(coin_sim_table.to_string(index=False))
    coin_sim_table.to_csv(OUT / "coin_simulation.csv", index=False)

    p_mean = float(plays["good"].mean())
    drive_rows = []
    for length in [4, 6, 8, 10, 12, 16]:
        for k in [2, 3, 4, 5]:
            exact = prob_at_least_one_run(length, p_mean, k)
            drive_rows.append({
                "drive_length": length,
                "p": p_mean,
                "k": k,
                "at_least_one": exact,
                "p_power_k": p_mean ** k,
                "ratio": exact / (p_mean ** k),
            })
    drive_table = pd.DataFrame(drive_rows)
    print(drive_table.to_string(index=False))
    drive_table.to_csv(OUT / "drive_length_streak.csv", index=False)

    drives = (
        plays.groupby(["game_id", "fixed_drive"])
        .agg(
            posteam=("posteam", "first"),
            score=("drive_ended_with_score", "max"),
            epa_sum=("epa", "sum"),
            n_plays=("play_id", "size"),
        )
        .reset_index()
    )
    drives["good_epa"] = (drives["epa_sum"] > 0).astype(int)
    drives = drives.sort_values(["game_id", "posteam", "fixed_drive"])
    drives["prev_score"] = drives.groupby(["game_id", "posteam"])["score"].shift(1)
    drives["prev_good_epa"] = drives.groupby(["game_id", "posteam"])["good_epa"].shift(1)

    drive_summary = {
        "n_drives": int(len(drives)),
        "score_rate": float(drives["score"].mean()),
        "score_rate_after_score": float(drives.loc[drives["prev_score"] == 1, "score"].mean()),
        "score_rate_after_no_score": float(drives.loc[drives["prev_score"] == 0, "score"].mean()),
        "good_epa_rate": float(drives["good_epa"].mean()),
        "good_epa_rate_after_good": float(drives.loc[drives["prev_good_epa"] == 1, "good_epa"].mean()),
        "good_epa_rate_after_bad": float(drives.loc[drives["prev_good_epa"] == 0, "good_epa"].mean()),
    }
    print(json.dumps(drive_summary, indent=2))
    pd.DataFrame([drive_summary]).to_csv(OUT / "drive_streaks.csv", index=False)

    workers = min(12, mp.cpu_count())
    n_sim = 500
    hists_play = run_simulation(pred, new_drive, n_sim, workers, 1000)
    drive_rate = plays.groupby(["game_id", "fixed_drive"])["good"].transform("mean").to_numpy(dtype=np.float64)
    hists_drive = run_simulation(drive_rate, new_drive, n_sim, workers, 2000)

    play_table, play_mean, play_std, play_low, play_high = summarize_simulation(hists_play, real_hist)
    drive_table_sim, drive_mean, drive_std, drive_low, drive_high = summarize_simulation(hists_drive, real_hist)
    play_table["model"] = "play_context"
    drive_table_sim["model"] = "drive_rate"
    combined = pd.concat([play_table, drive_table_sim], ignore_index=True)
    print(combined.to_string(index=False))
    combined.to_csv(OUT / "streak_simulation.csv", index=False)

    summary = {
        "n_plays": int(len(plays)),
        "n_drives": int(new_drive.sum()),
        "coin_n": 20,
        "coin_k": 5,
        "coin_exact": float(prob_at_least_one_run(20, 0.5, 5)),
        "coin_p_power_k": float(0.5 ** 5),
        "coin_ratio": float(prob_at_least_one_run(20, 0.5, 5) / (0.5 ** 5)),
        "coin_simulated": simulated[5],
        "p_mean": p_mean,
        "drive_summary": drive_summary,
        "streak_real": real_hist.tolist(),
        "streak_play_model_mean": play_mean.tolist(),
        "streak_play_model_low": play_low.tolist(),
        "streak_play_model_high": play_high.tolist(),
        "streak_drive_model_mean": drive_mean.tolist(),
        "streak_drive_model_low": drive_low.tolist(),
        "streak_drive_model_high": drive_high.tolist(),
        "n_simulations": n_sim,
    }
    with open(OUT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
