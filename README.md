# NFL Momentum

A data analysis project that tests the claims about "momentum" in American football made in the video [You will never look at momentum the same (we were all wrong)](https://www.youtube.com/watch?v=2uUsTUJSE9g). The video argues that momentum cannot be detected in player or team performance once game situation and randomness are taken into account, and that the one measurable effect is a change in decisions after a notable play. This repository reproduces those claims on NFL play-by-play data.

## Data

Play-by-play data from [nflverse](https://github.com/nflverse/nflverse-data), covering the 1999 to 2026 seasons. After filtering to scrimmage plays and removing kneels, spikes, aborted plays, two-point attempts and records without EPA, the analysis set contains 922,081 plays and 165,356 drives. Raw files live in `data/raw/` and the cleaned dataset is `data/processed/plays.parquet`.

## Analyses

### Task 01: Do good plays lead to good plays?

A good play is defined as `EPA > 0`. Consecutive good plays within a drive are counted as a streak. A 5-fold cross-fitted `HistGradientBoostingClassifier` predicts the probability of a good play from the game situation (down, distance, field position, score, time, teams, play type and other context). The observed good-play rate is then compared with the predicted rate at each streak length, with confidence intervals from game-clustered bootstraps.

After two good plays, the observed rate is 45.22% against an expected 45.49%, a difference of -0.27 percentage points with a 95% confidence interval of [-0.56, +0.07]. Every streak group has a confidence interval containing zero, and adding streak as a feature changes AUC from 0.61989 to 0.61968. Streaks carry no predictive information beyond the situation.

See `task01_good_play_streak/README.md` for the full results.

### Task 02: Randomness and the coin model

This task reproduces the video's coin example and tests whether apparent streaks can arise by chance. A dynamic-programming calculation gives a 24.99% probability of at least one run of five heads in 20 fair flips, against 3.125% for a fixed position, an eightfold gap. A million simulations confirm the exact value.

Simulating 500 seasons from the play-level model produces fewer long streaks than the real data, while a model using one fixed success probability per drive produces more. Drive-level heterogeneity accounts for the difference, matching the video's discussion of the 2022 study. Consecutive scoring rates move by about two percentage points and do not keep rising with longer streaks.

See `task02_coin_model/README.md` for the full results.

### Task 03: Changes in decisions

Five independent analyses measure behavior after a notable play. A leading team's pass rate rises by about 3.2 percentage points on the drive after the opponent scores. After a bad play, the offense's pass rate above expectation (`pass_oe`) falls by about 1.6 percentage points relative to a good play. A running back who leads his team in first-half rushing yards takes 57.7% to 60.6% of second-half carries across the tested filters. A running back's next carry after a gain of at least 10 yards succeeds 1.9 percentage points less often, and the sack rate on the next dropback after a sack is 0.8 percentage points higher. These effects are visible in the data, even though the underlying performance shows no momentum.

See `task03_behavior_change/README.md` for the full results.

## Overall result

Across plays, drives, individual players and both sides of the ball, performance after a streak matches what the situation and random variation predict. The measurable pattern is in behavior: teams and play callers react to what happened previously. The project reproduces the video's conclusion that momentum is difficult to measure in performance, while the response to perceived momentum shows up in decisions.

## Repository layout

- `video.md`: Chinese summary of the source video.
- `data/`: raw seasonal play-by-play files and the processed dataset.
- `task01_good_play_streak/`: streak analysis scripts, README and outputs.
- `task02_coin_model/`: coin calculations, simulations, README and outputs.
- `task03_behavior_change/`: decision analysis scripts, README and outputs.

Each task directory contains its own README with methods, results and file descriptions. Those READMEs are written in Chinese.

## Running the analysis

The scripts require Python with `numpy`, `pandas`, `scikit-learn`, `numba` and `pyarrow`. Run them from the repository root in this order, since later tasks use files produced by earlier ones:

```bash
python task01_good_play_streak/scripts/download_pbp.py
python task01_good_play_streak/scripts/preprocess.py
python task01_good_play_streak/scripts/analyze_streak.py
python task02_coin_model/scripts/analyze_coin.py
python task03_behavior_change/scripts/analyze_behavior.py
```

All results are written to the `outputs/` directory of the corresponding task.
