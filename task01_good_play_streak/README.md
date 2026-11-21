# Task 01：好 play 是否带来下一个好 play

## 目标

复现视频里的第一个结论：控制橡皮筋效应和场上情境之后，一个好 play 不会提高下一个 play 的成功率。

## 数据

nflverse 的 play by play 数据，1999 到 2026 共 28 个赛季，来源是 `nflverse-data` 的 `pbp` release。过滤到 scrimmage plays，排除 kneel、spike、aborted、two-point 和没有 EPA 的记录，得到 922,081 个 play。2026 赛季只覆盖到当前周。

原始文件保存在 `../data/raw`，清洗后的数据保存在 `../data/processed/plays.parquet`。

## 方法

good play 定义为 `EPA > 0`，与 nflfastR 的 `success` 字段定义一致。streak 在同一 drive 内按 `play_id` 顺序计算，`streak=2` 表示当前 play 之前连续两个 play 都是 good。

控制变量包括 down、ydstogo、yardline_100、goal_to_go、score_differential、half_seconds_remaining、qtr、play_type、shotgun、no_huddle、posteam、defteam、双方剩余暂停、spread_line、total_line 和 season。用 5 折交叉拟合的 `HistGradientBoostingClassifier` 得到每个 play 的 `P(good | 情境)`，再按 streak 分组比较实际 good 率与平均预测概率。置信区间用按比赛聚类的 bootstrap 计算，重复 300 次。

橡皮筋效应通过 `score_differential` 和 `play_type` 进入模型。比分较低的球队传球更多，控制这两个变量之后，传球选择带来的差异会进入预期值。

## 结果

drive 内 streak 的结果：

- `streak=0`：570,300 个 play，实际 40.87%，预期 40.77%，差 +0.10 个百分点，95% 置信区间 [-0.02, +0.22]
- `streak=1`：209,580 个 play，实际 45.30%，预期 45.46%，差 -0.16 个百分点，95% 置信区间 [-0.36, +0.03]
- `streak=2`：86,408 个 play，实际 45.22%，预期 45.49%，差 -0.27 个百分点，95% 置信区间 [-0.56, +0.07]
- `streak=3`：34,589 个 play，实际 46.13%，预期 46.05%，差 +0.08 个百分点，95% 置信区间 [-0.45, +0.63]
- `streak=4+`：21,204 个 play，实际 46.55%，预期 46.36%，差 +0.19 个百分点，95% 置信区间 [-0.50, +0.86]

所有置信区间都包含 0。`streak=2` 的实际 45.22% 对应视频里的 45%。

跨 drive 定义的 streak 结果一致。`streak=2` 有 95,406 个 play，实际 45.05%，预期 45.26%，差 -0.22 个百分点，95% 置信区间 [-0.51, +0.10]。

稳健性检查：

- 排除 2026 赛季，`streak=2` 的差是 -0.28 个百分点。
- 只算常规赛，`streak=2` 的差是 -0.24 个百分点。
- 传球 play，`streak=2` 的差是 -0.51 个百分点；跑球 play，差是 -0.05 个百分点。
- 按比分差分桶，绝大多数单元格的差在正负 1 个百分点以内。
- 按赛季，差值正负混合，2017 赛季是 -2.4 个百分点，2022 赛季是 +1.1 个百分点。

把 streak 作为额外特征加入模型后，交叉拟合的 AUC 从 0.61989 变为 0.61968，log loss 从 0.65888 变为 0.65896。

## 结论

控制情境变量后，连续好 play 之后的下一 play 成功率与模型预期一致，所有分组的置信区间都包含 0。原始成功率从 `streak=0` 的 40.87% 上升到 `streak=1` 的 45.30%，这一段上升完全由情境差异解释，模型的预期值同样从 40.77% 上升到 45.46%。

streak 没有提供增量预测能力。视频的第一个结论在 1999 到 2026 的数据上成立。

## 文件

- `scripts/download_pbp.py`：下载 28 个赛季的原始数据
- `scripts/preprocess.py`：清洗并生成 streak
- `scripts/analyze_streak.py`：交叉拟合建模与统计
- `outputs/plays_with_pred.parquet`：带 `pred_base` 和 `pred_streak` 的 play 数据
- `outputs/streak_overall.csv`：drive 内 streak 的主结果
- `outputs/streak_game_overall.csv`：跨 drive streak 的结果
- `outputs/streak_subsets.csv`：排除 2026 与常规赛的结果
- `outputs/streak2_by_season.csv`：按赛季的 `streak=2` 结果
- `outputs/streak_by_play_type.csv`：传球与跑球的对照
- `outputs/streak_by_score.csv`：按比分差分桶的对照
- `outputs/model_metrics.json`：AUC 与 log loss

## 运行

```bash
python scripts/download_pbp.py
python scripts/preprocess.py
python scripts/analyze_streak.py
```
