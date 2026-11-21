# Task 02：硬币模型与随机性

## 目标

复现视频里的第二个结论：人低估了随机序列中连续成功出现的频率，因此会把随机产生的 streak 误认为 momentum。

## 数据

使用 Task 01 的 `../task01_good_play_streak/outputs/plays_with_pred.parquet`，以及 `../data/processed/plays.parquet` 中的 drive 结果字段。共 922,081 个 play，165,356 个 drive。

## 方法

用动态规划精确计算长度为 n、单次成功概率为 p 的序列中至少出现一次 k 连成功的概率。用 100 万次模拟验证解析值。

计算不同 drive 长度下至少出现一次 k 连 good 的概率，与 `p^k` 对比。`p` 取全数据集的 good 率 0.4261。

在 drive 层面计算连续得分率和连续 EPA good drive 率。同一支球队的连续 drive 按 `fixed_drive` 排序。

用 Task 01 的 `pred_base` 作为每个 play 的硬币概率，独立抽样 500 次，比较模拟与真实的 streak 分布。500 次模拟按进程拆分，用 12 个进程并行，每次模拟处理全部 922,081 个 play。

再用每个 drive 的实际 good 率作为该 drive 的硬币概率，做同样的模拟，检验 drive 级异质性的影响。

## 结果

硬币：

- 20 次公平投掷中至少出现一次 5 连正面的概率是 24.99%，视频给出的数字是 25%。
- 指定位置连续 5 次正面的概率是 3.125%，两者相差 8.0 倍。
- 100 万次模拟得到 24.96%，与解析值一致。

drive 长度：

- 10 个 play 的 drive 中，至少出现一次 3 连 good 的概率是 36.26%，指定位置连续 3 个 good 的概率是 7.74%，相差 4.69 倍。
- 16 个 play 的 drive 中，至少出现一次 3 连 good 的概率是 53.77%。
- drive 越长，至少出现一次 streak 的概率与 `p^k` 的比值越大。

drive 层面：

- drive 得分率 36.35%，上一波得分后再次得分 38.47%，上一波未得分后得分 35.04%。
- EPA good drive 率 35.66%，上一波 good 后再次 good 38.21%，上一波非 good 后 34.30%。

play 层面模拟：

- 真实数据中 streak 为 2、3、4+ 的 play 分别是 86,408、34,589、21,204 个。
- play 级情境模型的均值是 75,694、25,659、13,840 个，z 分数分别是 51、67、48。
- drive 级固定概率模型的均值是 78,191、36,166、35,368 个。
- 真实数据的 streak 聚集程度高于 play 级独立模型，低于 drive 级固定概率模型。

## 结论

硬币部分复现成功。20 次投掷至少出现一次 5 连正面的概率是 25%，人对这个概率的直觉接近 3.125%，两者相差 8 倍。这个差距可以解释为什么人看到比赛里的 streak 会觉得它不可能是随机产生的。

NFL 数据中，只控制 play 级情境的独立模型低估了长 streak 的频率，差异来自 drive 级异质性。同一个 drive 内的 play 共享球队当天的状态，`pred_base` 只包含 play 级情境，缺少这部分信息。视频里 2022 年论文的结论是控制当天球队实力后 streak 与随机一致，这里的模拟显示了同一件事。

drive 层面的连续得分率与无条件得分率相差约 2 个百分点，没有随连续次数上升。

## 文件

- `scripts/analyze_coin.py`：硬币计算、drive 统计与并行模拟
- `outputs/coin_exact.csv`：n=20、k=1 到 10 的精确概率与比值
- `outputs/coin_simulation.csv`：100 万次模拟与解析值的对照
- `outputs/drive_length_streak.csv`：不同 drive 长度下至少出现一次 k 连 good 的概率
- `outputs/drive_streaks.csv`：drive 层面的得分率与 EPA good 率
- `outputs/streak_simulation.csv`：play 级情境模型与 drive 级模型的模拟结果
- `outputs/summary.json`：全部关键数字

## 运行

```bash
python scripts/analyze_coin.py
```
