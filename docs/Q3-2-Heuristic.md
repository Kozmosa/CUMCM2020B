# Q3.2 提交级经验均衡求解方案

> 本文说明 `python -m q3.solve_q3_2 --backend heuristic` 的最终质量口径。
>
> 完整数值模型、联合状态和严格最佳反应语义仍由 [`Q3-2-Solve.md`](./Q3-2-Solve.md) 定义。`heuristic` 不声称完整动作空间的马尔可夫完美均衡，而是求解一个覆盖主要路线、购买、补给和挖矿行为的宽参数化策略类，并用独立天气样本进行提交级审计。

---

## 1. 终点，而不是中间结果

旧实现只在固定 16 策略库中寻找经验均衡。该实现能够精确回放多人规则，但参考仓库中的分流路线作为库外偏离时，可为玩家带来数百至数千元收益。因此：

```text
HEURISTIC_PURE
```

不再视为正式终点。

新实现只有同时满足以下条件，才返回：

```text
SUBMISSION_READY_EMPIRICAL_EQ
```

| 指标 | 默认门槛 |
|---|---:|
| 每名玩家成功率的单侧 95% 下界 | `>= 0.9999` |
| 最大单边偏离收益均值 | `<= 100` 元 |
| 最大单边偏离收益单侧 95% 上界 | `<= 200` 元 |
| 策略响应搜索 | 连续 2 个筛选窗口无超过 50 元的响应 |
| 策略上限 | 未因 32 策略上限提前停止 |
| 均衡类型 | 经验纯均衡 |
| 随机种子稳定性 | 2 个额外训练种子的角色组合一致 |

任一条件未满足时，状态为：

```text
EMPIRICAL_EQ_NOT_READY
```

程序仍输出当前最佳策略、支付、成功率和未决 gap，但 CLI 返回非零退出码，避免把中间结果误当成竞赛结论。

---

## 2. 保留的精确语义

提交级后端仍复用 `q3.transition`，以下规则不近似：

- 三名玩家同步行动；
- 只有同一天走同一条有向边的玩家形成道路拥堵；
- 道路资源倍率为 `2k`；
- 同矿挖矿收益按人数平分，资源消耗为三倍；
- 村庄单人购买使用两倍价格，多人购买使用四倍价格；
- 沙暴禁止移动；
- 水、食物、现金和负重均按原始整数单位保存；
- 到达终点立即结算剩余资源退款；
- 失败支付为当前现金减 `M`；
- 默认 `M=10^6`，天气概率为 `0.5/0.4/0.1`。

近似只发生在策略空间覆盖和 Monte Carlo 期望估计，不发生在游戏规则。

---

## 3. 参数化完整策略

一个 `HeuristicPolicy` 是可直接执行的完整策略，包含：

```text
route
initial_water, initial_food
village_water_target, village_food_target
mine_days
safety_factor
yield_when_crowded
mine_only_alone
```

每日行动仍根据当前天气、公开联合状态、自己位置、库存和已挖矿天数生成。路线是主要移动模板；沙暴、资源不足、村庄补给和吸收状态会改变实际逐日行动。响应策略还可以在多人位于同一节点时主动等待以错开潜在同边拥堵，或只在矿山独处时挖矿，从而利用公开对手位置，而不是只执行固定开环路线。

### 3.1 路线全集

默认枚举起点到终点之间：

- 不重复节点；
- 总移动步数不超过 12；
- 满足地图邻接关系；

的全部简单路径。

第六关共有：

```text
804 条
```

该全集包含参考仓库的重要分流路线：

```text
1-6-11-16-17-18-23-24-25
```

也覆盖直达、经过村庄、经过矿山、村庄后矿山、矿山后村庄和其他绕行结构。

### 3.2 资源候选

对每条路线至少检查：

- 按路线平均消耗生成的稳健资源量；
- `200 水 / 200 食物` 经验锚点；
- 安全系数 `1.0, 1.5, 2.0, 2.5`；
- 超过现金或负重限制时的严格可行缩放。

经过村庄的策略同时生成剩余路线目标库存。村庄购买始终先按最不利的四倍价格检查可行性，实际支付仍由联合转移根据同时购买人数结算。

### 3.3 挖矿候选

经过矿山的路线检查：

```text
0, 1, 2, 3, 4 天
```

策略在到达矿山后完成目标挖矿天数；沙暴日允许挖矿。所有收益分成和资源消耗由精确联合转移处理。

---

## 4. 初始受限博弈

求解从 16 个可解释种子策略开始：

- 直达；
- 经村庄；
- 经矿山；
- 村庄后矿山；
- 不同第一跳；
- 低、高载荷；
- 2 天和 4 天挖矿。

训练默认使用：

```text
5000 条共同随机天气序列
```

同一轮所有 profile 使用相同天气样本，利用配对差降低策略比较方差。

玩家与规则对编号对称，因此只实际模拟：

```text
a <= b <= c
```

的 canonical profile，再通过玩家置换填充完整支付张量。

训练 profile 的逐天气支付保存在追加式缓存中。策略库扩张时只模拟包含新策略的 canonical profile，不重复计算旧 profile。

---

## 5. 策略响应搜索

每轮执行以下过程：

1. 在当前策略库上求全部经验纯 Nash profile；
2. 按总成功率、最低玩家支付、总支付和策略编码选择代表均衡；
3. 固定另外两名玩家策略，分别为每名玩家搜索库外响应；
4. 对 804 条路线先检查稳健库存和 `200/200` 锚点；
5. 选出前 24 条路线，扩展安全系数、村庄库存和 `0-4` 天挖矿候选；
6. 同时检查“同节点立即移动 / 主动等待”和“同矿继续挖矿 / 只在独处时挖矿”；
7. 每名玩家保留前 16 个响应用于后续审计；
8. 将超过 50 元的最强响应加入策略库；
9. 每轮最多加入 3 个策略，策略库最多 32 个；
10. 连续两个不同训练天气窗口均无超过 50 元的响应后停止。

若存在超过门槛的响应但策略库已达到 32，设置：

```text
policy_cap_reached=true
response_complete=false
```

此时无论有限库内部是否存在纯均衡，都不能通过提交门槛。

若最终没有经验纯均衡，程序可计算有限库混合 NashConv 解用于诊断，但当前提交质量门槛要求可解释、可独立审计的纯 profile，因此状态仍为 `EMPIRICAL_EQ_NOT_READY`。

---

## 6. 独立 holdout 审计

训练完成后不复用训练天气，而是使用：

```text
3 组独立随机种子
每组 100000 条天气序列
```

审计对象包括：

- 最终策略库中的全部单边偏离；
- 每轮路线响应搜索保留的最强库外候选；
- 最终均衡 profile 本身。

候选策略完全由训练样本选出，holdout 只用于评估，避免在验证集上继续调策略。

在正式 holdout 前，程序还固定最终策略库，使用两个额外随机种子、每个 500 条天气重新求有限博弈。玩家编号允许置换，但三种角色策略的组合必须与主训练一致；否则 `stability_complete=false`，不能通过提交门槛。

### 6.1 支付区间

`value_lower/value_upper` 是三组独立 holdout 中最保守的正态近似区间包络；`audit_value_mean` 为三组均值的平均。

### 6.2 成功率下界

每组独立审计分别计算单侧 Wilson 下界，最终取三组中的最小值。

当 100000 条样本中零失败时，失败概率的单侧 95% 上界约为：

$$
3\times10^{-5}.
$$

在 `M=10^6` 下，对应未观测失败造成的期望罚金不确定性约为 30 元。

### 6.3 偏离收益

对玩家 $i$ 和候选策略 $d$，逐天气计算配对差：

$$
D_{i,d,e}
=
G_i(d,\pi_{-i};\omega_e)
-
G_i(\pi;\omega_e).
$$

对所有“玩家 × 候选偏离”比较使用 Bonferroni 修正的单侧置信上界。三组审计分别计算后取最坏结果。

提交门槛为：

$$
\max_i \widehat R_i\le 100,
\qquad
\max_i R^{95\%,U}_i\le 200.
$$

---

## 7. 状态语义

### `SUBMISSION_READY_EMPIRICAL_EQ`

表示：

- 当前训练博弈存在经验纯均衡；
- 策略响应搜索完整停止；
- 没有因为策略数量上限停止；
- 三组独立审计的成功率下界均达标；
- 最大偏离收益均值和上界均达标。

这是可以直接写入竞赛论文的终点。

### `EMPIRICAL_EQ_NOT_READY`

表示已有合法策略和数值结果，但至少一项提交门槛失败。常见原因包括：

- 发现显著库外响应；
- 策略上限耗尽；
- 只得到混合诊断解；
- 成功率下界不足；
- regret 均值或上界过大。

### `SEARCH_STOPPED`

表示墙钟、RSS、profile 预算或取消信号中断。`heuristic` 不保存递归状态检查点，重新运行由固定随机种子复现。

所有 heuristic 结果仍保持：

```text
selection_complete=false
full_action_regret_certified=false
```

因为提交级经验均衡不等于完整动作空间的严格均衡。

---

## 8. 正式命令

```bash
.venv/bin/python -m q3.solve_q3_2 \
  --backend heuristic --mode level6-initial \
  --penalty 1000000 --p-storm 0.10 \
  --heuristic-episodes 5000 \
  --heuristic-audit-episodes 100000 \
  --heuristic-audit-replicates 3 \
  --heuristic-stability-episodes 500 \
  --heuristic-stability-replicates 2 \
  --heuristic-initial-policies 16 \
  --heuristic-max-policies 32 \
  --heuristic-route-max-moves 12 \
  --heuristic-response-screening-episodes 128 \
  --heuristic-response-route-candidates 24 \
  --heuristic-response-audit-candidates 16 \
  --heuristic-response-additions 3 \
  --heuristic-response-rounds 8 \
  --heuristic-response-stable-rounds 2 \
  --heuristic-response-training-regret 50 \
  --heuristic-submission-mean-regret 100 \
  --heuristic-submission-upper-regret 200 \
  --heuristic-submission-success-lower 0.9999 \
  --equilibrium pure-mixed \
  --seed 20260728 \
  --wall-hours 8 --memory-gib 16 \
  --output q3/output/q3_2_submission
```

这些参数也是 CLI 默认值。`--quality-regret 10` 属于完整 `adaptive` 后端，不作为 `heuristic` 的提交门槛。

---

## 9. 结果字段

正式 JSON 至少包含：

```text
status
value_lower / value_upper
success
player_regret_lower / player_regret_upper
policy.players
policy.library
policy.audit_response_candidates
stats.route_universe
stats.response_history
stats.response_complete
stats.policy_cap_reached
stats.audit_success_lower
stats.audit_policy_regret_mean
stats.audit_best_deviations
stats.audit_replicate_results
stats.stability_results
stats.submission_quality_met
```

论文应直接引用三组独立审计结果，而不是训练博弈中的支付。

---

## 10. 计算规模

第六关 12 步内路线数为 804。最终库上限 `K=32` 时：

```text
有序 profile      32^3 = 32768
canonical profile C(34,3) = 5984
```

追加式缓存只保存 5984 个 canonical profile 的训练样本。按 5000 天气样本估算，训练样本数组约 1.4 GiB，加上响应筛选、Python 对象和审计缓冲，16 GiB 内存预算充足。

相对旧 `K=16, N=2000` 的约 8 分钟运行，正式训练预计为数小时；三组独立审计只计算最终 profile 和有限偏离，不构造完整 `32^3 × 100000` 张量。

---

## 11. 可以和不可以声称的结论

通过门槛后可以写：

> 在包含全部 12 步以内简单路径、资源配置、村庄补给和矿山策略的参数化反馈策略类中，采用策略空间响应搜索得到经验近似 Nash 均衡。三组各 100000 条独立天气序列显示，三名玩家成功率的单侧 95% 下界均不低于 99.99%，最大单边经验偏离收益均值不超过 100 元，单侧 95% 上界不超过 200 元。

仍不可以写：

1. 已证明完整动作空间中的严格 Nash 均衡；
2. 已证明所有离轨公开状态都满足子博弈精炼；
3. regret 上界是 Bellman 意义下的确定性证明界；
4. 任意未参数化的复杂反馈策略都不可能改善。

该口径比参考仓库的固定对手轨迹和局部 Monte Carlo 评分严格得多，也比完整状态空间认证更符合竞赛计算预算。

---

## 12. 自动验证

测试覆盖：

- 第六关 12 步内简单路线恰为 804 条；
- 参考仓库的重要分流路线属于路线全集；
- 初始策略库路线、负重和现金合法；
- 精确联合回放与终点退款一致；
- 固定策略库基准继续确定性复现；
- 缩小实例能够通过提交级成功率和 regret 门槛；
- 预算取消返回 `SEARCH_STOPPED`；
- 缩小版第六关在发现强响应或达到策略上限时返回 `EMPIRICAL_EQ_NOT_READY`，不会误标为可提交。
