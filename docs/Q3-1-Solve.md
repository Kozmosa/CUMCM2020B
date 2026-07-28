# Q3.1 实现说明

> Q3 共用的状态规范化、无损剪枝、并行和检查点正确性统一见 [[Q3-Exactness]]。

> 对应题目：多名玩家、完整天气序列已知，所有玩家在第 0 天同时确定完整行动方案，此后不能修改；求解附件「第五关」。
>
> 数学模型见 [`Solution/Q3-1.md`](./Solution/Q3-1.md)。
>
> 本文主要说明数值算法、数据结构、计算流程和均衡验证方法。

---

## 1. 求解目标与实现口径

Q3.1 被写成一个完全信息静态博弈。玩家 $i$ 的纯策略

$$
\pi_i=(a_{i,0},a_{i,1},\ldots,a_{i,T})
$$

包含第 0 天初始购买和第 1 至 $T$ 天的全部行动。目标是求一个纯策略开环纳什均衡

$$
\Pi^*=(\pi_1^*,\ldots,\pi_n^*),
$$

使任意玩家都不能在其他玩家策略不变时，通过单独替换自己的完整策略提高支付。

实现采用以下统一口径：

1. 天气序列在求解前全部给定。
2. 玩家在第 0 天按基准价格购买，起点购买不受多人村庄价格影响。
3. 村庄购买默认采用 Q1/Q2 的 `start_of_day` 语义：当天开始时已在村庄，先购买，再执行当天主要行动。
4. 到达终点后该玩家的游戏结束，不再参与后续道路、矿山和村庄人数统计。
5. 成功支付为终点现金加剩余资源半价退款；失败支付与最新 Q2 保持一致，取失败时现金减去 $M$。
6. 外层使用迭代最佳反应；固定其他玩家策略时，玩家 $i$ 的最佳反应使用 Q1 型前向动态规划计算。
7. 最佳反应阶段把其他玩家已承诺的行动序列视为固定时间表；最终均衡候选必须通过联合回放，确认所有被承诺的行动在实际多人消耗下均可执行。

第五关有 $n=3$ 名玩家、$T=10$，地图与第三关相同，没有村庄。因此第五关的最佳反应内核只需处理：

- 同一有向边上的移动人数对资源消耗的影响；
- 同一矿山挖矿人数对挖矿收入的影响。

---

## 2. 关卡数据

第五关参数为：

| 参数 | 数值 |
|---|---:|
| 玩家数 | 3 |
| 节点数 | 13 |
| 起点 | 1 |
| 终点 | 13 |
| 矿山 | 9 |
| 村庄 | 无 |
| 截止日期 | 10 |
| 初始资金 | 10000 |
| 负重上限 | 1200 kg |
| 基础收益 | 200/天 |

地图邻接直接复用 `q2.data` 中第三关的 13 节点无向图。天气序列为

```text
晴朗, 高温, 晴朗, 晴朗, 晴朗,
晴朗, 高温, 高温, 高温, 高温
```

水和食物的质量、价格及基础消耗也与第三关相同。

实现时建议建立独立的 Q3 `LevelConfig`，由第三关配置复制地图和资源参数，再增加：

```text
n_players = 3
known_weather = True
```

避免直接修改 Q1/Q2 的配置类。

---

## 3. 策略与轨迹表示

每名玩家的策略保存为定长数组：

```text
Strategy
├── buy0_water
├── buy0_food
├── action[1..T]       # stay / mine / move:destination
├── buy_water[1..T]
└── buy_food[1..T]
```

策略经联合规则回放后，得到逐日轨迹：

```text
Trajectory
├── position[0..T]
├── water[0..T]
├── food[0..T]
├── cash_scaled[0..T]
├── finished_day
└── failed_day
```

行动编码应使用小整数而不是字符串：

```text
0                 stay
1                 mine
2 ... 2+degree-1  move 到对应邻接点
```

输出和调试时再转换为 `stay`、`mine`、`move:u`。

---

## 4. 资金的定点整数表示

多人同时挖矿时，每人的收入为 $R/k$。第五关中 $R=200$、$k\in\{1,2,3\}$，因此可能出现 $200/3$，不能继续直接用 Q1 的整数元资金。

为避免浮点误差影响最大值比较、策略判等和均衡验证，使用定点整数：

$$
L=\operatorname{lcm}(2,1,\ldots,n).
$$

其中因子 $2$ 用于精确表示终点半价退款。第五关 $n=3$，可取

$$
L=6.
$$

所有金额在程序内部乘以 $L$：

```text
initial_cash_scaled = L * initial_cash
price_scaled        = L * price
mine_income_scaled  = L * R / k
refund_scaled       = L * price * remaining / 2
```

这样所有状态值仍可使用 `int32` 或 `int64` 精确计算。最终输出时除以 $L$。

第五关金额较小，`int32` 足够；为便于未来扩展，最佳反应内核也可以统一使用 `int64`。

---

## 5. 联合策略回放

联合回放函数

```text
evaluate_profile(config, strategies) -> ProfileResult
```

用于计算给定策略组合的真实轨迹和每名玩家支付。每天必须先读取所有仍在游戏中的玩家行动，再统一计算交互人数，不能逐玩家立即更新。

### 5.1 每日计算顺序

第 $t$ 天按以下顺序处理：

1. 收集所有未结束玩家当天的购买和主要行动。
2. 若存在村庄，按购买地点统计实际购买人数并确定单价。
3. 对每条有向边 $(u,v)$ 统计移动人数 $k_t^E(u,v)$。
4. 对每个矿山 $m$ 统计挖矿人数 $k_t^M(m)$。
5. 根据统一统计结果，分别计算每名玩家的资源消耗、购买成本和挖矿收入。
6. 同时更新所有玩家的位置、库存和现金。
7. 标记当天到达终点或失败的玩家。

统计移动人数时，$(u,v)$ 与 $(v,u)$ 必须分开；相向移动不属于“从同一区域 A 到同一区域 B”。

### 5.2 玩家支付

若玩家 $i$ 在第 $\tau_i\leq T$ 天到达终点：

$$
U_i=C_{i,\tau_i}
+\frac12p_W W_{i,\tau_i}
+\frac12p_F F_{i,\tau_i}.
$$

若玩家在时刻 $\kappa_i$ 失败：

$$
U_i=C_{i,\kappa_i}-M.
$$

联合回放既是策略组合的支付计算器，也是独立于动态规划内部表的规则验证器。

---

## 6. 固定其他玩家后的交互参数

计算玩家 $i$ 的最佳反应时，固定其他玩家策略 $\Pi_{-i}$，预先汇总其他玩家的行动事件。

### 6.1 同路人数表

定义其他玩家在第 $t$ 天从 $u$ 移动到 $v$ 的人数：

$$
m_{-i,t}^{E}(u,v)
=\sum_{j\ne i}
\mathbf 1\{v_{j,t-1}=u,
z_{j,t}=\operatorname{move}(v)\}.
$$

若玩家 $i$ 也从 $u$ 移动到 $v$，总人数为

$$
k_t^E(u,v)=1+m_{-i,t}^{E}(u,v),
$$

玩家 $i$ 的移动消耗倍率为

$$
\lambda_{i,t}^{\mathrm{move}}(u,v)
=2\left(1+m_{-i,t}^{E}(u,v)\right).
$$

### 6.2 同矿人数表

定义其他玩家在矿山 $m$ 挖矿的人数：

$$
m_{-i,t}^{M}(m)
=\sum_{j\ne i}
\mathbf 1\{v_{j,t-1}=m,
z_{j,t}=\operatorname{mine}\}.
$$

玩家 $i$ 在该矿山挖矿时：

$$
k_t^M(m)=1+m_{-i,t}^{M}(m),
\qquad
R_{i,t}=\frac{R}{k_t^M(m)}.
$$

挖矿资源消耗倍率始终为 $3$，不随人数改变。

### 6.3 村庄购买价格表

一般关卡中，定义

$$
b_{-i,t}^{V}(v)
=\mathbf 1\{\text{第 }t\text{ 天有其他玩家在村庄 }v\text{ 购买}\}.
$$

玩家 $i$ 在该村庄购买时的价格为

$$
(p_{W,i,t},p_{F,i,t})
=
\begin{cases}
(2p_W,2p_F),&b_{-i,t}^{V}(v)=0,\\
(4p_W,4p_F),&b_{-i,t}^{V}(v)=1.
\end{cases}
$$

第五关没有村庄，因此该表为空。

这些数组都只依赖时间、地点和固定的其他玩家策略，可在每次最佳反应 DP 前一次性生成。

---

## 7. 最佳反应：前向动态规划

固定 $\Pi_{-i}$ 后，玩家 $i$ 的问题仍使用 Q1 的现金压缩：

$$
\operatorname{cash}_t(v,W,F)
=\max\{C:\text{玩家 }i\text{ 在第 }t\text{ 天结束时处于 }(v,W,F,C)\}.
$$

在固定库存和位置下，更高现金不会降低可行性，终点支付也随现金单调增加，因此只保留最大现金仍然是精确的。

### 7.1 第 0 天购买

初始化

```text
seed[0, 0] = initial_cash_scaled
cash[start] = BuyKnapsack(seed, base_prices)
```

其中 `BuyKnapsack` 与 Q1 相同，对水、食物两个方向分别做一次一维扫描。

第五关也可以直接遍历所有满足

$$
3W+2F\leq1200
$$

的初始购买组合。共有 120601 个可行库存组合。

### 7.2 第 $t$ 天转移

设当天天气基础消耗为

$$
(c_W(\theta_t),c_F(\theta_t)).
$$

对每个节点 $v$ 的整张库存平面执行以下转移。

#### 停留

$$
(c_W,c_F)
=\left(c_W(\theta_t),c_F(\theta_t)\right),
$$

```text
next[v, :-cW, :-cF]
    = maximum(next[v, :-cW, :-cF], cash[v, cW:, cF:])
```

#### 移动

对每个邻接点 $u\in N(v)$，先读取预计算同路人数：

$$
\lambda=2\left(1+m_{-i,t}^E(v,u)\right),
$$

再令

$$
c_W=\lambda c_W(\theta_t),
\qquad
c_F=\lambda c_F(\theta_t),
$$

并把切片写入 `next[u]`。沙暴日不生成移动转移。

#### 挖矿

仅当玩家在当天开始时已经位于矿山，才允许挖矿。消耗为基础消耗的 3 倍，收入为

$$
R_{i,t}=\frac{R}{1+m_{-i,t}^{M}(v)}.
$$

定点化后转移为

```text
candidate = cash[v, 3*cW:, 3*cF:] + income_scaled[t, v]
```

### 7.3 村庄购买

一般关卡在 `start_of_day` 模式下，先根据其他玩家是否同时购买确定单价，再调用 Q1 的 `BuyKnapsack`：

```text
work[v] = BuyKnapsack(cash[v], village_price[t, v])
```

第五关没有村庄，不执行此步骤。

### 7.4 终点与剪枝

继续沿用 Q1 的两项剪枝：

1. 已到终点的状态吸收，不再生成后续行动。
2. 使用忽略沙暴的 BFS 最短距离。如果到终点的最少步数大于剩余天数，则丢弃该状态或移动目的地。

每一天结束后，对终点层计算

$$
J_i=C_i+\frac12p_W W_i+\frac12p_F F_i,
$$

并保存当前最优终点状态。

### 7.5 回溯

每个被改善的状态保存：

```text
previous_node
previous_water
previous_food
action_code
bought_water
bought_food
cash_after
```

从最佳终点状态反向回溯到第 0 天，得到玩家 $i$ 的完整最佳反应策略。

---

## 8. 最佳反应伪代码

```text
best_response(player_i, profile):
    externality = build_externality_tables(profile without player_i)

    cash = NEG
    cash[start] = day0_purchase(base_price)
    best_terminal = none

    for t = 1 .. T:
        work = cash

        if start_of_day village purchase is available:
            for village v:
                price = externality.village_price[t, v]
                work[v] = buy_knapsack(cash[v], price)

        next = NEG

        for node v:
            apply stay transition

            if v is a mine:
                income = externality.mine_income[t, v]
                apply mine transition

            if weather[t] is not sandstorm:
                for u in adjacency[v]:
                    multiplier = externality.move_multiplier[t, v, u]
                    apply move transition v -> u

        update best terminal state
        save parent information
        cash = next

    return backtrack(best_terminal)
```

---

## 9. 迭代最佳反应

### 9.1 初始策略

忽略多人交互，分别调用已知天气单人 DP，得到每名玩家的初始策略：

$$
\Pi^{(0)}
=(\pi_1^{(0)},\ldots,\pi_n^{(0)}).
$$

第五关三名玩家参数相同，因此初始策略可能完全相同。后续顺序更新会使较早更新的玩家首先对拥堵作出反应。

### 9.2 Gauss–Seidel 顺序更新

第 $r$ 轮中依次更新玩家 $1,2,\ldots,n$：

$$
\pi_i^{(r+1)}
=\operatorname{BR}_i
(\pi_1^{(r+1)},\ldots,\pi_{i-1}^{(r+1)},
\pi_{i+1}^{(r)},\ldots,\pi_n^{(r)}).
$$

即本轮已经更新过的玩家使用新策略，尚未更新的玩家仍使用上一轮策略。

与全体玩家同时更新相比，顺序更新通常更稳定，并且每次更新后都能立即重新生成交互人数表。

### 9.3 策略改进量

更新玩家 $i$ 前，先用联合回放计算其当前支付 $U_i^{\mathrm{old}}$；最佳反应完成后计算

$$
\Delta_i
=U_i(\operatorname{BR}_i(\Pi_{-i}),\Pi_{-i})
-U_i(\Pi).
$$

如果 $\Delta_i>\varepsilon$，接受新策略；否则保留原策略。定点整数计算时可直接取 $\varepsilon=0$，浮点输出仅用于展示。

### 9.4 停止条件

一轮更新后，若

$$
\max_i\Delta_i\leq\varepsilon,
$$

则得到一个 $\varepsilon$-纳什均衡候选。

为避免相同价值的策略反复切换，最佳反应 DP 使用固定的确定性平局规则，例如依次偏好：

1. 更高支付；
2. 更早到达终点；
3. 更高终点现金；
4. 更小行动编码。

外层同时保存策略组合的哈希值；若哈希重复则说明进入循环，应停止并报告当前循环，而不是无限迭代。

### 9.5 外层伪代码

```text
profile = independent_single_player_initialisation()

for round = 1 .. max_rounds:
    max_gain = 0

    for i = 1 .. n:
        old_payoff = evaluate_profile(profile).payoff[i]
        candidate = best_response(i, profile)

        trial_profile = profile with strategy[i] = candidate
        trial_result = evaluate_profile(trial_profile)
        gain = trial_result.payoff[i] - old_payoff

        if gain > epsilon:
            profile = trial_profile

        max_gain = max(max_gain, gain)

    if max_gain <= epsilon:
        break

return verify_equilibrium(profile)
```

---

## 10. 均衡验证

迭代停止后必须独立执行以下检查。

### 10.1 联合规则回放

逐日验证：

- 初始购买资金和负重合法；
- 移动满足邻接关系；
- 沙暴日没有移动；
- 到达矿山当天没有挖矿；
- 同路人数及消耗倍率正确；
- 同矿人数及收入分配正确；
- 购买地点、购买价格、现金和负重正确；
- 到达终点后不再行动；
- 终点退款和失败支付正确。

### 10.2 单边偏离验证

对最终策略组合中的每名玩家重新求一次最佳反应：

$$
\delta_i
=U_i(\operatorname{BR}_i(\Pi_{-i}^*),\Pi_{-i}^*)
-U_i(\Pi^*).
$$

定义最大单边偏离收益

$$
\delta_{\max}=\max_i\delta_i.
$$

只有当 $\delta_{\max}\leq\varepsilon$ 时，才把结果报告为 $\varepsilon$-纳什均衡。若所有金额使用定点整数且 $\delta_{\max}=0$，则在当前纯策略搜索模型下为精确最佳反应不等式成立。

### 10.3 输出内容

第五关至少输出：

1. 每名玩家第 0 天购买量；
2. 每名玩家逐日位置、行动、水、食物和现金；
3. 每日同路人数和同矿人数；
4. 每名玩家到达日及最终支付；
5. 每名玩家的最佳反应支付和偏离收益 $\delta_i$；
6. 总支付、最低玩家支付等辅助指标。

---

## 11. CPU、向量化与 Numba 实现

### 11.1 数组类型

建议使用：

| 数据 | 类型 |
|---|---|
| DP 现金表 | `int32` 或 `int64` |
| 水、食物索引 | `int16` |
| 节点和行动编码 | `int8` 或 `int16` |
| 交互人数 | `int8` |
| 父指针 | 小整数结构化数组 |

不要在最佳反应热循环中使用字符串或 Python 对象。

### 11.2 向量化转移

同一动作对整张 `(W,F)` 平面的转移仍使用 Q1 的切片平移和 `maximum`。多人交互只改变每天、每条边的平移距离和挖矿收入，不改变数组转移结构。

对于第五关，一次最佳反应最多处理：

- 10 个时间步；
- 13 个节点；
- 52 条有向移动边；
- 每日一项停留转移和矿山处一项挖矿转移。

### 11.3 Numba 使用位置

Numba 主要用于：

1. 联合策略逐日回放；
2. 构造同路、同矿人数表；
3. 对紧凑可行库存索引执行转移；
4. 批量进行独立的均衡初值或参数实验。

若继续使用整张 NumPy 切片，`np.maximum` 已在底层 C 循环中运行，未必需要把每个切片再包装成 Numba 循环。应以基准测试决定使用 NumPy 切片还是 Numba 紧凑状态循环。

### 11.4 并行化

单次最佳反应内存在多个来源写入同一目标状态的 `max` 竞争，不应直接对来源节点使用无保护的 `prange`。

安全的并行层级包括：

- 不同初始策略实验；
- 不同参数灵敏度实验；
- 最终均衡验证中的独立玩家最佳反应；
- 使用线程私有 `next_cash` 后再归并。

第五关状态较小，单次最佳反应更可能受内存带宽和父指针写入限制，而不是纯算力限制。

---

## 12. 复杂度与预期资源

可行库存组合数为

$$
N_{WF}
=\#\{(W,F):3W+2F\leq1200\}
=120601.
$$

固定其他玩家后的单次最佳反应状态上界为

$$
O(T\,|V|\,N_{WF}).
$$

动作转移量还需乘以节点平均可选行动数。第五关约有 $1.57\times10^7$ 个时间—节点—库存状态格；按 52 条有向移动边、13 项停留和 1 项挖矿转移估算，单次最佳反应的整平面单元更新上界约为 $8\times10^7$。

采用滚动两层现金表时，核心价值数组约几十 MB；加上父指针和临时数组，建议把单次最佳反应内存控制在 0.1–0.8 GB。避免使用以元组为键的大型 Python 字典保存所有父指针。

在 8 核 CPU、8 GB 内存环境中，第五关适合直接运行。JIT 编译完成后，单次最佳反应预期为亚秒到数秒量级；总时间主要取决于迭代轮数和最终最佳反应验证次数。

---

## 13. 建议代码结构

```text
q3/
├── __init__.py
├── data.py             # 第五/六关参数与地图
├── model.py            # 多玩家状态、策略、支付、定点金额
├── interaction.py      # 同路、同矿、同村人数表
├── replay.py           # 联合策略同步回放
├── best_response.py    # Q1 型前向 DP 最佳反应
├── equilibrium.py      # 迭代最佳反应与偏离收益验证
├── verify.py           # 独立规则检查
└── solve_q3_1.py       # 第五关入口与结果输出
```

Q1 的地图无关 DP 逻辑可以提取为公共工具，但不应直接修改 Q1 已验证的求解器；Q3 的多人倍率、定点资金和交互人数应保持在独立模块中。

---

## 14. 完整计算流程

第五关的端到端计算顺序为：

1. 读取第三关地图和第五关参数、天气序列。
2. 建立三名玩家的独立单人最优初始策略。
3. 联合回放初始策略组合，计算真实交互人数和支付。
4. 固定玩家 2、3，构造玩家 1 的交互参数表并运行最佳反应 DP。
5. 接受玩家 1 的改进策略并重新联合回放。
6. 依次对玩家 2、3 重复最佳反应更新。
7. 重复整轮更新，直到最大策略改进量不超过 $\varepsilon$。
8. 对三名玩家分别重新运行一次最佳反应，计算 $\delta_i$。
9. 独立逐日回放最终策略，验证所有规则和支付。
10. 输出逐日策略表、最终支付和纳什偏离检验结果。

该流程直接复用 Q1 已验证的现金压缩和向量化转移思想，并在外层加入多人交互统计、联合回放和纳什均衡验证。
