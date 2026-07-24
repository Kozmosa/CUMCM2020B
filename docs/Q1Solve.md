# Q1 实现说明

> 对应题目：一名玩家，**天气全程已知**，求最优策略；求解附件「第一关」「第二关」，填入 `Result.xlsx`。  
> 数学模型见 [`Solution/Q1.md`](./Solution/Q1.md)。  
> 代码目录：[`q1/`](../q1/)。

---

## 1. 问题要点（实现视角）

| 规则 | 实现处理 |
|------|----------|
| 第 0 天只能在起点买物资，不能行动 | `day 0` 单独做一次购买 knapsack，无消耗 |
| 每天一个主行动：停留 / 前进 / 挖矿 | 消耗倍率 1 / 2 / 3；沙暴禁止前进 |
| 到达矿山**当天**不能挖矿 | 仅当**当天开始**已在矿山（`v_{t-1}∈M`）才允许 `mine` |
| 村庄可购水/食物，价格 = 2×基准价 | 见 §3 购买时机 flag |
| 负重 `3W+2F ≤ 1200` | 转移前用 mask 过滤 |
| 到达终点后游戏结束，剩余资源按**半价**退回 | 终值 `J = C + ½ p_W W + ½ p_F F` |
| 须在截止日期 `T` 或之前到达 | 未到终点且 `t=T` 的状态价值为 −∞（不可行） |

目标：在 `τ ≤ T` 到达终点，**最大化** `J`。

---

## 2. 算法：前向 DP（与 Bellman 等价）

### 2.1 为何不用直接逆向表

`Solution/Q1.md` 给出的是逆向 Bellman：

$$
V_t(s_t)=\max_{a} V_{t+1}(g(s_t,a,\theta_{t+1}))
$$

状态含资金 `C`，取值范围大（初始 10000 + 挖矿累计），不适合作为离散表格的一维。

### 2.2 关键观察：固定 `(t,v,W,F)` 只需保留 max C

- **可行性**：村庄购买预算 `p_W q^W + p_F q^F ≤ C`，C 更大不会更差。  
- **目标**：`J` 对 C 严格递增。  

因此定义

$$
\text{cash}_t(v,W,F) \;=\; \max\{\,C \mid \text{第 }t\text{ 天结束时状态为 }(v,W,F,C)\,\}
$$

前向递推 `cash_t`，在任意 `t` 若 `v = v_end`，用 `terminal_value` 更新全局最优。  
这与逆向 Bellman 在「C 单调」下数学等价，且天然支持「任意 `τ≤T` 到达」。

### 2.3 状态与转移

```
cash[v, W, F]  ∈ ℤ ∪ {−∞}     # 节点 id 直接作下标
```

**第 0 天**

```
seed[0,0] = C_init
cash[start] = BuyKnapsack(seed; 基准价)
```

**第 t 天**（`t = 1..T`）

```
work ← cash                                          # 昨日结束
if mode == start_of_day:
    for v in villages:
        work[v] ← BuyKnapsack(work[v]; 村庄价)

next ← −∞
for each reachable (v, W, F) in work:               # 向量化切片实现
    stay : next[v, W−c_w, F−c_f] ← max(..., C)
    mine : if v∈mines: +R, 消耗 3×
    move : if not storm: next[u, W−2c_w, F−2c_f] ← max(..., C)  ∀u∼v

if mode == after_arrival:
    for v in villages:
        next[v] ← BuyKnapsack(next[v]; 村庄价)

若 next[end] 有可达格: 用终值更新 best
cash ← next
```

消耗 `c_w, c_f` 由当天天气决定（晴朗/高温/沙暴）。

### 2.4 购买子问题 `BuyKnapsack`

在固定节点上，从已有库存 `(W₀,F₀)` 出发，允许再买任意非负整数箱：

$$
\max_{q^W,q^F\ge 0}\;
C - p_W q^W - p_F q^F
\quad\text{s.t.}\quad
3(W_0+q^W)+2(F_0+q^F)\le 1200,\;
C \ge p_W q^W + p_F q^F
$$

实现上对整张 `(W,F)` 表做两次一维扫描（先加水、再加食物）。单价恒定，两趟扫描精确。  
同时维护 `origin_w/f`，把每个买后格子指回买前库存，供父指针记录购买量。

### 2.5 剪枝

1. **到终点最短路**（BFS，忽略沙暴，乐观）：若 `dist(u) > T−t`，则今天移到 `u` 后不可能按时到达，跳过。  
2. **终点吸收**：已在终点的状态不再行动（游戏已结束）。  
3. **不可达格**：`cash == NEG` 的格子不扩展。

未做跨 `(W,F)` 的 Pareto 剪枝：多持资源会占满负重，削弱未来村庄购买能力，简单「资源多就优」不安全。

### 2.6 回溯

每个改进过的 `(t,v,W,F)` 记录父指针：

```text
(prev_v, prev_W, prev_F, action, buy_W, buy_F, cash_after)
```

从最优终点状态沿父指针回到 day 0，得到完整逐日轨迹，写入 `Result.xlsx`。

---

## 3. 村庄购买时机 flag

```python
class VillagePurchaseMode(str, Enum):
    START_OF_DAY  = "start_of_day"   # 默认；对齐 Solution/Q1.md
    AFTER_ARRIVAL = "after_arrival"  # 宽松解读「经过…可随时购买」
```

| 模式 | 含义 | CLI |
|------|------|-----|
| `start_of_day` | **当天开始**已在村庄才可买；当天刚走到村庄须次日再买 | `--purchase-mode start_of_day` |
| `after_arrival` | 当天行动并消耗后，若**结束位置**在村庄则可买 | `--purchase-mode after_arrival` |

两模式在第一关上终值相同（10470），仅购买发生日不同。默认使用 `start_of_day`。

---

## 4. 代码结构

```text
q1/
├── __init__.py     # 对外导出
├── data.py         # LevelConfig、天气、两关地图
├── model.py        # 消耗/价格/终值、VillagePurchaseMode、DayRecord
├── dp.py           # BuyKnapsack + 前向 DP + 回溯
├── verify.py       # 轨迹规则回放校验
├── solve.py        # CLI、打印、写 Result.xlsx
├── Result.xlsx     # 输出
└── README.md
```

| 模块 | 职责 |
|------|------|
| `data.py` | 关卡常量；第一关边表；第二关 8×8 交错六边形邻接 |
| `model.py` | 规则层小函数，无搜索逻辑 |
| `dp.py` | 核心求解；numpy 切片做 stay/mine/move |
| `verify.py` | 不依赖 DP 内部表，按规则重放轨迹 |
| `solve.py` | 入口；按官方模板列 A–E / G–K 填表 |

### 4.1 地图

- **第一关（27 点）**：无向边表写在 `data._EDGES_L1`，与官方附图及公开题解交叉核对。  
  特殊点：起点 1，终点 27，矿山 12，村庄 15。  
- **第二关（64 点）**：`8×8` 交错网格（hex-offset）。  
  `id = y*8 + x + 1`；偶数行对角连 `(y±1,x−1)`，奇数行连 `(y±1,x+1)`。  
  特殊点：起点 1，终点 64，矿山 {30,55}，村庄 {39,62}。

> 附件 markdown 中「22×3」是图面标注方式；拓扑上是 8×8 交错邻接，不是简单矩形四连通。

### 4.2 关键实现细节

1. **向量化转移**：对节点 `v` 的整张 `src[W,F]`，消耗 `(cw,cf)` 后写入  
   `next[v_to, :W+1-cw, :F+1-cf] = max(..., src[cw:, cf:] (+ income))`。  
2. **父指针只写改进格**：`np.where(better)`，避免全表 Python 循环。  
3. **资金用 `int32`**，不可达用 `NEG = -10^9`（低于任何真实资金）。  
4. **Result 注 2**：表中「剩余」= 当日消耗（及购买）完成后的量；到达终点后后续日期留空。

---

## 5. 运行方式

依赖：仓库根目录 `uv` 环境（`numpy`、`openpyxl`）。

```bash
# 默认：两关 + start_of_day → q1/Result.xlsx
uv run python -m q1.solve

uv run python -m q1.solve --level 1
uv run python -m q1.solve --level 2
uv run python -m q1.solve --purchase-mode after_arrival
uv run python -m q1.solve --quiet
uv run python -m q1.solve --out path/to/out.xlsx
```

库调用示例：

```python
from q1 import load_level, solve, VillagePurchaseMode

cfg = load_level("1")
res = solve(cfg, purchase_mode=VillagePurchaseMode.START_OF_DAY, verbose=False)
print(res.best_value, res.arrival_day, res.trajectory)
```

---

## 6. 数值结果（`start_of_day`）

| 关卡 | 最优终值 J | 到达日 τ | 终点 (C, W, F) |
|------|------------|----------|----------------|
| 第一关 | **10470** | 24 | (10470, 0, 0) |
| 第二关 | **12730** | 30 | (12730, 0, 0) |

### 6.1 第一关策略概要

1. 第 0 天买水 178、食物 333（耗资 4220，剩 5780）。  
2. 经 25→24→23→21→9 于第 8 天到村庄 15。  
3. 第 9 天在村庄补水后前往矿山 12；第 11–16、19 日挖矿（沙暴日 17–18 停留）。  
4. 经 13→15 再补给，9→21→27，第 24 天到达终点，资源耗尽，终值 10470。

### 6.2 第二关策略概要

1. 第 0 天买水 130、食物 405。  
2. 经上方路径至矿山 30 旁村庄 39 补给，在 30 挖矿数日。  
3. 再经 39 补给，前往矿山 55 继续挖矿。  
4. 第 30 天 55→56→64 到达终点，终值 12730。

轨迹经 `verify.py` 逐日回放：资源非负、负重、邻接、沙暴禁行、资金与购买均一致，终值与 DP 一致。

---

## 7. 与 `Solution/Q1.md` 的对应关系

| 模型符号 | 代码 |
|----------|------|
| 状态 \(s_t=(v_t,W_t,F_t,C_t)\) | `cash[v,W,F]` 存 max \(C_t\) |
| 决策 \(a_t=(z_t,q^W_t,q^F_t)\) | `action` + knapsack 购买 |
| \(\lambda(z)\in\{1,2,3\}\) | `cons["stay"/"move"/"mine"]` |
| \(Z_t\)（沙暴 / 矿山） | `is_storm`、`v in mines` 分支 |
| 终值 \(J\) | `terminal_value` |
| \(V_0(s_0)\) 最优 | `SolveResult.best_value` |
| 策略 \(\pi\) | `SolveResult.trajectory` |

前向 DP 是 Bellman 最优性方程在「C 维取 max」压缩后的实现，不改变最优值。

---

## 8. 复杂度与性能

- 单日状态上界：约 \(N \times \#\{(W,F):3W+2F\le 1200\}\) ≈ \(N \times 1.2\times 10^5\)。  
- 实际可达远小于上界；第一关峰值约 \(2\times 10^6\) 格/天，第二关约 \(5\times 10^6\)。  
- 转移以 numpy 切片为主，购买为 \(O(W_{\max}+F_{\max})\) 扫表；本机两关合计约数分钟量级（含父指针稀疏写入）。

---

## 9. 扩展与注意

- 改购买语义：只改 `--purchase-mode`，不必动转移核。  
- 第三问（天气未知）不在本包内：需把「已知 \(\theta_{1:T}\)」换成期望/鲁棒目标。  
- 勿对 `(W,F,C)` 做朴素三维 Pareto 剪枝（见 §2.5）。  
- 第二关地图必须用交错六边形邻接，矩形四连通会得到错误最优值。
