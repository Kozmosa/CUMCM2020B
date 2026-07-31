# Q3.2 实现说明

> 对应题目：三名玩家仅知道当天天气，每天行动结束后公开各玩家当日行动和剩余资源，随后根据新的公开状态制定下一天行动；求解附件「第六关」。
>
> 数学模型见 [`Solution/Q3-2.md`](./Solution/Q3-2.md)。
>
> 当前实现提供三个后端：`exact` 保留完整无损枚举；`adaptive` 只限制购买候选集，但会对固定候选均衡执行完整单边动作扫描，并以向上舍入的资源上界给出未展开偏离的 regret 上界；`heuristic` 在覆盖全部 12 步以内简单路线的参数化策略类中执行策略响应扩张，并用三组独立 holdout 样本进行成功率与 regret 审计。前两者不量化状态、资源、现金或天气，第三个后端保留精确联合转移规则，但不提供完整动作空间认证。详细提交门槛见 [`Q3-2-Heuristic.md`](./Q3-2-Heuristic.md)。
>
> 当前已完成的第六关数值结果和论文用表格见 [`Q3-2-Result.md`](./Q3-2-Result.md)。

---

## 1. 求解目标与范围

Q3.2 建模为有限时域完全信息随机博弈。第 $t$ 天开始时，所有玩家已经知道上一日结束后的公开联合状态 $S_{t-1}$，并观察到当天天气 $\theta_t$，随后同时选择行动。

完整数值求解器优先求**纯策略马尔可夫完美均衡**。`exact` 后端完整枚举并选择纯均衡；`adaptive` 后端在受限博弈中求纯均衡，然后逐玩家扫描完整动作空间，把最强有利偏离加入候选集。若受限阶段没有纯均衡，`pure-mixed` 模式在有限支持上最小化 NashConv，并明确标记为 `APPROX_MIXED`。

完整后端结果严格区分 `EXACT_SELECTED`、`CERTIFIED_PURE`、`APPROX_PURE`、`APPROX_MIXED` 和 `SEARCH_STOPPED`。只有根状态每名玩家的完整偏离收益上界不超过质量门槛时，才通过正式验收；默认门槛为 10 元。

`heuristic` 后端只在提交质量门槛全部通过时报告 `SUBMISSION_READY_EMPIRICAL_EQ`；否则报告 `EMPIRICAL_EQ_NOT_READY`。默认门槛为每名玩家成功率的单侧 95% 下界不低于 99.99%、最大偏离收益均值不超过 100 元、Bonferroni 修正的单侧 95% 上界不超过 200 元，并要求连续两个响应筛选窗口无超过 50 元的库外策略，以及两个额外训练种子的角色组合保持一致。其 `value_lower/value_upper` 和 regret 来自训练后独立生成的 holdout 天气，仍不能替代上述 10 元完整动作认证。原问题层面的 `selection_complete` 始终保持为 `false`。

玩家 $i$ 的随机路径支付沿用最新 Q2：

$$
G_{i,M}
=
\begin{cases}
C_{i,\tau_i}
+\dfrac12p_W W_{i,\tau_i}
+\dfrac12p_F F_{i,\tau_i},
&\tau_i\leq T,\\[8pt]
C_{i,\kappa_i}-M,
&\text{玩家 }i\text{ 失败}.
\end{cases}
$$

其中 $\tau_i$ 为到达终点的日期，$\kappa_i$ 为资源不足或截止日失败被确认的时刻。每名玩家最大化自己的惩罚期望支付

$$
\mathbb E[G_{i,M}].
$$

程序同时传播该均衡策略对应的成功概率，但成功概率不参与阶段行动的优劣比较。

---

## 2. 第六关数据与统一假设

第六关参数为：

| 参数 | 数值 |
|---|---:|
| 玩家数 | 3 |
| 节点数 | 25 |
| 起点 | 1 |
| 终点 | 25 |
| 村庄 | 14 |
| 矿山 | 18 |
| 截止日期 | 30 |
| 初始资金 | 10000 |
| 负重上限 | 1200 kg |
| 基础收益 | 1000/天 |

地图和资源参数直接复用 `q2.data.level4()`：

- 水每箱 3 kg，基准价格 5；
- 食物每箱 2 kg，基准价格 10；
- 晴朗基础消耗为水 3、食物 4；
- 高温基础消耗为水 9、食物 9；
- 沙暴基础消耗为水 10、食物 10。

天气继续采用 Q2 的 i.i.d. 基准分布：

$$
p(\mathrm{sunny})=0.5,
\qquad
p(\mathrm{hot})=0.4,
\qquad
p(\mathrm{storm})=0.1.
$$

基准失败参数取 $M=10^6$，并在最终数值分析中扫描天气概率和 $M$。

村庄购买默认采用 `start_of_day` 语义：玩家在当天开始时已经位于村庄，观察天气后决定购买量和当天主要行动；当天刚到达村庄不能在同日到达后购买。

---

## 3. 公开联合状态

玩家 $i$ 在第 $t$ 天结束时的个人状态定义为

$$
x_{i,t}
=(d_{i,t},v_{i,t},W_{i,t},F_{i,t},C_{i,t}),
$$

其中状态标记

$$
d_{i,t}
\in
\{\mathrm{active},\mathrm{finished},\mathrm{failed}\}.
$$

联合状态为

$$
S_t=(x_{1,t},x_{2,t},x_{3,t}).
$$

状态语义为：

- `active`：玩家仍需继续行动；
- `finished`：玩家已经到达终点，支付固定，不再参与交互；
- `failed`：玩家已经失败，支付固定，不再参与交互。

题目直接公开其他玩家的行动和剩余资源。现金可由共同已知的初始资金、购买记录、矿山收入和历史交互人数推算，因此实现中把现金纳入公开状态。

### 3.1 资金的定点整数表示

三名玩家同时挖矿时可能获得 $1000/3$。为避免浮点比较影响均衡判定，沿用 Q3.1 的定点金额：

$$
L=\operatorname{lcm}(1,2,3)=6.
$$

所有资金、价格、收入、退款和失败参数在内部乘以 $L$，使用 `int64` 精确保存：

```text
cash_scaled        = 6 * cash
price_scaled       = 6 * price
mine_income_scaled = 6 * R / k
refund_scaled      = 6 * price * remaining / 2
M_scaled           = 6 * M
```

期望价值需要乘以天气概率。天气概率可保存为浮点数，也可以在基准分布下写成分母为 10 的有理数。首版建议：

- 状态转移和单条路径支付使用精确定点整数；
- 期望价值使用 `float64`，三个天气分支按固定顺序求和；
- 纯策略最佳反应比较使用统一容差，例如 `1e-10`；
- 最终用高精度标量实现复核所有均衡不等式。

### 3.2 不直接删除现金维

Q2 使用 $V=C+U$，其前提是现金约束在对应单人关卡不绑定。Q3.2 中多人村庄价格可升至基准价格的四倍，且挖矿收入可能被三人平分，因此不能未经验证直接删除所有玩家的现金。

Q3.2 的状态键保留精确现金。Q2 的单人价值表只用于确定性乐观上界，不作为精确联合价值替代品。

---

## 4. 每日联合行动与同步转移

第 $t$ 天每名活动玩家的行动为

$$
a_{i,t}
=(q_{i,t}^{W},q_{i,t}^{F},z_{i,t}),
$$

其中 $z_{i,t}$ 为停留、挖矿或移动到某个邻接点。联合行动为

$$
A_t=(a_{1,t},a_{2,t},a_{3,t}).
$$

联合转移必须先读取全部玩家行动，再统一统计交互人数，最后同时更新所有状态。不能逐个玩家立即更新。

### 4.1 同路人数

对有向边 $(u,v)$：

$$
k_t^E(u,v)
=\sum_{i=1}^{3}
\mathbf 1\{d_{i,t-1}=\mathrm{active},
v_{i,t-1}=u,
z_{i,t}=\operatorname{move}(v)\}.
$$

移动玩家的消耗倍率为

$$
\lambda_{i,t}=2k_t^E(u,v).
$$

有向边 $(u,v)$ 和 $(v,u)$ 分开统计。

### 4.2 同矿人数

$$
k_t^M(m)
=\sum_{i=1}^{3}
\mathbf 1\{d_{i,t-1}=\mathrm{active},
v_{i,t-1}=m,
z_{i,t}=\operatorname{mine}\}.
$$

挖矿消耗倍率始终为 3，收入为

$$
R_{i,t}=\frac{R}{k_t^M(m)}.
$$

### 4.3 同村购买人数

$$
k_t^V(v)
=\sum_{i=1}^{3}
\mathbf 1\{d_{i,t-1}=\mathrm{active},
v_{i,t-1}=v,
q_{i,t}^W+q_{i,t}^F>0\}.
$$

若玩家购买，单价为

$$
(p_{W,i,t},p_{F,i,t})
=
\begin{cases}
(2p_W,2p_F),&k_t^V(v)=1,\\
(4p_W,4p_F),&k_t^V(v)\geq2.
\end{cases}
$$

### 4.4 每日计算顺序

每个联合行动按以下顺序计算：

1. 根据日初位置检查购买地点和主要行动类型是否合法。
2. 统计购买人数，确定购买价格。
3. 检查购买资金和买后负重。
4. 统计同路人数和同矿人数。
5. 根据最终人数计算每名玩家的资源消耗和挖矿收入。
6. 检查买后库存是否足以执行主要行动。
7. 同时更新位置、水、食物和现金。
8. 标记当天到达终点或失败的玩家。

---

## 5. 无损状态压缩

Q3.2 不建立三人联合状态的密集数组，只保存精确且确实被求解过程访问的状态。

### 5.1 吸收状态

到达终点或失败后，玩家后续状态不再变化。缓存中只需保存：

```text
status
fixed_payoff
```

不再为其枚举位置、资源、现金和行动。

若三名玩家都已吸收，联合状态直接返回支付向量和成功标记。

### 5.2 活动玩家降维

若只剩一名活动玩家，后续不再存在多人交互。精确实现应调用保留现金可行性约束的单人随机 DP。只有在已经证明从该状态开始现金约束不绑定时，才可直接查询现有 Q2 的 $C+U$ 表。

若剩两名活动玩家，联合行动和联合状态自然降为两人维度。

### 5.3 玩家置换规范化

玩家参数完全相同。将三个个人状态按固定字典序排序：

```text
(status, position, water, food, cash_scaled, fixed_payoff_scaled)
```

得到规范化联合状态

$$
\operatorname{canon}(S_t).
$$

缓存键只保存规范化状态，同时记录从原玩家编号到规范化编号的置换。读取价值和行动时应用逆置换恢复玩家身份。

该操作最多减少约 $3!=6$ 倍状态，且不改变任何支付或均衡关系。

### 5.4 可行库存紧凑索引

单人合法库存只包含

$$
\mathcal R
=\{(W,F):3W+2F\leq1200\}.
$$

共有 120601 个组合。预先建立：

```text
resource_id -> (water, food)
(water, food) -> resource_id
```

状态记录使用 `resource_id`，不存储矩形数组中的超重格。

### 5.5 稀疏记忆化

价值缓存键为

```text
packed(day, canonical_joint_state)  # 三人时固定 68 bytes
```

天气在日初观察后进入阶段博弈，因此阶段策略缓存键为

```text
(packed_state_key, weather)
```

不同历史到达相同精确联合状态时，共享同一价值和均衡行动。

### 5.6 精确状态编码

每个个人状态可以编码为固定宽度整数结构：

```text
status        uint8
position      uint8
resource_id   uint32
cash_scaled   int64
fixed_payoff_scaled  int64
```

三人状态使用固定宽度 bytes 作为哈希键：日期为 `uint16`，每名玩家为上述 22-byte 记录，总计 68 bytes。检查点写入时可以无损解码回逐日 NumPy 层；热路径不再保留嵌套 `PlayerState` 元组作为字典键。

---

## 6. 精确行动枚举

### 6.1 主要行动

活动玩家的主要行动为：

- `stay`；
- 若日初在矿山，则 `mine`；
- 若不是沙暴，则移动到任一邻接点。

第六关节点最大度数为 4，因此每名玩家通常只有 1 至 6 个主要行动。非村庄状态的主要联合行动最多约为

$$
6^3=216.
$$

### 6.2 第 0 天购买

第 0 天三名玩家均在起点，以基准价格独立购买。对每名玩家精确枚举所有满足

$$
3q_i^W+2q_i^F\leq1200,
$$

以及

$$
p_Wq_i^W+p_Fq_i^F\leq C_{\mathrm{init}}
$$

的整数购买组合。

第 0 天没有道路、矿山和村庄交互，但三名玩家的购买组合会影响后续联合状态，因此不能分别独立取单人最优购买。

### 6.3 村庄购买者集合分解

在村庄状态，先精确枚举购买者集合

$$
B\subseteq\mathcal I_{\mathrm{active}}.
$$

三名玩家最多有 8 个购买者集合。给定 $B$ 后：

- $i\notin B$ 强制 $q_i^W=q_i^F=0$；
- 若 $|B|=1$，购买者使用二倍价格；
- 若 $|B|\geq2$，购买者使用四倍价格；
- 每名购买者精确枚举所有资金和负重允许的正购买量。

这种分解只是把价格制度提前固定，不删除任何合法购买行动。

### 6.4 购买后库存枚举

对当前库存 $(W_i,F_i)$ 和已知价格 $(p_{W,i},p_{F,i})$，购买动作可等价地枚举所有购买后库存 $(W_i',F_i')$：

$$
W_i'\geq W_i,
\qquad
F_i'\geq F_i,
$$

$$
3W_i'+2F_i'\leq1200,
$$

$$
p_{W,i}(W_i'-W_i)
+p_{F,i}(F_i'-F_i)
\leq C_i.
$$

预先建立从每个 `resource_id`、价格制度和现金上限到合法购买后库存的迭代器。该迭代器可以按分块返回结果，避免一次性建立巨型行动列表。

### 6.5 重复行动转移合并

若两个联合行动产生完全相同的下一联合状态和同一即时支付变化，则它们具有相同 continuation value。只保留确定性平局规则下编码较小的联合行动。

这是无损合并。

---

## 7. 逆向归纳与稀疏递归

定义

```text
solve_state(t, S_t)
```

返回：

```text
value[3]       # 三名玩家的均衡期望支付
success[3]     # 对应均衡策略的成功概率
policy[theta]  # 每种下一天天气下的均衡联合行动
```

### 7.1 终端条件

当 $t=T$ 时：

$$
V_{i,T}(S_T)
=
\begin{cases}
U_i^{\mathrm{fixed}},
&d_{i,T}=\mathrm{finished},\\[8pt]
C_{i,T}-M,
&d_{i,T}=\mathrm{active},\\[8pt]
U_i^{\mathrm{fixed}},
&d_{i,T}=\mathrm{failed}.
\end{cases}
$$

成功概率分别为 1 或 0。

### 7.2 非终端递推

对于每一种正概率天气 $\theta$：

1. 生成该天气下的完整可行联合行动集合。
2. 对每个联合行动精确计算下一状态 $S_{t+1}$。
3. 查询或递归计算 `solve_state(t+1, S_{t+1})`。
4. 构造阶段支付张量或分块支付流。
5. 搜索纯策略阶段纳什均衡。
6. 按确定性规则选择一个均衡行动 $A_{t+1}^*(S_t,\theta)$。

随后计算

$$
V_{i,t}(S_t)
=\sum_{\theta\in\Theta}
p(\theta)
V_{i,t+1}
\left(
G_{t+1}(S_t,A_{t+1}^*(S_t,\theta),\theta)
\right).
$$

成功概率在同一均衡行动下按天气概率加权，不能单独对成功概率重新选行动。

### 7.3 多均衡选择

若同一阶段存在多个纯策略纳什均衡，固定采用：

1. 成功玩家期望人数最大；
2. 最低玩家价值最大；
3. 三名玩家总价值最大；
4. 联合行动编码最小。

选择规则必须在所有状态和线程中保持一致，才能得到确定性的马尔可夫策略。经过玩家置换规范化的状态先在规范化玩家顺序下选择均衡，再通过逆置换恢复原玩家编号。

---

## 8. 纯策略阶段均衡搜索

由于移动消耗和购买价格由联合行动共同决定，个人可行行动集合依赖其他玩家的行动。实现中先构造联合可行掩码 `valid`；对固定 $A_{-i}$，玩家 $i$ 的最佳反应只在满足

$$
(a_i,A_{-i})\in\mathcal A_t(S_{t-1},\theta_t)
$$

的单边偏离中取最大值。候选均衡联合行动本身也必须联合可行。该处理等价于在耦合可行集合上检查纯策略最佳反应条件。

### 8.1 小行动集：支付张量

当三名玩家的行动数分别为 $A_1,A_2,A_3$，且支付张量可以放入内存时，保存

```text
payoff.shape  = (A1, A2, A3, 3)
success.shape = (A1, A2, A3, 3)
valid.shape   = (A1, A2, A3)
```

玩家 1 的最佳反应掩码为

```python
payoff_valid = where(valid, payoff, -inf)
br1 = isclose(
    payoff_valid[..., 0],
    max(payoff_valid[..., 0], axis=0),
    atol=epsilon,
    rtol=0,
)
```

玩家 2、3 分别沿自己的行动轴取最大值。每名玩家的最大值只在给定对手行动后形成的**联合可行单边偏离**中计算。纯策略纳什均衡掩码为

```python
nash = valid & br1 & br2 & br3
```

比较时应使用统一的浮点容差函数，而不是裸 `==`。

### 8.2 大行动集：分块精确搜索

第 0 天和村庄购买可能产生巨大的行动集合，不能构造完整支付张量。对三人博弈采用分块精确搜索：

```text
for each opponent action pair (a2, a3) in chunks:
    scan all feasible a1 in chunks
    compute player 1 exact best-response value and BR action set

    for each a1 in BR1(a2, a3):
        scan all feasible a2' and verify a2 is a best response
        scan all feasible a3' and verify a3 is a best response

        if all three conditions hold:
            record pure Nash equilibrium
```

玩家编号可以轮换，优先让行动数最少的玩家作为外层循环。该算法最坏时间复杂度仍很高，但内存有界且不漏掉纯策略均衡。

对每个固定的对手行动组合，自己的购买量和主要行动必须全部扫描，不能只取 Q2 或单人价值下的购买最优解。

### 8.3 无纯策略均衡

若完整行动集合已扫描且没有任何联合行动同时满足三人的最佳反应条件，则记录该状态不存在纯策略阶段纳什均衡。首版求解器在该状态终止并上报，不自动引入混合策略或近似行动。

---

## 9. 保守的确定性剪枝

本节所有剪枝都必须有“不可能影响纯策略均衡或价值”的证明。

### 9.1 终点和失败吸收

已经吸收的玩家不再生成行动。全部玩家吸收时直接返回，不再递归。

### 9.2 最短路截止日剪枝

预计算忽略沙暴的 BFS 最短距离 $d_{\mathrm{end}}(v)$。若活动玩家满足

$$
d_{\mathrm{end}}(v_i)>T-t,
$$

则即使未来没有沙暴也无法按时到达，可直接结算失败。

对于移动目的地 $u$，若

$$
d_{\mathrm{end}}(u)>T-(t+1),
$$

该移动不可能形成成功路径。由于失败支付仍与现金有关，只有在该动作的失败 continuation 已被精确计算或可确定等价时，才可合并到失败动作；不能简单把所有失败动作无条件删除。

### 9.3 乐观资源不可达剪枝

预计算在以下乐观条件下，到终点或下一村庄所需的最低资源：

- 未来天气均取资源消耗最低的晴朗；
- 玩家始终单独移动；
- 路径不受其他玩家阻碍；
- 到村庄后允许按规则购买。

若玩家不在村庄，并且当前水、食物即使在该乐观模型下也无法到达终点或任何可补给村庄，则不存在成功路径，可直接结算失败。

该剪枝必须同时考虑水和食物，不能只使用总重量。

### 9.4 独立行动合法性剪枝

在组成联合行动前，删除以下个人行动：

- 沙暴日移动；
- 非矿山节点挖矿；
- 目的地不相邻；
- 玩家已吸收却仍行动；
- 购买地点不合法；
- 在最低多人影响下仍无法支付或执行的动作。

如果个人动作在单独移动或最低价格下都不可行，则在任何联合行动中都不可行。

### 9.5 联合行动可行性剪枝

联合人数确定后，立即批量检查：

- 购买成本；
- 买后负重；
- 同路移动消耗；
- 挖矿消耗；
- 沙暴限制；
- 到达矿山当天禁止挖矿。

任何一名活动玩家的动作不合法，该联合行动即从阶段博弈中删除。

### 9.6 零概率天气删除

若 $p(\theta)=0$，该天气分支不参与期望，不生成状态。正概率天气分支不得按概率大小截断。

### 9.7 重复后继状态合并

同一天气下，若多个联合行动产生完全相同的规范化后继状态、现金变化和玩家编号置换，则 continuation value 相同。按确定性平局规则保留一个代表行动。

### 9.8 严格支配行动删除

若已取得完整阶段支付，并能证明对所有使相关联合行动可行的对手行动都有

$$
Q_i(a_i,A_{-i})
\leq
Q_i(b_i,A_{-i}),
$$

且至少一个对手行动下严格小于，则 $a_i$ 不可能属于纯策略纳什均衡，可以删除。

删除后可重复检查，直到没有新的严格支配行动。

### 9.9 确定性上下界剪枝

对玩家 $i$，忽略所有多人负面交互可得到乐观上界。可以使用：

$$
UB_i^{\mathrm{analytic}}
=C_i+(T-t)R+J_{\mathrm{refund}}^{\max},
$$

或在相同天气分布下查询放松现金约束、无拥堵、完整矿山收入和二倍村庄价格的 Q2 单人价值：

$$
UB_i^{Q2}=C_i+U_t^{Q2}(v_i,W_i,F_i).
$$

多人交互只会增加消耗、提高价格或降低收益，因此这些量是乐观上界。若已有一条精确可行 continuation 给出下界 $LB_i$，并且对固定 $A_{-i}$ 有

$$
UB_i(a_i,A_{-i})<LB_i(b_i,A_{-i}),
$$

则 $a_i$ 不可能是最佳反应，可安全删除。

所有上下界剪枝都必须保存证明所需的界值，并在调试模式下支持关闭，以便与未剪枝标量结果对照。

### 9.10 明确禁止的剪枝

首版不允许：

- 水、食物或现金量化；
- 只保留高概率天气路径；
- Beam Search；
- Top-K 状态或购买行动；
- 按近似价值删除状态；
- 未经证明的跨库存 Pareto 删除；
- 限定预设路线或预设策略库。

---

## 10. 联合行动向量化

### 10.1 批量结构

把一批 $B$ 个联合行动保存为结构分离数组：

```text
action_type  int8   (B, 3)
source       int8   (B, 3)
destination  int8   (B, 3)
buy_water    int16  (B, 3)
buy_food     int16  (B, 3)
```

避免 `List[Action]` 和字符串。

### 10.2 有向边事件编码

移动事件编码为

```text
edge_code = source * (n_nodes + 1) + destination
```

非移动玩家使用不会与合法边冲突的特殊编码。广播比较

```python
same_edge = edge_code[:, :, None] == edge_code[:, None, :]
```

得到 `(B,3,3)` 布尔数组，再沿最后一维求和，得到每名玩家的同路人数。

### 10.3 同矿与同村编码

同矿人数由

```python
is_mine[:, :, None]
& is_mine[:, None, :]
& (source[:, :, None] == source[:, None, :])
```

批量计算。

同村购买人数使用

```python
is_buyer = (buy_water + buy_food) > 0
```

以及相同的地点广播比较计算。

### 10.4 批量可行性

根据人数批量得到：

```text
consumption_multiplier  (B, 3)
water_needed            (B, 3)
food_needed             (B, 3)
purchase_cost           (B, 3)
mine_income_scaled      (B, 3)
```

再生成

```python
valid = (
    valid_action
    & valid_cash
    & valid_weight
    & valid_resource
).all(axis=1)
```

只对 `valid` 行生成后继状态。

### 10.5 天气维向量化

天气只有三种，可以把天气作为批量维：

```text
weather_action_batch  (n_weather, B, 3, fields)
```

但天气观察后行动集合不同，尤其沙暴没有移动。实现时也可以分别构造三批行动，再统一对 continuation value 做概率加权。两种方式必须产生完全相同的标量结果。

### 10.6 分块处理

第 0 天和村庄购买的联合行动数可能远超内存。使用固定大小块：

```text
block_size = 2^14 ... 2^18
```

每个块依次执行：

1. 生成行动数组；
2. 计算交互人数；
3. 过滤不可行行动；
4. 生成后继状态；
5. 规范化并合并块内重复状态；
6. 查询 continuation value；
7. 更新最佳反应最大值或均衡候选。

分块只改变执行顺序，不改变枚举集合。

---

## 11. CPU 与 Numba 并行化

### 11.1 适合并行的层级

安全的并行任务包括：

- 同一时间层的不同联合状态；
- 不同正概率天气分支；
- 不同购买者集合；
- 大行动集搜索中的不同对手行动对；
- 最终验证中的不同单边偏离；
- 蒙特卡洛验证中的独立天气样本。

天气分支可以并行计算，但三个分支的期望值必须在主线程中按照固定天气顺序归并，避免线程完成顺序改变浮点加法顺序。

### 11.2 Numba 热核

建议使用 `@njit(cache=True)` 或 `@njit(parallel=True, cache=True)` 实现：

```text
build_joint_action_block
count_interactions
filter_joint_actions
apply_joint_transition
canonicalize_state_batch
best_response_scan
pure_nash_mask
replay_profile_batch
```

哈希表、递归调度和磁盘缓存留在 Python 层；大规模整数循环和数组运算放入 Numba。

当前资源感知上界已经按日期层实现 `@njit(parallel=True)`。普通 CPython 仍使用一个 Python successor worker，但 `--bound-threads` 可以独立使用最多 64 个原生线程；第六关完整上界后缀最多约 636 MiB，并只按实际请求的最早日期向前扩展。

购买偏离扫描另有 `--purchase-oracle auto|cpu|cuda|off`。`auto` 和 `cpu`
使用精确区域 max-pyramid：固定对手与行动骨架后，把购买格点写成“常数 +
资源残值 - 线性购买成本”，为水食矩形区域保存向上舍入最大值，一次比较
即可认证删除整块。只有不能删除的叶子才进入融合 Numba 转移和递归 DP。
`cuda` 使用 Numba CUDA 完整扫描每个格点，保留与 CPU 逐位一致的独立实现；
参考 A800 上完整点扫描受分支发散、传输和短 kernel 启动影响，慢于区域
oracle，因此 `auto` 选择区域算法而不强制使用 GPU。`off` 保留旧完整扫描
作为回归基准。

### 11.3 避免并行写竞争

多个线程不得直接向同一缓存或同一 `max` 数组无保护写入。采用：

1. 每线程私有输出缓冲；
2. 第一遍统计输出数量；
3. 前缀和计算每线程写入区间；
4. 第二遍填充扁平数组；
5. 主线程统一排序、去重和归并。

### 11.4 时间层批处理

虽然价值递推按时间逆序，但同一时间层中已知 continuation value 的状态可以并行求解。建议维护工作队列：

```text
requested states at day t
    -> collect missing successors at day t+1
    -> solve unique successors
    -> return to evaluate day-t stage games in parallel
```

相同后继状态只加入一次队列。

### 11.5 线程数

阶段行动评估主要受内存带宽限制。建议基准测试：

- 4 个线程；
- 性能核心数量；
- 全部逻辑核心。

不要同时启用 NumPy/BLAS 多线程和外层 Numba `prange` 嵌套并行。

---

## 12. 缓存与内存管理

### 12.1 分层缓存

按时间保存独立缓存：

```text
value_cache[t]
policy_cache[t][weather]
success_cache[t]
```

时间只向前转移，因此完成更早时间层后，可以将不再需要的临时行动数据释放。

### 12.2 状态批量去重

后继状态先编码为结构化数组，按以下字段排序：

```text
player1 state fields
player2 state fields
player3 state fields
```

使用 `np.lexsort`、结构化 `np.unique` 或自定义 Numba radix/sort 合并完全相同的状态。

### 12.3 值与策略分离

价值缓存只保存：

```text
value[3]    float64
success[3]  float64
```

策略缓存单独保存每种天气下的联合行动编码。求解阶段不保存完整联合行动支付张量，除非当前状态的行动规模足够小。

### 12.4 可选磁盘缓存

长时间运行时，可以在完成某个时间层后，把只读价值层写入内存映射文件。磁盘缓存不得改变浮点类型或状态键，重新加载后需保持位级一致。

---

## 13. 求解伪代码

```text
solve_state(t, state):
    state, inverse_perm = canonicalize(state)

    if state in value_cache[t]:
        return inverse_permute(value_cache[t][state])

    absorb finished and failed players

    if t == T or all players absorbed:
        return terminal_value(state)  # analytic leaf; do not cache

    state = absorb_deterministically_doomed_players(state)

    if all players absorbed:
        return terminal_value(state)

    expected_value = [0, 0, 0]
    expected_success = [0, 0, 0]

    for weather with positive probability:
        individual_actions = enumerate_all_exact_actions(state, weather)
        individual_actions = safe_individual_prune(individual_actions)

        if action sets are small:
            payoff_tensor = evaluate_all_joint_actions_vectorized(...)
            remove_strictly_dominated_actions(payoff_tensor)
            equilibria = pure_nash_from_tensor(payoff_tensor)
        else:
            equilibria = exact_chunked_pure_nash_search(...)

        if equilibria is empty:
            report no pure equilibrium for (t, state, weather)

        equilibrium = deterministic_equilibrium_selection(equilibria)

        expected_value += p[weather] * equilibrium.value
        expected_success += p[weather] * equilibrium.success
        policy_cache[t, state, weather] = equilibrium.action

    result = (expected_value, expected_success)
    value_cache[t, state] = expected_value
    success_cache[t, state] = expected_success
    return inverse_permute(result)
```

第 0 天在起点状态上调用同一个阶段博弈框架，但行动只包含基准价格初始购买，不包含天气和主要行动。

---

## 14. 正向策略执行

逆向求解完成后，实际游戏按公开状态查询策略：

```text
state = initial state before day-0 purchases
action0 = policy_day0[state]
state = apply_day0_purchase(action0)

for day = 1 .. T:
    weather = observed_weather[day]
    action = policy_cache[day-1, state, weather]
    state = apply_joint_transition(state, action, weather)

    record positions, resources, cash and interaction counts

    if all players absorbed:
        stop
```

由于天气只在当天观察，输出的完整策略是状态查询规则；某一条演示天气序列只能产生一条策略实现轨迹，不能代表整个反馈策略。

---

## 15. 验证

### 15.1 标量转移对照

实现一个不使用向量化、剪枝和并行的三人标量联合转移函数。随机生成小状态和联合行动，逐项比较：

- 交互人数；
- 购买价格；
- 消耗量；
- 挖矿收入；
- 后继状态；
- 终点和失败标记。

向量化结果必须与标量结果完全一致。

### 15.2 剪枝等价性

每种剪枝提供开关。在缩小的测试关卡上分别运行：

```text
all pruning disabled
one pruning enabled
all safe pruning enabled
```

比较价值、成功概率和均衡行动。除确定性平局下的等价行动编码外，结果必须一致。

### 15.3 阶段均衡验证

对保存的每个均衡联合行动，穷举每名玩家所有合法单边偏离，验证

$$
Q_i(A^*)
\geq
Q_i(a_i,A_{-i}^*)-\varepsilon.
$$

购买状态的验证也必须扫描全部合法购买量。

### 15.4 贝尔曼恒等式

对访问状态验证：

$$
V_{i,t}(S_t)
=
\sum_{\theta}p(\theta)
V_{i,t+1}
\left(G_{t+1}(S_t,A_{t+1}^*,\theta)\right).
$$

### 15.5 联合规则回放

独立回放演示轨迹，验证：

- 邻接和沙暴禁行；
- 到达矿山当天不能挖矿；
- 同路、同矿、同村人数；
- 购买资金和负重；
- 失败支付和终点退款；
- 到达或失败后不再行动。

### 15.6 蒙特卡洛

在已经求出的精确反馈策略上采样大量 i.i.d. 天气序列。比较：

- 蒙特卡洛平均支付与逆向价值；
- 蒙特卡洛成功率与伴随成功概率；
- 不同线程数下的结果一致性。

蒙特卡洛只用于验证，不参与策略搜索或剪枝。

---

## 16. 复杂度与可计算性

单名玩家位置—库存状态数约为

$$
25\times120601
=3.015025\times10^6.
$$

三人完整联合状态的理论数量约为

$$
(3.015025\times10^6)^3
\approx2.74\times10^{19},
$$

且尚未计入现金和状态标记。因此无损压缩和确定性剪枝只能降低实际访问量，不能消除问题的指数复杂度。

非村庄状态的主要行动集合较小，向量化后通常不是主要瓶颈。真正的瓶颈为：

1. 第 0 天三人初始购买组合；
2. 多名玩家同时位于村庄时的完整购买组合；
3. 30 天随机天气下递归产生的精确联合状态。

求解器应持续记录：

```text
states requested per day
unique states solved per day
joint actions generated
joint actions pruned as infeasible
duplicate successors merged
cache hit rate
pure-equilibrium states / no-pure-equilibrium states
wall time and peak memory
```

如果完整第六关超出 CPU 或内存预算，应报告精确算法运行到的规模和阻塞点，不能在不修改模型说明的情况下自动启用有损近似。

---

## 17. 建议代码结构

```text
q3/
├── __init__.py
├── data.py                 # 第五/六关配置、天气概率、地图
├── model.py                # 状态、行动、定点金额、支付
├── resource_index.py       # 120601 个可行库存及索引
├── action_enum.py          # 精确个人行动和购买枚举
├── profile_enum.py         # 行动骨架分组、联合可行块枚举
├── interaction.py          # 同路、同矿、同村批量统计
├── transition.py           # 标量与向量化联合转移
├── canonical.py            # 玩家置换规范化和状态编码
├── pruning.py              # 仅无损确定性剪枝
├── stage_game.py           # 支付张量、分块搜索、纯纳什验证
├── stochastic_dp.py        # 稀疏逆向归纳、缓存与工作队列
├── policy.py               # 联合状态和天气到行动的查询
├── simulate.py             # 正向反馈策略执行
├── verify.py               # 规则、均衡、贝尔曼与剪枝验证
└── solve_q3_2.py           # 第六关入口、统计和结果输出
```

Q3.1 和 Q3.2 可以共享状态、联合回放、交互统计、定点金额和规则验证模块；Q3.2 独有随机天气逆向归纳和阶段博弈均衡求解器。

---

## 18. 实现顺序

建议按以下顺序开发：

1. 建立第六关配置、三人状态和定点金额。
2. 实现标量联合转移和规则回放。
3. 实现无购买状态下的完整主要行动枚举。
4. 实现向量化联合转移，并与标量转移随机对照。
5. 实现小行动集支付张量和纯策略纳什检测。
6. 实现终端条件和短时域稀疏逆向归纳。
7. 加入吸收、最短路、乐观资源不可达和联合可行性剪枝。
8. 实现玩家置换规范化和后继状态去重。
9. 实现第 0 天与村庄的完整购买枚举。
10. 实现大行动集分块纯纳什搜索。
11. 加入 Numba 多状态并行、线程私有缓冲和批量去重。
12. 在缩小的截止日期和库存上完成穷举基准验证。
13. 逐步扩大到完整 30 天第六关，并记录状态、行动、时间和内存规模。
14. 最后进行均衡偏离检查、贝尔曼检查、联合回放和蒙特卡洛验证。

该顺序先建立可穷举验证的精确内核，再增加无损优化，确保所有加速步骤都可以与标量基准逐项对照。

### 18.1 当前实现进度

当前代码已经提供：

- `Q3Config.weather_sequence` 与第五关已知天气；
- Q3.1 联合回放、完整对手状态最佳反应、Gauss--Seidel、策略级 double-oracle 和混合兜底；
- Q3.2 `exact|adaptive|heuristic` 三后端：前两者提供紧凑数组购买动作、确定性候选、完整纯偏离扫描、资源感知向上舍入上界、精确购买格点 max-pyramid、Numba/CUDA 双实现和 NashConv 混合兜底；`heuristic` 提供 804 条短简单路线全集、购买/村庄/挖矿参数化策略、追加式 canonical-profile 缓存、策略响应扩张和三组独立 holdout 审计；
- 对称轨道约简、候选全集覆盖认证和经完整最佳反应验证的纯均衡提前路径；
- `PolicyEntry`、`SolveReport` 和明确的均衡状态；
- v2 目录检查点（manifest、逐日 NumPy 层、阶段元数据）以及 v1 pickle 迁移读取；
- 墙钟/RSS/状态/profile 预算和原子检查点。

小规模穷举与回归测试已覆盖标量/批量/紧凑数组转移、deadline 数组直结算、偏离导致对手提前失败、上界支配、不同原生线程数的按位确定性、exact/adaptive 一致性、无纯均衡混合解、CPU/CUDA 购买上界逐位一致以及 v1/v2 恢复。普通 CPython 3.13、一个 Python worker 和 64 个上界原生线程下，旧完整点扫描的 60 秒根状态基准执行 2.136 亿次偏离并评价 60,290 个状态；精确 max-pyramid 在同样 60 秒内认证 7.702 亿次偏离并评价 254,841 个状态，分别提升 3.61 倍和 4.23 倍，峰值 RSS 约 331 MiB。新的一分钟进度已接近旧五分钟的约 25.8 万状态，因此该前沿的端到端加速约 4.9 倍。

完整第六关 `adaptive` 正式实验已在 2026-07-29 按人工请求安全暂停。停止时已运行 38,095 秒（约 10.58 小时），保留 5,131,301 个非终止状态、直接结算 138,876,780 个终止叶，并扫描约 $1.804\times10^{11}$ 次完整单边偏离上界；峰值 RSS 约 7.07 GiB，v2 检查点约 1.8 GiB。实验尚未返回根状态策略或有限 regret，因此状态正确记录为 `SEARCH_STOPPED`，不能再按早期前沿线性外推声称“约 5 小时完成”。本地检查点和恢复命令保留在 `q3/output/level6-formal/`，需要完整认证时可继续运行。

旧固定 `K=16, N=2000` 经验纯均衡已被参考路线证明存在显著库外偏离，因此不再作为竞赛终点。新的提交级默认值为最多 32 个响应生成策略、5000 条训练天气，以及三组各 100000 条独立审计天气。第六关 12 步内共有 804 条简单路线；`K=32` 时只需缓存 5984 个 canonical profile。最终只有状态 `SUBMISSION_READY_EMPIRICAL_EQ` 可以进入论文结论，所有 `EMPIRICAL_EQ_NOT_READY` 结果都必须继续保留明确 gap。
