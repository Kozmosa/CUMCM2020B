# Q3 计算产物

本目录区分已认证结果与预算探针。第五关已经取得完整最佳反应认证的纯策略均衡；第六关根状态仍未达到质量门槛。

## 已认证验证结果

- `q3_1_level5_optimized/`：正式第五关 `CERTIFIED_PURE`。三名玩家均成功，支付分别为 9535、9480、9510 元，完整最佳反应 regret 均为 0；`selection_complete=false` 表示没有枚举所有其他可能均衡。
- `q3_1_smoke/`：已知天气微型三人开环博弈，`CERTIFIED_PURE`，三名玩家完整偏离 regret 均为 0。
- `q3_2_smoke/`：三人晚期微型反馈状态，`CERTIFIED_PURE`，支付均为 40，成功率均为 1，完整动作偏离 regret 为 0。
- `q3_2_day29_exact/` 与 `q3_2_day29_adaptive/`：第六关 day 29 状态 exact/adaptive 对照。两者支付均为 `-89839.5`、成功率均为 `0.9`；晴朗/高温移动到 25，沙暴停留。adaptive 完整偏离 regret 为 0。

## 正式关卡探针

- `q3_1_level5_probe/result.json`：旧 Python 字典前沿的失败基线，保留用于对照。紧凑 Numba 内核已取代该热路径。
- `q3_1_level5_optimized_probe/best_response.json`：正式第五关单次完整最佳反应基准；最大前沿约 114 万，累计检查约 875 万状态，峰值 RSS 约 397 MiB。
- `q3_2_level6_probe/result.json`：第六关根状态短预算探针。完成 216 个缓存状态后因墙钟预算停止；尚无根策略，故 `max_regret_upper = Infinity`。
- `q3_2_level6_probe/resumed.json`：从同一 v2 目录检查点恢复，`checkpoint_loads = 1`，证明恢复链路有效；仍因短预算停止。

正式运行命令见 [q3/README.md](../README.md)。第五关以三名玩家完整最佳反应 regret 为 0 通过纯均衡认证。第六关只有结果状态为 `EXACT_SELECTED`，或根状态 `CERTIFIED_PURE` 且 `max_regret_upper <= 10` 元时，才通过正式质量门槛。
