# Q3 计算产物

本目录区分“已认证的小规模/晚期状态结果”和“正式关卡预算探针”。预算探针没有达到根状态质量门槛，因此不能作为第五、六关最终数值答案。

## 已认证验证结果

- `q3_1_smoke/`：已知天气微型三人开环博弈，`CERTIFIED_PURE`，三名玩家完整偏离 regret 均为 0。
- `q3_2_smoke/`：三人晚期微型反馈状态，`CERTIFIED_PURE`，支付均为 40，成功率均为 1，完整动作偏离 regret 为 0。
- `q3_2_day29_exact/` 与 `q3_2_day29_adaptive/`：第六关 day 29 状态 exact/adaptive 对照。两者支付均为 `-89839.5`、成功率均为 `0.9`；晴朗/高温移动到 25，沙暴停留。adaptive 完整偏离 regret 为 0。

## 正式关卡探针

- `q3_1_level5_probe/result.json`：第五关完整最佳反应在 100000 个联合前沿状态上限处停止，状态为 `SEARCH_STOPPED`。另一次 2000000 上限的单次最佳反应试跑在约 4 分钟时 RSS 已超过 2.2 GiB，说明当前 Python 字典热路径仍需按计划改为结构分离数组/Numba，未生成认证第五关均衡。
- `q3_2_level6_probe/result.json`：第六关根状态短预算探针。完成 216 个缓存状态后因墙钟预算停止；尚无根策略，故 `max_regret_upper = Infinity`。
- `q3_2_level6_probe/resumed.json`：从同一 v2 目录检查点恢复，`checkpoint_loads = 1`，证明恢复链路有效；仍因短预算停止。

正式运行命令见 [q3/README.md](../README.md)。只有结果状态为 `EXACT_SELECTED`，或根状态 `CERTIFIED_PURE` 且 `max_regret_upper <= 10` 元时，才通过正式质量门槛。
