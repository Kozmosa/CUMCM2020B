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

## 第六关经验近似结果

- `q3_2_heuristic/result.json`：固定 16 策略、2000 条共同天气样本的基线结果，状态为 `HEURISTIC_PURE`。该结果只在原有限策略库内零经验 regret，未执行宽路线库外审计。
- `q3_2_submission/result.json`：覆盖 804 条 12 步以内简单路线的策略响应结果。最终库包含 29 个策略，使用 5000 条训练天气和三组各 100000 条独立审计天气；审计期望支付为 8018.19、7762.56、6359.93 元，最大经验偏离均值为 1306.76 元，状态为 `EMPIRICAL_EQ_NOT_READY`。
- 两份结果的论文表格、策略解释和准确结论范围见 [`docs/Q3-2-Result.md`](../../docs/Q3-2-Result.md)。

## 第六关正式计算

- `level6-formal/`：完整 `adaptive` 正式实验已在运行约 10.58 小时后按人工请求安全停止，状态为 `SEARCH_STOPPED`。
- 停止时完成 144,008,081 次状态评价，其中 138,876,780 次为直接终止结算；保留 5,131,301 个非终止状态，累计扫描约 $1.804\times10^{11}$ 个完整单边偏离上界，峰值 RSS 约 7.07 GiB。
- 约 1.8 GiB 的 checkpoint、运行日志和停止结果继续作为本地恢复数据保存，不随经验近似结果上传。

正式运行命令见 [q3/README.md](../README.md)。第五关以三名玩家完整最佳反应 regret 为 0 通过纯均衡认证。第六关只有结果状态为 `EXACT_SELECTED`，或根状态 `CERTIFIED_PURE` 且 `max_regret_upper <= 10` 元时，才通过正式质量门槛。
