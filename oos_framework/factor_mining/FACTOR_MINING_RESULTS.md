# 因子挖掘第四章 · 四大方向运行结果

- 样本: **150 只** A 股 | 变量池: 19 | 区间: 2018-10-23 ~ 2026-07-20

## 方向1 · 算子 + 变量网格搜索

正 IC 候选 Top10 (耗时 103.4s):

| 因子表达式 | IC(20d) | ICIR(20d) | 换手率 |
|---|---|---|---|
| `ts_mean(dist_high_20,20)` | +0.023 | +0.14 | 0.022 |
| `cs_rank(ts_mean(dist_high_20,20))` | +0.023 | +0.14 | 0.022 |
| `ts_rank(arate_5,5)` | +0.010 | +0.09 | 0.223 |
| `cs_rank(ts_rank(arate_5,5))` | +0.010 | +0.09 | 0.223 |
| `ts_mean(dist_high_20,10)` | +0.014 | +0.09 | 0.032 |
| `cs_rank(ts_mean(dist_high_20,10))` | +0.014 | +0.09 | 0.032 |
| `ts_rank(arate_5,10)` | +0.007 | +0.06 | 0.215 |
| `cs_rank(ts_rank(arate_5,10))` | +0.007 | +0.06 | 0.215 |
| `ts_mean(dist_high_20,5)` | +0.009 | +0.06 | 0.046 |
| `cs_rank(ts_mean(dist_high_20,5))` | +0.009 | +0.06 | 0.046 |

## 方向2 · 遗传规划因子工厂 (NSGA-II 多目标)

Pareto 前沿规模: **7** | 最优单体: `cs_rank(cs_rank(amount)) corr dist_high_20 div volume` | 耗时 17.5s

| 因子表达式(Pareto) | ICIR(20d) | 换手率 | IC(20d) |
|---|---|---|---|
| `cs_rank(cs_rank(amount)) corr dist_high_20 div volume` | +0.61 | 0.052 | +0.075 |
| `vrate_5 div ts_pct(vrate_5,20) sub ts_max(amount div high,10)` | +0.57 | 0.022 | +0.082 |
| `range_20 sub ts_max(amount div high,10)` | +0.56 | 0.022 | +0.082 |
| `cs_rank(open) corr dist_high_20 div volume sub ts_max(amount div cs_rank(low),10)` | +0.52 | 0.021 | +0.070 |
| `ts_max(ts_std(high,10),5) sub ts_max(amount div cs_rank(low),10)` | +0.52 | 0.020 | +0.070 |
| `ts_max(ts_delta(dist_high_20,60),10) sub ts_max(open,10)` | +0.25 | 0.004 | +0.046 |
| `ts_max(ts_max(cs_rank(low),20),20)` | -0.25 | 0.002 | -0.044 |

## 方向3 · LLM + MCTS 公式化挖掘

LLM 钩子: **未配置(本地启发式)** | 候选数: 119 | 耗时 6.2s

| 因子表达式(MCTS) | ICIR(20d) | IC(20d) | 换手率 | 访问次数 |
|---|---|---|---|---|
| `vol_20 div ret_1 mul vrate_5 div ret_1` | +0.32 | +0.040 | 0.313 | 1 |
| `vol_20 div log_ret_1 mul vrate_5 div ret_1` | +0.32 | +0.040 | 0.313 | 1 |
| `vol_20 div ret_10 mul vrate_5 sub amount` | +0.27 | +0.033 | 0.162 | 1 |
| `vol_20 div ret_10 mul vrate_5 sub volume` | +0.26 | +0.030 | 0.167 | 1 |
| `vol_20 div ret_10 mul vrate_5 div ret_10` | +0.19 | +0.025 | 0.171 | 1 |
| `vol_20 sub amp_1 mul vrate_5 div ret_1` | +0.18 | +0.020 | 0.329 | 1 |
| `vol_20 div ret_10 mul vrate_5 div ret_5` | +0.15 | +0.015 | 0.262 | 1 |
| `vol_20 div ret_10 mul vrate_5 sub high` | +0.14 | +0.015 | 0.175 | 1 |

## 方向4 · 微观结构挖掘 (Level-2 / Tick, 合成数据演示)

合成 6 只股票 tick 的微观结构特征 (耗时 0.0s)。**接入真实 Level-2 时需替换 demo 的 tick 输入, 特征函数可直接复用。**

| 股票 | trade_count_autocorr | single_trade_intercept | vp_euclidean | realized_spread | order_imbalance |
|---|---|---|---|---|---|
| SYN000 | -0.0014 | -4.5224 | 0.0102 | 0.0134 | 0.4795 |
| SYN001 | 0.0035 | -3.9926 | 0.0106 | 0.0227 | 0.5166 |
| SYN002 | 0.0031 | -5.4908 | 0.0102 | 0.0049 | 0.5197 |
| SYN003 | -0.0132 | -3.7480 | 0.0121 | 0.0268 | 0.5065 |
| SYN004 | -0.0066 | -4.2905 | 0.0124 | 0.0224 | 0.5031 |
| SYN005 | 0.0190 | -5.1136 | 0.0113 | 0.0056 | 0.5054 |

---
*总耗时 129.6s · 由 `factor_mining/run_mining.py` 自动生成*