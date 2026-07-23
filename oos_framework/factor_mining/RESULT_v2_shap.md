# P6 · SHAP 可解释性验证 (三维度 v2 · 末折 2024-25 近期)

- 解释模型: 末折 IS 2021-10~2024-10 训练(目标 20日), OOS 40000 样本 TreeExplainer 精确 SHAP
- dir_corr>0=因子值越大越利多(SHAP↑), <0=越大越利空; 验证是否符合经济直觉

## SHAP Top15 (按 mean|SHAP|)

| 排名 | 因子 | mean|SHAP| | 方向corr | gain排名 | chip? |
|---|---|---|---|---|---|
| 1 | `ts_max(chip_conc70,10) sub high sub cs_rank(ts_rank(dist_high_20,60))` | 0.1083 | +0.41 | 3 |  |
| 2 | `0.0 sub chip_cost_dev div vol_20` | 0.0564 | -0.81 | 16 |  |
| 3 | `chip_disp` | 0.0563 | +0.54 | 8 | ✓ |
| 4 | `0.0 sub cs_rank(chip_cost_dev) mul cs_rank(vol_20)` | 0.0560 | +0.44 | 15 |  |
| 5 | `chip_conc90` | 0.0539 | -0.53 | 9 | ✓ |
| 6 | `ts_rank(ret_10,20) sub high sub chip_conc70` | 0.0484 | +0.45 | 4 |  |
| 7 | `0.0 sub ret_20 div vol_20` | 0.0405 | -0.72 | 37 |  |
| 8 | `chip_profit_ratio` | 0.0405 | +0.48 | 12 | ✓ |
| 9 | `0.0 sub cs_rank(chip_conc90) sub cs_rank(chip_disp)` | 0.0394 | +0.63 | 7 |  |
| 10 | `cs_rank(chip_disp) mul 1.0 sub cs_rank(ret_20)` | 0.0384 | +0.60 | 1 |  |
| 11 | `cs_rank(vol_20 sub ret_20)` | 0.0376 | +0.77 | 36 |  |
| 12 | `cs_rank(vol_20 div amount)` | 0.0360 | +0.38 | 2 |  |
| 13 | `ts_min(ret_1 sub amount div ret_5 div log_ret_1,10)` | 0.0332 | +0.07 | 10 |  |
| 14 | `cs_rank(chip_disp) sub cs_rank(vol_20)` | 0.0325 | -0.78 | 35 |  |
| 15 | `0.0 sub cs_rank(ret_20) mul cs_rank(vol_20)` | 0.0324 | +0.28 | 6 |  |

## 筹码(chip)6 维 SHAP 表现

| 因子 | SHAP排名 | mean|SHAP| | 方向corr |
|---|---|---|---|
| `chip_disp` | 3 | 0.0563 | +0.54 |
| `chip_conc90` | 5 | 0.0539 | -0.53 |
| `chip_profit_ratio` | 8 | 0.0405 | +0.48 |
| `chip_disp div vol_20` | 17 | 0.0280 | +0.47 |
| `chip_skew` | 18 | 0.0277 | -0.30 |
| `chip_cost_dev` | 22 | 0.0252 | -0.72 |
| `chip_conc70` | 25 | 0.0248 | +0.31 |

- **SHAP 排序 vs gain 排序 Spearman = 0.57** (中等)
- chip 6 维平均 SHAP 排名: 14 / 46

*耗时 6.3min · shap_validate_v2.py*