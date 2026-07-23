*生成耗时 10.4s, 样本 100 只*

# 因子图书馆 · 生命周期状态

- 因子总数: **40** | 当前冻结: **21** | 衰减: **4**
- 冻结点: 2021-10-23 | 再冻结次数: 0 | 最近监控: 2026-04-21

## 冻结因子集(ICIR 权重)

| 因子 | 家族 | IS_IC | ICIR | 权重 | 近期IC | 状态 |
|---|---|---|---|---|---|---|
| `cs_rank(ts_std(dist_low_20,5) sub amount)` | 未知 | 0.0494 | 4.095 | 0.0455 | +0.0532 | ✅ |
| `cs_rank(vol_20 sub amount)` | 未知 | 0.0492 | 4.077 | 0.0456 | +0.0532 | ✅ |
| `cs_rank(vol_20 div amount)` | 未知 | 0.0407 | 3.405 | 0.0376 | -0.0068 | ⚠️衰减 |
| `0.0 sub cs_rank(ret_20) mul cs_rank(vol_20)` | 未知 | 0.0494 | 4.413 | 0.0561 | +0.1138 | ✅ |
| `ts_min(ret_1 sub amount div ret_5 div log_ret_1,10` | 未知 | 0.0323 | 2.912 | 0.0227 | +0.0471 | ✅ |
| `0.0 sub amp_1 corr dist_low_20` | 未知 | 0.0432 | 4.936 | 0.0588 | +0.1027 | ✅ |
| `0.0 sub cs_rank(amp_1) mul cs_rank(vol_20)` | 未知 | 0.0643 | 4.971 | 0.0685 | +0.1091 | ✅ |
| `0.0 sub cs_rank(amp_1) mul cs_rank(dist_low_20)` | 未知 | 0.0597 | 5.11 | 0.0707 | +0.0975 | ✅ |
| `0.0 sub cs_rank(dist_low_20) mul cs_rank(vol_20)` | 未知 | 0.0552 | 4.642 | 0.0636 | +0.1261 | ✅ |
| `cs_rank(vol_20 sub ret_20)` | 未知 | 0.0313 | 2.491 | 0.0392 | +0.0559 | ✅ |
| `0.0 sub ret_20 div vol_20` | 未知 | 0.0318 | 2.683 | 0.0409 | +0.0259 | ✅ |
| `0.0 sub amp_1 div dist_low_20` | 未知 | 0.0628 | 5.483 | 0.0783 | +0.0823 | ✅ |
| `0.0 sub cs_rank(vol_20) mul cs_rank(vrate_20)` | 未知 | 0.0595 | 5.586 | 0.0782 | +0.0696 | ✅ |
| `cs_rank(vol_20 sub dist_low_20)` | 未知 | 0.0397 | 3.27 | 0.0488 | +0.0688 | ✅ |
| `cs_rank(vol_20 sub volume)` | 未知 | 0.0437 | 4.141 | 0.0395 | +0.0919 | ✅ |
| `dist_low_20 div vol_20` | 未知 | 0.0523 | 4.034 | 0.0506 | +0.1232 | ✅ |
| `cs_rank(vol_20 sub ret_10)` | 未知 | 0.0255 | 2.029 | 0.0354 | +0.0452 | ✅ |
| `cs_rank(vol_20 sub range_20)` | 未知 | 0.0422 | 3.326 | 0.0442 | +0.0875 | ✅ |
| `cs_rank(vol_20 sub high)` | 未知 | 0.0178 | 1.61 | 0.0268 | -0.0886 | ⚠️衰减 |
| `cs_rank(vol_20 sub open)` | 未知 | 0.0162 | 1.466 | 0.0247 | -0.0902 | ⚠️衰减 |
| `cs_rank(vol_20 sub low)` | 未知 | 0.0157 | 1.424 | 0.0244 | -0.0905 | ⚠️衰减 |

## 全因子健康度

- `0.0 sub cs_rank(amp_1) mul cs_rank(vol_20)`: IC=0.0643 近期IC=0.1091 ✅
- `0.0 sub amp_1 div dist_low_20`: IC=0.0628 近期IC=0.0823 ✅
- `0.0 sub cs_rank(amp_1) mul cs_rank(dist_low_20)`: IC=0.0597 近期IC=0.0975 ✅
- `0.0 sub cs_rank(vol_20) mul cs_rank(vrate_20)`: IC=0.0595 近期IC=0.0696 ✅
- `0.0 sub cs_rank(dist_low_20) mul cs_rank(vol_20)`: IC=0.0552 近期IC=0.1261 ✅
- `dist_low_20 div vol_20`: IC=0.0523 近期IC=0.1232 ✅
- `cs_rank(ts_std(dist_low_20,5) sub amount)`: IC=0.0494 近期IC=0.0532 ✅
- `0.0 sub cs_rank(ret_20) mul cs_rank(vol_20)`: IC=0.0494 近期IC=0.1138 ✅
- `cs_rank(vol_20 sub amount)`: IC=0.0492 近期IC=0.0532 ✅
- `cs_rank(vol_20 sub volume)`: IC=0.0437 近期IC=0.0919 ✅
- `0.0 sub amp_1 corr dist_low_20`: IC=0.0432 近期IC=0.1027 ✅
- `cs_rank(vol_20 sub range_20)`: IC=0.0422 近期IC=0.0875 ✅
- `cs_rank(vol_20 div amount)`: IC=0.0407 近期IC=-0.0068 ⚠️衰减
- `cs_rank(vol_20 sub dist_low_20)`: IC=0.0397 近期IC=0.0688 ✅
- `ts_min(ret_1 sub amount div ret_5 div log_ret_1,10)`: IC=0.0323 近期IC=0.0471 ✅
- `0.0 sub ret_20 div vol_20`: IC=0.0318 近期IC=0.0259 ✅
- `cs_rank(vol_20 sub ret_20)`: IC=0.0313 近期IC=0.0559 ✅
- `cs_rank(vol_20 sub ret_10)`: IC=0.0255 近期IC=0.0452 ✅
- `cs_rank(vol_20 sub high)`: IC=0.0178 近期IC=-0.0886 ⚠️衰减
- `cs_rank(vol_20 sub open)`: IC=0.0162 近期IC=-0.0902 ⚠️衰减
- `cs_rank(vol_20 sub low)`: IC=0.0157 近期IC=-0.0905 ⚠️衰减
- `cs_rank(chip_conc90 sub amount mul chip_conc70)`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) mul 1.0 sub cs_rank(ret_20)`: IC=None 近期IC=None ✅
- `0.0 sub cs_rank(chip_cost_dev) mul cs_rank(vol_20)`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) mul 1.0 sub cs_rank(dist_low_20)`: IC=None 近期IC=None ✅
- `0.0 sub cs_rank(chip_conc90) mul cs_rank(chip_cost_dev)`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) mul 1.0 sub cs_rank(vol_20)`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) sub cs_rank(ret_20)`: IC=None 近期IC=None ✅
- `0.0 sub chip_cost_dev div vol_20`: IC=None 近期IC=None ✅
- `chip_disp div vol_20`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) sub cs_rank(dist_low_20)`: IC=None 近期IC=None ✅
- `0.0 sub chip_conc90 div chip_disp`: IC=None 近期IC=None ✅
- `cs_rank(chip_disp) sub cs_rank(vol_20)`: IC=None 近期IC=None ✅
- `0.0 sub cs_rank(chip_conc90) mul cs_rank(vol_20)`: IC=None 近期IC=None ✅
- `0.0 sub cs_rank(chip_conc90) sub cs_rank(chip_disp)`: IC=None 近期IC=None ✅
- `cs_rank(vol_20 sub chip_cost_dev)`: IC=None 近期IC=None ✅
- `cs_rank(vol_20 sub chip_profit_ratio)`: IC=None 近期IC=None ✅
- `ts_max(chip_conc70,10) sub high sub cs_rank(ts_rank(dis`: IC=None 近期IC=None ✅
- `cs_rank(ts_min(ret_20 sub chip_profit_ratio,20))`: IC=None 近期IC=None ✅
- `ts_rank(ret_10,20) sub high sub chip_conc70`: IC=None 近期IC=None ✅