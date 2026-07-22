# XGBoost 模型 SHAP 解释性分析（修正版）

- 数据: /workspace/quant_proto/v4proto_out 落盘, WFA 4 折 OOS 子样本(各 <=3000 行)
- 尺度说明: SHAP 值为 **log-odds(边际)** 贡献, 非概率; 标签为『未来收益排前30%的概率』(**排序/相对**, 非绝对收益方向)。
- base value(正类, log-odds) 各折: 1:-0.841, 2:-0.841, 3:-0.839, 4:-0.834

## 一、特征族总贡献 (Mean|SHAP| 归族, 缓和共线导致的信用瓜分)

- `ix_*`: 0.0628
- `mkt_*`: 0.0066
- `crowd_mom*`: 0.0035
- `pca_absorp*`: 0.0034
- `bvps*`: 0.0030
- `chip_*`: 0.0017
- `crowd_liq*`: 0.0015
- `amp*`: 0.0013
- `roe*`: 0.0009
- `netprofit*`: 0.0006
- `ret_*`: 0.0005
- `ocf*`: 0.0005
- `amt_*`: 0.0005
- `debt*`: 0.0004
- `eps*`: 0.0004

## 二、Top 单特征 Mean|SHAP| (log-odds 尺度)

- `mkt_vol_mean`: 0.0045
- `ix_mkt_amp_mean__netprofit_yoy`: 0.0032
- `bvps`: 0.0030
- `crowd_mom`: 0.0025
- `ix_mkt_roe_mean__chip_cc`: 0.0022
- `ix_mkt_vol_mean__ret_60`: 0.0019
- `ix_mkt_vol_mean__netprofit_yoy`: 0.0017
- `ix_mkt_roe_mean__ocf_netprofit`: 0.0017
- `pca_absorp_pct`: 0.0016
- `ix_mkt_roe_mean__roe`: 0.0016
- `ix_mkt_adv__netprofit_yoy`: 0.0015
- `ix_mkt_ret20_std__vol_20`: 0.0015
- `amp_20`: 0.0013
- `ix_mkt_ret20_std__roe`: 0.0013
- `pca_absorp`: 0.0013
- `ix_mkt_ret20_std__chip_cc`: 0.0012
- `ix_mkt_roe_mean__ret_20`: 0.0011
- `ix_mkt_ret20_std__chip_cb_short`: 0.0011
- `ix_mkt_amp_mean__bvps`: 0.0010
- `ix_mkt_ret20_std__bvps`: 0.0010
- `ix_mkt_roe_mean__ret_60`: 0.0010
- `ix_mkt_ret20_mean__eps`: 0.0010
- `ix_mkt_roe_mean__bvps`: 0.0009
- `ix_mkt_amp_mean__ret_60`: 0.0009
- `ix_mkt_roe_mean__chip_cb`: 0.0009

## 三、跨折方向一致性 (稳健性, 非均值)

> 判定: 某因子在多数折中 median(SHAP) 同号 => 方向稳定; flip_folds 多 => 疑似伪Alpha(应剔除)。

| 因子 | 稳定方向 | 正向折 | 负向折 | 翻转折 | Mean|SHAP| |
|---|---|---|---|---|---|
| `mkt_vol_mean` | 正向 | 2 | 2 | 2 | 0.0045 |
| `ix_mkt_amp_mean__netprofit_yoy` | 正向 | 2 | 2 | 2 | 0.0032 |
| `bvps` | 正向 | 2 | 2 | 2 | 0.0030 |
| `crowd_mom` | 正向 | 2 | 2 | 2 | 0.0025 |
| `ix_mkt_roe_mean__chip_cc` | 负向 | 1 | 3 | 1 | 0.0022 |
| `ix_mkt_vol_mean__ret_60` | 负向 | 0 | 4 | 0 | 0.0019 |
| `ix_mkt_vol_mean__netprofit_yoy` | 正向 | 2 | 2 | 2 | 0.0017 |
| `ix_mkt_roe_mean__ocf_netprofit` | 正向 | 2 | 2 | 2 | 0.0017 |
| `pca_absorp_pct` | 正向 | 2 | 2 | 2 | 0.0016 |
| `ix_mkt_roe_mean__roe` | 负向 | 0 | 4 | 0 | 0.0016 |
| `ix_mkt_adv__netprofit_yoy` | 负向 | 0 | 4 | 0 | 0.0015 |
| `ix_mkt_ret20_std__vol_20` | 正向 | 4 | 0 | 0 | 0.0015 |
| `amp_20` | 正向 | 3 | 1 | 1 | 0.0013 |
| `ix_mkt_ret20_std__roe` | 负向 | 1 | 3 | 1 | 0.0013 |
| `pca_absorp` | 正向 | 2 | 2 | 2 | 0.0013 |
| `ix_mkt_ret20_std__chip_cc` | 负向 | 0 | 4 | 0 | 0.0012 |
| `ix_mkt_roe_mean__ret_20` | 正向 | 3 | 1 | 1 | 0.0011 |
| `ix_mkt_ret20_std__chip_cb_short` | 正向 | 3 | 1 | 1 | 0.0011 |
| `ix_mkt_amp_mean__bvps` | 负向 | 1 | 3 | 1 | 0.0010 |
| `ix_mkt_ret20_std__bvps` | 正向 | 2 | 2 | 2 | 0.0010 |
| `ix_mkt_roe_mean__ret_60` | 负向 | 0 | 4 | 0 | 0.0010 |
| `ix_mkt_ret20_mean__eps` | 正向 | 2 | 2 | 2 | 0.0010 |
| `ix_mkt_roe_mean__bvps` | 正向 | 3 | 1 | 1 | 0.0009 |
| `ix_mkt_amp_mean__ret_60` | 负向 | 0 | 4 | 0 | 0.0009 |
| `ix_mkt_roe_mean__chip_cb` | 正向 | 3 | 1 | 1 | 0.0009 |

## 四、关键因子依赖图(带 bootstrap CI)

- `pca_absorp_pen`: 依赖图见 shap_dep_pca_absorp_pen.png（含跨折 bootstrap CI）
- `crowd_mom_pct`: 依赖图见 shap_dep_crowd_mom_pct.png（含跨折 bootstrap CI）
- `pca_absorp_x_macro`: 依赖图见 shap_dep_pca_absorp_x_macro.png（含跨折 bootstrap CI）

## 五、交互解耦 (shap_interaction_values)

末折交互贡献 Top8 (|SHAP interaction| 合计, 已剔除自交互):
  - `ix_mkt_ret20_std__vol_20`: 21.1069
  - `ix_mkt_amp_mean__netprofit_yoy`: 17.6955
  - `crowd_mom`: 13.0517
  - `ix_mkt_vol_mean__netprofit_yoy`: 12.0638
  - `crowd_mom_pct`: 11.9864
  - `ix_mkt_ret20_mean__ret_60`: 11.9421
  - `ix_mkt_ret20_std__netprofit_yoy`: 11.7340
  - `ix_mkt_roe_mean__ret_5`: 11.7059

## 六、结论与警示

- SHAP 仅解释**模型逻辑**, 不等同真值; 须与 OOS 的 AUC/IC 一并看。
- 同源特征族(pca_absorp_*/crowd_mom_*) 的 Mean|SHAP| 已归族汇总, 单特征方向可能被共线稀释, 解读单特征时谨慎。
- 依赖图尾部(>0.8/0.9 分位)样本稀疏, '断崖'多为噪声, **不得直接作为硬性止损线**; 阈值须带 CI 并经样本外验证。
- 标签为排序概率, SHAP 推高的是『进入前30%收益组』的概率, 不是绝对涨跌, 交易解读勿混淆。