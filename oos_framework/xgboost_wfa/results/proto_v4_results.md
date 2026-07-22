# XGBoost 因子挖掘 · WFA 原型 v4（v3 + 筹码结构维度）

- 样本: 400 只 | 面板: 654,618 行 | 特征: 217 个
- Alpha 18 + 市场环境 8 + 交互 167 + 拥挤度 18 + 筹码 6
- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；
  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过
- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；
  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)
- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项
- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 4 折
- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离

## 各折结果

- 折1.0: 单周期5日 AUC=0.526 | 融合 AUC=[0.533,0.548,0.531] IC=[+0.0411,+0.0515,+0.0403]
- 折2.0: 单周期5日 AUC=0.528 | 融合 AUC=[0.524,0.540,0.569] IC=[+0.0250,+0.0455,+0.0517]
- 折3.0: 单周期5日 AUC=0.536 | 融合 AUC=[0.533,0.551,0.549] IC=[+0.0628,+0.1189,+0.0897]
- 折4.0: 单周期5日 AUC=0.516 | 融合 AUC=[0.522,0.549,0.551] IC=[+0.0631,+0.1092,+0.0941]

## 汇总均值

- auc_single5: 0.5264
- ic_single5: 0.0098
- auc_fuse_5: 0.5277
- ic_fuse_5: 0.0480
- auc_fuse_20: 0.5470
- ic_fuse_20: 0.0813
- auc_fuse_60: 0.5499
- ic_fuse_60: 0.0690

## 因子重要性 Top20 (by Gain)

- pca_absorp_pen: gain=30.263 
- mkt_gm_mean: gain=25.125 
- mkt_amp_mean: gain=24.643 
- ix_mkt_amp_mean__ret_60: gain=23.627 
- ix_mkt_gm_mean__netprofit_yoy: gain=22.137 
- pca_absorp_x_macro: gain=21.596 
- ix_mkt_ret20_std__vol_20: gain=20.927 
- ix_mkt_gm_mean__chip_cb: gain=19.625 
- crowd_liq_delta_x_macro: gain=19.588 
- pca_absorp: gain=19.549 
- crowd_mom: gain=19.434 【拥挤度】
- crowd_liq: gain=19.333 
- crowd_mom_pct: gain=19.214 【拥挤度】
- pca_absorp_pct: gain=19.100 
- crowd_mom_x_macro: gain=19.003 【拥挤度】
- crowd_mom_roc: gain=18.222 【拥挤度】
- ix_mkt_vol_mean__eps: gain=17.477 
- crowd_liq_pct: gain=17.441 
- ix_mkt_gm_mean__eps: gain=17.367 
- ix_mkt_gm_mean__vol_20: gain=17.236 

## 拥挤度基础指标相关性(§3.4)

- corr(crowd_mom,pca_absorp) = +0.075
- corr(crowd_liq,crowd_mom) = -0.069
- corr(crowd_liq,pca_absorp) = -0.512

## 特征Gain占比(末折重要性)

- 拥挤度: 0.032
- 筹码结构: 0.025

## 结论

在 v2(交互+多周期融合+IS调参) 基础上先加因子拥挤度(v3)，再加筹码结构维度(v4)；
对比 v1(AUC≈0.528)、v2 与 v3，观察两个新维度的边际贡献。
IC 略负属正常(原始因子含噪声)；重点看 AUC 是否提升及拥挤度/筹码特征重要性。