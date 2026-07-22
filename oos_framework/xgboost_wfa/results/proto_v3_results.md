# XGBoost 因子挖掘 · WFA 原型 v3（v2 + 因子拥挤度维度）

- 样本: 400 只 | 面板: 654,618 行 | 特征: 179 个
- Alpha 18 + 市场环境 8 + 交互 135 + 拥挤度 18
- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；
  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过
- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项
- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 4 折
- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离

## 各折结果

- 折1.0: 单周期5日 AUC=0.525 | 融合 AUC=[0.531,0.537,0.511] IC=[+0.0399,+0.0346,-0.0117]
- 折2.0: 单周期5日 AUC=0.522 | 融合 AUC=[0.520,0.533,0.545] IC=[+0.0240,+0.0384,+0.0226]
- 折3.0: 单周期5日 AUC=0.531 | 融合 AUC=[0.530,0.550,0.548] IC=[+0.0558,+0.0948,+0.0737]
- 折4.0: 单周期5日 AUC=0.514 | 融合 AUC=[0.521,0.541,0.536] IC=[+0.0506,+0.0823,+0.0616]

## 汇总均值

- auc_single5: 0.5231
- ic_single5: -0.0024
- auc_fuse_5: 0.5253
- ic_fuse_5: 0.0426
- auc_fuse_20: 0.5402
- ic_fuse_20: 0.0626
- auc_fuse_60: 0.5350
- ic_fuse_60: 0.0366

## 因子重要性 Top20 (by Gain)

- pca_absorp_x_macro: gain=28.078 
- pca_absorp_pen: gain=21.942 
- mkt_amp_mean: gain=19.433 
- mkt_gm_mean: gain=19.161 
- ix_mkt_amp_mean__ret_60: gain=18.455 
- crowd_liq_delta_x_macro: gain=17.976 
- ix_mkt_gm_mean__netprofit_yoy: gain=17.203 
- ix_mkt_ret20_std__vol_20: gain=17.149 
- ix_mkt_gm_mean__eps: gain=16.398 
- crowd_mom_pct: gain=16.335 【拥挤度】
- pca_absorp: gain=16.261 
- crowd_liq: gain=15.909 
- pca_absorp_pct: gain=15.810 
- crowd_mom: gain=15.733 【拥挤度】
- crowd_mom_x_macro: gain=15.488 【拥挤度】
- ix_mkt_amp_mean__vol_20: gain=15.394 
- crowd_mom_roc: gain=15.390 【拥挤度】
- ix_mkt_gm_mean__net_margin: gain=14.770 
- ix_mkt_vol_mean__eps: gain=14.735 
- crowd_liq_pct: gain=14.720 

## 拥挤度基础指标相关性(§3.4)

- corr(crowd_mom,pca_absorp) = +0.075
- corr(crowd_liq,crowd_mom) = -0.069
- corr(crowd_liq,pca_absorp) = -0.512

## 结论

在 v2(交互+多周期融合+IS调参) 基础上加入因子拥挤度维度；
对比 v1(AUC≈0.528) 与 v2，观察拥挤度维度的边际贡献。
IC 略负属正常(原始因子拥挤/噪声)；重点看 AUC 是否提升及拥挤度特征重要性。