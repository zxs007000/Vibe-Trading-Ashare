# XGBoost 因子挖掘 · WFA 原型 v4 全量版（v3 + 筹码结构维度，全市场）

- 样本: 250 只 | 面板: 341,326 行 | 特征: 217 个
- Alpha 17 + 市场环境 8 + 交互 168 + 拥挤度 18 + 筹码 6
- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；
  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过
- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；
  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)
- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项
- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 4 折
- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离

## 各折结果

- 折1.0: 单周期5日 AUC=0.533 | 融合 AUC=[0.529,0.535,0.526] IC=[+0.0054,+0.0272,+0.0058]
- 折2.0: 单周期5日 AUC=0.529 | 融合 AUC=[0.534,0.555,0.535] IC=[+0.0396,+0.0728,+0.0223]
- 折3.0: 单周期5日 AUC=0.523 | 融合 AUC=[0.515,0.545,0.538] IC=[+0.0443,+0.0853,+0.0841]
- 折4.0: 单周期5日 AUC=0.530 | 融合 AUC=[0.528,0.563,0.587] IC=[+0.0627,+0.1042,+0.1228]

## 汇总均值

- auc_single5: 0.5287
- ic_single5: -0.0043
- auc_fuse_5: 0.5265
- ic_fuse_5: 0.0380
- auc_fuse_20: 0.5493
- ic_fuse_20: 0.0724
- auc_fuse_60: 0.5466
- ic_fuse_60: 0.0587

## 因子重要性 Top20 (by Gain)

- ix_mkt_amp_mean__netprofit_yoy: gain=22.014 
- pca_absorp_pen: gain=21.656 
- ix_mkt_ret20_std__vol_20: gain=21.415 
- pca_absorp_x_macro: gain=20.486 
- ix_mkt_vol_mean__current_ratio: gain=17.915 
- ix_mkt_adv__current_ratio: gain=17.880 
- mkt_roe_mean: gain=17.562 
- pca_absorp_pct: gain=17.513 
- mkt_amp_mean: gain=17.501 
- crowd_mom: gain=17.028 【拥挤度】
- current_ratio: gain=16.969 
- ix_mkt_ret20_std__ma_dev_20: gain=16.959 
- ix_mkt_vol_mean__amp_20: gain=16.808 
- pca_absorp: gain=16.756 
- ix_mkt_ret20_std__netprofit_yoy: gain=16.605 
- ix_mkt_roe_mean__roe: gain=16.523 
- ix_mkt_vol_mean__netprofit_yoy: gain=16.397 
- ix_mkt_amp_mean__ret_5: gain=16.387 
- crowd_mom_x_macro: gain=16.188 【拥挤度】
- ix_mkt_vol_mean__roe: gain=16.170 

## 拥挤度基础指标相关性(§3.4)

- corr(crowd_mom,pca_absorp) = +0.058
- corr(crowd_liq,crowd_mom) = -0.005
- corr(crowd_liq,pca_absorp) = -0.630

## 特征Gain占比(末折重要性)

- 拥挤度: 0.031
- 筹码结构: 0.026

## 结论

在 v2(交互+多周期融合+IS调参) 基础上先加因子拥挤度(v3)，再加筹码结构维度(v4)；
对比 v1(AUC≈0.528)、v2 与 v3，观察两个新维度的边际贡献。
IC 略负属正常(原始因子含噪声)；重点看 AUC 是否提升及拥挤度/筹码特征重要性。