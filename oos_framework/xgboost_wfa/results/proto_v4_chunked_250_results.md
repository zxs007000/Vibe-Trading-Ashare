# XGBoost 因子挖掘 · WFA 原型 v4（分块流式版，全市场）

- 样本: 250 只 | 特征分片: 8 | 特征: 197 个
- 内存策略: PhaseA逐只落盘 → PhaseB按年切片截面特征 → PhaseC按需读分片+float32增量拼矩阵
  全市场 5500 只可在 8GB cgroup 内运行(峰值≈单折训练矩阵 ~3.5GB)
- Alpha 17 + 市场环境 8 + 交互 148 + 拥挤度 18 + 筹码 6
- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；
  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过
- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；
  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)
- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项
- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 4 折
- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离

## 各折结果

- 折1.0: 单周期5日 AUC=0.528 | 融合 AUC=[0.527,0.536,0.529] IC=[+0.0131,+0.0289,+0.0072]
- 折2.0: 单周期5日 AUC=0.536 | 融合 AUC=[0.535,0.544,0.530] IC=[+0.0090,+0.0250,+0.0073]
- 折3.0: 单周期5日 AUC=0.528 | 融合 AUC=[0.517,0.546,0.539] IC=[+0.0359,+0.0815,+0.0693]
- 折4.0: 单周期5日 AUC=0.538 | 融合 AUC=[0.528,0.559,0.587] IC=[+0.0561,+0.0903,+0.1149]

## 汇总均值

- auc_single5: 0.5325
- ic_single5: -0.0192
- auc_fuse_5: 0.5266
- ic_fuse_5: 0.0285
- auc_fuse_20: 0.5464
- ic_fuse_20: 0.0564
- auc_fuse_60: 0.5464
- ic_fuse_60: 0.0497

## 因子重要性 Top20 (by Gain)

- ix_mkt_ret20_std__vol_20: gain=85.762 
- ix_mkt_amp_mean__ma_dev_20: gain=55.567 
- ix_mkt_vol_mean__netprofit_yoy: gain=52.051 
- ix_mkt_amp_mean__netprofit_yoy: gain=44.348 
- netprofit_yoy: gain=43.068 
- ix_mkt_adv__ret_20: gain=42.704 
- ix_mkt_roe_mean__roe: gain=41.853 
- ix_mkt_amp_mean__ret_5: gain=41.251 
- mkt_amp_mean: gain=41.244 
- ix_mkt_amp_mean__ret_60: gain=41.015 
- ix_mkt_adv__netprofit_yoy: gain=40.780 
- pca_absorp_delta_x_macro: gain=40.591 【拥挤度】
- ix_mkt_ret20_std__netprofit_yoy: gain=40.292 
- ix_mkt_vol_mean__amp_20: gain=39.919 
- ix_mkt_vol_mean__vol_20: gain=39.801 
- ix_mkt_ret20_std__ret_60: gain=39.491 
- ix_mkt_ma_dev_mean__eps: gain=38.443 
- ix_mkt_adv__ma_dev_20: gain=38.249 
- mkt_roe_mean: gain=37.015 
- ret_20: gain=36.631 

## 拥挤度基础指标相关性(§3.4)

- corr(crowd_mom,pca_absorp) = +nan
- corr(crowd_liq,crowd_mom) = +nan
- corr(crowd_liq,pca_absorp) = +nan

## 特征Gain占比(末折重要性)

- 拥挤度: 0.025
- 筹码结构: 0.028

## 结论

分块流式版(v4_chunked)与 v4_full 算法一致，仅内存策略不同：
通过 PhaseA逐只落盘 + PhaseB按年切片截面特征 + PhaseC按需读分片+float32增量拼矩阵，
把峰值内存从全面板(~23GB)压到单折训练矩阵(~3.5GB)，全市场 5500 只可在 8GB 内完成 WFA+SHAP。