# XGBoost 因子挖掘 · WFA 原型 v4（分块流式版，全市场）

- 样本: 5445 只 | 特征分片: 8 | 特征: 218 个
- 内存策略: PhaseA逐只落盘 → PhaseB按年切片(回看5年历史算截面) → PhaseC按需读分片+float32+IS子抽样(≤500,000)
  全市场 5500 只可在 8GB cgroup 内运行(峰值≈训练矩阵 ~1-2GB + DMatrix ~1-2GB)
- Alpha 17 + 市场环境 8 + 交互 169 + 拥挤度 18 + 筹码 6
- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；
  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过
- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；
  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)
- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项
- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 4 折
- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离
- 注: IS 训练做 ≤500,000 行随机子抽样以控内存(全市场横截面特征已含所有股票, 不影响覆盖)

## 各折结果

- 折1.0: 单周期5日 AUC=0.535 | 融合 AUC=[0.538,0.541,0.522] IC=[-0.0017,-0.0079,-0.0349]
- 折2.0: 单周期5日 AUC=0.534 | 融合 AUC=[0.538,0.538,0.530] IC=[+0.0109,+0.0021,-0.0206]
- 折3.0: 单周期5日 AUC=0.529 | 融合 AUC=[0.525,0.552,0.550] IC=[+0.0487,+0.0758,+0.0378]
- 折4.0: 单周期5日 AUC=0.536 | 融合 AUC=[0.522,0.553,0.570] IC=[+0.0543,+0.0866,+0.1023]

## 汇总均值

- auc_single5: 0.5335
- ic_single5: -0.0135
- auc_fuse_5: 0.5307
- auc_fuse_20: 0.5459
- auc_fuse_60: 0.5429
- ic_fuse_5: 0.0280
- ic_fuse_20: 0.0392
- ic_fuse_60: 0.0211

## 因子重要性 Top20 (by Gain)

- pca_absorp_pen: gain=60.318 【拥挤度】
- ix_mkt_vol_mean__vol_20: gain=48.330 
- ix_mkt_amp_mean__ret_60: gain=41.725 
- crowd_liq_pct: gain=37.696 【拥挤度】
- mkt_amp_mean: gain=36.333 
- netprofit_yoy: gain=34.166 
- mkt_roe_mean: gain=33.640 
- crowd_mom_pct: gain=31.266 【拥挤度】
- crowd_liq_x_macro: gain=30.975 【拥挤度】
- crowd_mom: gain=28.921 【拥挤度】
- ix_mkt_amp_mean__vol_20: gain=28.739 
- mkt_vol_mean: gain=28.142 
- pca_absorp_x_macro: gain=27.906 【拥挤度】
- crowd_liq: gain=27.609 【拥挤度】
- pca_absorp_pct: gain=27.373 【拥挤度】
- crowd_mom_delta_x_macro: gain=26.859 【拥挤度】
- ix_mkt_roe_mean__vol_20: gain=26.387 
- mkt_ret20_std: gain=25.853 
- pca_absorp: gain=25.403 【拥挤度】
- ix_mkt_ret20_std__vol_20: gain=25.167 

## 拥挤度基础指标相关性(§3.4)


## 特征Gain占比(末折重要性)

- 拥挤度: 0.129
- 筹码结构: 0.018

## 结论

分块流式版(v4_chunked)与 v4_full 算法一致，仅内存策略不同：
PhaseA逐只落盘 + PhaseB按年切片(回看5年历史以支撑 ZYEARS/PCT_WIN 滚动) + PhaseC按需读分片+float32+IS子抽样，
把峰值内存从全面板(~23GB)压到单折训练矩阵(~1-2GB)，全市场 5500 只可在 8GB 内完成 WFA+SHAP。