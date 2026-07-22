# XGBoost 因子挖掘 · WFA 原型结果

- 样本股票数: 400（三表齐全子集，MAX_STOCKS=400）
- 面板规模: 674,528 行
- 特征: 17 个（量价8 + 基本面9）
- 标签: 排序打分（截面前30%未来5日收益=1，T+1起算）
- WFA: 训练3年/测试1年/步长1年, 共4折

## WFA 各折结果

- 折1: AUC=0.529, 合并IC=-0.0463
- 折2: AUC=0.529, 合并IC=-0.0423
- 折3: AUC=0.523, 合并IC=-0.0197
- 折4: AUC=0.532, 合并IC=-0.0113
- **均值 AUC=0.528**, **均值IC=-0.0299**

## 因子重要性 (Top, by Gain)

- vol_20: gain=27.656, weight=487
- netprofit_yoy: gain=22.119, weight=500
- gross_margin: gain=21.642, weight=525
- eps: gain=21.573, weight=456
- net_margin: gain=21.527, weight=366
- debt_ratio: gain=20.575, weight=462
- operate_income_yoy: gain=19.742, weight=449
- ocf_netprofit: gain=19.551, weight=526
- amp_20: gain=19.246, weight=493
- ret_60: gain=19.212, weight=463
- ret_20: gain=18.078, weight=385
- ma_dev_20: gain=17.908, weight=317

## 结论

链路验证：数据湖 → 特征 → WFA → XGBoost → AUC/IC/因子重要性 已跑通。
均值 AUC=0.528（>0.5 有效，>0.6 合格，>0.7 优秀），均值IC=-0.0299（>0 方向有效）。
注：原型仅用部分数据(三表齐全子集)+固定超参，未做参数寻优；全量三大表到位后可扩展至全市场并扩充交互特征。