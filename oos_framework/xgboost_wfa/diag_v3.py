#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 v3 拥挤度：是否为全 NaN / 日期覆盖 / dropna 后存活行数。"""
import numpy as np, pandas as pd
import xgb_wfa_proto_v3 as M

codes = M.eligible_codes()[:30]
print("codes:", len(codes))
panels = [M.build_stock_panel(c) for c in codes]
panel = pd.concat([p for p in panels if p is not None], ignore_index=True)
print("panel rows:", len(panel), "date", panel['date'].min(), "~", panel['date'].max())

env = M.market_env(panel)
base, crowd = M.build_crowding(panel, env)
print("\n=== base_crowd non-NaN per column ===")
print(base.notna().sum())
print("base date range:", base.index.min(), "~", base.index.max())
print("\n=== crowd_feat non-NaN per column (head) ===")
print(crowd.notna().sum())
print("crowd date range:", crowd.index.min(), "~", crowd.index.max())

# 模拟 main 中的 dropna 行为
panel = panel.drop(columns=["_amount", "_ret"], errors="ignore")
from xgb_wfa_proto_v3 import ALPHA_FEATS
M.winsorize(panel, ALPHA_FEATS)
M.cross_section_zscore(panel, ALPHA_FEATS)
panel, FEATS = M.add_interactions(panel, env)
_kdate = panel["date"].astype("int64").values
_cf = crowd.copy(); _cf.index = _cf.index.astype("int64")
_cf_dates = _cf.index
for col in list(crowd.columns):
    panel[col] = pd.Series(_kdate).map(dict(zip(_cf_dates, _cf[col]))).values
FEATS = FEATS + list(crowd.columns)
print("\nFEATS count:", len(FEATS))
print("panel rows before dropna:", len(panel))
surv = panel.dropna(subset=FEATS)
print("panel rows after dropna(subset=FEATS):", len(surv))
print("surv date range:", surv['date'].min(), "~", surv['date'].max())
# 逐折看
folds = M.wfa_folds(panel["date"])
for i,(is_s,is_e,oos_s,oos_e) in enumerate(folds,1):
    isd = panel[(panel.date>=is_s)&(panel.date<is_e)].dropna(subset=FEATS)
    osd = panel[(panel.date>=oos_s)&(panel.date<oos_e)].dropna(subset=FEATS)
    print(f"fold{i}: IS rows={len(isd)} OOS rows={len(osd)}")
print("\nsample crowd row (last valid):")
print(crowd.dropna().tail(2).to_string())
