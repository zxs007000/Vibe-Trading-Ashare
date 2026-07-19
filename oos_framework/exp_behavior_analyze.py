"""exp_behavior_analyze.py — 基于已落盘的行为面板做 IC 显著性 + 正交性诊断"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
panel = pd.read_parquet(os.path.join(CACHE, "behavior_panel.parquet"))
FACTORS = ["max1","max3","max5","skew","ivol","lottery","anchor52",
           "anchor_int","zt_dist","cgo","avol"]

def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    n = len(a)
    if n < 5: return np.nan
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    da = ra - ra.mean(); db = rb - rb.mean()
    den = np.sqrt(float((da**2).sum()) * float((db**2).sum()))
    return float((da*db).sum()/den) if den > 0 else np.nan

rows = []
for f in FACTORS:
    ic20 = panel.groupby("date").apply(lambda x: _spearman(x[f].values, x["fwd20"].values))
    ic60 = panel.groupby("date").apply(lambda x: _spearman(x[f].values, x["fwd60"].values))
    for lbl, ic in (("IC20", ic20), ("IC60", ic60)):
        ic = ic.dropna()
        mean = ic.mean(); sd = ic.std(); n = len(ic)
        t = mean/(sd/np.sqrt(n)) if sd > 0 else np.nan
        pos = (ic > 0).mean()
        rows.append((f, lbl, round(mean,4), round(sd,4), n, round(t,2), round(pos,3)))
df = pd.DataFrame(rows, columns=["factor","horizon","IC","IC_std","ndays","tstat","pct_pos"])

print("\n" + "="*86)
print("行为因子 IC 显著性 (截面rank-IC, PIT) — HS300")
print("="*86)
print(df.to_string(index=False))
print("-"*86)
print("tstat: |t|>2 近似显著; pct_pos: 正IC天数占比(>0.55 偏单向)")
print("\n因子相关系数矩阵 (日频截面z后按股均值):")
sub = panel.dropna(subset=FACTORS)
corr = sub[FACTORS].corr().round(2)
print(corr.to_string())
