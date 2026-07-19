"""exp_behavior_factor.py — 行为金融因子试点 (锚定/处置效应参考点)

只用已缓存 HS300 价(D:/stcok-worm/yjyg_hs300_prices.parquet), 无网络, 秒级.
因子:
  f_anchor  = close / rolling(252d)最高收盘 - 1   (George&Hwang 2004 锚定/参考点效应:
            股价越接近52周高点=越"显眼/强势锚定", 行为上倾向延续; 预期 IC>0)
  f_reversal = - 过去20日收益                    (注意力/显著性代理: 散户追显眼股→短期高估→反转;
             NOTE: 此即价量衰减池里的反转, 仅作"行为机制"对照, 非独立慢熵)

IC: 截面 rank-IC vs 前向20/60d, 全样本 + 2023-25.
注: HS300 机构化, 行为 edge 应弱于小盘; 本脚本仅给第一手读数, 真测需小盘价.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

PRICE_CACHE = "D:/stcok-worm/yjyg_hs300_prices.parquet"


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    da = ra - ra.mean(); db = rb - rb.mean()
    den = np.sqrt(float((da ** 2).sum()) * float((db ** 2).sum()))
    return float((da * db).sum() / den) if den > 0 else np.nan


def main():
    p = pd.read_parquet(PRICE_CACHE)
    recs = []
    for code, g in p.groupby("code"):
        g = g.sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="last")]
        if len(g) < 252:
            continue
        g["fwd20"] = g["close"].shift(-20) / g["close"] - 1.0
        g["fwd60"] = g["close"].shift(-60) / g["close"] - 1.0
        hi252 = g["close"].rolling(252, min_periods=120).max()
        g["f_anchor"] = g["close"] / hi252 - 1.0          # 锚定: 越近高点越大
        g["f_reversal"] = -(g["close"].pct_change(20))      # 注意力/反转代理
        gg = g.dropna(subset=["fwd20", "fwd60", "f_anchor", "f_reversal"])
        if gg.empty:
            continue
        gg = gg.reset_index()
        gg["code"] = code
        recs.append(gg[["date", "code", "f_anchor", "f_reversal", "fwd20", "fwd60"]])
    panel = pd.concat(recs, ignore_index=True)
    print(f"样本: {panel.shape[0]} (date,code) | 代码 {panel['code'].nunique()} | "
          f"{panel['date'].min():%Y-%m-%d}~{panel['date'].max():%Y-%m-%d}")

    facs = ["f_anchor", "f_reversal"]
    res = {}
    for f in facs:
        for fwd, lab in (("fwd20", "20"), ("fwd60", "60")):
            ic_full = panel.groupby("date").apply(lambda x: _spearman(x[f].values, x[fwd].values)).mean()
            sub = panel[panel["date"] >= pd.Timestamp("2023-01-01")]
            ic_25 = sub.groupby("date").apply(lambda x: _spearman(x[f].values, x[fwd].values)).mean()
            res[(f, lab)] = (ic_full, ic_25)

    print("\n" + "=" * 64)
    print("行为金融因子试点 — 锚定(52周高点距离) / 反转(注意力代理) [HS300]")
    print("=" * 64)
    print(f"{'因子':14s} {'IC20全':>9s} {'IC60全':>9s} {'IC20 23-25':>11s} {'IC60 23-25':>11s}")
    for f in facs:
        print(f"{f:14s} {res[(f,'20')][0]:+9.3f} {res[(f,'60')][0]:+9.3f} "
              f"{res[(f,'20')][1]:+11.3f} {res[(f,'60')][1]:+11.3f}")
    print("-" * 64)
    print("f_anchor 预期>0(近高点延续); f_reversal 预期<0(追涨反转, 但属衰减池)")
    print("=" * 64)


if __name__ == "__main__":
    main()
