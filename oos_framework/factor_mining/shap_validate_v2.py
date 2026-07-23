#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P6 · SHAP 可解释性验证 (三维度 v2 因子)
==========================================
复现 WFA 末折(2024-10~2025-10, IC 最强的近期折)模型, 用 TreeExplainer 在 OOS 上
算 SHAP 值, 验证:
  1) 特征贡献排序(mean|SHAP|) —— 对比 gain importance, 看是否一致.
  2) 方向性 —— 每个因子的 SHAP 与因子值的相关号, 验证是否符合经济学直觉
     (如 chip_profit_ratio 获利盘高应为负贡献=未来抛压).
  3) chip 维度真实贡献 —— chip 6 维在 SHAP 排序里的位置.
用户熵模型验证: 慢熵(基本面/筹码)应有稳定单调方向, 非过拟合伪相关.
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from factor_mining.factor_wfa import _hp_search, HORIZONS, RANDOM_STATE

FEAT_DIR = "factor_mining/wfa_v2_store"
RESULT_MD = "factor_mining/RESULT_v2_shap.md"
TARGET_HZ = 20          # 用 20 日标签(IC 最强)训练解释模型
SHAP_SAMPLE = 40000     # OOS 上采样多少条算 SHAP(TreeExplainer 精确, 控内存)


def main():
    import shap
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(FEAT_DIR, "feat_*.parquet")))
    if not files:
        raise RuntimeError("没有 feat 分片")

    # 特征列: 排除 id/标签列
    cols0 = pd.read_parquet(files[0]).columns.tolist()
    non_feat = ["date", "code", "fwd_ret_1"] + [f"cls_{h}" for h in HORIZONS] \
        + [f"fwd_ret_{h}" for h in HORIZONS]
    feat_cols = [c for c in cols0 if c not in non_feat]
    print(f"[shap] 特征 {len(feat_cols)} 列", flush=True)

    # 末折: IS=2021-10~2024-10, OOS=2024-10~2025-10 (对齐 wfa_folds 折4)
    IS_S, IS_E = pd.Timestamp("2021-10-23"), pd.Timestamp("2024-10-23")
    OOS_S, OOS_E = pd.Timestamp("2024-10-23"), pd.Timestamp("2025-10-23")
    read_cols = ["date", "code", f"cls_{TARGET_HZ}", f"fwd_ret_{TARGET_HZ}"] + feat_cols
    is_parts, oos_parts = [], []
    for f in files:
        df = pd.read_parquet(f, columns=read_cols)
        is_parts.append(df[(df["date"] >= IS_S) & (df["date"] < IS_E)])
        oos_parts.append(df[(df["date"] >= OOS_S) & (df["date"] < OOS_E)])
    isd = pd.concat(is_parts, ignore_index=True)
    osd = pd.concat(oos_parts, ignore_index=True)
    print(f"[shap] 末折 IS={len(isd)} OOS={len(osd)} | {(time.time()-t0):.0f}s", flush=True)

    if len(isd) > 2_000_000:
        isd = isd.sample(2_000_000, random_state=RANDOM_STATE)
    Xis = isd[feat_cols].values.astype("float32")
    yis = isd[f"cls_{TARGET_HZ}"].values
    best = _hp_search(Xis, yis)
    print(f"[shap] 最佳参数 {best}", flush=True)
    m = XGBClassifier(n_estimators=300, nthread=4, eval_metric="auc",
                      random_state=RANDOM_STATE, use_label_encoder=False, **best)
    m.fit(Xis, yis)
    print(f"[shap] 模型训练完 | {(time.time()-t0):.0f}s", flush=True)

    # gain importance(对照)
    _raw = m.get_booster().get_score(importance_type="gain")
    gain = {feat_cols[int(k[1:])]: v for k, v in _raw.items()
            if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feat_cols)}

    # OOS 采样算 SHAP
    if len(osd) > SHAP_SAMPLE:
        osd = osd.sample(SHAP_SAMPLE, random_state=RANDOM_STATE)
    Xos = osd[feat_cols].values.astype("float32")
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(Xos)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    print(f"[shap] SHAP 计算完 {sv.shape} | {(time.time()-t0):.0f}s", flush=True)

    # 汇总: mean|SHAP| + 方向(SHAP 与特征值的相关号)
    mean_abs = np.abs(sv).mean(axis=0)
    rows = []
    for j, c in enumerate(feat_cols):
        xj = Xos[:, j].astype("float64")
        s = sv[:, j].astype("float64")
        ok = np.isfinite(xj) & np.isfinite(s)
        corr = np.corrcoef(xj[ok], s[ok])[0, 1] if ok.sum() > 100 and xj[ok].std() > 0 else np.nan
        rows.append({"feat": c, "mean_abs_shap": float(mean_abs[j]),
                     "dir_corr": float(corr), "gain": float(gain.get(c, 0.0)),
                     "is_chip": c.startswith("chip_")})
    rdf = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    rdf["shap_rank"] = rdf.index + 1
    rdf["gain_rank"] = rdf["gain"].rank(ascending=False).astype(int)

    # 落盘
    rdf.to_parquet("factor_mining/shap_summary_v2.parquet", index=False)
    chip_rows = rdf[rdf["is_chip"]]
    lines = ["# P6 · SHAP 可解释性验证 (三维度 v2 · 末折 2024-25 近期)\n",
             f"- 解释模型: 末折 IS 2021-10~2024-10 训练(目标 {TARGET_HZ}日), "
             f"OOS {len(osd)} 样本 TreeExplainer 精确 SHAP",
             "- dir_corr>0=因子值越大越利多(SHAP↑), <0=越大越利空; 验证是否符合经济直觉\n",
             "## SHAP Top15 (按 mean|SHAP|)\n",
             "| 排名 | 因子 | mean|SHAP| | 方向corr | gain排名 | chip? |",
             "|---|---|---|---|---|---|"]
    for _, r in rdf.head(15).iterrows():
        lines.append(f"| {r['shap_rank']} | `{r['feat']}` | {r['mean_abs_shap']:.4f} | "
                     f"{r['dir_corr']:+.2f} | {r['gain_rank']} | {'✓' if r['is_chip'] else ''} |")
    lines.append(f"\n## 筹码(chip)6 维 SHAP 表现\n")
    lines.append("| 因子 | SHAP排名 | mean|SHAP| | 方向corr |")
    lines.append("|---|---|---|---|")
    for _, r in chip_rows.iterrows():
        lines.append(f"| `{r['feat']}` | {r['shap_rank']} | {r['mean_abs_shap']:.4f} | {r['dir_corr']:+.2f} |")
    # SHAP vs gain 排序一致性
    spearman = rdf[["shap_rank", "gain_rank"]].corr(method="spearman").iloc[0, 1]
    lines.append(f"\n- **SHAP 排序 vs gain 排序 Spearman = {spearman:.2f}** "
                 f"({'高度一致' if spearman > 0.7 else '中等' if spearman > 0.4 else '分歧大, gain 有偏'})")
    lines.append(f"- chip 6 维平均 SHAP 排名: {chip_rows['shap_rank'].mean():.0f} / {len(feat_cols)}")
    lines.append(f"\n*耗时 {(time.time()-t0)/60:.1f}min · shap_validate_v2.py*")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[3:]), flush=True)
    print(f"\n[shap] 结果 → {RESULT_MD} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
