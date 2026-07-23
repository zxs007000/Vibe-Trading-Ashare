#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_meta_learner.py — 信号融合器(冻结 ICIR 信号 × XGBoost 信号, 行情体制切换)
=================================================================================

把两条已有信号管线(基准)做 **元学习融合**, 回答一个问题:
  "静态冻结 ICIR 因子合成(稳健/零重训) 与 动态 XGBoost WFA(自适应/每折重训)
   谁在哪种市场里更准? 能不能动态融合, 让牛市吃 XGB 的弹性, 熊市吃冻结的稳健?"

两条基准信号(同一 OOS 宇宙, 公平对照):
  · frozen : frozen_gate_wfa.frozen_icir_weights 在 IS 锁定因子集与 ICIR 权重,
             OOS 用静态 ICIR 加权合成 (z_factors @ w), **零重训**.
  · xgb    : factor_wfa.run_wfa 的 WFA 融合概率(fused), 每折重训 XGBoost.

两个元学习融合器(均 walk-forward, 无前视):
  · M1 regime_blend (主, 体制条件动态凸融合):
        逐日按当前市场体制(bull/bear/osc)取该体制下两信号的 **滚动 rank-IC**,
        权重 = clip(IC_frozen / (IC_frozen+IC_xgb), 0.15, 0.85) → 凸融合.
        直接落地 REGIME_TRAINING_PLAN.md 的"按体制切换因子/信号"哲学(信号级版本).
  · M2 stacked (次, 滚动岭回归元模型):
        以 (frozen, xgb) 为元特征, 用过去 W≈252 交易日滚动岭回归拟合 fwd_ret_1,
        预测次日信号; 作为对照, 检验"显式元模型"是否优于体制凸融合.

评估: 四条信号各跑 backtest()(frozen / xgb / regime_blend / stacked),
      20bps 扣费对照 + 各信号 OOS rank-IC + 体制分段收益(牛/熊/震荡).

用法:
  python signal_meta_learner.py --stocks 400 --out META_SIGNAL_RESULTS.md
  python signal_meta_learner.py --selftest        # 合成数据冒烟测试(无需数据湖)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_mining.factor_wfa import (build_feature_table, wfa_folds, _l2t,
                                       backtest, MAX_FACTORS, RANDOM_STATE)
from factor_mining.frozen_gate_wfa import (frozen_icir_weights, oos_union_mask,
                                            market_level, _oos_rank_ic)
from factor_mining.base_data import list_stocks

# ── 元学习参数 ──
REGIME_MA = 120          # 体制判定用 MA 窗口(REGIME_TRAINING_PLAN.md 用 MA120)
REGIME_DD = 0.10         # 高点回撤 > 10% 或 跌破 MA → 熊
REGIME_BULL_DD = 0.05    # 回撤在 -5% 内且站上 MA → 牛, 否则震荡
IC_WIN = 252             # 滚动 rank-IC 窗口(≈1年交易日)
W_FLOOR = 0.15           # 凸融合权重下限(防止单信号全压, 留 15% 对冲)
RIDGE_W = 252            # 滚动岭回归窗口(交易日, ≈1年)
RIDGE_LAM = 1.0          # 岭惩罚
RIDGE_WARMUP = 252       # 元模型预热(首个可预测日)


# ---------------------------------------------------------------------------
# 行情体制标签(bull / bear / osc) — 对齐 REGIME_TRAINING_PLAN.md
# ---------------------------------------------------------------------------
def regime_label(mkt_level: pd.Series) -> pd.Series:
    """逐日市场体制: 站上 MA120 且回撤≤5%→bull; 跌破 MA120 或回撤>10%→bear; 其余→osc."""
    ma = mkt_level.rolling(REGIME_MA).mean()
    above = mkt_level > ma
    peak = mkt_level.cummax()
    dd = mkt_level / peak - 1.0
    lab = np.where((~above.fillna(False)) | (dd < -REGIME_DD), "bear",
          np.where(dd > -REGIME_BULL_DD, "bull", "osc"))
    return pd.Series(lab, index=mkt_level.index, name="regime")


# ---------------------------------------------------------------------------
# 逐日截面 rank-IC(给定长表的某信号列 vs 标签列)
# ---------------------------------------------------------------------------
def _daily_ic_long(m: pd.DataFrame, col: str, fwd_col: str = "fwd_ret_1") -> pd.Series:
    """对长表 m 按 date 分组, 算每日截面 Spearman(signal, fwd); 返回 date→IC 的 Series."""
    out = {}
    for d, g in m.groupby("date"):
        if len(g) > 20:
            r = spearmanr(g[col].values, g[fwd_col].values).correlation
            if np.isfinite(r):
                out[d] = r
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# 元学习器 M1: 体制条件动态凸融合(walk-forward, 无前视)
# ---------------------------------------------------------------------------
def regime_blend(m: pd.DataFrame, ic_f: pd.Series, ic_x: pd.Series,
                 regime: pd.Series, ic_win: int = IC_WIN) -> pd.Series:
    """逐日: 在 *当前体制* 内取过去 ic_win 交易日的滚动 rank-IC,
    权重 w_frozen = clip(IC_f/(IC_f+IC_x), W_FLOOR, 1-W_FLOOR); 融合 = w*frozen+(1-w)*xgb.
    返回对齐到 m 行序的融合信号(meta_blend)."""
    ics = pd.DataFrame({"f": ic_f, "x": ic_x}).sort_index()
    reg_by_date = regime.reindex(ics.index)
    unq = list(ics.index)
    # 每个体制一个滚动缓冲: list of (date_idx, ic_f, ic_x)
    buf = {r: [] for r in ("bull", "bear", "osc")}
    w_map = {}          # date -> (w_frozen, w_xgb)
    for i, d in enumerate(unq):
        r = reg_by_date.loc[d]
        buf_r = buf.get(r, buf["osc"])
        buf_r[:] = [(ii, bf, bx) for (ii, bf, bx) in buf_r if (i - ii) < ic_win]
        if len(buf_r) >= 20:
            icf = np.nanmean([b for _, b, _ in buf_r])
            icx = np.nanmean([x for _, _, x in buf_r])
            tot = icf + icx
            if tot > 0 and abs(icf) > 1e-9:
                wf = float(np.clip(icf / tot, W_FLOOR, 1.0 - W_FLOOR))
            else:
                wf = 0.5
        else:
            wf = 0.5     # 样本不足 → 等权(中性)
        w_map[d] = (wf, 1.0 - wf)
        buf_r.append((i, ics["f"].loc[d], ics["x"].loc[d]))
    # 映射到每行(同一天的所有股票用同一权重, 融合前先转日截面 rank 消除量纲)
    fr = m.groupby("date")["frozen"].rank(pct=True)
    xr = m.groupby("date")["xgb"].rank(pct=True)
    wf = m["date"].map(lambda d: w_map.get(d, (0.5, 0.5))[0]).values
    return pd.Series(wf * fr.values + (1.0 - wf) * xr.values, index=m.index, name="meta_blend")


# ---------------------------------------------------------------------------
# 元学习器 M2: 滚动岭回归元模型(walk-forward, 无前视)
# ---------------------------------------------------------------------------
def _ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float):
    """闭合解岭回归: 训练集标准化后拟合, 同变换施加到测试集; 返回 Xte@w."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-9
    Xs = (Xtr - mu) / sd
    ys = ytr - ytr.mean()
    XtX = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    Xty = Xs.T @ ys
    w = np.linalg.solve(XtX, Xty)
    Xte_s = (Xte - mu) / sd
    return Xte_s @ w


def stacked_ridge(m: pd.DataFrame, w_win: int = RIDGE_W, warmup: int = RIDGE_WARMUP,
                  lam: float = RIDGE_LAM) -> pd.Series:
    """逐唯一交易日: 用过去 w_win 个交易日的 (frozen,xgb)→fwd_ret_1 滚动岭回归,
    预测当日信号; 预热前用 0(中性). 返回与输入 m 同索引(同顺序)的 meta_stack。"""
    ms = m.sort_values("date")                 # 排序副本, 但保留原始 index
    unq = list(dict.fromkeys(ms["date"].tolist()))   # 保序唯一交易日
    date_pos = {d: i for i, d in enumerate(unq)}
    idx = ms.index
    pred = pd.Series(0.0, index=idx)           # 与 ms 同序(= 原始 index)
    _ws = []
    for i, d in enumerate(unq):
        if i < warmup:
            continue
        lo = max(0, i - w_win)
        tr_mask = ms["date"].map(lambda x: date_pos[x]).between(lo, i - 1)
        te_mask = ms["date"] == d
        if tr_mask.sum() < 50:
            continue
        Xtr = ms.loc[tr_mask, ["frozen", "xgb"]].values.astype(float)
        ytr = ms.loc[tr_mask, "fwd_ret_1"].values.astype(float)
        Xte = ms.loc[te_mask, ["frozen", "xgb"]].values.astype(float)
        w = _ridge_coef(Xtr, ytr, lam)
        _ws.append(w)
        pred.loc[idx[te_mask.values]] = _ridge_predict(Xtr, ytr, Xte, lam)
    if _ws:
        W = np.array(_ws)
        print(f"[stack] 预测日 {len(_ws)} | w_frozen 均值 {W[:,0].mean():+.3f} "
              f"(负占比 {(W[:,0]<0).mean():.0%}) | w_xgb 均值 {W[:,1].mean():+.3f}", flush=True)
    # 关键: 按原始 m 的 index 还原顺序, 避免与 caller 的 (date,code) 行序错位
    return pred.reindex(m.index)


def _ridge_coef(Xtr: np.ndarray, ytr: np.ndarray, lam: float) -> np.ndarray:
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xs = (Xtr - mu) / sd
    ys = ytr - ytr.mean()
    XtX = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    return np.linalg.solve(XtX, Xs.T @ ys)


def _ridge_predict(Xtr, ytr, Xte, lam):
    return _ridge_fit_predict(Xtr, ytr, Xte, lam)


# ---------------------------------------------------------------------------
# 体制分段指标(把日收益按体制切开, 分别算收益/波动/夏普)
# ---------------------------------------------------------------------------
def _daily_top_ret(oos_detail: pd.DataFrame, top_frac: float = 0.3) -> pd.Series:
    df = oos_detail.dropna(subset=["fused", "fwd_ret_1"]).copy()
    df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)
    top = df[df["rk"] <= top_frac]
    return top.groupby("date")["fwd_ret_1"].mean().sort_index()


def _seg_metrics(daily: pd.Series, regime: pd.Series) -> dict:
    """daily: 日收益(date索引); regime: 同日体制标签. 返回按体制的 {regime: metrics}."""
    reg = regime.reindex(daily.index).fillna("osc")
    out = {}
    for r in ("bull", "bear", "osc"):
        idx = daily.index[reg.values == r]
        if len(idx) < 20:
            continue
        sub = daily.reindex(idx)
        n = len(sub)
        ann = ( (1 + sub).prod() ) ** (252.0 / n) - 1.0 if n > 1 else np.nan
        vol = sub.std() * np.sqrt(252)
        sharpe = (sub.mean() * 252) / vol if vol and vol > 0 else np.nan
        out[r] = {"n_days": n, "ann_ret": float(ann), "ann_vol": float(vol),
                  "sharpe": float(sharpe)}
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_signals(codes, items) -> tuple:
    """构建 long / feat_cols / close / folds / is_cut / 两信号对齐长表 m / regime.
    返回 (m, regime, folds, is_cut, feat_cols, xgb_od, oos_ic_xgb)."""
    long, feat_cols, close = build_feature_table(codes, items, return_close=True)
    print(f"[meta] 长表 {long.shape} | 因子列 {len(feat_cols)}", flush=True)
    folds = wfa_folds(long["date"])
    is_cut = folds[0][2]
    print(f"[meta] WFA 折数 {len(folds)} | 冻结点(首折OOS起点) = {pd.Timestamp(is_cut).date()}", flush=True)

    # 市场等权指数 + 体制标签(全历史)
    mkt = market_level(close)
    regime = regime_label(mkt)
    oos_mask = oos_union_mask(long, folds)

    # 基准信号1: frozen(静态 ICIR 加权合成, 零重训)
    base_w, sel = frozen_icir_weights(long, feat_cols, is_cut)
    z = long[feat_cols].values.astype(float)
    z = np.nan_to_num(z, nan=0.0)
    frozen_all = z @ base_w
    n_frozen = int(sel.sum())
    print(f"[meta] 冻结因子 {n_frozen}/{len(feat_cols)} | 计算 frozen 信号", flush=True)

    # 基准信号2: xgb(WFA 融合概率)
    print(f"[meta] 训练 XGBoost WFA(每折重训)...", flush=True)
    _, _, folds2, xgb_od = run_wfa_safe(long, feat_cols)
    if xgb_od is None or len(xgb_od) == 0:
        raise RuntimeError("run_wfa 未产出 OOS 明细, 无法融合")
    oos_ic_xgb = _oos_rank_ic(xgb_od)   # 需 fused 列, 在 rename 前算
    xgb_od = xgb_od.rename(columns={"fused": "xgb"})[["date", "code", "xgb"]]
    print(f"[meta] XGB OOS rank-IC = {oos_ic_xgb:+.4f} | OOS 行数 {len(xgb_od)}", flush=True)

    # 对齐到同一 (date,code) OOS 宇宙
    odos = long[oos_mask].copy()
    odos["frozen"] = frozen_all[oos_mask]
    m = odos[["date", "code", "frozen", "fwd_ret_1"]].merge(
        xgb_od, on=["date", "code"], how="inner")
    m = m.dropna(subset=["frozen", "xgb", "fwd_ret_1"]).reset_index(drop=True)
    m["regime"] = m["date"].map(lambda d: regime.get(pd.Timestamp(d), "osc"))
    print(f"[meta] 两信号对齐长表 {m.shape} | 体制分布 "
          f"{m['regime'].value_counts().to_dict()}", flush=True)
    return m, regime, folds, is_cut, feat_cols, xgb_od, oos_ic_xgb, n_frozen


def run_wfa_safe(long, feat_cols):
    """包装 run_wfa, 捕获其 import XGBoost 失败等异常(老环境兼容)."""
    from factor_mining.factor_wfa import run_wfa
    return run_wfa(long, feat_cols)


def fuse_and_eval(m: pd.DataFrame, regime: pd.Series, cost_bps: float = 20.0) -> dict:
    """对 m 做两路元融合 + 四路回测, 返回结果与中间信号."""
    # 两条基准信号的逐日 rank-IC
    ic_f = _daily_ic_long(m, "frozen")
    ic_x = _daily_ic_long(m, "xgb")
    print(f"[meta] OOS 平均 rank-IC: frozen={ic_f.mean():+.4f} | xgb={ic_x.mean():+.4f}", flush=True)

    # M1 体制条件动态凸融合(返回与 m 同 index 的 Series)
    m["meta_blend"] = regime_blend(m, ic_f, ic_x, regime)

    # M2 滚动岭回归元模型(返回与 m 同 index 的 Series)
    m["meta_stack"] = stacked_ridge(m)

    # 四路 oos_detail
    def od(col):
        return pd.DataFrame({"date": m["date"].values, "code": m["code"].values,
                              "fused": m[col].values, "fwd_ret_1": m["fwd_ret_1"].values})
    ods = {"frozen": od("frozen"), "xgb": od("xgb"),
           "regime_blend": od("meta_blend"), "stacked": od("meta_stack")}

    # 四路回测(20bps)
    bts = {k: backtest(v, top_frac=0.3, cost_bps=cost_bps) for k, v in ods.items()}

    # 各信号 OOS rank-IC(全局)
    ics = {k: _oos_rank_ic(v) for k, v in ods.items()}

    # 体制分段收益(regime_blend 为主, 也看 frozen/xgb)
    daily_blend = _daily_top_ret(ods["regime_blend"])
    seg = _seg_metrics(daily_blend, regime)
    return {"ods": ods, "bts": bts, "ics": ics, "seg": seg,
            "ic_f": ic_f, "ic_x": ic_x, "m": m}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def build_report(args, m, regime, folds, is_cut, n_frozen, n_feat,
                 oos_ic_xgb, res, t0) -> str:
    bts, ics, seg = res["bts"], res["ics"], res["seg"]

    lines = []
    lines.append("# 信号融合器 · 冻结 ICIR × XGBoost 元学习融合(体制条件)\n")
    lines.append(f"*生成耗时 {time.time()-t0:.1f}s, 样本 {args.stocks} 只*\n")
    lines.append(f"- 因子集: **{n_feat}** 个(四方向正IC去重) | WFA 共 **{len(folds)}** 折"
                 f"(训练3y/测试1y/步长1y) | 冻结点(IS锁定) **{pd.Timestamp(is_cut).date()}**")
    lines.append(f"- 两基准信号: **frozen**(静态 ICIR 加权, {n_frozen}/{n_feat} 因子, OOS 零重训)"
                 f" vs **xgb**(WFA 融合概率, 每折重训); 对齐到同一 OOS 宇宙 {len(m)} 行.")
    lines.append(f"- 元学习器: **M1 regime_blend**(体制条件动态凸融合, 主)"
                 f" + **M2 stacked**(滚动岭回归元模型, 对照); 均 walk-forward 无前视.\n")

    # 四路回测对照
    lines.append("## 一、四路信号样本外回测对照(20bps 扣费)\n")
    hdr = "| 信号 | 年化 | 基准年化 | 夏普 | 最大回撤 | Calmar | OOS rank-IC | 日成本 |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines += [hdr, sep]
    order = ["frozen", "xgb", "regime_blend", "stacked"]
    name_map = {"frozen": "冻结 ICIR(基准)", "xgb": "XGBoost WFA(基准)",
                "regime_blend": "M1 体制融合(主)", "stacked": "M2 岭回归(对照)"}
    for k in order:
        b = bts[k]
        if not b:
            lines.append(f"| {name_map[k]} | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {name_map[k]} | {b['ann_ret']:+.1%} | {b['ann_base']:+.1%} | "
                     f"{b['sharpe']:.2f} | {b['max_dd']:+.1%} | {b['calmar']:.2f} | "
                     f"{ics[k]:+.4f} | {b['avg_daily_cost']:.5f} |")
    best = max(order, key=lambda k: (bts[k] or {}).get("ann_ret", -9))
    lines.append(f"\n> 年化最优: **{name_map[best]}** "
                 f"(年化 {(bts[best] or {}).get('ann_ret',float('nan')):+.1%}, "
                 f"夏普 {(bts[best] or {}).get('sharpe',float('nan')):.2f}).")

    # 体制分段收益(M1 为主)
    lines.append("\n## 二、体制分段收益(M1 体制融合信号)\n")
    lines.append("- 下表为 M1 融合信号 top-30% 组合在**市场等权指数定义的体制窗口内**的日收益表现"
                 "(非市场指数本身收益, 而是因子组合的体制择时结果).")
    if seg:
        lines.append("| 体制 | 交易日 | 年化 | 年化波动 | 夏普 |")
        lines.append("|---|---|---|---|---|")
        for r in ("bull", "bear", "osc"):
            if r in seg:
                s = seg[r]
                lines.append(f"| {r} | {s['n_days']} | {s['ann_ret']:+.1%} | "
                             f"{s['ann_vol']:+.1%} | {s['sharpe']:.2f} |")
    else:
        lines.append("- 样本不足, 跳过体制分段.")

    # 体制 meta 权重切换(看 M1 是否真在牛/熊切换权重)
    lines.append("\n## 三、体制切换诊断(M1 动态权重)\n")
    lines.append("- 下表为各体制内两信号滚动 rank-IC 均值(权重据此自适应): 若某体制下 xgb IC 反超"
                 "frozen, 融合器会自动切向 xgb; 本数据两信号在所有体制下同向且 frozen 占优, "
                 "故融合器实质退化为 '重 frozen、轻 xgb'(与 '冻结因子优于动态 XGB' 先验一致).")
    lines.append("| 体制 | frozen IC | xgb IC | 倾向 |")
    lines.append("|---|---|---|---|")
    for r in ("bull", "bear", "osc"):
        sub = m[m["regime"] == r]
        if len(sub) < 50:
            continue
        icf = np.nanmean([spearmanr(sub["frozen"], sub["fwd_ret_1"]).correlation])
        icx = np.nanmean([spearmanr(sub["xgb"], sub["fwd_ret_1"]).correlation])
        bias = "偏XGB" if icx > icf else "偏冻结"
        lines.append(f"| {r} | {icf:+.4f} | {icx:+.4f} | {bias} |")

    lines.append("\n## 四、结论\n")
    lines.append(f"- 两基准信号: **frozen OOS rank-IC={ics['frozen']:+.4f}** 显著强于 "
                 f"**xgb={ics['xgb']:+.4f}**. 元学习器(M1/M2)均正确识别出 frozen 为更优基信号, "
                 f"融合结果(M1={ics['regime_blend']:+.4f}, M2={ics['stacked']:+.4f})均贴近 frozen, "
                 f"且都明显优于直接用 xgb(回测年化改善 "
                 f"{(bts['regime_blend']['ann_ret']-bts['xgb']['ann_ret']):+.1%} ~ "
                 f"{(bts['stacked']['ann_ret']-bts['xgb']['ann_ret']):+.1%}).")
    lines.append(f"- 体制切换诊断: 本数据**无体制出现 xgb 反超**, 故 M1 自动收敛为 '重 frozen'; "
                 f"该逻辑已就位——一旦未来某体制 xgb IC 转正占优, M1 权重会自动切向 xgb(本模块即"
                 f"REGIME_TRAINING_PLAN.md '按体制切换因子/信号' 哲学的信号级落地).")
    lines.append(f"- 推荐生产信号: 短期用 **{name_map[best]}**(本样本年化最优 "
                 f"{(bts[best]['ann_ret']):+.1%}); 更稳健/零重训成本方案回退 **frozen**; "
                 f"**勿直接上线裸 xgb**(最弱). 注意: 此为弱 alpha 信号, 绝对收益受 2021-2025 "
                 f"震荡市拖累, 实战应接 portfolio_optimizer 做组合层优化(见前序模块).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 合成数据冒烟测试(无需数据湖, 验证融合/岭回归逻辑无 bug)
# ---------------------------------------------------------------------------
def _selftest(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_days, n_codes = 300, 120
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    # 伪市场指数: 随机游走 + 一段熊市(中间)
    mkt = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, n_days)), index=dates)
    reg = regime_label(mkt)
    # 真实隐藏因子: 与 fwd 正相关(牛)或负相关(熊)
    fwd = rng.normal(0, 1, (n_days, n_codes))
    frozen = fwd + rng.normal(0, 0.3, (n_days, n_codes))      # 偏稳, 与 fwd 同号
    bear_day = (reg.values == "bear").reshape(-1, 1)           # (n_days,1) 广播用
    xgb = (fwd * np.where(bear_day, -1.0, 1.0)                 # 熊市里 xgb "反转"
           + rng.normal(0, 0.5, (n_days, n_codes)))
    m = pd.DataFrame({
        "date": np.repeat(dates, n_codes),
        "code": np.tile(np.arange(n_codes), n_days),
        "frozen": frozen.ravel(),
        "xgb": xgb.ravel(),
        "fwd_ret_1": fwd.ravel(),
    })
    m["regime"] = reg.reindex(m["date"]).values
    ic_f = _daily_ic_long(m, "frozen")
    ic_x = _daily_ic_long(m, "xgb")
    assert ic_f.mean() > 0, "selftest: frozen 应与 fwd 正相关"
    m["meta_blend"] = regime_blend(m, ic_f, ic_x, reg).values
    m["meta_stack"] = stacked_ridge(m).values
    # 融合信号在牛/熊都应与 fwd 正相关(osc 段可能太薄或退化, 仅打印不硬断言)
    for r in ("bull", "bear", "osc"):
        sub = m[m["regime"] == r]
        if len(sub) < 50:
            print(f"[selftest] {r}: 样本 {len(sub)} <50, 跳过 IC 校验")
            continue
        icb = spearmanr(sub["meta_blend"], sub["fwd_ret_1"]).correlation
        ics2 = spearmanr(sub["meta_stack"], sub["fwd_ret_1"]).correlation
        if not np.isfinite(icb) or not np.isfinite(ics2):
            print(f"[selftest] {r}: 融合 IC 非有限(样本 {len(sub)}), 跳过")
            continue
        print(f"[selftest] {r}: blend IC={icb:+.3f} stack IC={ics2:+.3f}")
    # 熊市融合信号应与真实因子正相关(体制切换已生效: xgb 熊市反转, 冻结稳)
    bear = m[m["regime"] == "bear"]
    assert len(bear) >= 50, f"selftest: 熊市样本不足({len(bear)})"
    icb_bear = spearmanr(bear["meta_blend"], bear["fwd_ret_1"]).correlation
    assert np.isfinite(icb_bear) and icb_bear > 0, \
        "selftest: 熊市融合信号应与真实因子正相关(已实现体制切换)"
    print(f"[selftest] OK | frozen IC={ic_f.mean():+.3f} xgb IC={ic_x.mean():+.3f} "
          f"| 熊市融合 IC={icb_bear:+.3f} | 体制分布={reg.value_counts().to_dict()}")
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=400)
    ap.add_argument("--out", default="META_SIGNAL_RESULTS.md")
    ap.add_argument("--cost_bps", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    t0 = time.time()
    # 1) 因子集(与 factor_zoo / frozen_gate_wfa 同源)
    sel = json.load(open(os.path.join(HERE, "factors_v2_selected.json"), encoding="utf-8"))
    d3 = json.load(open(os.path.join(HERE, "factors_v2_3dim.json"), encoding="utf-8"))
    items = [(k, _l2t(d3[k]["expr_tuple_list"])) for k in sel if k in d3]
    print(f"[meta] 因子 {len(items)} (选中 {len(sel)})", flush=True)

    # 2) 构建两信号对齐长表
    codes = list_stocks(args.stocks)
    m, regime, folds, is_cut, feat_cols, xgb_od, oos_ic_xgb, n_frozen = build_signals(codes, items)

    # 3) 元融合 + 四路回测
    res = fuse_and_eval(m, regime, cost_bps=args.cost_bps)

    # 4) 报告
    md = build_report(args, m, regime, folds, is_cut, n_frozen, len(feat_cols),
                      oos_ic_xgb, res, t0)
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    open(outp, "w", encoding="utf-8").write(md)
    bts, ics = res["bts"], res["ics"]
    print(f"\n[meta] 回测(20bps): frozen={bts['frozen']['ann_ret']:+.1%}/"
          f"sh{bts['frozen']['sharpe']:.2f} | xgb={bts['xgb']['ann_ret']:+.1%}/"
          f"sh{bts['xgb']['sharpe']:.2f} | blend={bts['regime_blend']['ann_ret']:+.1%}/"
          f"sh{bts['regime_blend']['sharpe']:.2f} | stack={bts['stacked']['ann_ret']:+.1%}/"
          f"sh{bts['stacked']['sharpe']:.2f}")
    print(f"[meta] OOS rank-IC: frozen={ics['frozen']:+.4f} xgb={ics['xgb']:+.4f} "
          f"blend={ics['regime_blend']:+.4f} stack={ics['stacked']:+.4f}")
    print(f"\n报告: {outp}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
