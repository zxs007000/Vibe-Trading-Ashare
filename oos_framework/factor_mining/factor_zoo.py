#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_zoo.py — 因子图书馆(统一因子生命周期管理)
==============================================

把分散在各处的「因子定义 / 冻结权重 / 衰减监控 / 再冻结」收敛到一个**有状态的图书馆**,
实现完整生命周期:

    register ──▶ freeze(IS 期算 ICIR 权重, 锁定因子集) ──▶ monitor(滚动 IC 衰减监控)
                     ▲                                            │
                     │                                            ▼ 衰减占比 > 阈值
                     └────────────── refreeze(用近期 IS 重算权重)◀──┘

设计要点
--------
- 状态落盘 `factor_zoo_state.json`: 每个因子的定义 + 生命周期元数据(IS_IC/ICIR/近期IC/
  历史IC/衰减标志/冻结权重/冻结次数), 以及全局 frozen_set / freeze_date / refreeze_count。
- freeze / refreeze 复用 `frozen_gate_wfa.frozen_icir_weights`(IC>0 & ICIR>0 → 权重=ICIR)。
- monitor 复用向量化截面 rank-IC(与 factor_decay_monitor 同源口径), 判衰减:
  近窗 IC < 历史 × DECAY_FRAC 或 近窗转负 → 衰减。
- `weights_vector(feat_cols)` 直接产出对齐的冻结权重数组, 供 `frozen_oos_detail` 使用,
  与现有冻结策略管线零摩擦对接。

用法
----
  from factor_mining.factor_zoo import FactorZoo
  zoo = FactorZoo()
  zoo.register("cs_rank(vol_20 sub amount)", expr_tuple, family="波动", source="mine")
  zoo.freeze(long, feat_cols, is_cut)
  zoo.monitor(long, feat_cols)
  zoo.maybe_refreeze(long, feat_cols, is_cut)
  w = zoo.weights_vector(feat_cols)         # 对齐的冻结权重

  python factor_zoo.py [--stocks 400]       # 真实数据跑完整生命周期 demo
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

from factor_mining.base_data import list_stocks, load_base_data, derive_variables, forward_returns
from factor_mining.operators import evaluate_expr
from factor_mining.factor_wfa import _l2t, wfa_folds, build_feature_table
from factor_mining.frozen_gate_wfa import frozen_icir_weights
from factor_mining.universe import load_universe

# 监控/再冻结参数(与 factor_decay_monitor 同源口径)
ROLL_IC_WIN = 60          # 滚动 IC 窗口(≈ 3 月)
BASE_IC_WIN = 250         # 基准 IC 窗口(≈ 1 年)
DECAY_FRAC = 0.30         # 近窗 IC < 历史 × 30% → 衰减
REFREEZE_THRESHOLD = 0.40 # 衰减因子占比 > 40% → 触发再冻结
STATE_FILE = os.path.join(HERE, "factor_zoo_state.json")


# ---------------------------------------------------------------------------
# 向量化截面 rank-IC(直接从 long 格式算, 避免反复 pivot 大面板)
# ---------------------------------------------------------------------------
def _daily_rank_ic_panel(long: pd.DataFrame, cols: list[str], fwd_col: str = "fwd_ret_1") -> pd.DataFrame:
    """对每个因子列, 逐日截面 Spearman(factor.rank(), fwd.rank()), 返回 date×factor 的日频 IC 矩阵。"""
    fwd_p = long.pivot(index="date", columns="code", values=fwd_col).sort_index()
    vr = fwd_p.rank(axis=1)
    vc = vr.sub(vr.mean(axis=1), axis=0)
    out = {}
    for c in cols:
        fp = long.pivot(index="date", columns="code", values=c).reindex(
            index=fwd_p.index, columns=fwd_p.columns)
        fr = fp.rank(axis=1)
        fc = fr.sub(fr.mean(axis=1), axis=0)
        num = (fc * vc).sum(axis=1)
        den = np.sqrt((fc ** 2).sum(axis=1) * (vc ** 2).sum(axis=1))
        out[c] = num / den.replace(0, np.nan)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 因子图书馆
# ---------------------------------------------------------------------------
class FactorZoo:
    """统一因子生命周期管理器(有状态, 可落盘)。"""

    def __init__(self, state: Optional[dict] = None):
        self.state = state or {
            "version": "1.0",
            "factors": {},          # name -> {expr_tuple, family, source, direction, ...}
            "frozen_set": [],       # 当前冻结因子名
            "freeze_date": None,    # 冻结点(IS 截止日)
            "refreeze_count": 0,    # 再冻结次数
            "last_monitor": None,
        }

    # ---- 注册 ----
    def register(self, name: str, expr_tuple, family: str = "未知", source: str = "mine",
                 direction: int = 1):
        """登记一个因子定义(幂等, 重复 register 覆盖元数据但保留历史统计)。"""
        prev = self.state["factors"].get(name, {})
        self.state["factors"][name] = {
            "expr_tuple": list(_l2t(expr_tuple)) if isinstance(expr_tuple, (list, tuple)) else expr_tuple,
            "family": family, "source": source, "direction": direction,
            "registered": prev.get("registered"),
            "is_ic": prev.get("is_ic"), "icir": prev.get("icir"),
            "frozen": prev.get("frozen", False), "frozen_weight": prev.get("frozen_weight", 0.0),
            "recent_ic": prev.get("recent_ic"), "base_ic": prev.get("base_ic"),
            "decay_flag": prev.get("decay_flag", False),
            "n_freezes": prev.get("n_freezes", 0), "last_freeze": prev.get("last_freeze"),
            "last_monitor": prev.get("last_monitor"),
        }
        return self

    def register_many(self, items: list[tuple], families: Optional[dict] = None,
                      source: str = "mine"):
        """批量登记: items = [(name, expr_tuple), ...]。families 可选 name->family 映射。"""
        for name, expr in items:
            fam = (families or {}).get(name, "未知")
            self.register(name, expr, family=fam, source=source)
        return self

    # ---- 冻结 ----
    def freeze(self, long: pd.DataFrame, feat_cols: list[str], is_cut, tag: str = "IS"):
        """在 IS 期(is_cut 前)算逐因子 ICIR, 冻结 IC>0 & ICIR>0 的因子, 权重=ICIR(归一)。

        更新 state: frozen_set / freeze_date / 各因子 is_ic/icir/frozen/frozen_weight。
        """
        w, sel = frozen_icir_weights(long, feat_cols, is_cut)
        frozen_names = [feat_cols[i] for i in range(len(feat_cols)) if sel[i]]
        # 各因子 IS 统计(供监控对比基线): 原始 IC / ICIR
        icd = _daily_rank_ic_panel(long, feat_cols)
        is_ic = icd.mean()
        icir = icd.mean() / (icd.std() + 1e-9) * np.sqrt(252)
        for i, name in enumerate(feat_cols):
            f = self.state["factors"].get(name)
            if f is None:
                continue
            f["is_ic"] = round(float(is_ic.get(name, np.nan)), 4) if pd.notna(is_ic.get(name)) else None
            f["icir"] = round(float(icir.get(name, 0.0)), 3) if sel[i] else 0.0
            f["frozen"] = bool(sel[i])
            f["frozen_weight"] = round(float(w[i]), 4) if sel[i] else 0.0
            f["n_freezes"] = int(f.get("n_freezes", 0)) + 1
            f["last_freeze"] = str(pd.Timestamp(is_cut).date())
        self.state["frozen_set"] = frozen_names
        self.state["freeze_date"] = str(pd.Timestamp(is_cut).date())
        print(f"[zoo] 冻结完成: {len(frozen_names)}/{len(feat_cols)} 因子 | 冻结点 {self.state['freeze_date']}",
              flush=True)
        return frozen_names, w

    # ---- 监控(衰减) ----
    def monitor(self, long: pd.DataFrame, feat_cols: list[str], fwd_horizon: int = 20) -> pd.DataFrame:
        """对全部因子算滚动 IC, 标记衰减, 更新 state 中各因子的 recent_ic/base_ic/decay_flag。"""
        missing = [n for n in feat_cols if n not in self.state["factors"]]
        if missing:
            print(f"[zoo] 警告: {len(missing)} 个因子不在 state 中, 已跳过: {missing[:3]}", flush=True)
        fwd_col = f"fwd_ret_{fwd_horizon}"
        if fwd_col not in long.columns:
            # 回退: 用 forward_returns 现算
            close = long.pivot(index="date", columns="code", values="close")
            fwd = forward_returns(close, horizons=(fwd_horizon,))[fwd_horizon]
            long = long.copy()
            long = long.merge(fwd.reset_index().melt(id_vars="index", var_name="code", value_name=fwd_col)
                              .rename(columns={"index": "date"}), on=["date", "code"], how="left")
        icd = _daily_rank_ic_panel(long, feat_cols, fwd_col=fwd_col)
        roll = icd.rolling(ROLL_IC_WIN, min_periods=20).mean()
        base = icd.rolling(BASE_IC_WIN, min_periods=60).mean()
        rows = []
        for name in feat_cols:
            f = self.state["factors"].get(name)
            if f is None:
                continue
            recent = float(roll[name].iloc[-1]) if len(roll) else np.nan
            b = float(base[name].iloc[-1]) if len(base) else (f.get("is_ic") or np.nan)
            decay_ratio = recent / b if (b and abs(b) > 1e-9) else np.nan
            decay_flag = (np.isfinite(recent) and np.isfinite(b) and b > 0 and recent < b * DECAY_FRAC)
            if np.isfinite(recent) and recent < 0:
                decay_flag = True
            f["recent_ic"] = round(recent, 4) if np.isfinite(recent) else None
            f["base_ic"] = round(b, 4) if np.isfinite(b) else None
            f["decay_flag"] = bool(decay_flag)
            f["last_monitor"] = str(long["date"].max().date()) if "date" in long else None
            rows.append({"name": name, "is_ic": f.get("is_ic"), "icir": f.get("icir"),
                         "recent_ic": f["recent_ic"], "base_ic": f["base_ic"],
                         "decay_ratio": round(decay_ratio, 2) if np.isfinite(decay_ratio) else None,
                         "decay_flag": decay_flag, "frozen": f.get("frozen", False)})
        self.state["last_monitor"] = str(long["date"].max().date()) if "date" in long else None
        df = pd.DataFrame(rows)
        n_decay = int(df["decay_flag"].sum())
        print(f"[zoo] 监控完成: {n_decay}/{len(df)} 因子衰减 "
              f"({n_decay/len(df):.0%}) | 阈值 {REFREEZE_THRESHOLD:.0%}", flush=True)
        return df

    # ---- 再冻结触发 ----
    def maybe_refreeze(self, long: pd.DataFrame, feat_cols: list[str], is_cut,
                       refreeze_threshold: float = REFREEZE_THRESHOLD) -> tuple:
        """若衰减占比 > 阈值, 用 IS 期重算 ICIR 权重(再冻结), 递增 refreeze_count。返回 (是否触发, df)。"""
        df = self.monitor(long, feat_cols)
        n_decay = int(df["decay_flag"].sum())
        rate = n_decay / len(df) if len(df) else 0
        if rate > refreeze_threshold:
            print(f"[zoo] 🔥 衰减占比 {rate:.0%} > {refreeze_threshold:.0%} → 触发再冻结", flush=True)
            self.freeze(long, feat_cols, is_cut, tag="refreeze")
            self.state["refreeze_count"] = int(self.state.get("refreeze_count", 0)) + 1
            return True, df
        print(f"[zoo] 未触发再冻结: 衰减占比 {rate:.0%} ≤ {refreeze_threshold:.0%}", flush=True)
        return False, df

    # ---- 取对齐权重 ----
    def weights_vector(self, feat_cols: list[str]) -> np.ndarray:
        """产出对齐 feat_cols 的冻结权重数组(未冻结因子=0), 供 frozen_oos_detail 使用。"""
        return np.array([self.state["factors"].get(c, {}).get("frozen_weight", 0.0) or 0.0
                         for c in feat_cols], dtype=float)

    # ---- 持久化 ----
    def save(self, path: str = STATE_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
        print(f"[zoo] 状态已落盘: {path}", flush=True)

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "FactorZoo":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ---- 报告 ----
    def report(self) -> str:
        s = self.state
        fs = s["factors"]
        n = len(fs)
        n_frozen = sum(1 for v in fs.values() if v.get("frozen"))
        n_decay = sum(1 for v in fs.values() if v.get("decay_flag"))
        L = ["# 因子图书馆 · 生命周期状态\n",
             f"- 因子总数: **{n}** | 当前冻结: **{n_frozen}** | 衰减: **{n_decay}**",
             f"- 冻结点: {s.get('freeze_date')} | 再冻结次数: {s.get('refreeze_count', 0)} | "
             f"最近监控: {s.get('last_monitor')}\n",
             "## 冻结因子集(ICIR 权重)\n",
             "| 因子 | 家族 | IS_IC | ICIR | 权重 | 近期IC | 状态 |",
             "|---|---|---|---|---|---|---|"]
        for name in s.get("frozen_set", []):
            v = fs.get(name, {})
            ri = f"{v.get('recent_ic'):+.4f}" if isinstance(v.get('recent_ic'), (int, float)) else "—"
            L.append(f"| `{name[:50]}` | {v.get('family','?')} | {v.get('is_ic')} | "
                     f"{v.get('icir')} | {v.get('frozen_weight')} | {ri} | "
                     f"{'⚠️衰减' if v.get('decay_flag') else '✅'} |")
        L.append("\n## 全因子健康度\n")
        for name, v in sorted(fs.items(), key=lambda kv: -(kv[1].get("is_ic") or 0)):
            L.append(f"- `{name[:55]}`: IC={v.get('is_ic')} 近期IC={v.get('recent_ic')} "
                     f"{'⚠️衰减' if v.get('decay_flag') else '✅'}")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# 真实数据 demo: 完整生命周期
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=400)
    ap.add_argument("--out", default="FACTOR_ZOO_REPORT.md")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    sel = json.load(open(os.path.join(HERE, "factors_v2_selected.json"), encoding="utf-8"))
    d3 = json.load(open(os.path.join(HERE, "factors_v2_3dim.json"), encoding="utf-8"))
    items = [(k, _l2t(d3[k]["expr_tuple_list"])) for k in sel if k in d3]
    families = {k: d3[k].get("family", "未知") for k in d3}
    print(f"[zoo] 因子 {len(items)} (选中 {len(sel)})", flush=True)

    codes = list_stocks(args.stocks)
    long, feat_cols = build_feature_table(codes, items)
    folds = wfa_folds(long["date"])
    is_cut = folds[0][2]
    print(f"[zoo] 长表 {long.shape} | 冻结点 {pd.Timestamp(is_cut).date()}", flush=True)

    zoo = FactorZoo()
    zoo.register_many(items, families=families, source="mine_v2")
    zoo.freeze(long, feat_cols, is_cut)
    triggered, df = zoo.maybe_refreeze(long, feat_cols, is_cut)
    if not args.no_save:
        zoo.save()
    md = zoo.report()
    md = f"*生成耗时 {time.time()-t0:.1f}s, 样本 {args.stocks} 只*\n\n" + md
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    open(outp, "w", encoding="utf-8").write(md)
    print(f"\n报告: {outp} | 再冻结触发: {triggered} | 耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
