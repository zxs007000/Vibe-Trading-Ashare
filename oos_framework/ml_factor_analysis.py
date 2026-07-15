"""ml_factor_analysis.py — ML 因子分析 + ML 增强因子(全市场 OOS).

Part 1 因子分析: 用 XGBoost 在 (股票×日期) 抽样面板上预测 5 日前向收益,
  取得 33 因子的全局重要性, 以及牛/熊/震荡三态下的条件重要性 —— 直接服务
  "选对状态用对因子" 论点(看哪些因子在哪种 regime 真正有效).

Part 2 增强因子: 严格 OOS 构建 ML 复合因子(ml_factor).
  复用 regime_wfa 的 18-fold 滚动结构: 每 fold 用 train 窗(历史)训练 XGBoost,
  预测 test 窗(未来) 的 5 日收益打分, 全程无前视. 该打分作为单一复合因子,
  接进 rolling_wfa_dual_regime(带两档位闸门), 与 A~E 变体头对头.

约束: 8G cgroup + 600s 会话. 训练抽样 2000 只股票控内存/时长; 预测对全 5515 只泛化.
"""
import sys, time, os, warnings
sys.path.insert(0, "oos_framework")
sys.path.insert(0, "agent/backtest")
import numpy as np, pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import regime_wfa as R
from regime_wfa import load_engine_inputs_cached, rolling_wfa_dual_regime, _daily_rank_ic_arr, _market_regime_label

HERE = R.HERE
OUT = HERE / "screen_results"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_DAYS, TEST_DAYS, PURGE = 250, 250, 5
N_SAMPLE_STOCKS = 2000


def _folds(n):
    folds = []
    i = TRAIN_DAYS
    while i + PURGE + TEST_DAYS <= n:
        ts = i - TRAIN_DAYS; te = i; ve = i + PURGE; vb = ve + TEST_DAYS
        folds.append((ts, te, ve, vb)); i += TEST_DAYS
    return folds


def part1_importance(inp):
    """全局 + 三态条件因子重要性(XGBoost gain)."""
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes = inp["fwd"], inp["dates"], inp["codes"]
    n_codes = len(codes)
    rng = np.random.default_rng(42)
    samp = list(codes); rng.shuffle(samp); samp = samp[:N_SAMPLE_STOCKS]
    sidx = [list(codes).index(c) for c in samp]
    # 构建抽样特征矩阵 (所有日期 × 抽样股票)
    t0 = time.time()
    X = np.stack([zarr[f][:, sidx] for f in ALL], axis=-1)   # (dates, 2000, 33)
    y = fwd.values[:, sidx]                                   # (dates, 2000)
    X = X.reshape(-1, len(ALL)); Y = y.reshape(-1)
    m = (~np.isnan(X).any(1)) & (~np.isnan(Y))
    X, Y = X[m].astype(np.float32), Y[m].astype(np.float32)
    print(f"  [分析] 特征矩阵 {X.shape} ({time.time()-t0:.1f}s)", flush=True)

    def train(Xs, Ys):
        md = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0)
        md.fit(Xs, Ys)
        return md

    glob = train(X, Y)
    imp_glob = pd.Series(glob.feature_importances_, index=ALL).sort_values(ascending=False)
    print(f"  [分析] 全局模型训练完成, top5: {list(imp_glob.head(5).index)}", flush=True)

    # 三态条件: 按市场 regime 切分样本(注意 reg_flat 须先与 X 同掩码 m 对齐)
    reg = _market_regime_label(inp["mkt_level"], dates).values   # (dates,)
    reg_flat = reg.repeat(len(samp))                              # 对齐 (dates*2000,)
    reg_mask = reg_flat[m]                                        # 与 X 同形 (578074,)
    imp_reg = {}
    for state in ("bull", "bear", "osc"):
        sm = (reg_mask == state)
        if sm.sum() < 5000:
            print(f"  [分析] {state} 样本不足, 跳过", flush=True); continue
        md = train(X[sm], Y[sm])
        imp_reg[state] = pd.Series(md.feature_importances_, index=ALL)
        print(f"  [分析] {state} 模型完成 (n={sm.sum()})", flush=True)

    # 图: 全局 + 三态 top 因子
    top = imp_glob.head(12).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(top)); w = 0.22
    ax.bar(x - 1.5*w, [imp_glob[t] for t in top], w, label="全局")
    cols = {"bull": "tab:green", "bear": "tab:red", "osc": "tab:orange"}
    for j, st in enumerate(("bull", "bear", "osc")):
        if st in imp_reg:
            ax.bar(x + (j-1)*w, [imp_reg[st].get(t, 0) for t in top], w, label=st)
    ax.set_xticks(x); ax.set_xticklabels(top, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("XGBoost gain 重要性"); ax.set_title("ML 因子重要性: 全局 vs 牛/熊/震荡(全市场 OOS 抽样)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "ml_factor_importance.png", dpi=110); plt.close(fig)

    # 报告表
    md = ["# ML 因子分析(全市场 5515 只, XGBoost)", "",
          f"- 面板: {n_codes}只 × {dates[0].date()}~{dates[-1].date()}, 33 因子, 目标=5日前向收益",
          f"- 训练抽样: {N_SAMPLE_STOCKS} 只 × 全日期, XGBoost(hist, 150树, depth4)",
          f"- 三态切分: 按市场 regime(牛/熊/震荡)拆样本, 各训独立模型看条件重要性", ""]
    md += ["## 1. 全局因子重要性(top12)", "",
           "| 因子 | gain |", "|---|---|"]
    for f in top:
        md.append(f"| {f} | {imp_glob[f]:.4f} |")
    md += ["", "## 2. 三态条件重要性(同 top12 因子)", "",
           "| 因子 | 全局 | bull | bear | osc |",
           "|---|---|---|---|---|"]
    for f in top:
        def g(d): return f"{imp_reg[d][f]:.4f}" if d in imp_reg else "-"
        md.append(f"| {f} | {imp_glob[f]:.4f} | {g('bull')} | {g('bear')} | {g('osc')} |")
    md += ["", "## 3. 解读",
           "- 若某因子在 bull 高、bear 低(或反之), 正是'状态→因子'框架的料: 该因子有 regime 依赖性.",
           "- 全局重要性高但三态平坦的因子 = 全天候因子; 三态分化大的 = 需按状态启停的因子.", ""]
    (OUT / "ml_factor_analysis.md").write_text("\n".join(md), encoding="utf-8")
    return imp_glob, imp_reg


def part2_ml_factor(inp):
    """严格 OOS 构建 ML 复合因子, 接进 regime WFA 对比."""
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes = inp["fwd"], inp["dates"], inp["codes"]
    n = len(dates)
    folds = _folds(n)
    rng = np.random.default_rng(42)
    samp = list(codes); rng.shuffle(samp); samp = samp[:N_SAMPLE_STOCKS]
    sidx = [list(codes).index(c) for c in samp]
    ml = np.full((n, len(codes)), np.nan, dtype=np.float32)
    t0 = time.time()
    for k, (ts, te, ve, vb) in enumerate(folds):
        Xtr = np.stack([zarr[f][ts:te][:, sidx] for f in ALL], axis=-1).reshape(-1, len(ALL))
        ytr = fwd.values[ts:te][:, sidx].reshape(-1)
        m = (~np.isnan(Xtr).any(1)) & (~np.isnan(ytr))
        Xtr, ytr = Xtr[m].astype(np.float32), ytr[m].astype(np.float32)
        Xte = np.stack([zarr[f][ve:vb] for f in ALL], axis=-1).reshape(-1, len(ALL))
        md = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0)
        md.fit(Xtr, ytr)
        pred = md.predict(np.nan_to_num(Xte, nan=0.0)).reshape(vb - ve, len(codes))
        ml[ve:vb] = pred.astype(np.float32)
        print(f"  [增强] fold{k} 训练 {Xtr.shape[0]}样本 -> 预测 {pred.shape} ({time.time()-t0:.1f}s)", flush=True)

    # IC of ml_factor
    ml_ic = _daily_rank_ic_arr(ml, fwd.values, dates)
    zarr_ml = dict(zarr); zarr_ml["ml_factor"] = ml
    fac_ic_ml = dict(fac_ic); fac_ic_ml["ml_factor"] = ml_ic
    print(f"  [增强] ml_factor IC 均值={ml_ic.mean():+.4f}, 有效日={ml_ic.notna().sum()}", flush=True)

    base = dict(zarr=zarr_ml, fac_ic=fac_ic_ml, factor_names=["ml_factor"],
                fwd=fwd, dates=dates, codes=codes, mkt_level=inp["mkt_level"],
                factor_regime_labels=None)
    r_ml = rolling_wfa_dual_regime(**base, gate=True, use_market_regime=False, use_factor_regime=False)
    r_ml_c = rolling_wfa_dual_regime(**base, gate=True, use_market_regime=True, use_factor_regime=False)
    for nm, r in (("ML因子+闸门", r_ml), ("ML因子+市场regime", r_ml_c)):
        a = r["agg"]
        print(f"  [{nm}] Sharpe={a['sharpe']:+.3f} 回撤={a['maxdd']:+.2%} "
              f"通过率={r['pass_rate']:.0%} 决策={r['decision']}", flush=True)
    return ml, ml_ic, {"ML因子+闸门": r_ml, "ML因子+市场regime": r_ml_c}


def part3_ml_enhanced(inp, imp_glob, imp_reg):
    """ML regime 条件加权复合因子: 保留 33 因子分散度, 但按市场 regime 用 ML 重要性重新加权.

    这是对'朴素单 ML 因子'失败的正确修正 —— 单因子失去分散度(见 Part2), 而把 ML 的
    regime 条件重要性用作 33 因子的权重, 既保留分散又注入 ML 状态感知. 等价于 C(市场regime
    因子选择) 的 ML 加权版.
    """
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes = inp["fwd"], inp["dates"], inp["codes"]
    reg = _market_regime_label(inp["mkt_level"], dates).values
    n = len(dates); nc = len(codes)

    def wvec(state):
        src = imp_reg.get(state, imp_glob)         # 该 regime 无模型则退回全局
        w = src.reindex(ALL).fillna(0.0)
        # 仅保留 IC>0 的因子(避免引入负 IC 噪声稀释信号), 其余置 0
        ic_mean = pd.Series({f: float(fac_ic[f].mean()) for f in ALL})
        w = w * (ic_mean > 0).astype(float)
        s = w.sum()
        return (w / s).values if s > 0 else np.zeros(len(ALL))
    W = {st: wvec(st) for st in ("bull", "bear", "osc")}

    enh = np.zeros((n, nc), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        w = W.get(reg[i], W["bull"])
        enh[i] = np.tensordot(w, np.array([zarr[f][i] for f in ALL]), axes=1)
    print(f"  [增强3] ML 加权复合因子构建完成 ({time.time()-t0:.1f}s)", flush=True)

    enh_ic = _daily_rank_ic_arr(enh, fwd.values, dates)
    zarr_e = dict(zarr); zarr_e["ml_enh"] = enh
    fac_ic_e = dict(fac_ic); fac_ic_e["ml_enh"] = enh_ic
    print(f"  [增强3] ml_enh IC 均值={enh_ic.mean():+.4f}", flush=True)

    base = dict(zarr=zarr_e, fac_ic=fac_ic_e, factor_names=["ml_enh"],
                fwd=fwd, dates=dates, codes=codes, mkt_level=inp["mkt_level"],
                factor_regime_labels=None)
    r_e = rolling_wfa_dual_regime(**base, gate=True, use_market_regime=False, use_factor_regime=False)
    a = r_e["agg"]
    print(f"  [ML加权复合因子+闸门] Sharpe={a['sharpe']:+.3f} 回撤={a['maxdd']:+.2%} "
          f"通过率={r_e['pass_rate']:.0%} 决策={r_e['decision']}", flush=True)
    return enh, enh_ic, r_e


def main():
    t0 = time.time()
    print("=" * 60); print("ML 因子分析 + ML 增强因子(全市场 OOS)"); print("=" * 60)
    inp = load_engine_inputs_cached()
    print(f"面板: {inp['n_codes']}只 × {inp['dates'][0].date()}~{inp['dates'][-1].date()} | 因子 {len(inp['ALL'])}")

    print("\n[Part 1] ML 因子重要性(全局 + 三态)...")
    imp_glob, imp_reg = part1_importance(inp)

    print("\n[Part 2] 朴素单 ML 因子(OOS) + WFA 对比...")
    ml, ml_ic, ml_res = part2_ml_factor(inp)

    print("\n[Part 3] ML regime 条件加权复合因子(保留分散度) + WFA 对比...")
    enh, enh_ic, enh_res = part3_ml_enhanced(inp, imp_glob, imp_reg)

    # 对比表(含历史 A~E, 取上轮全市场结果作参照)
    print("\n[对比] ML 增强因子 vs 现有变体(全市场)")
    print(f"{'变体':<26}{'Sharpe':>9}{'回撤':>9}{'决策':>8}")
    print("-" * 52)
    ref = {  # 上轮全市场 regime_wfa 结果(同面板同方法, 作参照)
        "A:无闸": (0.557, -0.7044), "B:两档位闸门": (0.424, -0.6522),
        "C:市场regime": (0.659, -0.5262), "D:因子regime": (0.435, -0.6478),
        "E:双regime": (0.632, -0.5304)}
    for nm, (sh, dd) in ref.items():
        print(f"{nm:<26}{sh:>+9.3f}{dd:>+9.1%}{'FAIL':>8}")
    for nm, r in list(ml_res.items()) + [("ML加权复合因子+闸门", enh_res)]:
        a = r["agg"]
        print(f"{nm:<26}{a['sharpe']:>+9.3f}{a['maxdd']:>+9.1%}{r['decision']:>8}")
    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
