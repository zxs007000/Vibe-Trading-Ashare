# -*- coding: utf-8 -*-
"""选项 A: 把双尾 ML 闸叠到"选股策略净值"上演示护盘效果。

约束: Vibe defensive_gating 的基线 A 真实净值(pA)需跑完整个 WFA 因子流水线才能拿到,
而本环境无 _wfa_cache(重算需 12G 因子, 会 OOM), 故用与基线 A 统计对齐的重构净值:
  真实沪深300路径 + 杠杆(LEV) + 每日 alpha(ALPHA_DAILY), 调到 年化≈+23.2% / MaxDD≈-68.1%。
闸门逻辑(特征/标签/WFA/LightGBM/确定性闸)完全复用 ml_double_tail_proto, 不重新发明。
"""
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "stockworm"))

from oos_engine import load_market_index, market_regime, compute_metrics, DEFENSIVE_PATH
from ml_double_tail_proto import (build_features, build_label, wfa_predict, rolling_pctile,
                                  get_macro, ML_COEF, W_FLOOR, RULE_CAP)

# 对齐基线 A (+23.2% / -68.1%): 沪深300 自身年化~5.1%, MaxDD~-46.7%
LEV = 1.75
ALPHA_DAILY = 0.142 / 252


def main():
    t0 = time.time()
    # ── 1. 真实沪深300 路径 ──
    idx = load_market_index("sh000300", index=None)
    r_m = idx.pct_change().fillna(0.0)
    # ── 2. 重构选股净值: 杠杆 + alpha, 对齐基线 A ──
    r_stock = (LEV * r_m + ALPHA_DAILY).clip(-0.15, 0.15)
    nav = (1 + r_stock).cumprod()
    nav = nav[nav.index >= pd.Timestamp("2010-01-01")]
    r_stock = r_stock.reindex(nav.index)
    eq = nav

    # ── 3. 防御底仓(真实) ──
    try:
        def_series = pd.read_parquet(DEFENSIVE_PATH)["defensive"]
        def_daily = def_series.reindex(nav.index).ffill().fillna(0.0)
        dsrc = "防御组合parquet"
    except Exception:
        def_daily = pd.Series(0.05 ** (1 / 252) - 1, index=nav.index)
        dsrc = "合成5%"
    print(f"  防御收益源: {dsrc}")

    # ── 4. 特征 + 宏观腿(估值/流动性) ──
    feats = build_features(nav)
    buffett, m2 = get_macro(nav.index)
    feats["buffett"] = buffett
    feats["buffett_pctile"] = rolling_pctile(buffett)
    feats["buffett_high"] = (feats["buffett_pctile"] > 0.80).astype(float)
    feats["m2_yoy"] = m2
    feats["m2_resonance"] = (feats["buffett_high"].astype(bool) &
                              (m2 < m2.rolling(250).mean())).astype(float)
    feats = feats.dropna()
    y = build_label(nav).reindex(feats.index)

    # ── 5. WFA 训练 LightGBM, 输出每日尾部概率 ──
    pred, _ = wfa_predict(feats, y)

    # ── 5b. 重构选股净值(回测窗口 = WFA 有预测覆盖的日期, 与下方对比表 BH 同源) ──
    bt_idx = pred.dropna().index
    r_book = r_stock.reindex(bt_idx)
    r_cum = (1 + r_book).cumprod()
    ann = r_cum.iloc[-1] ** (252 / len(r_book)) - 1
    maxdd = (r_cum / r_cum.cummax() - 1).min()
    print(f"[重构选股净值·回测窗口] CAGR={ann:+.2%}  MaxDD={maxdd:+.2%}  "
          f"窗口={bt_idx[0]:%Y-%m-%d}~{bt_idx[-1]:%Y-%m-%d}  "
          f"(目标对齐 基线A +23.2%/-68.1%, 此窗口即闸门生效区间)")

    # ── 6. 闸门权重 ──
    w_ml = (1 - ML_COEF * pred).clip(W_FLOOR, 1.0)
    bear = market_regime(nav, trend_window=250, dd_thr=-0.15).shift(1).reindex(nav.index).fillna(False)
    w_rule = pd.Series(np.where(bear, RULE_CAP, 1.0), index=nav.index)
    w_mlrule = pd.concat([w_ml, w_rule], axis=1).min(axis=1)

    # ── 7. 日频回测(四档) ──
    mkt_ret = nav.pct_change()
    r_bh = mkt_ret.shift(-1)
    r_rule = w_rule.shift(-1) * mkt_ret.shift(-1) + (1 - w_rule.shift(-1)) * def_daily.shift(-1)
    r_ml = w_ml.shift(-1) * mkt_ret.shift(-1) + (1 - w_ml.shift(-1)) * def_daily.shift(-1)
    r_mr = w_mlrule.shift(-1) * mkt_ret.shift(-1) + (1 - w_mlrule.shift(-1)) * def_daily.shift(-1)
    cmp = pd.DataFrame({"BH": r_bh, "Rule": r_rule, "ML": r_ml, "ML+Rule": r_mr}).dropna()
    m = compute_metrics(cmp, periods_per_year=252)

    # ── 8. 校准 ──
    df = pd.DataFrame({"p": pred.reindex(cmp.index).dropna(),
                       "tail": y.reindex(cmp.index).dropna()}).dropna()
    bins = pd.qcut(df["p"], 5, duplicates="drop")
    cal = df.groupby(bins, observed=True)["tail"].agg(["mean", "count"])
    pred_mean = df.groupby(bins, observed=True)["p"].mean()

    # ── 9. 输出 ──
    print("\n" + "=" * 64)
    print("双尾 ML 闸 · 叠到选股净值(对齐基线A) · 路径=沪深300+杠杆")
    print("=" * 64)
    n_days = len(cmp)
    print(f"{'策略':<10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>10}{'累计':>10}")
    for k in ["BH", "Rule", "ML", "ML+Rule"]:
        v = m.get(k, {})
        cum = v.get('cumulative_return', 0)
        geom = (1 + cum) ** (252 / n_days) - 1   # 几何年化(与首行一致), 非算术均值×252
        print(f"{k:<10}{geom:>8.2%}{v.get('sharpe', 0):>9.2f}"
              f"{v.get('max_drawdown', 0):>10.2%}{cum:>10.1%}")
    print(f"\n[校准] 无条件尾部率(全样本)={y.reindex(cmp.index).mean():.2%}  ← ML分位应显著高于它才算有效")
    print("[校准] 预测概率分位 → 真实未来20日尾部率:")
    for b, row in cal.iterrows():
        pm = pred_mean.loc[b]
        print(f"  预测{pm:.2f} → 真实尾部率={row['mean']:.2%}  n={int(row['count'])}")
    print(f"\n总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
