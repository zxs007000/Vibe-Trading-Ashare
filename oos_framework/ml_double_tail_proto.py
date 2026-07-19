# -*- coding: utf-8 -*-
"""双尾防御 ML 门控 · 原型 (用户 2026-07-19 批准试做)

目的: 用 ML 预测"市场尾部风险"(前瞻回撤), 把预测概率作为一道概率型闸门,
      叠加在现有确定性回撤闸(趋势破位/深度回撤→减仓)之上, 演示 ML 如何运作。

为什么 ML 适合这层(接熵模型):
  - 预测的是二阶矩(回撤/波动), 比预测收益方向(一阶矩)平稳得多;
  - 标签是"未来20日最大回撤<-8%"的尾部事件, 不会被交易掉(无对手消耗信号);
  - 特征含估值腿(巴菲特指标=总市值/GDP, PIT对齐) + 波动/动量腿 + 流动性共振(M2)。

纪律(防过拟合/前视):
  - 扩展窗口 Walk-Forward: 只用 t 之前的样本训练, 每20日重训, 预测 t 之后;
  - 所有特征滚动统计量仅用历史(因果); 巴菲特指标按发布日+60天对齐(PIT);
  - 闸门权重 w 只用 t 及之前信息, 赚的是 t→t+1 的收益(无前视)。

对比四档:
  BH        = 满仓买入持有(基准)
  Rule      = 仅确定性回撤闸(现有 market_regime)
  ML        = 仅 ML 概率闸门
  ML+Rule   = 两者取严(min), 系统级组合
"""
import os, sys, time, importlib.util, json
os.environ["LIGHTGBM_VERBOSITY"] = "0"   # 抑制 LightGBM 训练刷屏
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "stockworm"))

# 注意: oos_engine 改为惰性导入(仅在 main() 内), 使本模块可被 defensive_gating
# 跨仓 import ml_gate_weight 而**不触发 oos_engine 解析**(避免与 Vibe 自带 oos_engine 冲突).
# ml_gate_weight 及其依赖(build_features/label/tail_magnitude/fit_gpd_mom/wfa_predict/get_macro)
# 均不依赖 oos_engine, 故跨仓调用零冲突.

# ── 本地 macro 模块(巴菲特指标), 走 importlib 避免包名差异 ──
_macro_path = os.path.join(ROOT, "stockworm", "macro.py")
_spec = importlib.util.spec_from_file_location("macro_local", _macro_path)
macro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(macro)

HOLD_FWD = 20          # 前瞻窗口(交易日): 标签=未来20日最大回撤
TAIL_THR = -0.08       # 尾部事件阈值: 前瞻回撤 < -8%
MIN_TRAIN = 500        # 最小训练样本(日)
RETRAIN_STEP = 20      # 每20日扩展重训
ML_COEF = 1.5          # w = 1 - ML_COEF * p_tail
W_FLOOR = 0.30         # 最低股票仓位(不全清, 保留弹性)
RULE_CAP = 0.50        # 确定性闸触发时股票仓位上限


def get_macro(index):
    """巴菲特指标 + M2同比, 优先读本地 CSV(避免每次重拉 akshare)。"""
    csv_dir = os.path.join(ROOT, "stockworm", "data", "macro")
    os.makedirs(csv_dir, exist_ok=True)
    bpath = os.path.join(csv_dir, "buffett_ratio.csv")
    mpath = os.path.join(csv_dir, "m2_growth.csv")
    if os.path.exists(bpath) and os.path.exists(mpath):
        b = pd.read_csv(bpath, index_col=0, parse_dates=True)["buffett_ratio"]
        m = pd.read_csv(mpath, index_col=0, parse_dates=True)["m2_growth"]
        print(f"  [macro] 读本地缓存: 巴菲特 {len(b)} 点, M2 {len(m)} 点")
    else:
        print("  [macro] 实时拉取 akshare (GDP+总市值+M2)...")
        b = macro.buffett_ratio_daily(index=index)
        m = macro.m2_growth_daily(index=index)
        b.rename("buffett_ratio").to_frame().to_csv(bpath)
        m.rename("m2_growth").to_frame().to_csv(mpath)
        print(f"  [macro] 已缓存: 巴菲特末值 {float(b.iloc[-1]):.3f}")
    return b.reindex(index).ffill(), m.reindex(index).ffill()


def build_features(nav: pd.Series, breadth: pd.Series = None):
    """从市场净值构造市场尾部特征(全部因果, 仅用历史)。"""
    ret = nav.pct_change().fillna(0.0)
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    mom20 = nav / nav.shift(20) - 1.0
    mom60 = nav / nav.shift(60) - 1.0
    dd = nav / nav.cummax() - 1.0
    ma20 = nav.rolling(20).mean()
    ma60 = nav.rolling(60).mean()
    ma20_dev = nav / ma20 - 1.0
    ma60_dev = nav / ma60 - 1.0
    feats = pd.DataFrame({
        "vol20": vol20, "vol60": vol60,
        "vol_ratio": vol20 / (vol60 + 1e-9),
        "mom20": mom20, "mom60": mom60,
        "dd": dd,
        "ma20_dev": ma20_dev, "ma60_dev": ma60_dev,
    })
    if breadth is not None:
        feats["breadth_disp"] = breadth.reindex(nav.index)
    return feats


def build_label(nav: pd.Series):
    """标签: 未来 HOLD_FWD 日最大回撤 < TAIL_THR 记为尾部事件(1)。

    用反转滚动取'未来窗口最小值', 仅用于训练目标(预测时不用)。
    """
    n = len(nav)
    fut_min = nav.copy()
    # 从后往前滚动取最小值 = 每个 t 之后20日的最低净值
    rev = nav[::-1]
    rmin = rev.rolling(HOLD_FWD, min_periods=1).min()[::-1]
    fwd_dd = rmin / nav - 1.0
    return (fwd_dd < TAIL_THR).astype(int)


def build_tail_magnitude(nav: pd.Series):
    """尾部幅值序列(连续, 负值): 未来 HOLD_FWD 日最大回撤. 标签 build_label 的连续版,
    供 GPD 拟合超限幅值(>TAIL_THR 的部分). 仅用历史(因果: 取 t 之后窗口, 预测时不用)."""
    rev = nav[::-1]
    rmin = rev.rolling(HOLD_FWD, min_periods=1).min()[::-1]
    return rmin / nav - 1.0


def rolling_pctile(s: pd.Series, win: int = 1260):
    """5年滚动分位(估值去趋势用)。"""
    return s.rolling(win, min_periods=250).apply(lambda x: (x[-1] <= x).mean(), raw=True)


def fit_gpd_mom(exceedances: pd.Series):
    """广义帕累托(GPD) 矩估计(Method of Moments), 拟合超限样本 y_i = x_i - u (正数).

    返回 (xi, sigma) 或 None(样本不足/矩估计不稳定). 仅对 |xi|<0.5 稳定.
    无外部依赖(不调 scipy), 适合在 WFA 折内逐折扩展拟合.
    """
    y = exceedances.dropna().values.astype(float)
    if len(y) < 10:
        return None
    mean_y = y.mean()
    mean_y2 = (y ** 2).mean()
    if mean_y <= 0 or mean_y2 <= 0:
        return None
    xi = 0.5 * (1.0 - (mean_y ** 2) / mean_y2)
    if xi >= 0.5 or xi <= -0.5:       # MoM 仅对 |xi|<0.5 稳定
        return None
    sigma = 0.5 * mean_y * (1.0 + (mean_y ** 2) / mean_y2)
    if sigma <= 0:
        return None
    return float(xi), float(sigma)


def wfa_predict(X: pd.DataFrame, y: pd.Series, magnitude=None,
               min_train=MIN_TRAIN, step=RETRAIN_STEP):
    """扩展窗口 WFA: 只用历史训练, 周期重训, 输出每日尾部概率 + GPD 尾部强度.

    magnitude: 连续的尾部幅值序列(未来 HOLD_FWD 日最大回撤, 负值). 提供时,
        每折用**训练窗内**超限样本(剔除最后 HOLD_FWD 天避免标签跨折泄漏)拟 GPD,
        把"条件期望超限幅度"映射成逐日 severity 乘子(默认中性 1.0).
        这样 ML 腿在"概率高 且 历史尾部更肥"的日子降仓更狠 —— 即 GPD 尾部风险定价.
    """
    import lightgbm as lgb
    dates = X.index.tolist()
    pred = pd.Series(0.0, index=X.index)
    sev = pd.Series(1.0, index=X.index)     # GPD 尾部强度(中性 1.0)
    i = min_train
    n_models = 0
    while i < len(dates):
        tr = dates[:i]
        te = dates[i: min(i + step, len(dates))]
        Xtr, ytr = X.loc[tr].dropna(), y.loc[tr].dropna()
        Xtr, ytr = Xtr.align(ytr, join="inner", axis=0)
        if len(Xtr) < min_train:
            i += step
            continue
        # 类不平衡: 尾部事件稀有 → 加权重(PR-AUC 友好)
        pos = max(ytr.mean(), 0.05)
        scale = np.where(ytr.values == 1, 1.0 / pos, 1.0 / (1 - pos))
        model = lgb.LGBMClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=30, n_jobs=1, random_state=42,
            class_weight="balanced",
        )
        model.fit(Xtr.values, ytr.values)
        te_X = X.loc[te].dropna()
        pred.loc[te_X.index] = model.predict_proba(te_X.values)[:, 1]
        # ── GPD 尾部强度: 训练窗内超限幅值(剔除最后 HOLD_FWD 天, 防标签跨折) ──
        if magnitude is not None:
            mag_tr = magnitude.loc[tr].dropna()
            mag_tr = mag_tr.iloc[: max(0, len(mag_tr) - HOLD_FWD)]   # 防跨折
            exc = mag_tr[mag_tr < TAIL_THR] - TAIL_THR             # 超限幅度(正数)
            g = fit_gpd_mom(exc)
            if g is not None:
                xi, sigma = g
                es = sigma / max(1e-6, 1.0 - xi)             # 条件期望超限幅度
                sev_val = float(np.clip(es / abs(TAIL_THR), 0.5, 2.5))
                sev.loc[te_X.index] = sev_val
        n_models += 1
        i += step
    print(f"  [WFA] 共重训 {n_models} 次, 预测覆盖 {int((pred > 0).sum())} 日")
    return pred, sev


def ml_gate_weight(nav: pd.Series, buffett=None, m2=None):
    """双尾 ML 腿: 产出日频仓位权重 w_ml∈[W_FLOOR,1].

    特征/标签/宏观腿与 main() 完全一致; 宏观可外部传入(复用调用方已加载的
    巴菲特/M2, 避免二次拉取). 返回对齐 nav.index 的权重序列(无信号日=1.0 满仓).
    """
    feats = build_features(nav)
    if buffett is None or m2 is None:
        buffett, m2 = get_macro(nav.index)
    feats["buffett"] = buffett
    feats["buffett_pctile"] = rolling_pctile(buffett)
    feats["buffett_high"] = (feats["buffett_pctile"] > 0.80).astype(float)
    feats["m2_yoy"] = m2
    feats["m2_resonance"] = (feats["buffett_high"].astype(bool) &
                              (m2 < m2.rolling(250).mean())).astype(float)
    feats = feats.dropna()
    y = build_label(nav).reindex(feats.index)
    mag = build_tail_magnitude(nav).reindex(feats.index)
    pred, sev = wfa_predict(feats, y, magnitude=mag)
    w_ml = (1 - ML_COEF * pred * sev).clip(W_FLOOR, 1.0)
    w_ml = w_ml.reindex(nav.index).fillna(1.0)
    print(f"  [ML腿] 权重覆盖 {int((w_ml < 1).sum())} 日(均值 {w_ml.mean():.3f}, "
          f"最小 {w_ml.min():.3f}); GPD 强度均值 {sev.reindex(w_ml.index).mean():.3f}")
    return w_ml


def main():
    t0 = time.time()
    from oos_engine import (compute_metrics, load_market_index,
                                market_regime, DEFENSIVE_PATH)
    # ── 1. 市场净值(优先真实上证综指, 否则等权全A代理) ──
    print("[1] 加载市场净值...")
    code = os.environ.get("ML_INDEX", "sh000001")
    idx = load_market_index(code, index=None)
    if idx is not None and len(idx) > 200:
        nav = (1 + idx.pct_change().fillna(0)).cumprod()
        src = f"{code}"
    else:
        # 等权全A代理: 用试点面板
        print("  [WARN] 指数缺失, 改用试点面板等权净值")
        from panel_builder import load_panel
        panel = load_panel([
            "600519", "000858", "300750", "601318", "600036", "600276",
            "601012", "600900", "000333", "600887", "601899", "300059",
        ], "2018-01-01", "2026-06-30")
        nav = (1 + panel["close"].pct_change().fillna(0).mean(axis=1)).cumprod()
        src = "试点面板等权"
    nav = nav[nav > 0]
    # 截取近窗口(默认2010起), 排除早期流动性缺失时代, 校准更干净
    START_DATE = os.environ.get("ML_START", "2010-01-01")
    nav = nav[nav.index >= pd.Timestamp(START_DATE)]
    print(f"  净值源: {src}, {nav.index[0].date()}~{nav.index[-1].date()}, {len(nav)} 日 (窗口≥{START_DATE})")

    # ── 2. 特征 + 估值腿(巴菲特指标) + 流动性腿(M2) ──
    print("[2] 构造特征 + 巴菲特指标(估值腿) + M2(流动性腿)...")
    feats = build_features(nav)
    buffett, m2 = get_macro(nav.index)
    feats["buffett"] = buffett
    feats["buffett_pctile"] = rolling_pctile(buffett)
    feats["buffett_high"] = (feats["buffett_pctile"] > 0.80).astype(float)
    feats["m2_yoy"] = m2
    feats["m2_resonance"] = (feats["buffett_high"].astype(bool) & (m2 < m2.rolling(250).mean())).astype(float)
    feats = feats.dropna()
    print(f"  特征矩阵: {feats.shape[1]} 维 × {feats.shape[0]} 日")

    # ── 3. 标签(前瞻20日回撤尾部事件) ──
    y = build_label(nav).reindex(feats.index)
    print(f"  尾部事件占比: {y.mean():.2%} (标签=未来20日回撤<-8%)")

    # ── 4. WFA 训练, 输出每日尾部概率 ──
    print("[4] 扩展窗口 WFA 训练 LightGBM...")
    pred, _ = wfa_predict(feats, y)

    # 可解释性: 用全样本(除最后一年)训一个模型看特征重要性
    split = feats.index[int(len(feats) * 0.85)]
    Xtr, ytr = feats.loc[:split].dropna(), y.loc[:split].dropna()
    Xtr, ytr = Xtr.align(ytr, join="inner", axis=0)
    import lightgbm as lgb
    imp_model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    class_weight="balanced", n_jobs=1, random_state=42)
    imp_model.fit(Xtr.values, ytr.values)
    imp = sorted(zip(feats.columns, imp_model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("  [重要性 gini] " + " | ".join(f"{k}:{int(v)}" for k, v in imp[:8]))

    # ── 5. 闸门权重 ──
    print("[5] 构造闸门权重...")
    w_ml = (1 - ML_COEF * pred).clip(W_FLOOR, 1.0)
    # 确定性回撤闸(现有规则): 趋势破位或深度回撤 → 减仓
    bear = market_regime(nav, trend_window=250, dd_thr=-0.15).shift(1).reindex(nav.index).fillna(False)
    w_rule = pd.Series(np.where(bear, RULE_CAP, 1.0), index=nav.index)
    w_mlrule = pd.concat([w_ml, w_rule], axis=1).min(axis=1)

    # ── 6. 回测(日频, w[t] 决定 t→t+1 持仓) ──
    print("[6] 日频回测(对比四档)...")
    mkt_ret = nav.pct_change()                      # ret[d] = 当天收益
    # 防御收益: 优先读防御组合, 否则合成5%年化(对应50万低风险盘)
    try:
        def_series = pd.read_parquet(DEFENSIVE_PATH)["defensive"]
        def_daily = def_series.reindex(nav.index).ffill().fillna(0.0)
        dsrc = "防御组合parquet"
    except Exception:
        def_daily = pd.Series(0.05 ** (1 / 252) - 1, index=nav.index)
        dsrc = "合成5%年化"
    # t 日权重赚 t→t+1 收益(无前视)
    r_bh = mkt_ret.shift(-1)
    r_rule = w_rule.shift(-1) * mkt_ret.shift(-1) + (1 - w_rule.shift(-1)) * def_daily.shift(-1)
    r_ml = w_ml.shift(-1) * mkt_ret.shift(-1) + (1 - w_ml.shift(-1)) * def_daily.shift(-1)
    r_mr = w_mlrule.shift(-1) * mkt_ret.shift(-1) + (1 - w_mlrule.shift(-1)) * def_daily.shift(-1)
    cmp = pd.DataFrame({"BH": r_bh, "Rule": r_rule, "ML": r_ml, "ML+Rule": r_mr}).dropna()
    m = compute_metrics(cmp, periods_per_year=252)
    print(f"  防御收益源: {dsrc}")

    # ── 7. 校准(预测概率 vs 真实尾部频率) ──
    print("[7] 可靠性校准(预测分位 → 真实尾部率)...")
    df = pd.DataFrame({"p": pred.reindex(cmp.index).dropna(), "tail": y.reindex(cmp.index).dropna()})
    df = df.dropna()
    bins = pd.qcut(df["p"], 5, duplicates="drop")
    cal = df.groupby(bins, observed=True)["tail"].agg(["mean", "count"])
    pred_mean = df.groupby(bins, observed=True)["p"].mean()
    for b, row in cal.iterrows():
        pm = pred_mean.loc[b]
        print(f"    分位{b}  预测均值={float(pm):.2f}  真实尾部率={row['mean']:.2%}  n={int(row['count'])}")

    # ── 8. 输出 ──
    print("\n" + "=" * 70)
    print(f"双尾 ML 门控 · 回测对比 (标签=未来{HOLD_FWD}日回撤<-{int(-TAIL_THR*100)}% | 市场:{src})")
    print("=" * 70)
    print(f"{'策略':<10}{'年化':>9}{'Sharpe':>9}{'MaxDD':>10}{'累计':>10}")
    for k in ["BH", "Rule", "ML", "ML+Rule"]:
        v = m.get(k, {})
        print(f"{k:<10}{v.get('annual_return',0):>8.2%}{v.get('sharpe',0):>9.2f}"
              f"{v.get('max_drawdown',0):>10.2%}{v.get('cumulative_return',0):>10.1%}")
    print(f"\n总耗时 {time.time()-t0:.1f}s")

    out = {"src": src, "n_days": len(nav),
           "tail_rate": float(y.mean()),
           "metrics": {k: m.get(k, {}) for k in ["BH", "Rule", "ML", "ML+Rule"]},
           "importance": {k: int(v) for k, v in imp}}
    with open("ml_double_tail_proto.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("已保存 ml_double_tail_proto.json")


if __name__ == "__main__":
    main()
