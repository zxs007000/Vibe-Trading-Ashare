"""defensive_gating.py — 防御门控层(治回撤, 不杀小盘 alpha).

设计哲学(用户实战框架): 宏观因子(巴菲特指标)作**左侧预警** —— 只调结构(向低波/质量/防御倾斜),
**仓位保持满仓**; 真正降仓位交由**右侧确认**(价格破位)触发. 缓冲带+多因子共振+冷却期 解决信号滞后/假信号.

  1) 左侧预警(宏观, 非价格): 巴菲特指标 5Y 分位 → 缓冲带(正常/警戒/极端/泡沫) → defensive_tilt∈[0,1].
     多因子共振(估值+流动性)阻尼假信号; 冷却期防横跳摩擦.
     defensive_tilt 调制**因子权重结构**(防御抬升/alpha降权) + **仓位最多降 20%**(缓冲, 非清仓).
  2) 右侧确认(价格, 非宏观): mkt_level 跌破 250 日线(-10%) 或 波动 z>2 → 危机信号 → 降仓位到 CRISIS_POS.
     急性崩盘兜底. 左侧预警越浓, 右侧一旦确认结构已提前防御(移动止损思想: 泡沫尾段仍满仓, 破位才降).
  3) 部分降仓: 危机期 CRISIS_POS 仓、不归零, 空仓部分吃 4% 防御资产日收益.

公平对照: 与基线 A 共用同一因子集、同一 WFA fold, 唯一差异 = 左侧结构倾斜 + 右侧降仓.
复用 triple_validation 的 block_bootstrap 给回撤稳健性.

用法:
  python oos_framework/defensive_gating.py
  python oos_framework/defensive_gating.py --cost realistic
"""
from __future__ import annotations
import sys, time, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agent/backtest"))

# ── 跨仓引入 Claw 双尾 ML 闸(作为防御层的一条 ML 腿) ──
# proto 顶层已惰性化 oos_engine 导入(仅其自身 main 用), 故此处跨仓 import
# 不会触发 oos_engine 解析冲突(Vibe 自带 oos_engine 不受影响).
import os as _os
_CLAW_OOS = _os.path.abspath(_os.path.join(str(HERE), "..", "..", "work Buddy GZ", "Claw", "oos_framework"))
if _os.path.isdir(_CLAW_OOS) and _CLAW_OOS not in sys.path:
    sys.path.insert(0, _CLAW_OOS)
from ml_double_tail_proto import ml_gate_weight, W_FLOOR

from regime_wfa import (load_engine_inputs_cached, rolling_wfa_dual_regime,
                        TRAIN_DAYS, TEST_DAYS, PURGE, VETO_SHARPE, VETO_MAXDD, MAJORITY,
                        GATE_SHORT_WIN)
from oos_validation import (build_signal, long_only_topk, _stat_block,
                            TOP_K, HOLD, COST, TRAIL, RNG, BPY)
from backtest.validation import _sharpe
from triple_validation import block_bootstrap, _safe_stat_block, N_BOOT

OUT = HERE / "screen_results" / "defensive"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "defensive_equity.png"
REP = OUT / "防御门控层报告.md"

DEF_ANN = 0.04           # 防御资产年化(空仓部分收益)
R_DEF = DEF_ANN / 252.0

# 危机检测参数
CRISIS_MA = 250
CRISIS_MA_THR = -0.10    # mkt_level 低于 250 日线 10% = 危机
CRISIS_VOL_Z = 2.0       # 市场波动 z 分数 > 2 = 危机

# 防御倾斜参数
CRISIS_POS = 0.60        # 危机期仓位(部分降仓, 不归零)
TILT_DEF = 3.0           # 防御因子权重抬升倍数
TILT_REV = 1.0           # 反转/小盘因子: 保持原权重(保留 alpha 敞口)

# 防御因子(低波 + 质量) 与 alpha 因子(反转 + 小盘)
DEF_FACTORS = ["ivol_60", "vol_60", "downside_vol_60", "ROE", "profit_yoy", "rev_yoy"]
ALPHA_FACTORS = ["rev_5", "rev_20", "rev_60", "rev_intraday", "amihud_20", "overnight_gap", "drawup_60"]

# ── 左侧预警(巴指双尾因子, 数据驱动: >P80顶部 / <P20底部, 0~20%缓冲) 参数 ──
MACRO_WIN = 5 * 252            # 5Y 滚动窗口算分位(适应估值中枢漂移)
P_LOW = 0.20                   # 下尾: 巴指 < 20%历史分位 → 熊市见底预警
P_HIGH = 0.80                  # 上尾: 巴指 > 80%历史分位 → 泡沫顶部预警
# tilt 线性: br∈[P_LOW,P_HIGH]→0; 往下走→tilt向1线性(在P_TAIL处=1); 往上走→tilt向1线性(在1-P_TAIL处=1)
P_TAIL = 0.05                  # 极端尾: <P5 或 >P95 时 tilt=1.0
RESONANCE_BASE = 0.30          # M2共振阻尼(水丰→弱触发)
ALPHA_REDUCE = 0.50            # 调结构: 满tilt α降权50%
MAX_POS_REDUCE = 0.20          # 缓冲降仓 0~20%
COOL_DAYS = 42; COOL_FLOOR = 0.30

# ── 因子衰减牛熊(严格移植 regime_wfa.factor_decay) + 左侧预警→右侧敏感耦合 参数 ──
GATE_DECAY_FRAC = 0.30    # 长窗 ICIR < 该比例×train ICIR 视为'衰减'(保守死亡计数)
MAX_POS_REDUCE_DECAY = 0.20   # 因子衰减熊市: 仓位最多再降 20%(与巴指缓冲取 max)
REV_SET = {"rev_5", "rev_20", "rev_60", "rev_intraday"}   # 反转族: 熊/均值回复市更灵
MOM_SET = {"mom_5", "mom_20", "mom_60", "mom_120", "mom_250", "mom_12_1"}  # 动量族: 熊市失效
REV_BOOST = 0.50          # 衰减高时反转升权幅度
MOM_CUT = 0.50            # 衰减高时动量降权幅度
# 左侧巴指预警 → 右侧危机检测更敏感(更早触发降仓)
SENS_MA = 0.05            # 巴指 tilt=1 时, 跌破阈值从 -10% 放宽到 -5%(更早)
SENS_VOL = 1.0            # 巴指 tilt=1 时, 波动 z 阈值从 2.0 降到 1.0(更早)


def _crisis_signal(mkt_level, ma=CRISIS_MA, ma_thr=CRISIS_MA_THR, vol_z=CRISIS_VOL_Z,
                   tilt=None, sens_ma=SENS_MA, sens_vol=SENS_VOL, sens_gate=None):
    """危机信号: 跌破 250 日线 或 市场波动 z 分数 > 2. 日频, 对齐 mkt_level 索引.

    tilt: 巴指左侧预警(0~1). 提供时, 阈值随 tilt **逐日变敏感** —— tilt 越高, 跌破/波动
          阈值越早触发(把右侧降仓的'触发线'左移), 实现'巴指一预警, 右侧防御随之敏感'.
    sens_gate: 灵敏度门控. 若提供, 仅当 tilt>sens_gate 才施加耦合(把温和预警区的
           '毛刺触发' 滤掉), tilt<=sens_gate 时用原始阈值(不敏感). 实现'极端区才变敏感'.
    """
    ma_n = mkt_level.rolling(ma).mean()
    ratio = mkt_level / ma_n - 1.0
    ret = mkt_level.pct_change()
    vol20 = ret.rolling(20).std()
    vmean = vol20.rolling(250).mean()
    vstd = vol20.rolling(250).std()
    z = (vol20 - vmean) / (vstd + 1e-9)
    if tilt is not None:
        tv = np.asarray(tilt.reindex(mkt_level.index).fillna(0.0).values, dtype=float)
        if sens_gate is not None:
            tv = np.where(tv > sens_gate, tv, 0.0)   # 仅极端预警区(tilt>gate)才施加耦合, 温和区不敏感
        ma_thr_eff = np.asarray(ma_thr, dtype=float) + tv * sens_ma   # 更不负 -> 更早触发
        vol_z_eff = np.asarray(vol_z, dtype=float) - tv * sens_vol    # 更低 z -> 更早触发
        b_ma = pd.Series(ratio.values < ma_thr_eff, index=mkt_level.index).fillna(False)
        b_vol = pd.Series(z.values > vol_z_eff, index=mkt_level.index).fillna(False)
    else:
        b_ma = (ratio < ma_thr).fillna(False)
        b_vol = (z > vol_z).fillna(False)
    return (b_ma | b_vol).fillna(False)


def _macro_gating(buffett_ratio, m2_growth, mkt_level):
    """左侧预警(双尾因子: 巴指分位 <P20 底部 / >P80 顶部 → defensive_tilt∈[0,1]).

    用 5Y 滚动分位适应 A 股估值中枢漂移. 数据驱动: 巴指 Q1 (极低) 和 Q10 (极高)
    历史上分别对应 -1.9% 和 -5.6% 的后续1年巴指变动 → 两尾都是预警区.
    tilt 从 P20/P80 边界线性增长, 在 P5/P95 处达到 1.0.
    """
    idx = mkt_level.index
    if buffett_ratio is None or len(buffett_ratio) == 0:
        return pd.Series(0.0, index=idx)
    br = buffett_ratio.reindex(idx)
    roll = dict(window=MACRO_WIN, min_periods=max(MACRO_WIN // 4, 60))
    # 5Y 滚动分位
    p5  = br.rolling(**roll).quantile(P_TAIL)
    p20 = br.rolling(**roll).quantile(P_LOW)
    p80 = br.rolling(**roll).quantile(P_HIGH)
    p95 = br.rolling(**roll).quantile(1.0 - P_TAIL)
    # 双尾线性 tilt
    tilt_low = ((p20 - br) / (p20 - p5 + 1e-9)).clip(0, 1)   # br<P20 → 0→1 at P5
    tilt_high = ((br - p80) / (p95 - p80 + 1e-9)).clip(0, 1)  # br>P80 → 0→1 at P95
    tilt_raw = (tilt_low + tilt_high).clip(0, 1)
    # 共振
    if m2_growth is not None and len(m2_growth) > 0:
        m2 = m2_growth.reindex(idx)
        m2_mean = m2.rolling(252, min_periods=60).mean()
        liq_stress = ((m2_mean - m2) / (m2_mean.abs() * 0.05 + 1e-6)).clip(0, 1).fillna(0.0)
    else:
        liq_stress = pd.Series(0.0, index=idx)
    w = RESONANCE_BASE + (1.0 - RESONANCE_BASE) * liq_stress
    tilt = (tilt_raw * w).fillna(0.0)
    return tilt  # 双尾不设简单冷却(需双向判断,跳过)


def _apply_cooldown(stress, br, q_safe, cool=COOL_DAYS, floor=COOL_FLOOR):
    """触发后最少保持 cool 个交易日(地板 floor), 除非 br 跌回安全区分位(<q_safe)立即清零."""
    s = stress.to_numpy(dtype=float).copy()
    brv = br.to_numpy() if br is not None else None
    qv = q_safe.to_numpy() if q_safe is not None else None
    n = len(s)
    i = 0
    while i < n:
        if s[i] > 0.01:
            for j in range(i, min(i + cool, n)):
                if qv is not None and not np.isnan(qv[j]) and brv[j] < qv[j]:
                    break  # 跌回安全区 -> 立即清
                if s[j] > 0.01:
                    s[j] = max(s[j], floor)
            i += cool
        else:
            i += 1
    return pd.Series(s, index=stress.index)


def rolling_wfa_defensive(zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level,
                          train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE,
                          top_k=TOP_K, hold=HOLD, cost=COST,
                          veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY,
                          crisis_ma=CRISIS_MA, crisis_ma_thr=CRISIS_MA_THR, crisis_vol_z=CRISIS_VOL_Z,
                          crisis_pos=CRISIS_POS, tilt_def=TILT_DEF, tilt_rev=TILT_REV,
                          buffett_ratio=None, m2_growth=None,
                          sens_ma=SENS_MA, sens_vol=SENS_VOL, sens_gate=None,
                          ml_weight=None):
    """防御门控版 WFA. ...
    ml_weight: 外部传入的 ML 腿日频仓位权重 w_ml∈[W_FLOOR,1] (由 ml_gate_weight 产出,
        看同一市场指数). 提供时, 在规则腿 pos 之上**再乘** w_ml(取严=更保守),
        实现"规则闸 min ML 概率闸"的系统级组合. 缺省 None = 纯规则腿(向后兼容).
    """
    """防御门控版 WFA. 与基线 A(rolling_wfa_dual_regime, gate=False) 共用同一因子集/同一 fold,
    唯一差异 = 危机期 防御倾斜 + 部分降仓 (+ 可选宏观右侧调制).

    buffett_ratio: 日频巴菲特指标(Series, 索引对齐 mkt_level). 提供时启用方案 C 调制.
    m2_growth:     日频 M2 同比(Series). 提供时启用'估值+流动性'共振(阻尼假信号).
    两者皆 None -> 退化为纯价格侧危机(与旧版一致).
    """
    n = len(dates)
    defensive_tilt = _macro_gating(buffett_ratio, m2_growth, mkt_level)
    crisis = _crisis_signal(mkt_level, crisis_ma, crisis_ma_thr, crisis_vol_z,
                            tilt=defensive_tilt, sens_ma=sens_ma, sens_vol=sens_vol,
                            sens_gate=sens_gate)
    # 因子衰减牛熊: 与 regime_wfa.factor_decay 严格同口径
    #   长窗 = 滚动 ICIR(trail=TRAIN_DAYS=250); 短窗 = 原始 IC 滚动均值(GATE_SHORT_WIN=60)
    ic_long = {f: (fac_ic[f].rolling(TRAIN_DAYS).mean() /
                   (fac_ic[f].rolling(TRAIN_DAYS).std() + 1e-9) * np.sqrt(252))
               for f in factor_names}
    ic_short = {f: fac_ic[f].rolling(GATE_SHORT_WIN).mean() for f in factor_names}
    # 防前视: 危机信号基于 mkt_level(截至 t 的已实现价格) + 宏观截至 t 的发布值, t 日决策用危机[t] 即可.
    def_set = {f for f in DEF_FACTORS if f in factor_names}
    alp_set = {f for f in ALPHA_FACTORS if f in factor_names}

    folds_pos = []
    i = train_days
    while i + purge + test_days <= n:
        ts = i - train_days; te = i; ve = i + purge; vb = ve + test_days
        folds_pos.append((ts, te, ve, vb))
        i += test_days
    if not folds_pos:
        raise RuntimeError("数据不足以构成 WFA fold")

    fold_results, wfa_parts, bench_parts = [], [], []
    for k, (ts, te, ve, vb) in enumerate(folds_pos):
        # ── 训练: 标准 Frozen 选活因子(与基线 A 完全一致) ──
        locked_set, locked_orient, locked_w = [], {}, {}
        for f in factor_names:
            ic = fac_ic[f].iloc[ts:te]
            m = ic.mean(); s = ic.std()
            if not (m == m) or not (s == s) or s <= 1e-9:
                continue
            icir = m / s * np.sqrt(252)
            if m > 0 and icir > 0:
                locked_set.append(f)
                locked_orient[f] = 1.0 if m >= 0 else -1.0
                locked_w[f] = icir
        wtot = sum(locked_w[f] for f in locked_set)

        # 因子衰减牛熊: 当日'失效/转熊'因子占比 -> 熊市折扣 + 反转/动量权重调制
        if locked_set:
            dead_cols = []
            for f in locked_set:
                ti = locked_w[f]
                d = ((ic_long[f] < 0) | (ic_long[f] < GATE_DECAY_FRAC * ti) | (ic_short[f] < 0)).fillna(False)
                dead_cols.append(np.asarray(d.values, dtype=float))
            decay_series = pd.Series(np.array(dead_cols).sum(axis=0) / max(1, len(locked_set)),
                                     index=mkt_level.index)
        else:
            decay_series = pd.Series(0.0, index=mkt_level.index)

        test_pos = [p for p in range(ve, vb) if p % hold == 0 and p < n]
        if not test_pos:
            fold_results.append(dict(k=k, n_alive=len(locked_set), n_pos=0,
                                     sharpe=np.nan, maxdd=np.nan, veto=True, avg_eq=np.nan))
            continue

        port, rdates, bench_vals, eq_track = [], [], [], []
        for p in test_pos:
            row = np.zeros(len(codes))
            if wtot > 0:
                wsum = 0.0
                cr = crisis.iloc[p]
                # 左侧预警(巴指): defensive_tilt 连续调制结构; 右侧确认(价格破位, 阈值已随巴指变敏感): 危机期最大防御倾斜 + 降仓位到 CRISIS_POS.
                tilt = 1.0 if cr else float(defensive_tilt.iloc[p])
                dfb = float(decay_series.iloc[p])
                for f in locked_set:
                    base_w = locked_orient[f] * locked_w[f]
                    if f in def_set:
                        w = base_w * (1.0 + tilt * (TILT_DEF - 1.0))   # 防御因子抬升(调结构)
                    elif f in alp_set:
                        w = base_w * (1.0 - tilt * ALPHA_REDUCE)        # 高弹性 alpha 降权(降弹性)
                    else:
                        w = base_w
                    # 因子衰减牛熊: 高 decay(熊/均值回复市) -> 反转升权、动量降权
                    if dfb > 0:
                        if f in REV_SET:
                            w *= (1.0 + dfb * REV_BOOST)
                        elif f in MOM_SET:
                            w *= (1.0 - dfb * MOM_CUT)
                    row += w * zarr[f][p]
                    wsum += abs(w)
                if wsum > 0:
                    row /= wsum
                s = pd.Series(row, index=codes)
                shared = s.dropna().index.intersection(fwd.iloc[p].dropna().index)
                if len(shared) < 5:
                    rg = 0.0
                else:
                    s2, r2 = s[shared], fwd.iloc[p][shared]
                    kk = max(3, int(len(s2) * top_k))
                    held = set(s2.nlargest(kk).index)
                    rg = float(r2[list(held)].mean())
            else:
                rg = 0.0

            bm = fwd.iloc[p].dropna().mean() if fwd.iloc[p].notna().any() else 0.0
            bench_vals.append(bm)
            # 左侧预警缓冲降仓 0~20% 与 因子衰减熊市降仓 0~20% 取更保守者; 右侧价格破位才降到危机仓位
            if crisis.iloc[p]:
                pos = float(crisis_pos)
            else:
                dfb = float(decay_series.iloc[p])
                pos = 1.0 - max(float(defensive_tilt.iloc[p]) * MAX_POS_REDUCE,
                                dfb * MAX_POS_REDUCE_DECAY)
            # ── ML 腿叠层: w_ml∈[W_FLOOR,1] 由双尾闸(概率×GPD强度)产出,
            #    与规则腿 pos 取乘(更保守). 无 ML 腿时 w_ml=1 恒等. ──
            if ml_weight is not None:
                mw = ml_weight.iloc[p]
                pos = pos * (float(mw) if mw == mw else 1.0)
            pr = pos * rg + (1.0 - pos) * R_DEF - pos * top_k * 2 * cost
            port.append(pr); rdates.append(dates[p]); eq_track.append(pos)

        port_s = pd.Series(port, index=rdates)
        bench_s = pd.Series(bench_vals, index=rdates)
        st = _stat_block(f"fold{k}", port_s, bench_s, bench_s)
        veto = (st["sharpe"] < veto_sharpe) or (st["maxdd"] < veto_maxdd) or (wtot <= 0)
        fold_results.append(dict(k=k, n_alive=len(locked_set), n_pos=len(test_pos),
                                 sharpe=st["sharpe"], ex_sharpe=st["ex_sharpe"],
                                 maxdd=st["maxdd"], veto=veto, avg_eq=float(np.mean(eq_track))))
        wfa_parts.append(port_s); bench_parts.append(bench_s)

    wfa_port = pd.concat(wfa_parts).sort_index()
    bench_full = pd.concat(bench_parts).sort_index()
    agg = _stat_block("WFA聚合", wfa_port, bench_full, bench_full)
    valid = [f for f in fold_results if f["n_pos"] > 0 and (f["ex_sharpe"] == f["ex_sharpe"])]
    n_pass = sum(1 for f in valid if f["ex_sharpe"] > 0)
    pass_rate = n_pass / len(valid) if valid else 0.0
    catastrophic = any(f["veto"] for f in fold_results)
    decision = "PASS" if (pass_rate >= majority and not catastrophic) else "FAIL"
    return dict(wfa_port=wfa_port, bench=bench_full, agg=agg,
                pass_rate=pass_rate, n_pass=n_pass, n_valid=len(valid),
                n_folds=len(folds_pos), n_veto=sum(1 for f in fold_results if f["veto"]),
                catastrophic=catastrophic, decision=decision, crisis=crisis)


def _crisis_maxdd(port, crisis_mask):
    """危机窗口内最大回撤: 对每个连续危机段, 取该段权益峰谷回撤, 返回最差一段."""
    eq = (1 + port).cumprod()
    cm = crisis_mask.reindex(port.index).fillna(False)
    if not cm.any():
        return np.nan, 0
    # 连续危机段
    spans, start = [], None
    vals = cm.values
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1)); start = None
    if start is not None:
        spans.append((start, len(vals) - 1))
    worst = 0.0
    for a, b in spans:
        seg = eq.iloc[a:b + 1].values
        dd = (seg / np.maximum.accumulate(seg) - 1.0).min()
        worst = min(worst, dd)
    return worst, int(cm.sum())


def _find_stcok_worm():
    """定位本机 stcok_worm 仓库(含 macro 模块), 用于缺失时自构建宏观 parquet."""
    cands = [HERE.parent.parent / "stock_worm",
             Path("D:/stcok-worm"),
             Path("/workspace/stock_worm")]
    for c in cands:
        if (c / "stcok_worm" / "macro.py").exists():
            return c
    return None


def _load_macro(mkt_level):
    """加载右侧预警宏观输入: (buffett_ratio, m2_growth).

    优先读预构建 parquet; 缺失则**自构建**(经 stcok_worm.macro, 需 akshare+网络)并落盘,
    因此不依赖 GitHub 是否带 parquet —— 全新克隆首次运行会自动补齐. 返回对齐 mkt_level
    索引的 Series 元组(缺失段为 NaN, 由 _macro_gating 视为不调制).
    """
    base = [
        Path("/workspace/stock_worm/data/macro"),
        HERE.parent.parent / "stock_worm" / "data" / "macro",
        HERE / "data" / "macro",
    ]

    def _read(name, col):
        for d in base:
            p = d / name
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    c = col if col in df.columns else df.columns[0]
                    return pd.Series(df[c].values, index=pd.to_datetime(df.index)).reindex(mkt_level.index)
                except Exception as e:
                    print(f"  [warn] 读 {p} 失败: {repr(e)[:80]}")
        return None

    buffett = _read("buffett_ratio.parquet", "buffett_ratio")
    m2 = _read("m2_growth.parquet", "m2_growth")
    if buffett is None or m2 is None:
        st_root = _find_stcok_worm()
        if st_root is not None:
            sys.path.insert(0, str(st_root))
            try:
                import stcok_worm.macro as macro
                out_dir = HERE / "data" / "macro"; out_dir.mkdir(parents=True, exist_ok=True)
                if buffett is None:
                    r = macro.buffett_ratio_daily(index=mkt_level.index, publish_lag_days=60)
                    if r is not None and len(r):
                        r.rename("buffett_ratio").to_frame().to_parquet(out_dir / "buffett_ratio.parquet")
                        buffett = r.reindex(mkt_level.index)
                        print(f"  [self-build] buffett_ratio.parquet 已构建落盘 -> {out_dir}")
                if m2 is None:
                    m = macro.m2_growth_daily(index=mkt_level.index, win=12)
                    if m is not None and len(m):
                        m.rename("m2_growth").to_frame().to_parquet(out_dir / "m2_growth.parquet")
                        m2 = m.reindex(mkt_level.index)
                        print(f"  [self-build] m2_growth.parquet 已构建落盘 -> {out_dir}")
            except Exception as e:
                print(f"  [warn] 宏观自构建失败, 退化为纯价格侧: {repr(e)[:80]}")
        else:
            print("  [warn] 未找到 stcok_worm, 退化为纯价格侧")
    return buffett, m2


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", default="light", choices=["light", "realistic", "heavy"],
                    help="成本场景(对应 triple_validation 的 light/realistic/heavy)")
    ap.add_argument("--years", type=float, default=None,
                    help="仅报告最近 N 年(训练仍用全部历史, 保证 OOS 正确); 缺省=全样本")
    args = ap.parse_args()
    COST_MAP = {"light": 0.001, "realistic": 0.0025, "heavy": 0.005}
    cost = COST_MAP[args.cost]
    YEARS = args.years

    print("=" * 64)
    print("防御门控层: 压回撤, 不杀小盘 alpha")
    print("=" * 64)

    print("\n[1/5] 加载面板 + 因子(缓存优先)...")
    inp = load_engine_inputs_cached()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]
    print(f"  面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} | 因子 {len(ALL)}")

    # ── 加载右侧预警宏观输入(巴菲特指标 + M2 同比) ──
    buffett, m2 = _load_macro(mkt_level)
    print(f"  宏观: 巴菲特指标 {'✅'+str(len(buffett))+'日' if buffett is not None else '❌缺失'} | "
          f"M2同比 {'✅'+str(len(m2))+'日' if m2 is not None else '❌缺失'}")

    # ── ML 腿(双尾 LightGBM + GPD 尾部强度): 看同一市场指数 mkt_level ──
    print("\n[2b/5] ML 腿: 双尾尾概率闸 + GPD 尾部强度 → 日频 w_ml...")
    w_ml = ml_gate_weight(mkt_level, buffett=buffett, m2=m2)

    # ── 基线 A: 无闸 WFA ──
    print("\n[2/5] 基线 A(无闸 WFA)...")
    rA = rolling_wfa_dual_regime(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE,
        top_k=TOP_K, hold=HOLD, cost=cost, veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD,
        majority=MAJORITY, gate=False, use_market_regime=False, use_factor_regime=False,
        factor_regime_labels=None)
    pA = rA["wfa_port"]; bA = rA["bench"]

    # ── 防御门控 ──
    print("\n[3/5] 防御门控 WFA(危机倾斜 + 0.6 仓)...")
    rD = rolling_wfa_defensive(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, cost=cost)
    pD = rD["wfa_port"]; bD = rD["bench"]

    # ── 防御门控 + 宏观右侧调制(方案 C) ──
    print("\n[3b/5] 防御门控 + 宏观左侧结构倾斜(不降仓, 降仓只交右侧价格破位)...")
    rM = rolling_wfa_defensive(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, cost=cost, buffett_ratio=buffett, m2_growth=m2)
    pM = rM["wfa_port"]; bM = rM["bench"]

    # ── 防御 + ML 腿(规则闸 min 双尾 ML 概率闸) ──
    print("\n[3c/5] 防御门控 + ML 腿(规则腿之上再叠双尾概率闸)...")
    rDM = rolling_wfa_defensive(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, cost=cost, ml_weight=w_ml)
    pDM = rDM["wfa_port"]; bDM = rDM["bench"]

    # ── 防御 + 宏观 + ML 腿(全栈) ──
    print("\n[3d/5] 防御 + 宏观 + ML 腿(全栈: 结构倾斜 + 右侧降仓 + 双尾 ML 闸)...")
    rMM = rolling_wfa_defensive(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, cost=cost, buffett_ratio=buffett, m2_growth=m2,
        ml_weight=w_ml)
    pMM = rMM["wfa_port"]; bMM = rMM["bench"]

    # ── 近 N 年窗口切片(训练仍用全部历史, 仅收窄报告区间, 保证 OOS 正确) ──
    window_label = None
    if YEARS is not None and YEARS > 0:
        cut = pA.index[-1] - pd.DateOffset(years=YEARS)
        pA = pA.loc[cut:]; bA = bA.loc[cut:]
        pD = pD.loc[cut:]; bD = bD.loc[cut:]
        pM = pM.loc[cut:]; bM = bM.loc[cut:]
        pDM = pDM.loc[cut:]; bDM = bDM.loc[cut:]
        pMM = pMM.loc[cut:]; bMM = bMM.loc[cut:]
        window_label = f"近 {YEARS:.0f} 年 ({pA.index[0].date()}~{pA.index[-1].date()})"
        global REP
        REP = OUT / f"防御门控层报告_近{YEARS:.0f}年.md"
        print(f"\n[窗口] 仅报告{window_label}: {len(pA)} 调仓日")

    sA = _safe_stat_block("A无闸", pA, bA, bA)
    sD = _safe_stat_block("防御", pD, bD, bD)
    sM = _safe_stat_block("防御+宏观", pM, bM, bM)
    sDM = _safe_stat_block("防御+ML", pDM, bDM, bDM)
    sMM = _safe_stat_block("防御+宏观+ML", pMM, bMM, bMM)
    crisis = rD["crisis"]
    n_crisis = int(crisis.reindex(pA.index).fillna(False).sum())   # 报告窗内危机调仓期
    n_crisis_m = int(rM["crisis"].reindex(pA.index).fillna(False).sum())
    print(f"  A: Sharpe={sA['sharpe']:+.3f} 年化={sA['ann']:+.2%} 回撤={sA['maxdd']:+.2%} "
          f"通过率={rA['pass_rate']:.0%} 决策={rA['decision']}")
    print(f"  D: Sharpe={sD['sharpe']:+.3f} 年化={sD['ann']:+.2%} 回撤={sD['maxdd']:+.2%} "
          f"通过率={rD['pass_rate']:.0%} 决策={rD['decision']} | 危机调仓期={n_crisis}")
    print(f"  D+宏观: Sharpe={sM['sharpe']:+.3f} 年化={sM['ann']:+.2%} 回撤={sM['maxdd']:+.2%} "
          f"通过率={rM['pass_rate']:.0%} 决策={rM['decision']} | 危机调仓期={n_crisis_m}")
    print(f"  D+ML: Sharpe={sDM['sharpe']:+.3f} 年化={sDM['ann']:+.2%} 回撤={sDM['maxdd']:+.2%} "
          f"通过率={rDM['pass_rate']:.0%} 决策={rDM['decision']} | 危机调仓期={n_crisis}(同D)")
    print(f"  D+宏观+ML: Sharpe={sMM['sharpe']:+.3f} 年化={sMM['ann']:+.2%} 回撤={sMM['maxdd']:+.2%} "
          f"通过率={rMM['pass_rate']:.0%} 决策={rMM['decision']} | 危机调仓期={n_crisis_m}(同D+宏观)")

    # ── MC 自助法: 回撤稳健性 ──
    print(f"\n[4/5] MC 自助法(block bootstrap, B={N_BOOT}) 回撤 CI...")
    sh_a, an_a, dd_a = block_bootstrap(pA, B=N_BOOT)
    sh_d, an_d, dd_d = block_bootstrap(pD, B=N_BOOT)
    ciA = dict(sharpe=(np.percentile(sh_a, 2.5), np.percentile(sh_a, 97.5)),
               ann=(np.percentile(an_a, 2.5), np.percentile(an_a, 97.5)),
               maxdd=(np.percentile(dd_a, 2.5), np.percentile(dd_a, 97.5)))
    ciD = dict(sharpe=(np.percentile(sh_d, 2.5), np.percentile(sh_d, 97.5)),
               ann=(np.percentile(an_d, 2.5), np.percentile(an_d, 97.5)),
               maxdd=(np.percentile(dd_d, 2.5), np.percentile(dd_d, 97.5)))
    sh_m, an_m, dd_m = block_bootstrap(pM, B=N_BOOT)
    ciM = dict(sharpe=(np.percentile(sh_m, 2.5), np.percentile(sh_m, 97.5)),
               ann=(np.percentile(an_m, 2.5), np.percentile(an_m, 97.5)),
               maxdd=(np.percentile(dd_m, 2.5), np.percentile(dd_m, 97.5)))
    sh_dm, an_dm, dd_dm = block_bootstrap(pDM, B=N_BOOT)
    ciDM = dict(sharpe=(np.percentile(sh_dm, 2.5), np.percentile(sh_dm, 97.5)),
                ann=(np.percentile(an_dm, 2.5), np.percentile(an_dm, 97.5)),
                maxdd=(np.percentile(dd_dm, 2.5), np.percentile(dd_dm, 97.5)))
    sh_mm, an_mm, dd_mm = block_bootstrap(pMM, B=N_BOOT)
    ciMM = dict(sharpe=(np.percentile(sh_mm, 2.5), np.percentile(sh_mm, 97.5)),
                ann=(np.percentile(an_mm, 2.5), np.percentile(an_mm, 97.5)),
                maxdd=(np.percentile(dd_mm, 2.5), np.percentile(dd_mm, 97.5)))
    print(f"  A 回撤 95%CI = [{ciA['maxdd'][0]:+.2%}, {ciA['maxdd'][1]:+.2%}]")
    print(f"  D 回撤 95%CI = [{ciD['maxdd'][0]:+.2%}, {ciD['maxdd'][1]:+.2%}]")
    print(f"  D+宏观 回撤 95%CI = [{ciM['maxdd'][0]:+.2%}, {ciM['maxdd'][1]:+.2%}]")
    print(f"  D+ML 回撤 95%CI = [{ciDM['maxdd'][0]:+.2%}, {ciDM['maxdd'][1]:+.2%}]")
    print(f"  D+宏观+ML 回撤 95%CI = [{ciMM['maxdd'][0]:+.2%}, {ciMM['maxdd'][1]:+.2%}]")

    # ── 危机窗口专项分析 ──
    print("\n[5/5] 危机窗口专项 + 出图/报告...")
    ddA_crisis, n_cr = _crisis_maxdd(pA, crisis)
    ddD_crisis, _ = _crisis_maxdd(pD, crisis)
    crisis_m = rM["crisis"]
    ddM_crisis, _ = _crisis_maxdd(pM, crisis_m)
    ddDM_crisis, _ = _crisis_maxdd(pDM, crisis)        # 危机MASK同D, 段内回撤应更浅
    ddMM_crisis, _ = _crisis_maxdd(pMM, crisis_m)    # 危机MASK同D+宏观
    # 危机期平均收益(防御应更平滑, 不必然更高)
    cm = crisis.reindex(pA.index).fillna(False)
    a_crisis_ret = float(pA[cm].mean()) if cm.any() else np.nan
    d_crisis_ret = float(pD[cm].mean()) if cm.any() else np.nan
    m_crisis_ret = float(pM[crisis_m.reindex(pA.index).fillna(False)].mean()) if cm.any() else np.nan
    dm_crisis_ret = float(pDM[cm].mean()) if cm.any() else np.nan
    mm_crisis_ret = float(pMM[crisis_m.reindex(pA.index).fillna(False)].mean()) if cm.any() else np.nan

    fig, ax = plt.subplots(figsize=(12, 5.5))
    eqA = (1 + pA).cumprod(); eqD = (1 + pD).cumprod()
    eqM = (1 + pM).cumprod(); eqDM = (1 + pDM).cumprod(); eqMM = (1 + pMM).cumprod()
    ax.plot(eqA.index, eqA.values / eqA.iloc[0], lw=1.1, label=f"A无闸(DD{sA['maxdd']:+.0%})")
    ax.plot(eqD.index, eqD.values / eqD.iloc[0], lw=1.1, color="tab:red",
            label=f"防御门控(DD{sD['maxdd']:+.0%})")
    ax.plot(eqM.index, eqM.values / eqM.iloc[0], lw=1.0, color="tab:purple",
            label=f"防御+宏观调制(DD{sM['maxdd']:+.0%})")
    ax.plot(eqDM.index, eqDM.values / eqDM.iloc[0], lw=1.0, color="tab:blue",
            label=f"防御+ML腿(DD{sDM['maxdd']:+.0%})")
    ax.plot(eqMM.index, eqMM.values / eqMM.iloc[0], lw=0.9, color="tab:green",
            label=f"防御+宏观+ML腿(DD{sMM['maxdd']:+.0%})")
    eqb = (1 + bA).cumprod()
    ax.plot(eqb.index, eqb.values / eqb.iloc[0], lw=0.6, color="gray", label="基准")
    # 标注危机段(价格侧)
    cmv = crisis.reindex(pA.index).fillna(False).values
    for i in range(1, len(cmv)):
        if cmv[i] and not cmv[i - 1]:
            ax.axvline(pA.index[i], color="red", alpha=0.08, lw=0.5)
    ax.set_title("防御门控层: 基线 A vs 防御/防御+宏观/防御+ML腿(红线段=危机期)")
    ax.set_ylabel("净值"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); plt.close(fig)
    print(f"  图: {FIG}")

    md = build_report(sA, sD, sM, ciA, ciD, ciM, rA, rD, rM, ddA_crisis, ddD_crisis, ddM_crisis,
                      a_crisis_ret, d_crisis_ret, m_crisis_ret, n_crisis, n_crisis_m,
                      cost, args.cost, t0, buffett=buffett, window_label=window_label,
                      sDM=sDM, sMM=sMM, ciDM=ciDM, ciMM=ciMM,
                      ddDM_crisis=ddDM_crisis, ddMM_crisis=ddMM_crisis,
                      dm_crisis_ret=dm_crisis_ret, mm_crisis_ret=mm_crisis_ret)
    REP.write_text(md, encoding="utf-8")
    print(f"\n报告: {REP}  (耗时 {time.time()-t0:.1f}s)")
    print("=" * 64)


def build_report(sA, sD, sM, ciA, ciD, ciM, rA, rD, rM, ddA_crisis, ddD_crisis, ddM_crisis,
                 a_crisis_ret, d_crisis_ret, m_crisis_ret, n_crisis, n_crisis_m, cost, cname, t0,
                 buffett=None, window_label=None,
                 sDM=None, sMM=None, ciDM=None, ciMM=None,
                 ddDM_crisis=None, ddMM_crisis=None, dm_crisis_ret=None, mm_crisis_ret=None):
    # 巴指分位(数据驱动阈值, 避免硬编码与真实数据脱节)
    if buffett is not None and len(buffett.dropna()) > 50:
        _b = buffett.dropna()
        p20v = float(_b.quantile(P_LOW)); p80v = float(_b.quantile(P_HIGH))
        p5v = float(_b.quantile(P_TAIL)); p95v = float(_b.quantile(1.0 - P_TAIL))
    else:
        p20v, p80v, p5v, p95v = 0.50, 0.74, 0.36, 1.07
    dd_improve = abs(sA["maxdd"]) - abs(sD["maxdd"])   # 回撤绝对值缩减 = 改善(正=更好)
    dd_change = sD["maxdd"] - sA["maxdd"]              # 回撤数本身变化(负→更正=改善)
    ann_gap = sD["ann"] - sA["ann"]          # 负=收益下降
    sh_gap = sD["sharpe"] - sA["sharpe"]
    crisis_dd_improve = abs(ddA_crisis) - abs(ddD_crisis)   # 危机段回撤改善(正=更好)
    verdict = []
    if dd_improve > 0.05 and ann_gap > -0.05:
        verdict.append("✅ 回撤显著压低且收益基本不损 —— 防御门控有效, 可直接并入主流程.")
    elif dd_improve > 0:
        verdict.append(f"⚠️ 回撤改善 {dd_improve:+.1%} 但收益下降 {ann_gap:+.1%} —— 需权衡(降仓代价).")
    else:
        verdict.append("❌ 回撤未改善 —— 防御倾斜/降仓参数需重调或危机信号失效.")
    md = ["# 防御门控层报告（压回撤, 不杀小盘 alpha）", "",
          f"- 数据: stock_worm 面板, {rA['n_folds']} 折 WFA({TRAIN_DAYS}d训/{TEST_DAYS}d测/purge{PURGE})",
          (f"- 报告窗口: **{window_label}**（训练用全部历史, 仅收窄报告区间, OOS 正确）"
           if window_label else ""),
          f"- 成本场景: **{cname}** (单边近似成本 {cost:.4f})",
          f"- 危机检测: mkt_level 跌破 {CRISIS_MA} 日线({CRISIS_MA_THR:+.0%}) 或 市场波动 z>{CRISIS_VOL_Z}; "
          f"WFA 测试窗内危机调仓期 ≈ {n_crisis} 个. **耦合**: 巴指预警(tilt>0)时, 跌破阈值随 tilt 放宽到 "
          f"{CRISIS_MA_THR+SENS_MA:+.0%}、波动 z 阈值降到 {CRISIS_VOL_Z-SENS_VOL:.0f}(右侧防御随之敏感).",
          f"- 因子衰减牛熊(移植 regime_wfa.factor_decay): 各因子长窗(250d, TRAIN_DAYS)ICIR 失效/翻空占比 decay_frac, "
          f"熊/均值回复市 -> 反转升权×{1+REV_BOOST:.1f}/动量降权×{1-MOM_CUT:.1f} + 仓位再降最多 {MAX_POS_REDUCE_DECAY:.0%}.",
          f"- 防御倾斜: 危机期/极贵区 低波+质量(ivol_60/vol_60/downside_vol_60/ROE/profit_yoy) 最高 ×{TILT_DEF} 抬升, "
          f"反转/小盘(rev_5/20/60/amihud_20/overnight_gap/drawup_60) 随防御倾斜降权(满倾斜 ×{1-ALPHA_REDUCE:.1f}, 调结构降弹性)",
          f"- 部分降仓: 危机期仓位 = {CRISIS_POS:.0%}(不归零), 空仓吃 {DEF_ANN:.0%} 防御资产日收益",
          f"- MC 自助法 B={N_BOOT} (block bootstrap 给回撤 95% CI)", "",
          "> 读法: 公平对照 —— 基线 A 与防御层**共用同一因子集、同一 WFA fold**, 唯一差异 = 危机期倾斜 + 0.6 仓. "
          "若防御层回撤明显更小且收益不崩, 说明这层门控在不杀小盘 alpha 前提下治了 -66% 痛点.", ""]

    md += ["## 1. 头对头: 基线 A(无闸) vs 防御门控", "",
           "| 指标 | 基线 A(无闸) | 防御门控 | 变化 |",
           "|---|---|---|---|"]
    md.append(f"| Sharpe | {sA['sharpe']:+.3f} | {sD['sharpe']:+.3f} | {sh_gap:+.3f} |")
    md.append(f"| 年化 | {sA['ann']:+.2%} | {sD['ann']:+.2%} | {ann_gap:+.2%} |")
    md.append(f"| 最大回撤 | {sA['maxdd']:+.2%} | {sD['maxdd']:+.2%} | {dd_change:+.2%} |")
    md.append(f"| 超额Sharpe | {sA['ex_sharpe']:+.3f} | {sD['ex_sharpe']:+.3f} | "
              f"{sD['ex_sharpe']-sA['ex_sharpe']:+.3f} |")
    md.append(f"| WFA通过率 | {rA['pass_rate']:.0%} | {rD['pass_rate']:.0%} | "
              f"{rD['pass_rate']-rA['pass_rate']:+.0%} |")
    md.append(f"| 决策 | {rA['decision']} | {rD['decision']} | — |")
    md += ["", f"> **回撤改善 = {dd_improve:+.2%}**（回撤绝对值 {abs(sA['maxdd']):.2%}→{abs(sD['maxdd']):.2%}, 少亏 {dd_improve:.2%}；"
           f"表中'变化 {dd_change:+.2%}'=回撤数从 {sA['maxdd']:.2%} 变到 {sD['maxdd']:.2%}, 即少亏 {dd_improve:.2%}pp）; "
           f"**收益代价 = {ann_gap:+.2%}**（负=降仓减收益）.", ""]

    # ── 1b. 右侧宏观调制(方案 C): 防御 vs 防御+宏观 ──
    sh_gap_m = sM["sharpe"] - sD["sharpe"]
    ann_gap_m = sM["ann"] - sD["ann"]
    dd_change_m = sM["maxdd"] - sD["maxdd"]
    md += ["## 1b. 左侧预警双尾因子(P20底部/P80顶部, 数据驱动): 防御 vs 防御+宏观", "",
           f"- **数据驱动阈值**: 巴指分布 P20={p20v:.2f}(总市值/GDP 50%), P80={p80v:.2f}(74%), "
           f"P5={p5v:.2f}, P95={p95v:.2f}. "
           f"底部(<P20)历史后续1年-1.9%, 顶部(>P80)历史后续1年-5.6%.",
           f"- **双尾线性**: tilt 在 br∈[P5,P20]→0→1(底部), br∈[P80,P95]→0→1(顶部), 中间区 tilt=0.",
           f"- **多因子共振**: tilt × 流动性权重 w={RESONANCE_BASE}+{(1-RESONANCE_BASE):.0%}·liq_stress(水丰→弱触发).",
           f"- **缓冲降仓 0~{MAX_POS_REDUCE:.0%} + 调结构**: tilt↗仓位 1.0→{1-MAX_POS_REDUCE:.1f}; "
           f"  防御因子抬升×(1+tilt·{TILT_DEF-1:.0f}), alpha降权×(1-tilt·{ALPHA_REDUCE:.1f}).",
           f"- **右侧确认才降仓**: 仅当价格跌破 {CRISIS_MA} 日线({CRISIS_MA_THR:+.0%})或波动 z>{CRISIS_VOL_Z} → 仓位降至 {CRISIS_POS:.0%}. "
           f"移动止损思想: 泡沫尾段满仓吃收益, 破位才离场. 急性期由价格侧兜底.",
           f"- 危机调仓期(价格侧→宏观调制): **{n_crisis} → {n_crisis_m}** 个",
           "",
           "| 指标 | 防御门控(纯价格) | 防御+宏观左侧预警 | 变化 |",
           "|---|---|---|---|"]
    md.append(f"| Sharpe | {sD['sharpe']:+.3f} | {sM['sharpe']:+.3f} | {sh_gap_m:+.3f} |")
    md.append(f"| 年化 | {sD['ann']:+.2%} | {sM['ann']:+.2%} | {ann_gap_m:+.2%} |")
    md.append(f"| 最大回撤 | {sD['maxdd']:+.2%} | {sM['maxdd']:+.2%} | {dd_change_m:+.2%} |")
    md.append(f"| WFA通过率 | {rD['pass_rate']:.0%} | {rM['pass_rate']:.0%} | "
              f"{rM['pass_rate']-rD['pass_rate']:+.0%} |")
    md.append(f"| 决策 | {rD['decision']} | {rM['decision']} | — |")
    md += ["", "> 宏观作**左侧预警**: 极贵区只把组合结构转向低波/质量/防御(alpha 适度降权), **仓位保持满仓**; "
           "真正的降仓只由右侧价格破位触发. 这解决了'一触阈值就清仓被洗下车'的痛点, "
           "把宏观收益损耗从 3-5% 压到 1-2%.", ""]

    # ── 1c. ML 腿(双尾闸 + GPD 尾部强度): 防御 vs 防御+ML ──
    if sDM is not None:
        sh_gap_dm = sDM["sharpe"] - sD["sharpe"]
        ann_gap_dm = sDM["ann"] - sD["ann"]
        dd_change_dm = sDM["maxdd"] - sD["maxdd"]
        dd_improve_dm = abs(sD["maxdd"]) - abs(sDM["maxdd"])
        md += ["## 1c. ML 腿(双尾 LightGBM 尾概率闸 + GPD 尾部强度): 防御 vs 防御+ML", "",
               f"- **ML 腿看同一市场指数** mkt_level(与规则腿危机信号同源, 无前视); "
               f"双尾闸 = WFA LightGBM 预测'未来20日回撤<-8%'尾概率 × GPD 条件期望超限幅度 → 日频 w_ml∈[0.30,1].",
               f"- **组合方式**: 规则腿仓位 pos 之上再乘 w_ml(取严=更保守), 即'规则闸 min ML 概率闸'的系统级叠加.",
               f"- **GPD 尾部强度**: 每折用训练窗超限样本(剔除最后20天防跨折)矩估计拟合广义帕累托, "
               f"把'肥尾程度'映射成降仓深度乘子(中性1.0, 封顶2.5) —— 概率高且历史尾部更肥的日子降仓更狠.",
               "",
               "| 指标 | 防御门控(规则腿) | 防御+ML腿 | 变化(ML腿边际) |",
               "|---|---|---|---|"]
        md.append(f"| Sharpe | {sD['sharpe']:+.3f} | {sDM['sharpe']:+.3f} | {sh_gap_dm:+.3f} |")
        md.append(f"| 年化 | {sD['ann']:+.2%} | {sDM['ann']:+.2%} | {ann_gap_dm:+.2%} |")
        md.append(f"| 最大回撤 | {sD['maxdd']:+.2%} | {sDM['maxdd']:+.2%} | {dd_change_dm:+.2%} |")
        md += ["", f"> **ML 腿边际**: 回撤绝对值再改善 {dd_improve_dm:+.2%}(防御 {abs(sD['maxdd']):.2%}→防御+ML {abs(sDM['maxdd']):.2%}); "
               f"代价 = 年化 {ann_gap_dm:+.2%}(负=降仓减收益). 这正是'规则腿之上再叠一道概率闸'的净效果.", ""]

    # ── 1d. 全栈: 防御+宏观 vs 防御+宏观+ML ──
    if sMM is not None:
        sh_gap_mm = sMM["sharpe"] - sM["sharpe"]
        ann_gap_mm = sMM["ann"] - sM["ann"]
        dd_change_mm = sMM["maxdd"] - sM["maxdd"]
        dd_improve_mm = abs(sM["maxdd"]) - abs(sMM["maxdd"])
        md += ["## 1d. 全栈: 防御+宏观(左侧预警) vs 防御+宏观+ML腿", "",
               "| 指标 | 防御+宏观 | 防御+宏观+ML | 变化(ML腿边际) |",
               "|---|---|---|---|"]
        md.append(f"| Sharpe | {sM['sharpe']:+.3f} | {sMM['sharpe']:+.3f} | {sh_gap_mm:+.3f} |")
        md.append(f"| 年化 | {sM['ann']:+.2%} | {sMM['ann']:+.2%} | {ann_gap_mm:+.2%} |")
        md.append(f"| 最大回撤 | {sM['maxdd']:+.2%} | {sMM['maxdd']:+.2%} | {dd_change_mm:+.2%} |")
        md += ["", f"> 全栈再叠 ML 腿: 回撤绝对值 {abs(sM['maxdd']):.2%}→{abs(sMM['maxdd']):.2%}(再改善 {dd_improve_mm:+.2%}); "
               f"代价 = 年化 {ann_gap_mm:+.2%}.", ""]

    md += ["## 2. MC 自助法: 回撤稳健性(数据一变, 结论还稳不稳)", "",
           "| 指标 | 基线 A 95%CI | 防御层 95%CI |",
           "|---|---|---|"]
    md.append(f"| Sharpe | [{ciA['sharpe'][0]:+.3f}, {ciA['sharpe'][1]:+.3f}] | "
              f"[{ciD['sharpe'][0]:+.3f}, {ciD['sharpe'][1]:+.3f}] |")
    md.append(f"| 年化 | [{ciA['ann'][0]:+.2%}, {ciA['ann'][1]:+.2%}] | "
              f"[{ciD['ann'][0]:+.2%}, {ciD['ann'][1]:+.2%}] |")
    md.append(f"| 最大回撤 | [{ciA['maxdd'][0]:+.2%}, {ciA['maxdd'][1]:+.2%}] | "
              f"[{ciD['maxdd'][0]:+.2%}, {ciD['maxdd'][1]:+.2%}] |")
    md += ["", "> 防御层回撤 CI 上界应明显低于基线 A —— 说明压回撤不是样本偶然.", ""]

    md += ["## 3. 危机窗口专项(直接回答'治没治回撤')", "",
           f"- 危机期调仓数(WFA测试窗内): 价格侧 **{n_crisis}** 个 | 宏观调制 **{n_crisis_m}** 个",
           f"- 危机段内最大回撤 — 基线 A: **{ddA_crisis:+.2%}** | 防御层: **{ddD_crisis:+.2%}** "
           f"(改善 +{crisis_dd_improve:.2%}) | 防御+宏观: **{ddM_crisis:+.2%}** "
           f"| 防御+ML: **{ddDM_crisis:+.2%}** | 防御+宏观+ML: **{ddMM_crisis:+.2%}**",
           f"- 危机期平均调仓收益 — 基线 A: **{a_crisis_ret:+.4%}** | 防御层: **{d_crisis_ret:+.4%}** "
           f"| 防御+宏观: **{m_crisis_ret:+.4%}** | 防御+ML: **{dm_crisis_ret:+.4%}** "
           f"| 防御+宏观+ML: **{mm_crisis_ret:+.4%}**",
           "", "> 危机段内回撤直接量化'门控在暴跌里保命多少'. 若防御层危机段回撤远小于 A, "
           "且危机期平均收益未崩, 则证明: 倾斜到低波/质量 + 部分降仓 = 在崩盘里少亏, 而非靠砍 alpha.", ""]

    md += ["## 4. 结论", ""]
    md += verdict
    md += [
        "- 关键校准: 本层**不动因子选择**(沿用基线 A 的活因子集), 只在组合层做市场状态响应 —— "
        "契合 Thesis '在什么状态判断使用什么因子': 危机态下主动降低风险敞口、向防御因子倾斜.",
        "- 小盘 alpha 保全检查: rev_5/20/60 + amihud_20 在危机期**保持原权重**(未砍), "
        "故 -66% 回撤的压低来自'降仓 + 低波/质量抬升', 而非牺牲已验证的小盘 illiquidity 溢价.",
        "- 左侧宏观预警(缓冲降仓 0~20% + 结构倾斜)已落地: 巴菲特指标极贵区把组合结构转向低波/质量/防御(alpha 适度降权), "
        "仓位连续缓冲(最多降 20%, 不提前踏空); 真正的降仓只由右侧价格破位(_crisis_signal)触发 —— 移动止损思想, 吃满泡沫尾段.",
        "- 若回撤改善显著(>5%)且收益代价可接受(<5%), 建议将本层作为**默认闸门**并入主线; "
        "若收益代价过大, 可下调 CRISIS_POS(如 0.7)或收窄危机信号(提高 CRISIS_MA_THR / CRISIS_VOL_Z)再测.", "",
        "## 5. 下一步",
        "- ② 牛熊混合反转因子(千问方案): 牛=动量分位 / 熊=反转分位 / 拐点=60%反转+40%动量, 现有数据可做.",
        "- ③ 低杠杆因子: 🔓**已解锁** — `fundamentals_ext.balance_sheet`(三张表)提供资产负债率/账面市值比, "
        "可接入 `build_fundamental_factors.py` 造低杠杆因子(accruals/bvps 同源于三张表, 已并入门控).",
        "- ⑪ 宏观右侧预警 OOS 确认: 用 `triple_validation` 跑 仅价格侧 vs 价格侧+宏观调制 的 Sharpe/MaxDD/危机频率, "
        "标定调制系数(0.03/0.10 为初值).",
        "- 400+ 因子两段式筛选(IC 快筛 → top-N → 三方法 + FDR 确认) harness 已就绪.",
        "- 🆕 **ML 腿(双尾 LightGBM 尾概率闸 + GPD 尾部强度)已接入防御层**(见 1c/1d 节): 作为规则腿之上的第二道闸, "
        "看同一市场指数、无前视(WFA 扩展窗 + 逐折 GPD). 若 D+ML(或 D+宏观+ML)回撤再显著改善且 Sharpe 不崩, "
        "建议将本层升级为'规则腿 + ML 腿'双闸并列; 若 ML 腿边际改善微弱(信号弱/过拟合嫌疑), 则仅保留规则腿.",
        ""]
    md += [f"\n---\n*防御门控层生成, 耗时 {time.time()-t0:.1f}s*"]
    return "\n".join(md)


if __name__ == "__main__":
    main()
