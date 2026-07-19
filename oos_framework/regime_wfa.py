"""
regime_wfa.py — 双 Regime 条件 WFA 引擎
(REGIME_TRAINING Plan Phase 2 + REGIME_FACTOR Plan Phase 3 整合实现)

核心思路:
  1. 市场级 (REGIME_TRAINING): 按 test 窗的市场状态 (bull/bear/osc) 选匹配的因子冻结集.
  2. 因子级 (REGIME_FACTOR): 各因子独立判自身牛熊, 按自身状态调信号合成权重.
  3. 两层协同: 市场级选因子集 + 因子级调内部权重 + anti 闸门调总仓位.

变体 (公平对照: B/C/D/E 共用同一个已验证两档位闸门, 唯一变量是 regime 特征):
  A: 无闸 (裸多头 Frozen, 基线)
  B: 两档位闸门 (对齐 oos_wfa.py 的 C/D: 因子衰减正渐进 + 熊市 regime 状态机) — 公平基线
  C: B + 市场级 regime 条件因子集
  D: B + 因子级 regime 加权信号
  E: B + 双 regime (C+D)

用法:
  python oos_framework/regime_wfa.py
"""

import sys, time, os, warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = Path("/workspace/stock_worm/data")
SF_PANEL = DATA / "ashare_daily_panel_survivorfree.parquet"
ALIVE_PANEL = DATA / "ashare_daily_panel.parquet"
CSRC_MAP = DATA / "csrc_industry_map.parquet"
FUND_PARQUET = DATA / "fundamentals/fund_factors_daily.parquet"  # 旧 ROE 等(仅 zoo 面板注入用, 候选池已切 PIT 基本面)
OUT_DIR = HERE / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REGIME_CSV = OUT_DIR / "factor_regime.csv"

sys.path.insert(0, str(HERE.parent / "agent/backtest"))
from factor_zoo_daily import (build_factors, neutralize_factors, ALL_FACTOR_NAMES,
                             ZOO_FACTOR_NAMES, FUND_FACTOR_NAMES, daily_rank_ic,
                             set_market_regime, build_zoo_factors, build_fundamental_factors)
from oos_validation_corrected import load_wide_sf, build_zarr
from oos_validation import _stat_block, TOP_K, HOLD, COST, TRAIL, RNG, BPY
from backtest.validation import _sharpe

# ── 参数 ──
TRAIN_DAYS = TRAIL
TEST_DAYS = 250
PURGE = 5
MAJORITY = 2.0 / 3.0
VETO_SHARPE = -1.0
VETO_MAXDD = -0.35
BEAR_MA = 120
BEAR_THR = -0.10
BULL_THR = 0.05

# 因子级 regime 权重
W_BULL = 1.0
W_BEAR = 0.3
W_NEU = 0.7

# 防御资产
DEF_ANN = 0.04

# ── 两档位闸门参数 (原样对齐 oos_wfa.py 已验证的 C/D 闸门, 非二元硬闸) ──
GATE_BEAR_MA = BEAR_MA          # 熊市线 MA 窗口
GATE_DECAY_FRAC = 0.30          # 长窗: 因子近期 ICIR < 该比例×train ICIR 视为'衰减'(保守死亡计数)
GATE_SHORT_WIN = 60             # 短窗: 因子近 GATE_SHORT_WIN 日 IC<0 视为'刚报出熊市信号'(较激进)
GATE_DECAY_FLOOR = 0.50         # 正渐进下限: 因子全失效时仓位最低 50%(不归零); 熊信号才执行反渐进->0
POS_CAP = 1.0                   # 正渐进仓位上限(牛市可加满 100%)


def _recent_icir(fac_ic, trail=TRAIN_DAYS):
    """预计算每因子'截至各日的滚动 ICIR'(因果, 只用 ≤t 的 IC). 供因子衰减计数用."""
    out = {}
    for f, s in fac_ic.items():
        m = s.rolling(trail).mean()
        sd = s.rolling(trail).std()
        out[f] = (m / (sd + 1e-9) * np.sqrt(252))
    return out


def _alive_mkt_level(dates):
    alive = pd.read_parquet(ALIVE_PANEL)
    alive["_d"] = pd.to_datetime(alive["date"]).dt.normalize()
    cal = pd.to_datetime(dates)
    alive = alive[alive["_d"].isin(cal)]
    wide = alive.pivot(index="_d", columns="code", values="close").reindex(dates)
    ret = wide.pct_change()
    return (1.0 + ret.mean(axis=1, skipna=True).fillna(0.0)).cumprod()


def _load_data():
    """一站式数据加载, 返回所有组件."""
    w = load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))
    fac = build_factors(w)
    fac.update(build_zoo_factors(w))   # 候选池刷新: 并入 zoo 短名单因子
    fac.update(build_fundamental_factors(w))   # 候选池刷新 #2: PIT 基本面因子(并入后随价量一起行业中性化)
    fac = neutralize_factors(fac, ind_map)
    del w
    ALL = ALL_FACTOR_NAMES + ZOO_FACTOR_NAMES + FUND_FACTOR_NAMES
    zarr = build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_FACTOR_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)   # 写盘前补全基本面因子 NaN
    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    mkt_level = _alive_mkt_level(dates)
    return dict(zarr=zarr, fac_ic=fac_ic, ALL=ALL, fwd=fwd, dates=dates,
                codes=codes, n_codes=n_codes, cov=cov, ind_map=ind_map,
                mkt_level=mkt_level)


def _rss():
    try:
        import resource as _rs
        return _rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _daily_rank_ic_arr(z, v, dates, batch=300):
    """日期分批的逐日横截面 spearman rank-IC(输入 z/v 为 (dates×codes) numpy 数组).

    与 daily_rank_ic(逐日 rank -> pearson)数学等价, 但按日期分块处理, 使全市场 (4975×5515)
    不必一次性持有两因子宽表. 每批用 pandas rank(skipna) 复刻原语义.
    """
    n = len(dates)
    ic = np.full(n, np.nan)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        fz = pd.DataFrame(z[s:e]); vz = pd.DataFrame(v[s:e])
        fr = fz.rank(axis=1); vr = vz.rank(axis=1)
        fc = fr.sub(fr.mean(axis=1), axis=0); vc = vr.sub(vr.mean(axis=1), axis=0)
        num = (fc * vc).sum(axis=1)
        den = np.sqrt((fc ** 2).sum(axis=1) * (vc ** 2).sum(axis=1))
        ic[s:e] = (num / den.replace(0, np.nan)).values
    return pd.Series(ic, index=dates)


def load_engine_inputs_cached():
    """按股票分块流式加载器(应对 8G cgroup 上限 + 600s 会话上限).

    全市场 5515 只时, 原管线同时持有 w(1.3G)+fac(3.6G)+neu(3.6G)+zarr(3.6G)≈12G 必 OOM.
    改为: 预分配 zarr(float32, 3.6G) 一次, 按 ≤CHUNK 只股票切块, 每块独立 build_factors→
    中性化→z-score→写入 zarr 对应列并立即释放; 块间峰值仅 ~w(0.66G)+fwd(1.1G)+zarr(3.6G)
    +单块因子(≤2G)≈7G, 安全. zarr 逐因子缓存为 .npy(避免 3.6G 单 pickle 写入 2× 内存翻倍),
    fac_ic 缓存为 pkl; 被杀后重跑从 .done 标记恢复.
    """
    CACHE = HERE / "screen_results" / "_wfa_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    zarr_dir = CACHE / "zarr"; zarr_dir.mkdir(exist_ok=True)
    done = CACHE / "zarr.done"
    fic_p = CACHE / "fac_ic.pkl"
    t0 = time.time()

    w = load_wide_sf()
    w = {c: w[c].astype(np.float32) for c in w}   # float32 省内存(6表 0.66G)
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5).astype(np.float32)
    dates, codes = fwd.index, fwd.columns
    # 牛熊 regime: 全市场等权趋势(250d偏离), 注入 build_factors 供牛熊混合反转因子判定牛/熊
    mkt_trend = (w["close"].mean(axis=1) / w["close"].mean(axis=1).rolling(250).mean() - 1)
    set_market_regime(mkt_trend)
    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))
    ALL = ALL_FACTOR_NAMES + ZOO_FACTOR_NAMES + FUND_FACTOR_NAMES

    if done.exists() and all((zarr_dir / f"{f}.npy").exists() for f in ALL) and fic_p.exists():
        zarr = {f: np.load(zarr_dir / f"{f}.npy", mmap_mode="r") for f in ALL}  # mmap: 按需加载, 不占常驻
        fac_ic = pd.read_pickle(fic_p)
        print(f"  [cache] zarr({len(ALL)}因子)+fac_ic 命中 ({time.time()-t0:.1f}s)")
    else:
        nd, nc = len(dates), n_codes
        # memmap 落盘: 避免 66 因子(34原+32zoo)全量 zarr(>7G)占满 8G cgroup 匿名内存 -> OOM.
        # 文件映射内存可被 cgroup 回收, 比 np.empty 匿名内存安全.
        zarr = {}
        for f in ALL:
            zarr[f] = np.memmap(zarr_dir / f"{f}.mmap", dtype=np.float32, mode="w+", shape=(nd, nc))
        code_pos = {c: i for i, c in enumerate(codes)}
        CHUNK = 600
        chunks = [list(codes)[i:i + CHUNK] for i in range(0, nc, CHUNK)]
        print(f"  流式构建: {nc}只 / {len(chunks)}块(每块≤{CHUNK}) ...", flush=True)
        for ci, ch in enumerate(chunks):
            idx = [code_pos[c] for c in ch]
            w_c = {k: w[k][ch] for k in w}
            fac_c = build_factors(w_c)
            zoo_c = build_zoo_factors(w_c)   # 候选池刷新: 并入 zoo 短名单
            for k, v in zoo_c.items():
                fac_c[k] = v.reindex(index=dates, columns=ch).astype(np.float32)
            fund_c = build_fundamental_factors(w_c)   # 候选池刷新 #2: PIT 基本面因子
            for k, v in fund_c.items():
                fac_c[k] = v.reindex(index=dates, columns=ch).astype(np.float32)
            fac_c = neutralize_factors(fac_c, ind_map)
            for f in ALL:
                zf = fac_c[f]
                mu = zf.mean(axis=1); sd = zf.std(axis=1)
                zz = zf.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
                zarr[f][:, idx] = zz.reindex(index=dates, columns=ch).values.astype(np.float32)
                del zz
            del w_c, fac_c
            import gc as _gc; _gc.collect()
            print(f"    块{ci+1}/{len(chunks)} done (RSS={_rss():.0f}MB, {time.time()-t0:.1f}s)", flush=True)
        for f in FUND_FACTOR_NAMES:
            zarr[f] = np.nan_to_num(zarr[f], nan=0.0)   # 写盘前补全基本面因子 NaN, 否则落盘含 NaN
        fwdv = fwd.values
        fac_ic = {}
        for f in ALL:
            # 逐因子写盘: fsync 回写脏页 + posix_fadvise(DONTNEED) 丢弃页缓存, 避免写盘
            # 产生的干净页缓存在 cgroup 内累积撑爆 8G; 同时立即算 fac_ic 并释放该因子内存.
            with open(zarr_dir / f"{f}.npy", "wb") as fh:
                np.save(fh, np.asarray(zarr[f]))   # memmap -> 常规数组落盘
                fh.flush(); os.fsync(fh.fileno())
                try:
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
            fac_ic[f] = _daily_rank_ic_arr(zarr[f], fwdv, dates).reindex(dates)
            del zarr[f]
            try:
                (zarr_dir / f"{f}.mmap").unlink()   # 清理 memmap 中间文件
            except Exception:
                pass
        pd.to_pickle(fac_ic, fic_p)
        done.write_text("1")
        # 重新 mmap 懒加载(不占常驻), 供同进程 WFA 使用; 写盘内存已随 del 释放
        zarr = {f: np.load(zarr_dir / f"{f}.npy", mmap_mode="r") for f in ALL}
        print(f"  [构建] zarr+fac_ic 落盘(fsync+fadvise, {time.time()-t0:.1f}s)")

    mkt_level = _alive_mkt_level(dates)
    print(f"  [加载完成] {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 因子 {len(ALL)} | 总 {time.time()-t0:.1f}s")
    return dict(zarr=zarr, fac_ic=fac_ic, ALL=ALL, fwd=fwd, dates=dates, codes=codes,
                n_codes=n_codes, cov=cov, ind_map=ind_map, mkt_level=mkt_level)


def _bear_signal(mkt_level, mode="ma120"):
    """熊市信号 (对齐 oos_wfa.py)."""
    ma = mkt_level.rolling(BEAR_MA).mean()
    ratio = mkt_level / ma - 1
    b_ma = ratio < BEAR_THR
    if mode != "ensemble":
        return b_ma.fillna(False)
    ma60 = mkt_level.rolling(60).mean()
    b_ma60 = (mkt_level / ma60 - 1) < -0.08
    hi20 = mkt_level.rolling(20).max()
    b_dd = (mkt_level / hi20 - 1) < -0.12
    ret = mkt_level.pct_change()
    vol = ret.rolling(20).std()
    volt = vol.rolling(250).mean()
    b_vol = (vol > 2.5 * volt.replace(0, np.nan)).fillna(False)
    return (b_ma | b_ma60 | b_dd | b_vol).fillna(False)


def _market_regime_label(mkt_level, dates):
    """三态市场标签: bull/bear/osc, 对齐 rebalance_dates."""
    ma = mkt_level.rolling(BEAR_MA).mean()
    ratio = mkt_level / ma - 1
    bear = (ratio < BEAR_THR).fillna(False)
    # 震荡: 非熊 且 价格窄幅波动
    osc_dev = ((mkt_level - ma).abs() / (ma + 1e-9) < 0.07)
    hi20 = mkt_level.rolling(20).max()
    lo20 = mkt_level.rolling(20).min()
    osc_rng = ((hi20 - lo20) / (lo20 + 1e-9) < 0.12).fillna(False)
    osc = (~bear) & (osc_dev | osc_rng)
    labels = pd.Series("bull", index=dates)
    labels = labels.mask(bear, "bear")
    labels = labels.mask(osc, "osc")
    return labels


def _build_regime_factor_set(fac_ic, train_slice, regime_label, target_regime,
                             min_factors=3, candidates=None):
    """根据历史 IC 筛选在目标 regime 下表现好的因子 (REGIME_TRAINING Phase 2).

    在 train_slice 中, 只取 regime 匹配 target_regime 的日子,
    在这些日子里计算每个因子的 IC 均值, 取 IC>0 的因子.

    Args:
        fac_ic: dict[name → Series(IC)].
        train_slice: (ts, te) 索引范围.

        regime_label: 日频 regime 标签 Series.
        target_regime: "bull" 或 "bear" 或 "osc".
        min_factors: 最少因子数 (不够则放宽).
        candidates: 候选因子集合(默认 fac_ic 全部键). 单因子变体(如 ML 因子)须传入
                    factor_names, 否则会选出不在 factor_names 的因子, 导致后续
                    locked_orient 缺键 KeyError.

    Returns:
        list[str] 选出因子名.
    """
    if candidates is None:
        candidates = list(fac_ic.keys())
    ts, te = train_slice
    selected = []
    for f in candidates:
        ic = fac_ic[f].iloc[ts:te]
        if ic.empty or ic.isna().all():
            continue
        # 匹配 regime: 用 regime_label 中对应日期筛选
        mask = regime_label.reindex(ic.index) == target_regime
        ic_reg = ic[mask].dropna()
        if len(ic_reg) < 10:
            continue
        mu = ic_reg.mean()
        if mu > 0:
            selected.append(f)
    if len(selected) < min_factors:
        # 放宽: 用全量 IC (不按 regime 筛选)
        for f in candidates:
            ic = fac_ic[f].iloc[ts:te].dropna()
            if len(ic) < 10:
                continue
            mu = ic.mean()
            if mu > 0 and f not in selected:
                selected.append(f)
    return selected


def _load_factor_regime() -> Optional[pd.DataFrame]:
    """加载因子级 regime 标签 (由 diagnose_factor_regime.py 产出)."""
    if not REGIME_CSV.exists():
        return None
    df = pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)
    return df


def rolling_wfa_dual_regime(
    zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level,
    train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE, step=None,
    top_k=TOP_K, hold=HOLD, cost=COST,
    veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY,
    gate=False, bear_mode="ma120", addback_mode="factor_decay",
    use_market_regime=False,       # REGIME_TRAINING: 按市场状态选因
    use_factor_regime=False,       # REGIME_FACTOR: 按因子自身状态调权
    factor_regime_labels=None,     # DataFrame(dates × factors), 值 ∈ {1,0,-1}
):
    """双 regime 条件滚动 WFA.

    gate: 是否启用 anti 闸门.
    use_market_regime: 市场级 regime 因子集切换 (REGIME_TRAINING).
    use_factor_regime: 因子级 regime 加权信号 (REGIME_FACTOR).
    factor_regime_labels: 因子日频 bull(1)/neutral(0)/bear(-1) 标签.
    """
    n = len(dates)
    # 防前视: 因子 regime 标签基于 pnl, 而 pnl[p] 含前视收益 fwd[p](持仓 HOLD 日).
    # 决策日 p 只能用截至 p-HOLD 的 pnl(已实现的过去收益) -> 标签整体滞后 HOLD 个交易日, 去掉前视.
    if use_factor_regime and factor_regime_labels is not None:
        factor_regime_labels = factor_regime_labels.shift(HOLD)
    if step is None:
        step = test_days
    # WFA fold 划分
    folds_pos = []
    i = train_days
    while i + purge + test_days <= n:
        ts = i - train_days; te = i; ve = i + purge; vb = ve + test_days
        folds_pos.append((ts, te, ve, vb))
        i += step
    if not folds_pos:
        raise RuntimeError(f"数据不足以构成 WFA fold")

    bear = _bear_signal(mkt_level, bear_mode) if gate else None
    bull_sig = ((mkt_level / mkt_level.rolling(BEAR_MA).mean() - 1) > BULL_THR) if gate else None
    regime_label = _market_regime_label(mkt_level, dates) if use_market_regime else None
    # 因子衰减计数(两档位正渐进用): 长窗 ICIR 衰减 + 短窗近季 IC 翻空, 与 oos_wfa.py 同口径
    recent_icir = (_recent_icir(fac_ic, TRAIN_DAYS)
                   if (gate and addback_mode == "factor_decay") else None)
    recent_ic_short = ({f: fac_ic[f].rolling(GATE_SHORT_WIN).mean() for f in factor_names}
                       if (gate and addback_mode == "factor_decay") else None)
    r_def = DEF_ANN / 252.0

    fold_results, wfa_parts, bench_parts = [], [], []

    for k, (ts, te, ve, vb) in enumerate(folds_pos):
        # ── 训练: 选择因子集 ──
        if use_market_regime:
            # 预测 test 窗的市场状态 (用 train 窗最后的状态)
            last_regime = regime_label.iloc[te - 1] if te > 0 else "bull"
            train_slice = (ts, te)
            locked_set = _build_regime_factor_set(
                fac_ic, train_slice, regime_label, last_regime, candidates=factor_names)
        else:
            # 标准 Frozen: 用全量 IC 筛选
            locked_set = []
        locked_orient, locked_w = {}, {}
        if not locked_set:
            # 降级: 标准 Frozen 选择
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
        else:
            for f in locked_set:
                if f in factor_names:
                    ic = fac_ic[f].iloc[ts:te]
                    m = ic.mean(); s = ic.std()
                    if (m == m) and (s == s) and s > 1e-9:
                        locked_orient[f] = 1.0 if m >= 0 else -1.0
                        locked_w[f] = m / s * np.sqrt(252)
                    else:
                        locked_orient[f] = 1.0
                        locked_w[f] = 1.0
        wtot = sum(locked_w.get(f, 0) for f in locked_set)

        test_pos = [p for p in range(ve, vb) if p % hold == 0 and p < n]
        if not test_pos:
            fold_results.append(dict(k=k, train=f"{dates[ts].date()}~{dates[te].date()}",
                                      test="(无调仓日)", n_alive=len(locked_set), n_pos=0,
                                      sharpe=np.nan, ex_sharpe=np.nan, maxdd=np.nan,
                                      veto=True, avg_eq=np.nan))
            continue

        port, rdates, bench_vals, eq_track = [], [], [], []
        regime = False  # anti 闸门熊市 regime 状态

        for p in test_pos:
            # ── 选股信号合成 ──
            row = np.zeros(len(codes))
            if wtot > 0:
                if use_factor_regime and factor_regime_labels is not None:
                    # 因子级 regime 加权
                    total_w = 0.0
                    for f in locked_set:
                        orient = locked_orient.get(f, 1.0)
                        w_icir = locked_w.get(f, 1.0)
                        fl = factor_regime_labels.loc[dates[p], f] if dates[p] in factor_regime_labels.index else 0
                        regime_weight = W_BULL if fl == 1 else (W_BEAR if fl == -1 else W_NEU)
                        weight = orient * w_icir * regime_weight
                        row += weight * zarr[f][p]
                        total_w += abs(weight)
                    if total_w > 0:
                        row /= total_w
                else:
                    # 标准 ICIR 加权
                    for f in locked_set:
                        row += locked_orient[f] * locked_w[f] * zarr[f][p]
                    row /= wtot

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

            # ── 已验证两档位闸门 (原样对齐 oos_wfa.py 的 C/D, 非二元硬闸) ──
            #   反渐进: 熊信号触发 -> 持仓硬归0并进入熊市 regime;
            #   熊市 regime 内: 仓位封顶 50%, 由因子反弹信号在 0~50% 调制;
            #   牛市退出需'价格回升 + 因子全健康'双重确认 -> 放回 100%;
            #   正渐进(因子衰减): 无熊信号时仓位随因子报熊占比从 100% 滑到 50% 下限(不归零).
            if gate:
                if wtot <= 0:
                    pos = 0.0
                else:
                    # 因子报熊占比(长窗衰减∨近季翻空); 取更保守(减仓更多)
                    if addback_mode == "factor_decay":
                        decay = 0
                        for f in locked_set:
                            ri = recent_icir[f].iloc[p]; ti = locked_w.get(f, 0.0)
                            long_dead = (ri == ri) and (ri < 0 or ri < GATE_DECAY_FRAC * ti)
                            si = recent_ic_short[f].iloc[p]
                            short_bear = (si == si) and (si < 0)   # 近季刚报出熊市信号
                            if long_dead or short_bear:
                                decay += 1
                        decay_frac = decay / max(1, len(locked_set))
                    else:
                        decay_frac = 0.0
                    if bear.iloc[p]:
                        pos = 0.0
                        regime = True
                    elif bull_sig.iloc[p] and decay_frac == 0.0:
                        # 牛市确认需'价格回升 + 因子全健康'双重确认(避免死猫反弹里因子仍衰时误判满仓)
                        regime = False
                        pos = GATE_DECAY_FLOOR + (POS_CAP - GATE_DECAY_FLOOR) * (1.0 - decay_frac)
                    elif regime:
                        pos = GATE_DECAY_FLOOR * (1.0 - decay_frac)   # 熊市期内: 0~50%
                    else:
                        pos = GATE_DECAY_FLOOR + (POS_CAP - GATE_DECAY_FLOOR) * (1.0 - decay_frac)  # 牛市: 50%~100%
                rdef = r_def
                pr = pos * rg + (1.0 - pos) * rdef - pos * top_k * 2 * cost
                eq_track.append(pos)
            else:
                pr = rg - top_k * 2 * cost
            port.append(pr); rdates.append(dates[p])

        port_s = pd.Series(port, index=rdates)
        bench_s = pd.Series(bench_vals, index=rdates)
        st = _stat_block(f"fold{k}", port_s, bench_s, bench_s)
        veto = (st["sharpe"] < veto_sharpe) or (st["maxdd"] < veto_maxdd) or (wtot <= 0)
        avg_eq = float(np.mean(eq_track)) if eq_track else np.nan
        fold_results.append(dict(k=k,
            train=f"{dates[ts].date()}~{dates[te].date()}",
            test=f"{(dates[test_pos[0]]).date()}~{(dates[test_pos[-1]]).date()}",
            n_alive=len(locked_set), n_pos=len(test_pos),
            sharpe=st["sharpe"], ex_sharpe=st["ex_sharpe"],
            maxdd=st["maxdd"], veto=veto, avg_eq=avg_eq))
        wfa_parts.append(port_s)
        bench_parts.append(bench_s)

    wfa_port = pd.concat(wfa_parts).sort_index()
    bench_full = pd.concat(bench_parts).sort_index()
    agg = _stat_block("WFA聚合", wfa_port, bench_full, bench_full)
    valid = [f for f in fold_results if f["n_pos"] > 0 and (f["ex_sharpe"] == f["ex_sharpe"])]
    n_pass = sum(1 for f in valid if f["ex_sharpe"] > 0)
    pass_rate = n_pass / len(valid) if valid else 0.0
    catastrophic = any(f["veto"] for f in fold_results)
    decision = "PASS" if (pass_rate >= majority and not catastrophic) else "FAIL"
    return dict(folds=fold_results, wfa_port=wfa_port, bench=bench_full, agg=agg,
                pass_rate=pass_rate, n_pass=n_pass, n_valid=len(valid),
                n_folds=len(folds_pos), n_veto=sum(1 for f in fold_results if f["veto"]),
                catastrophic=catastrophic, decision=decision)


def main():
    t0 = time.time()
    print("=" * 60)
    print("双 Regime 条件 WFA 引擎")
    print(f"  REGIME_TRAINING Phase 2 + REGIME_FACTOR Phase 3")
    print("=" * 60)

    # ── 数据加载 ──
    print("\n[1/4] 加载面板 + 因子(阶段缓存可续跑)...")
    inp = load_engine_inputs_cached()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]
    print(f"  面板: {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()}")
    print(f"  因子: {len(ALL)}")

    # ── 因子级 regime 标签 ──
    print("\n[2/4] 加载因子级 regime 标签...")
    factor_regime = _load_factor_regime()
    if factor_regime is not None:
        print(f"  已加载: {factor_regime.shape}")
    else:
        print(f"  未找到 ({REGIME_CSV}), 跳过因子级 regime 加权")

    # ── 跑 5 个变体 ──
    print("\n[3/4] 跑 WFA 变体 (5 个)...")
    variants = {
        "A: 无闸": dict(gate=False, use_market_regime=False, use_factor_regime=False),
        "B: 两档位闸门": dict(gate=True, bear_mode="ma120",
                           use_market_regime=False, use_factor_regime=False),
        "C: 两档位+市场regime": dict(gate=True, bear_mode="ma120",
                                  use_market_regime=True, use_factor_regime=False),
        "D: 两档位+因子regime": dict(gate=True, bear_mode="ma120",
                                  use_market_regime=False, use_factor_regime=True),
        "E: 两档位+双regime": dict(gate=True, bear_mode="ma120",
                                use_market_regime=True, use_factor_regime=True),
    }
    base = dict(zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd,
                dates=dates, codes=codes, mkt_level=mkt_level,
                factor_regime_labels=factor_regime)

    results = {}
    for name, kw in variants.items():
        print(f"  [{name}] ...", end=" ", flush=True)
        r = rolling_wfa_dual_regime(**base, **kw)
        results[name] = r
        a = r["agg"]
        print(f"Sharpe={a['sharpe']:+.3f} 回撤={a['maxdd']:+.2%} 通过率={r['pass_rate']:.0%} "
              f"否决={r['n_veto']} 决策={r['decision']}")

    # ── 对比表 ──
    print("\n[4/4] 输出对比表")
    print(f"\n{'=' * 70}")
    print(f"{'变体':<20} {'Sharpe':>8} {'超额':>8} {'年化':>8} {'回撤':>8} {'通过率':>6} {'否决':>4}")
    print(f"{'-' * 70}")
    for name, r in results.items():
        a = r["agg"]
        print(f"{name:<20} {a['sharpe']:>+8.3f} {a['ex_sharpe']:>+8.3f} "
              f"{a['ann']:>+8.2%} {a['maxdd']:>+8.2%} {r['pass_rate']:>6.0%} {r['n_veto']:>4d}")

    # ── 图 ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    names = list(results.keys())
    sh = [results[n]["agg"]["sharpe"] for n in names]
    dd = [results[n]["agg"]["maxdd"] for n in names]
    colors = ["tab:red" if results[n]["decision"] == "FAIL" else "tab:green" for n in names]
    bars = ax.bar(range(len(names)), sh, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    for i, (s, d) in enumerate(zip(sh, dd)):
        ax.text(i, s, f"{s:+.2f}\n{d:+.0%}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("双 Regime WFA 变体对比 (红=FAIL, 绿=PASS)")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name in names:
        r = results[name]
        wp = r["wfa_port"]
        eq = (1 + wp).cumprod()
        ax.plot(eq.index, eq.values / eq.iloc[0], lw=0.9,
                label=f"{name}(DD{r['agg']['maxdd']:+.0%})")
    bf = results[names[0]]["bench"].reindex(wp.index).fillna(0.0)
    eqb = (1 + bf).cumprod()
    ax.plot(eqb.index, eqb.values / eqb.iloc[0], lw=0.6, color="gray", label="基准")
    ax.set_title("聚合净值")
    ax.set_ylabel("净值")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = OUT_DIR / "regime_wfa_comparison.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"\n  图: {fig_path}")

    print(f"\n总耗时: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
