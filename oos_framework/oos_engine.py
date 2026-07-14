"""
OOS 框架 · 验证引擎

核心流程:
1. 加载 panel + 计算因子 → 因子值矩阵
2. 计算前向收益 (FWD_HORIZON=5d)
3. 滚动 rank-IC 分析 (TRAIL=250d) → 每个因子的 IS IC/ICIR
4. IS 冻结 (SPLIT=2024-09-01) → frozen_set = {IS_IC>0 & IS_ICIR>0}
5. OOS 评估 → 逐因子 OOS IC/ICIR
6. 信号合成 + 选股回测
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  参数（与报告一致）
# ============================================================

SPLIT = "2024-09-01"       # 严格 OOS 分界
TOP_K = 0.30                # 每期取截面前 30%
HOLD = 5                    # 5 交易日再平衡
COST = 0.001                # 单边千一
REGIME_COST = COST          # 仓位在股票↔防御间摆动的单边费率
STOCK_CAP = 0.75            # 反渐进式: 熊市股票仓位上限(用户2026-07-14: 由0.50提至0.75, 信任因子+上证综指判熊)
BEAR_FLOOR = 0.50          # 反渐进式: 熊市股票仓位下限/底仓(用户2026-07-14: 由0.25提至0.50, 测试熊市保底仓位最优值)
DEFENSIVE_PATH = "D:/work Buddy GZ/Claw/stockworm/defensive/defensive_returns.parquet"  # 防御组合(70国债/25红利/5纳指)
INDEX_LAKE = "D:/work Buddy GZ/Claw/stockworm/index"  # 真实大盘指数日线(由 stock-worm/akshare 拉取)
FWD_HORIZON = 5             # 前向收益天数


def load_market_index(code: str = "sh000001", index: Optional[pd.Index] = None) -> Optional[pd.Series]:
    """加载真实大盘指数日线收盘(上证综指/沪深300等), 用于 ABC 闸门牛熊判定。

    数据源: stockworm/index/{code}.parquet (stock-worm 用 akshare 拉取, 见 build_index_lake.py)。
    对齐到给定交易日(index); 缺失回退 None(调用方降级为等权全A代理)。
    """
    p = Path(INDEX_LAKE) / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        s = df["close"].astype(float).copy()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")]
        if index is not None:
            idx = pd.to_datetime(index)
            s = s.reindex(idx).ffill().bfill()
        return s
    except Exception as e:
        warnings.warn(f"[load_market_index] {code} 加载失败: {e}")
        return None
TRAIL = 250                 # 滚动 IC 统计窗口
RNG = 42                    # 随机种子

np.random.seed(RNG)


# ============================================================
#  前向收益
# ============================================================

def compute_forward_returns(close: pd.DataFrame, horizon: int = FWD_HORIZON) -> pd.DataFrame:
    """计算前向收益矩阵。

    Args:
        close: DataFrame(index=date, columns=code)
        horizon: 前向天数

    Returns:
        DataFrame(index=date, columns=code)，值=未来 horizon 天的收益率
    """
    fwd = close.shift(-horizon) / close - 1
    return fwd


# ============================================================
#  Rank IC 分析
# ============================================================

def rank_ic(factor_values: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """计算每日截面 rank-IC (纯 numpy, 无 pandas 逐行循环, 零 float64 副本)。

    用 numpy argsort 实现行级排序 (本质 C 级), 再用全向量化协方差公式
    一次性算出所有日期的 Spearman 秩相关。避免了 pandas .rank(axis=1)
    在超宽矩阵 (4000×3881) 上创建 float64 中间 DataFrame 的内存/GC 压⼒。

    Args:
        factor_values: factor z-score DataFrame (index=date, columns=code)
        forward_returns: 前向收益 DataFrame

    Returns:
        Series(index=date, values=rank-IC)
    """
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_codes = factor_values.columns.intersection(forward_returns.columns)
    if len(common_dates) < 10 or len(common_codes) < 10:
        return pd.Series(dtype=float)

    # 子集对齐 → numpy (C 连续内存, 所有后续操作都在 C 层)
    fv = factor_values.loc[common_dates, common_codes].values.astype(np.float64, copy=False)
    fr = forward_returns.loc[common_dates, common_codes].values.astype(np.float64, copy=False)
    n_dates, n_codes = fv.shape

    # ---- 步骤 1: 行级排序 (C 级 argsort) ----
    ord_x = np.argsort(fv, axis=1)    # (n_dates, n_codes), int64
    ord_y = np.argsort(fr, axis=1)

    # 把 order 转为 rank: rk[row_i, ord_i[j]] = j+1
    # 向量化: 对每行的 order 列赋秩值
    row_ix = np.arange(n_dates)[:, None]  # (n_dates, 1)
    rk_x = np.empty_like(fv)
    rk_y = np.empty_like(fr)
    rk_x[row_ix, ord_x] = np.arange(1, n_codes + 1, dtype=np.float64)
    rk_y[row_ix, ord_y] = np.arange(1, n_codes + 1, dtype=np.float64)

    # NaN → rank also NaN
    rk_x[np.isnan(fv)] = np.nan
    rk_y[np.isnan(fr)] = np.nan

    # ---- 步骤 2: 全向量化 Spearman 相关 ----
    valid = ~np.isnan(rk_x) & ~np.isnan(rk_y)
    # keepdims: shape (n_dates, 1), 与 nansum(keepdims) 对齐, 避免广播成 (n_dates, n_dates)
    n = np.maximum(valid.sum(axis=1, keepdims=True).astype(np.float64), 1.0)

    # 去均值 (用 n 而非 nn−1, ddof 在相关中相消)
    cx = rk_x - (np.nansum(rk_x, axis=1, keepdims=True) / n)
    cy = rk_y - (np.nansum(rk_y, axis=1, keepdims=True) / n)
    np.nan_to_num(cx, copy=False)
    np.nan_to_num(cy, copy=False)

    # 所有聚合保持 keepdims, 否则 (n_dates,) 与 (n_dates,1) 广播成 (n_dates,n_dates) 方阵
    cov = (cx * cy).sum(axis=1, keepdims=True) / n
    sd_x = np.sqrt(np.maximum((cx * cx).sum(axis=1, keepdims=True) / n, 1e-15))
    sd_y = np.sqrt(np.maximum((cy * cy).sum(axis=1, keepdims=True) / n, 1e-15))

    ic = cov / (sd_x * sd_y)

    # 有效样本 < 10 → NaN
    ic[valid.sum(axis=1, keepdims=True) < 10] = np.nan

    return pd.Series(ic.ravel(), index=common_dates, name="rank_ic")


def rolling_ic_stats(
    ic_series: pd.Series,
    window: int = TRAIL,
) -> pd.DataFrame:
    """滚动 IC 统计。

    Returns:
        DataFrame(index=date, columns=["ic_mean", "ic_std", "icir"])
    """
    roll_mean = ic_series.rolling(window, min_periods=min(60, window)).mean()
    roll_std = ic_series.rolling(window, min_periods=min(60, window)).std()
    roll_icir = (roll_mean / roll_std.replace(0, np.nan))

    return pd.DataFrame({
        "ic_mean": roll_mean,
        "ic_std": roll_std,
        "icir": roll_icir,
    }, index=ic_series.index)


# ============================================================
#  IS/OOS 评估
# ============================================================

def evaluate_factors(
    factor_dict: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    split_date: str = SPLIT,
) -> pd.DataFrame:
    """对因子集做 IS/OOS 评估。

    Returns:
        DataFrame 每行一个因子:
        [name, IS_IC_mean, IS_ICIR, OOS_IC_mean, OOS_ICIR, alive]
    """
    results = []
    for name, fv in factor_dict.items():
        ic = rank_ic(fv, forward_returns)
        if ic.dropna().empty:
            results.append({
                "name": name,
                "IS_IC_mean": np.nan, "IS_ICIR": np.nan,
                "OOS_IC_mean": np.nan, "OOS_ICIR": np.nan,
                "alive": False,
            })
            continue

        is_ic = ic.loc[:split_date]
        oos_ic = ic.loc[split_date:]

        is_mean = is_ic.mean(skipna=True)
        is_icir_val = is_mean / is_ic.std() if is_ic.std() > 0 else 0
        oos_mean = oos_ic.mean(skipna=True)
        oos_icir_val = oos_mean / oos_ic.std() if oos_ic.std() > 0 else 0

        results.append({
            "name": name,
            "IS_IC_mean": round(float(is_mean), 6),
            "IS_ICIR": round(float(is_icir_val), 4),
            "OOS_IC_mean": round(float(oos_mean), 6),
            "OOS_ICIR": round(float(oos_icir_val), 4),
            "alive": bool(is_mean > 0 and is_icir_val > 0),
        })

    df = pd.DataFrame(results).set_index("name")
    df = df.sort_values("IS_ICIR", ascending=False)
    return df


def get_frozen_set(eval_df: pd.DataFrame) -> list[str]:
    """获取冻结因子集: IS_IC>0 且 IS_ICIR>0"""
    alive = eval_df[eval_df["alive"] == True]
    return alive.index.tolist()


# ============================================================
#  信号合成
# ============================================================

def build_signal(
    factor_dict: dict[str, pd.DataFrame],
    eval_df: pd.DataFrame,
    frozen_set: list[str],
    weight_src: str = "is",
) -> pd.DataFrame:
    """合成综合信号。

    composite = Σ (orient(f) × w(f) × z(f)) / Σ w(f)

    orient(f) = +1 if IS_IC ≥ 0 else -1
    w(f) = IS_ICIR(f) (weight_src="is")
    """
    if not frozen_set:
        return pd.DataFrame()

    # 获取公共日期和代码
    all_dates = None
    all_codes = None
    for name in frozen_set:
        if name not in factor_dict:
            continue
        fv = factor_dict[name]
        if all_dates is None:
            all_dates = fv.index
            all_codes = fv.columns
        else:
            all_dates = all_dates.intersection(fv.index)

    if all_dates is None or len(all_dates) == 0:
        return pd.DataFrame()

    numerator = pd.DataFrame(0.0, index=all_dates, columns=all_codes)
    denominator = 0.0

    for name in frozen_set:
        if name not in factor_dict or name not in eval_df.index:
            continue
        row = eval_df.loc[name]
        orient = 1 if row["IS_IC_mean"] >= 0 else -1
        w = abs(row["IS_ICIR"]) if weight_src == "is" else 1.0
        fv = factor_dict[name].reindex(index=all_dates, columns=all_codes)
        numerator = numerator.add(orient * w * fv.fillna(0), fill_value=0)
        denominator += w

    if denominator > 0:
        signal = numerator / denominator
    else:
        signal = numerator

    return signal


# ============================================================
#  选股回测
# ============================================================

def market_regime(
    nav: pd.Series,
    trend_window: int = 250,
    dd_thr: float = -0.15,
) -> pd.Series:
    """由市场净值判定熊市(recession)状态: 趋势破位 或 深度回撤 → True(应空仓)。

    仅使用滞后一期信息 (调用方需 .shift(1)), 避免前视偏差。
      - 趋势破位: nav < MA(trend_window)
      - 深度回撤: 自顶部回撤 < dd_thr (如 -15%)
    """
    ma = nav.rolling(trend_window, min_periods=20).mean()
    trend_down = nav < ma
    peak = nav.cummax()
    dd = nav / peak - 1.0
    deep_dd = dd < dd_thr
    bear = (trend_down | deep_dd).fillna(False)
    return bear


def bear_signal_fast(
    nav: pd.Series,
    trend_window: int = 40,
    dd_thr: float = -0.10,
    crash_drop: float = -0.05,
    crash_roll: int = 3,
    crash_roll_thr: float = -0.08,
    crash_dd_20: float = -0.10,
) -> pd.Series:
    """更快的熊市判定: 缩短 MA + 暴跌熔断触发, 让 50% 能"立即"降下来。

    相比 market_regime(MA120 + -10% 深度回撤, 滞后严重), 本函数:
      - 趋势破位: nav < MA(trend_window), 默认 40 (远快于 120)
      - 深度回撤: 自顶部回撤 < dd_thr
      - 暴跌熔断: 单日收益 < crash_drop(默认-5%)
                  或 滚动 crash_roll 日累计 < crash_roll_thr(默认-8%)
                  或 自 20 日高点回撤 < crash_dd_20(默认-10%)
                → 立即判熊, 不等趋势线确认
    仅用滞后一期信息(调用方需 .shift(1) 或延后到下一再平衡日读取)。
    """
    nav = nav.astype(float)
    ma = nav.rolling(trend_window, min_periods=10).mean()
    trend_down = nav < ma
    peak = nav.cummax()
    dd = nav / peak - 1.0
    deep_dd = dd < dd_thr
    ret = nav.pct_change().fillna(0.0)
    crash_day = ret < crash_drop
    roll = ret.rolling(crash_roll).sum()
    crash_roll_sig = roll < crash_roll_thr
    hi20 = nav.rolling(20, min_periods=5).max()
    dd20 = nav / hi20 - 1.0
    crash_dd = dd20 < crash_dd_20
    bear = (trend_down | deep_dd | crash_day | crash_roll_sig | crash_dd).fillna(False)
    return bear


# ── 新三态闸门常量 (熊/震荡/牛) — 用户最终 spec ──
BEAR_CAP = 0.50              # 熊市信号→立即降到 50%
BOUNCE_MAX = 0.75            # 反弹出现前最高加仓到此 (≤75%)
BEAR_ADD_STEP = (BOUNCE_MAX - BEAR_CAP) / 4   # 反弹前逐步加仓步长(约4期到75%)
REBEAR_CAP = 0.18            # 反弹后再判熊→快速减到 20% 以下
REBEAR_MAX_PERIODS = 12      # re_bear 安全出口: 熊信号连续消失≥此期数(仍非osc)→回满仓, 防永久踏空
GATE_OSC_RANGE = 0.12        # 震荡日线振幅上限 (12%)
GATE_OSC_DEV = 0.07          # 震荡偏离 MA 上限 (7%)
BOUNCE_GAIN_THR = 0.05       # bounce: 从熊市低点反弹 5% 视为"反弹出现"
OSC_MINUTE_BLEND = 0.60      # osc 状态下分钟短线信号在选股中的权重


# ── ABC 波浪闸门常量 (用户 2026-07-14 新 spec) ──
# 多信号共识: 用多个独立牛熊信号加权投票, 避免单点信号不稳定导致误判
# A浪(共识熊)减仓至50% → B浪(反弹)加仓至75% → C浪(再下跌/急跌)初期20%
# → 企稳敏感信号 → 分期加仓回75%
ABC_BULL_FULL = 1.0       # 企稳/牛市 满仓(用户2026-07-14纠正: 牛市应100%, 去掉"永久留25%防御")
ABC_A_CAP = 0.50          # A浪(下跌初)减仓至50%
ABC_B_FULL = 0.75         # B浪(反弹)加仓至75%
ABC_C_INIT = 0.20         # C浪(再下跌)初期20%
ABC_C_RAMP_STEP = (ABC_BULL_FULL - ABC_C_INIT) / 4   # C→企稳 分期加仓步长(约4期到75%)
ABC_BOUNCE_GAIN = 0.05    # 从低点反弹5%视为B浪(反弹出现)
ABC_CRASH_CONFIRM_REBOUND = 0.015  # 确认式熔断(用户2026-07-14方向1): 急跌后下一期反弹≥1.5%→视为V型假警报, 不砍
ABC_BEAR_SCORE_THR = 0.45 # bear_score≥此 → 共识熊(多角度验证, 需多信号同时触发)
ABC_BULL_SCORE_THR = 0.50 # bull_score≥此 → 企稳敏感信号(需多信号同时确认)
ABC_VOL_WIN = 20          # 波动率计算窗口
ABC_VOL_WIN_LONG = 60     # 波动率长期基准窗口


def oscillating_regime(
    close_daily: pd.Series,
    trend_window: int = 120,
    dd_thr: float = -0.10,
) -> pd.Series:
    """判定震荡市(ranging): 非熊市 & 价格窄幅。

    决策: 震荡 = NOT bear (by market_regime) AND 价格围绕均线窄幅波动。
    用两条件组合: |close - MA60|/MA60 < 7% 或 20日振幅 < 12%。
    """
    nav = (1 + close_daily.pct_change().fillna(0)).cumprod()
    bear = market_regime(nav, trend_window, dd_thr)

    ma60 = close_daily.rolling(60, min_periods=20).mean()
    deviation = (close_daily - ma60).abs() / (ma60 + 1e-9)
    tight = deviation < GATE_OSC_DEV

    hi20 = close_daily.rolling(20, min_periods=10).max()
    lo20 = close_daily.rolling(20, min_periods=10).min()
    range_pct = (hi20 - lo20) / (lo20 + 1e-9)
    low_range = range_pct < GATE_OSC_RANGE

    osc = (~bear) & (tight | low_range)
    return osc.fillna(False)


def build_regime_ensemble(
    bench_close_daily: pd.Series,
    rebalance_dates,
    bear_score_thr: float = ABC_BEAR_SCORE_THR,
    bull_score_thr: float = ABC_BULL_SCORE_THR,
    vw: int = ABC_VOL_WIN,
    vw_long: int = ABC_VOL_WIN_LONG,
) -> dict:
    """多信号牛熊共识引擎 (用户 2026-07-14: 牛熊信号不稳定 → 多角度验证).

    核心思想: 不再用单一信号判定牛熊, 而是用多个 *独立* 信号加权投票,
    只有多个信号 *同时* 触发才确认共识熊/共识牛, 从而大幅降低单点噪声造成的
    误判(旧 fast_bear 只用 MA40+回撤, 一次 wobble 就判熊 → 85%时间防御)。

    熊市信号(多角度, 加权, 越代表"真趋势"权重越高):
      ma20   : 价格 < MA20              (短趋势, 弱)
      ma60   : 价格 < MA60              (中趋势, 强)
      dd20   : 自20日高点回撤 < -6%     (中短期深度, 强)
      mom20  : 20日动量 < -4%           (动量, 弱)
      volx   : 20日波动率/60日 > 1.4    (波动率扩张=风险off, 中)
    牛市/企稳信号(敏感, 用于捕捉"行情企稳"):
      ma10   : 价格 > MA10              (短趋势翻多, 弱)
      mom10  : 10日动量 > +2%           (短动量翻多, 弱)
      volc   : 20日波动率 < 60日波动率  (波动率收缩=企稳, 强)
      rec    : 自历史高点回撤 > -2.5%   (回撤基本收复, 强)
    急跌熔断(立即逃, 不进共识, 直接覆盖):
      e1 单日收益 < -3.5%
      e2 3日累计 < -6%
      e3 自10日高点回撤 < -7%

    返回 dict (均已 reindex 到 rebalance_dates, 滞后0期=用当日收盘决策, 无前视):
      bear_score : float[0,1] 熊市加权共识度
      bull_score : float[0,1] 企稳加权共识度
      emergency   : bool       急跌熔断
      nav         : 等权全A净值(状态机跟踪低点用)
    """
    nav = (1 + bench_close_daily.pct_change().fillna(0)).cumprod().astype(float)
    ret = nav.pct_change().fillna(0.0)

    # ── 熊市信号 ──
    ma20 = nav.rolling(20, min_periods=10).mean()
    ma60 = nav.rolling(60, min_periods=20).mean()
    hi20 = nav.rolling(20, min_periods=10).max()
    mom20 = nav / nav.shift(20) - 1.0
    vol20 = ret.rolling(vw).std()
    vol60 = ret.rolling(vw_long, min_periods=20).std()
    bear_sig = {
        "ma20": (nav < ma20).fillna(False),
        "ma60": (nav < ma60).fillna(False),
        "dd20": ((nav / hi20 - 1) < -0.06).fillna(False),
        "mom20": (mom20 < -0.04).fillna(False),
        "volx": ((vol20 / (vol60 + 1e-9)) > 1.4).fillna(False),
    }
    bear_w = {"ma20": 1, "ma60": 2, "dd20": 2, "mom20": 1, "volx": 1}

    # ── 企稳/牛市信号(敏感) ──
    ma10 = nav.rolling(10, min_periods=5).mean()
    mom10 = nav / nav.shift(10) - 1.0
    hi_all = nav.cummax()
    dd_all = nav / hi_all - 1.0
    bull_sig = {
        "ma10": (nav > ma10).fillna(False),
        "mom10": (mom10 > 0.02).fillna(False),
        "volc": (vol20 < vol60).fillna(False),
        "rec": (dd_all > -0.025).fillna(False),
    }
    bull_w = {"ma10": 1, "mom10": 1, "volc": 2, "rec": 2}

    # ── 急跌熔断 ──
    e1 = ret < -0.035
    roll3 = ret.rolling(3).sum()
    e2 = roll3 < -0.06
    hi10 = nav.rolling(10, min_periods=5).max()
    e3 = ((nav / hi10 - 1) < -0.07).fillna(False)
    emergency = (e1 | e2 | e3).fillna(False)

    bw = sum(bear_w[k] * bear_sig[k] for k in bear_w)
    gw = sum(bull_w[k] * bull_sig[k] for k in bull_w)
    bear_score = (bw / sum(bear_w.values())).reindex(rebalance_dates).fillna(0.0)
    bull_score = (gw / sum(bull_w.values())).reindex(rebalance_dates).fillna(0.0)
    emergency_rb = emergency.reindex(rebalance_dates).fillna(False)
    nav_rb = nav.reindex(rebalance_dates).fillna(1.0)

    return {
        "bear_score": bear_score,
        "bull_score": bull_score,
        "emergency": emergency_rb,
        "nav": nav_rb,
    }


def bear_scale_series(
    signal: pd.DataFrame,
    hold: int = HOLD,
    top_frac: float = 0.10,
    win: int = 252,
    minp: int = 60,
) -> pd.Series:
    """熊市渐进加仓强度: 合成信号的多空价差(spread)滚动 z 分数, 截断到 [0, 1]。

    值越大 = 当期选股 alpha 越强 → 熊市可多配股票; 值=0 → 熊市股票清零, 全配防御。
    与 backtest_regime_compare.py 口径一致, 保证引擎与对比回测同口径。
    """
    topk = max(1, int(signal.shape[1] * top_frac))
    spread = signal.apply(
        lambda r: r.nlargest(topk).mean() - r.nsmallest(topk).mean(), axis=1
    )
    rmean = spread.rolling(win, min_periods=minp).mean()
    rstd = spread.rolling(win, min_periods=minp).std()
    scale = ((spread - rmean) / (rstd + 1e-9)).clip(0, 1)
    return scale


def run_backtest(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    top_k: float = TOP_K,
    hold: int = HOLD,
    cost: float = COST,
    rf_annual: float = 0.0,
    use_regime: bool = True,
    regime_trend_w: int = 120,
    regime_dd_thr: float = -0.10,
    gate_mode: str = "anti",
    stock_cap: float = STOCK_CAP,
    defensive_path: str = DEFENSIVE_PATH,
    signal_minute: Optional[pd.DataFrame] = None,
    fast_bear: bool = True,
    bear_trend_w: int = 40,
    abc_static_weight: Optional[float] = None,
    market_index: str = "sh000001",
    delist_risk: Optional[pd.DataFrame] = None,
    delist_thr: float = 0.50,
) -> pd.DataFrame:
    """基于信号的选股回测（反渐进式默认闸 + 双成本模型）。

    闸门模式 gate_mode:
      - "binary": 旧二元硬闸, 熊市直接空仓(收益≈rf), 无防御资产。
      - "anti"  (默认, 用户选定): 牛市满仓; 熊市(MA120/-10%)股票仓位 = 信号强度×stock_cap
            (上限 stock_cap, 信号弱则清零), 余下配防御组合(70%国债ETF+25%红利低波+5%纳指)。
      - "tri":   三态闸门(熊/震荡/牛): 快熊信号(缩短MA+暴跌熔断)→熊市立即50%→等待反弹加至≤75%
            →反弹再判熊快减<20%(非粘性, 持续反弹逐步回补); 震荡市用分钟短线因子参与(30%~70%)。

    成本模型(双, 已修正):
      - 换股成本 = w × cost×2×turnover  (仅对持股部分计费, 空仓 w=0 时=0)
      - 仓位摆动成本 = REGIME_COST×2×|Δw|  (股票↔防御 整体摆动, 捕捉反渐进式滑点)
      注: 旧版把换股成本按全组合扣、且漏计仓位摆动 → 空仓期被多扣(脏数), 已修正。

    关键修正(继承):
      - 持有期收益 = close[t+hold]/close[t] - 1, 每个再平衡日只取「一次」持有期收益。
      - 基准 = 当期所有可交易股票的等权持有期收益 (buy-everything)。
      - regime 闸: tri/anti 用等权全A基准净值判定市场状态(滞后一期避免前视);
      ABC 闸门改用真实大盘指数(默认上证综指 sh000001, 由 stock-worm 拉取)判定牛熊。

    Args:
        signal: 合成信号矩阵 (index=date, columns=code), 截面方向已带正负。
        close:  复权收盘价矩阵 (index=date, columns=code)。
        top_k:  每期选股比例。
        hold:   持有/再平衡天数。
        cost:   单边费率 (如 0.001 = 千一)。
        rf_annual: 年化无风险利率 (夏普分母用), A股近似 0。
        gate_mode: "anti"(反渐进式, 默认) 或 "binary"(二元硬闸)。
        stock_cap: 熊市股票仓位上限 (反渐进式)。
        defensive_path: 防御组合 parquet 路径。

    Returns:
        DataFrame(index=date, columns=["portfolio", "weight", "benchmark"])
        weight = 当期股票仓位(诊断/滑点分析用)。
    """
    if gate_mode not in ("binary", "anti", "tri", "abc"):
        raise ValueError(f"未知 gate_mode: {gate_mode}")

    # 持有期前向收益: 每个格子 = 从当日持有 hold 天的收益
    fwd = close.shift(-hold) / close - 1

    common_dates = sorted(set(signal.index) & set(fwd.index))
    if len(common_dates) < hold + 1:
        return pd.DataFrame()

    rebalance_dates = common_dates[::hold]
    n_select = max(1, int(signal.shape[1] * top_k))

    # 预计算每期等权全A基准收益 → 累积净值 → regime 状态 (滞后一期避免前视)
    _bench = []
    for _d in rebalance_dates:
        if _d in fwd.index:
            _r = fwd.loc[_d].dropna()
            _bench.append(_r.mean() if len(_r) else 0.0)
        else:
            _bench.append(0.0)
    _bench_ret = pd.Series(_bench, index=rebalance_dates)
    _bench_nav = (1 + _bench_ret.fillna(0)).cumprod()
    # 大盘指数净值(可选): 用于 anti 牛熊判定, 替代等权全A代理
    # 用户诊断(2026-07-14): 之前 anti 一直用等权全A净值当牛熊标的 → 应改为真实上证综指
    _idx_nav_rb = None
    if market_index:
        _idx_close = load_market_index(market_index, index=close.index)
        if _idx_close is not None:
            _idx_nav_daily = (1 + _idx_close.pct_change().fillna(0.0)).cumprod()
            _idx_nav_rb = _idx_nav_daily.reindex(rebalance_dates).ffill().fillna(1.0)
    _bear_nav = _idx_nav_rb if _idx_nav_rb is not None else _bench_nav
    _bear = market_regime(_bear_nav, regime_trend_w, regime_dd_thr).shift(1, fill_value=False) \
        if use_regime else pd.Series(False, index=rebalance_dates)
    # 日频快熊信号(缩短MA + 暴跌熔断): 让三态闸门的 50% 在暴跌后 ≤HOLD 天内即可触发
    _bench_close_daily = close.mean(axis=1)
    _bench_nav_daily = (1 + _bench_close_daily.pct_change().fillna(0)).cumprod()
    _bear_daily = bear_signal_fast(_bench_nav_daily, bear_trend_w, regime_dd_thr) \
        if (use_regime and fast_bear) else pd.Series(False, index=_bench_close_daily.index)
    _bear_daily_rb = _bear_daily.reindex(rebalance_dates).fillna(False)
    # 市场状态标签(bull/bear/osc), 用于分段回测 — 与闸门内部判定一致
    _regime_label = pd.Series("bull", index=rebalance_dates)
    _regime_label = _regime_label.mask(_bear, "bear").mask(_bear_daily_rb, "bear")
    rf_period = rf_annual / 252.0 * hold

    # 防御组合收益 (对齐再平衡日): 反渐进/三态/ABC 都用
    rdef = pd.Series(0.0, index=rebalance_dates)
    if gate_mode in ("anti", "tri", "abc"):
        try:
            rdef = pd.read_parquet(defensive_path)["defensive"].reindex(rebalance_dates).fillna(0.0)
        except Exception as e:
            print(f"  [WARN] 防御组合加载失败({defensive_path}): {e}, 熊市非股票部分按 rf=0")

    # 熊市渐进加仓强度 (信号多空价差滚动 z): 反渐进/三态/ABC 用
    scale_ser = bear_scale_series(signal) if gate_mode in ("anti", "tri", "abc") else None

    # ── 三态闸门: 震荡检测 + 反弹检测 ──
    if gate_mode == "tri":
        # 震荡: 用等权全A日频 close 做净值→偏离检测
        # close 原始以交易日为 index, 用 daily_close_nav 做检测
        _daily_close = close.mean(axis=1)  # 全A均价(代理日频)
        _osc = oscillating_regime(_daily_close, regime_trend_w, regime_dd_thr)
        _osc_rb = _osc.reindex(rebalance_dates).fillna(False) if not _osc.empty \
            else pd.Series(False, index=rebalance_dates)
        _regime_label = _regime_label.mask(_osc_rb, "osc")
        tri_state = "bull"       # bull / bear_50 / bear_add / re_bear / osc
        bear_low = 1.0           # 本次熊市区间阶段低点(净值)
        bear_periods = 0         # 进入 bear_add 后已过再平衡期数(逐步加仓用)
        reb_clear = 0            # re_bear 状态下熊信号连续消失期数(安全出口用)
        # osc 态统一使用日线级别因子(用户决策 2026-07-14: 不再接入分钟数据湖)
    elif gate_mode == "abc":
        # 多信号牛熊共识引擎: 多角度验证牛熊, 替代单点 fast_bear
        # 大盘参考改用真实上证综指(stock-worm 拉取), 而非等权全A代理
        _idx_close = load_market_index(market_index, index=close.index)
        if _idx_close is None:
            warnings.warn(f"[ABC] 指数 {market_index} 缺失, 降级为等权全A代理")
            _bench_close_daily = close.mean(axis=1)
        else:
            _bench_close_daily = _idx_close
        ens = build_regime_ensemble(_bench_close_daily, rebalance_dates)
        _regime_label = pd.Series("bull", index=rebalance_dates)
        _regime_label = _regime_label.mask(
            ens["bear_score"] >= ABC_BEAR_SCORE_THR, "bear")
        _regime_label = _regime_label.mask(ens["emergency"], "crash")
        abc_state = "bull"       # bull / A / B / C
        wave_low = 1.0           # 当前下跌浪阶段低点(净值), 用于反弹判定
        stab_ramp = 0            # C→企稳 分期加仓已过期权数
        pending_crash = False    # 确认式熔断: 急跌已挂起, 等下一期确认
        crash_ref_nav = 1.0      # 挂起时的净值, 用于判断下一期是否反弹收回
        # osc 态/选股统一使用日线级别因子(不再接入分钟数据湖)
    elif gate_mode == "anti":
        pass  # rdef 已在上方加载
    else:
        rdef = pd.Series(0.0, index=rebalance_dates)

    portfolio_returns = []
    benchmark_returns = []
    weight_series = []
    state_series = []  # 三态闸门状态记录
    prev_selected = set()
    prev_w = 1.0

    for date in rebalance_dates:
        if date not in signal.index:
            continue
        b = bool(_bear_daily_rb.get(date, False)) if gate_mode == "tri" else bool(_bear.get(date, False))

        # ── 股票仓位 w ──
        if gate_mode == "binary":
            w = 0.0 if b else 1.0
        elif gate_mode == "tri":
            # === 三态闸门状态机(用户最终 spec) ===
            # 熊→立即50% →反弹前逐步加至≤75% →反弹再判熊→<20%按住至震荡 →震荡用短线因子参与
            is_osc = bool(_osc_rb.get(date, False))
            cur_nav = float(_bench_nav.get(date, 1.0))
            sig_str = float(scale_ser.get(date, 0.5))  # 信号强度 (0~1)

            if is_osc:
                # 震荡市: 用日线级别因子(z-score 均值)调制参与强度(不再用分钟信号)
                tri_state = "osc"
                daily_z = signal.loc[date].dropna()
                osc_sig = float(daily_z.mean()) if len(daily_z) else 0.0
                w = float(np.clip(0.30 + osc_sig * 0.40, 0.30, 0.70))
            elif tri_state == "re_bear":
                # 反弹后再判熊→已快速减到 18% 以下; 按 spec 按住至市场进入震荡(osc 在上方优先处理)
                w = REBEAR_CAP
                if not b:
                    reb_clear += 1
                    if reb_clear >= REBEAR_MAX_PERIODS:
                        # 安全出口: 熊信号连续消失很久且仍未 osc → 回满仓, 防永久踏空
                        tri_state = "bull"
                        w = 1.0
                        bear_low = 1.0
                        bear_periods = 0
                        reb_clear = 0
                else:
                    reb_clear = 0
            elif not b:
                # 非熊市(且非 re_bear/非 osc): 满仓
                tri_state = "bull"
                w = 1.0
                bear_low = 1.0
                bear_periods = 0
            elif tri_state in ("bull", "osc"):
                # 刚进入熊市: 立即降到 50%
                tri_state = "bear_50"
                w = BEAR_CAP
                bear_low = cur_nav
                bear_periods = 0
            else:
                # 已在熊市(bear_50 / bear_add), 反弹尚未出现 → 从 50% 逐步加仓至 ≤75%
                bounced = (cur_nav / max(bear_low, 1e-9) - 1) > BOUNCE_GAIN_THR
                if bounced:
                    if b:
                        # 反弹后接着判断熊市 → 快速减仓至 20% 以下(按住至震荡)
                        tri_state = "re_bear"
                        w = REBEAR_CAP
                        bear_periods = 0
                        reb_clear = 0
                    else:
                        # 反弹 + 不再熊 → 确认复苏, 满仓
                        tri_state = "bull"
                        w = 1.0
                        bear_low = 1.0
                        bear_periods = 0
                else:
                    # 反弹未出现: 逐步加仓, 最高不超 75%
                    if tri_state == "bear_50":
                        tri_state = "bear_add"
                        bear_periods = 1
                    elif tri_state == "bear_add":
                        bear_periods += 1
                    w = float(min(BOUNCE_MAX, BEAR_CAP + bear_periods * BEAR_ADD_STEP))

            # 跟踪阶段低点
            if b:
                bear_low = min(bear_low, cur_nav)
        elif gate_mode == "anti":
            w = max(BEAR_FLOOR, float(scale_ser.get(date, 0.0)) * stock_cap) if b else 1.0
        elif gate_mode == "abc":
            # === ABC 波浪闸门状态机 (多信号加权共识驱动) ===
            # 设计要点(用户 2026-07-14):
            #  - 单点牛熊信号不稳定 → 用 bear_score/bull_score 加权共识(需多信号同时触发)
            #  - 急跌(e_m)立即逃到 C 浪(20%); 慢跌(非共识熊)保持满仓参与(因子强劲)
            #  - A浪(共识熊)减至50% → B浪(反弹)加至75% → C浪(再下跌/急跌)初期20%
            #    → 企稳敏感信号(bull_score) → 分期加仓回75%
            bs = float(ens["bear_score"].get(date, 0.0))
            gs = float(ens["bull_score"].get(date, 0.0))
            em = bool(ens["emergency"].get(date, False))
            cur_nav = float(ens["nav"].get(date, 1.0))
            cb = bs >= ABC_BEAR_SCORE_THR            # 共识熊(多角度验证)
            stable = (gs >= ABC_BULL_SCORE_THR) and (not cb) and (not em)  # 企稳敏感信号
            if abc_state in ("A", "B", "C"):
                wave_low = min(wave_low, cur_nav)
            bounced = (cur_nav / max(wave_low, 1e-9) - 1) > ABC_BOUNCE_GAIN

            # ── 确认式急跌熔断(用户2026-07-14方向1) ──
            #  暴跌当期不立即砍, 先挂起观察一期:
            #   - 下一期已反弹收回(≥1.5%) 或 急跌信号消失 → V型假警报, 不砍(避免砍在底部踏空)
            #   - 下一期仍未收回且续跌 → 确认真急跌, 逃到 C 浪(20%)
            crash_now = False
            if pending_crash:
                rebound = (cur_nav / max(crash_ref_nav, 1e-9) - 1) >= ABC_CRASH_CONFIRM_REBOUND
                pending_crash = False
                if (not rebound) and em:
                    crash_now = True          # 未收回且仍急跌 → 确认逃逸
                # 否则(已反弹 或 急跌解除) → 假警报, 维持原状态机
            if em and (not crash_now) and (not pending_crash):
                pending_crash = True          # 新出现急跌 → 先挂起, 下一期确认
                crash_ref_nav = cur_nav

            if crash_now:
                # 确认急跌: 逃到 C 浪(20%)
                abc_state = "C"
                w = ABC_C_INIT
                wave_low = cur_nav
                stab_ramp = 0
            elif abc_state == "bull":
                if cb:
                    abc_state = "A"          # A浪: 共识熊→减仓至50%
                    w = ABC_A_CAP
                    wave_low = cur_nav
                else:
                    w = ABC_BULL_FULL        # 非共识熊(含慢跌): 满仓参与, 上限75%
            elif abc_state == "A":
                if cb:
                    if bounced:
                        abc_state = "B"       # 反弹出现→B浪, 加至75%
                        w = ABC_B_FULL
                    else:
                        w = ABC_A_CAP         # 仍共识熊且未反弹→维持50%
                else:
                    abc_state = "bull"       # 熊信号解除→回牛市(75%)
                    w = ABC_BULL_FULL
                    wave_low = 1.0
            elif abc_state == "B":
                if cb and (cur_nav < wave_low * 0.995):
                    abc_state = "C"          # 再下跌(C浪): 创出新低→硬减至20%
                    w = ABC_C_INIT
                    wave_low = cur_nav
                    stab_ramp = 0
                elif stable:
                    abc_state = "bull"       # 企稳确认→回牛市
                    w = ABC_BULL_FULL
                    wave_low = 1.0
                else:
                    w = ABC_B_FULL           # 反弹延续→维持75%参与
            elif abc_state == "C":
                if stable:
                    stab_ramp += 1           # 企稳敏感信号→分期加仓回75%
                    w = float(min(ABC_BULL_FULL, ABC_C_INIT + stab_ramp * ABC_C_RAMP_STEP))
                    if w >= ABC_BULL_FULL - 1e-9:
                        abc_state = "bull"
                        wave_low = 1.0
                        stab_ramp = 0
                elif cb:
                    w = ABC_C_INIT
                    wave_low = min(wave_low, cur_nav)
                else:
                    abc_state = "bull"       # 熊信号解除但未确认企稳→谨慎回牛市
                    w = ABC_BULL_FULL
                    wave_low = 1.0

        # ── 择时关闭对照(恒定仓位, 其余路径完全同 abc: 同选股/同防御/同成本) ──
        # 用于隔离"择时"本身的贡献: 仅把 w 钉死, 状态机/信号/防御/手续费全不变。
        if gate_mode == "abc" and abc_static_weight is not None:
            w = float(abc_static_weight)

        # 选股: 统一使用日线级别信号(osc 态也用日线因子, 不再混合分钟信号)
        scores = signal.loc[date].dropna()
        # ── 退市风险过滤(可选): 选股前剔除高风险股, 隔离困境股敞口 ──
        if delist_risk is not None and date in delist_risk.index:
            _dr = delist_risk.loc[date].reindex(scores.index)
            scores = scores[_dr.fillna(0.0) < delist_thr]
        if len(scores) >= n_select:
            selected = set(scores.nlargest(n_select).index.tolist())
            row = fwd.loc[date]
            avail = [c for c in selected if c in row.index and pd.notna(row[c])]
        else:
            selected, row, avail = set(), None, []

        # 股票部分收益 + 换股成本(仅持股部分)
        if avail:
            sr = row[avail].mean()
            turnover = (len(selected ^ prev_selected) / max(1, len(selected))) if prev_selected else 1.0
            name_cost = w * cost * 2 * turnover
            prev_selected = selected
        else:
            sr = rf_period
            name_cost = 0.0

        if gate_mode == "abc":
            dr = float(rdef.get(date, 0.0))   # ABC 永久留防御仓位(恒 w≤0.75)
        elif gate_mode == "tri":
            dr = float(rdef.get(date, 0.0)) if w < 1.0 else 0.0
        elif gate_mode == "anti":
            dr = float(rdef.get(date, 0.0)) if b else 0.0
        else:
            dr = 0.0
        reg_to = abs(w - prev_w)
        regime_cost = REGIME_COST * 2 * reg_to
        ret = w * sr + (1 - w) * dr - name_cost - regime_cost

        all_row = fwd.loc[date].dropna() if date in fwd.index else pd.Series(dtype=float)
        bench_ret = all_row.mean() if len(all_row) else 0.0

        portfolio_returns.append({"date": date, "return": float(ret)})
        benchmark_returns.append({"date": date, "return": float(bench_ret)})
        weight_series.append({"date": date, "weight": float(w)})
        if gate_mode == "tri":
            state_series.append({"date": date, "state": tri_state})
        elif gate_mode == "abc":
            state_series.append({"date": date, "state": abc_state})
        prev_w = w

    if not portfolio_returns:
        return pd.DataFrame()

    port = pd.DataFrame(portfolio_returns).set_index("date")
    bench = pd.DataFrame(benchmark_returns).set_index("date")
    wt = pd.DataFrame(weight_series).set_index("date")
    result = pd.DataFrame({
        "portfolio": port["return"],
        "weight": wt["weight"],
        "benchmark": bench["return"],
    })
    if gate_mode in ("tri", "abc") and state_series:
        st = pd.DataFrame(state_series).set_index("date")
        result["gate_state"] = st["state"]
    if gate_mode in ("tri", "abc") and not _regime_label.empty:
        result["regime"] = _regime_label
    return result


# ============================================================
#  回测指标
# ============================================================

def compute_metrics(returns: pd.DataFrame, periods_per_year: float = 252 / HOLD,
                    rf_annual: float = 0.0) -> dict:
    """计算回测指标: 夏普, 年化, 最大回撤, 累计收益。

    Args:
        returns: 逐期收益序列 (已为持有期收益, 非日收益)。
        periods_per_year: 每年再平衡期数 (默认 252/HOLD)。
        rf_annual: 年化无风险利率, 用于夏普。
    """
    if returns.empty:
        return {}

    metrics = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 5:
            continue

        ann_ret = r.mean() * periods_per_year
        ann_vol = r.std() * np.sqrt(periods_per_year)
        excess = ann_ret - rf_annual
        sharpe = excess / ann_vol if ann_vol > 0 else 0

        equity = (1 + r).cumprod()
        cum = equity.iloc[-1] - 1
        cum_max = equity.cummax()
        dd = equity / cum_max - 1
        max_dd = dd.min()

        metrics[col] = {
            "sharpe": round(float(sharpe), 4),
            "annual_return": round(float(ann_ret), 4),
            "annual_vol": round(float(ann_vol), 4),
            "max_drawdown": round(float(max_dd), 4),
            "cumulative_return": round(float(cum), 4),
            "n_periods": len(r),
        }

    return metrics


# ============================================================
#  一站式 OOS 验证
# ============================================================

def run_full_oos(
    factor_dict: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    split_date: str = SPLIT,
    industry_map: dict = None,
    close: pd.DataFrame = None,
) -> dict:
    """一站式 OOS 验证流程。

    Args:
        factor_dict: 因子矩阵字典。
        forward_returns: 前向收益 (用于 IC 分析)。
        split_date: OOS 分界。
        industry_map: 行业映射。
        close: 复权收盘价矩阵 (回测用, 必须传入否则回测为空)。

    Returns:
        {
            "eval": DataFrame (逐因子 IS/OOS IC/ICIR),
            "frozen_set": list[str],
            "backtest_result": DataFrame (组合/基准收益),
            "metrics": dict (夏普/年化/回撤),
        }
    """
    print(f"\n{'='*60}")
    print(f"OOS 验证: {len(factor_dict)} 因子, SPLIT={split_date}")
    print(f"{'='*60}")

    # Step 1: 因子评估
    print("\n[1/4] 计算 IS/OOS IC/ICIR...")
    eval_df = evaluate_factors(factor_dict, forward_returns, split_date)

    frozen_set = get_frozen_set(eval_df)
    n_alive = len(frozen_set)
    print(f"  冻结集: {n_alive}/{len(factor_dict)} 因子")

    # Step 2: 信号合成
    print("\n[2/4] 合成 Frozen 信号...")
    signal = build_signal(factor_dict, eval_df, frozen_set, weight_src="is")
    print(f"  信号维度: {signal.shape}")

    # Step 3: 回测 (必须用复权收盘价, 不能用重叠前向收益复利)
    gate_mode = "anti"   # 用户选定默认闸门: 反渐进式 + 防御资产
    trend_w, dd_thr = 120, -0.10
    print("\n[3/4] 回测 (top_k={}, hold={}, cost={}, gate={}, regime=MA{}/{}%)...".format(
        TOP_K, HOLD, COST, gate_mode, trend_w, int(dd_thr * 100)))
    if close is None:
        print("  [WARN] 未传入 close, 跳过回测")
        bt = pd.DataFrame()
    else:
        bt = run_backtest(
            signal, close, TOP_K, HOLD, COST,
            gate_mode=gate_mode, regime_trend_w=trend_w, regime_dd_thr=dd_thr,
        )

    # Step 4: 指标
    print("\n[4/4] 计算回测指标...")
    metrics = compute_metrics(bt, periods_per_year=252 / HOLD) if not bt.empty else {}

    print(f"\n{'='*60}")
    print("冻结因子 (IS IC/ICIR → OOS IC/ICIR):")
    print(eval_df[eval_df.index.isin(frozen_set)].to_string())
    print(f"\n回测指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    return {
        "eval": eval_df,
        "frozen_set": frozen_set,
        "backtest_result": bt,
        "metrics": metrics,
    }
