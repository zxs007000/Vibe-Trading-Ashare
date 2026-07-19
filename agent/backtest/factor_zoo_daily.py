"""factor_zoo_daily.py — Branch 2(库优先): 异族因子动物园 × regime IC 矩阵.

策略(用户指示): 先去 466 因子库挑, 拿近似因子改(实现标准公式), 最后才重新算.
本脚本从库里挑 ICIR 最高的异族因子(momentum/reversal/volatility/liquidity),
在已落地的 20y 日线面板(1489 只, OHLCV+amount)上按标准公式实现,
计算 逐因子 × 逐年(=regime) 的 rank-IC 矩阵 + 热力图, 找出 2023-2026 仍活着的因子.

说明: 库里 academic_rmw/smb/cma/hml(质量/价值) 需基本面(市值/账面值), 日线面板没有,
故用 价格可算 的异族因子代替: 动量/反转/波动/流动性(Amihud 用 amount) + 特质波动.
这些家族与现有 7 个微观结构因子驱动源不同 -> 不同 regime 会有不同因子活着.

用法:
  python backtest/factor_zoo_daily.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backtest.validation import _sharpe  # noqa

PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
OUT_DIR = Path(__file__).parent / "screen_results"
MAT_CSV = OUT_DIR / "factor_zoo_regime_ic.csv"
HEAT = OUT_DIR / "factor_zoo_regime_heatmap.png"
CACHE = OUT_DIR / "factor_zoo_ic.pkl"
FWD_HORIZON = 5          # 5日持有 -> 用 5日前向收益算 IC(与回测框架一致)

# 牛熊 regime 注入: WFA 用全市场趋势算一次后 set 进来, build_factors 分块时复用
# (build_factors 按股票分块调用, 块内 close.mean 只是块均值, 不能当市场趋势, 故由外部注入)
_MARKET_TREND = None
def set_market_regime(s):
    """注入全市场趋势序列(date-indexed, 如 mkt_level/等权均值 的 250d 偏离),
    供牛熊混合反转因子判定牛/熊/拐点. 不注入时该因子退化为 20d 反转."""
    global _MARKET_TREND
    _MARKET_TREND = s


def load_wide():
    p = pd.read_parquet(PANEL)
    p = p.sort_values(["code", "date"])
    cols = ["open", "high", "low", "close", "volume", "amount"]
    wide = {c: p.pivot(index="date", columns="code", values=c) for c in cols}
    return wide


def build_factors(wide):
    """返回 dict: name -> date×code DataFrame(已为因子值, 未标准化).

    内存策略: 输入宽表降为 float32(全市场 5515 只时, 33 因子 float64 宽表 ~7.2GB 会逼近 8G cgroup 上限);
    float32 对 rank-IC/中性化/z-score 精度足够. 仅 beta 计算在 _beta_matrix 内显式转 float64 保精度.
    """
    close = wide["close"].astype(np.float32)
    open_ = wide["open"].astype(np.float32)
    low = wide["low"].astype(np.float32)
    high = wide["high"].astype(np.float32)
    vol = wide["volume"].astype(np.float32)
    amount = wide["amount"].astype(np.float32)
    ret = close.pct_change()
    f = {}
    # ---- 动量 momentum ----
    f["mom_5"]   = close / close.shift(5) - 1
    f["mom_20"]  = close / close.shift(20) - 1
    f["mom_60"]  = close / close.shift(60) - 1
    f["mom_120"] = close / close.shift(120) - 1
    f["mom_250"] = close / close.shift(250) - 1
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1      # Carhart 12-1 月
    # ---- 反转 reversal ----
    f["rev_5"]  = -f["mom_5"]
    f["rev_20"] = -f["mom_20"]
    f["rev_60"] = -f["mom_60"]
    f["rev_intraday"] = -(close - open_) / open_               # 隔夜/日内反转
    # ---- 牛熊混合反转(千问方案): 牛=动量分位 / 熊=反转分位 / 拐点=60%反转+40%动量 ----
    # 用交叉截面分位秩混合: 牛→动量分位, 熊→反转分位, 中间平滑过渡(拐点≈50/50).
    mom_lt = close / close.shift(250) - 1                       # 1年动量
    rev_mt = -(close / close.shift(20) - 1)                     # 20日反转
    mom_r = mom_lt.rank(axis=1, pct=True)
    rev_r = rev_mt.rank(axis=1, pct=True)
    if _MARKET_TREND is not None:
        mt = _MARKET_TREND.reindex(close.index)
        w_bull = ((mt + 0.15) / 0.30).clip(0, 1)                # 1=强牛(用动量) 0=熊(用反转)
        wbm = pd.DataFrame(np.tile(w_bull.values.reshape(-1, 1), (1, close.shape[1])),
                           index=close.index, columns=close.columns)
        f["bullbear_rev"] = wbm * mom_r + (1 - wbm) * rev_r
    else:
        f["bullbear_rev"] = rev_r                               # 无 regime 时退化为 20d 反转
    # ---- 波动 volatility ----
    f["vol_20"] = ret.rolling(20).std()
    f["vol_60"] = ret.rolling(60).std()
    f["ret_skew_60"] = ret.rolling(60).skew()
    # 特质波动 idiosyncratic vol(对等权市场收益的残差波动)
    mkt = ret.mean(axis=1)
    beta = _beta_matrix(ret, mkt, min_obs=250)   # 向量化, 与逐股票 np.cov 循环严格等价(单测 ~1e-15)
    beta_mat = pd.DataFrame({c: beta.get(c, np.nan) for c in ret.columns}, index=ret.index)
    resid = ret - beta_mat.mul(mkt, axis=0)
    f["ivol_60"] = resid.rolling(60).std()
    # ---- 流动性 liquidity(用 amount) ----
    f["amihud_20"] = (ret.abs() / amount).rolling(20).mean()     # |ret|/成交额
    dolvol = amount.rolling(20).mean()
    f["dolvol_trend"] = amount / dolvol - 1                       # 近期流动性 vs  trailing
    # ---- 技术面 technical ----
    ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    f["ma_dev_20"] = close / ma20 - 1
    f["ma_dev_60"] = close / ma60 - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close   # 标准化 MACD 柱
    gain = ret.clip(lower=0); loss = (-ret).clip(lower=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-12)
    f["rsi_14"] = 1 - 1 / (1 + rs)                                # 0~1
    s20 = close.rolling(20).std()
    f["boll_w"] = (ma20 + 2 * s20 - (ma20 - 2 * s20)) / ma20      # 布林带宽
    f["adx_14"] = _adx(high, low, close, 14)                      # 趋向强度
    # ---- 微观结构 microstructure ----
    f["high_52w"] = close / close.rolling(252).max() - 1          # 距52周新高
    f["overnight_gap"] = open_ / close.shift(1) - 1               # 今开/昨收
    f["intraday_range"] = (high - low) / close                    # 日内振幅
    f["downside_vol_60"] = ret.clip(upper=0).rolling(60).std()    # 下行波动
    f["drawup_60"] = close.rolling(60).max() / close - 1          # 距60日高跌幅(回撤近似)
    # ---- 量价 volume-price ----
    f["vol_ratio"] = vol / vol.rolling(20).mean()                 # 异常放量
    f["vol_price_corr"] = ret.rolling(20).corr(vol.pct_change())  # 量价相关
    f["amount_strength"] = amount / amount.rolling(60).mean() - 1 # 成交强度
    return f


def _beta_matrix(ret, mkt, min_obs=250):
    """向量化 beta = cov(mkt,ret)/var(mkt), 与逐股票 np.cov 循环严格等价(误差~1e-15).

    旧循环每列用各自 paired 子集 v=rc.notna()&mkt.notna(), 在子集上算均值/协方差.
    此处逐列对齐该子集语义: V 为逐列 paired 掩码, 无效行值先置 0(0.0*nan=nan, 必须 np.where 处理)
    再求加权矩. 公式为 beta = Sxy*n/(Sxx*(n-1)), 其中 Sxx=Σ(x-xbar)^2, Sxy=Σ(x-xbar)(y-ybar),
    恰等于 np.cov(x,y)[0,1]/np.var(x)(cov ddof=1, var ddof=0).
    全市场 5500 只时, 向量化约 0.02s/400只 vs 循环 0.25s/400只 ~10-25x, 且避开逐股票 Python 循环
    在 600s 会话硬限下跑不完的问题.
    """
    R = ret.values.astype(np.float64)
    Mc = mkt.values.astype(np.float64)[:, None]
    V = (ret.notna() & mkt.notna().to_numpy()[:, None]).values.astype(np.float64)  # 0/1
    Mc_s = np.where(V > 0.5, Mc, 0.0)
    R_s = np.where(V > 0.5, R, 0.0)
    n = V.sum(axis=0)
    mkt_sum = Mc_s.sum(axis=0)
    ret_sum = R_s.sum(axis=0)
    mkt_sq_sum = (Mc_s ** 2).sum(axis=0)
    xy_sum = (Mc_s * R_s).sum(axis=0)
    xbar = mkt_sum / n
    ybar = ret_sum / n
    Sxx = mkt_sq_sum - n * xbar ** 2
    Sxy = xy_sum - n * xbar * ybar
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = Sxy * n / (Sxx * (n - 1))
    beta = np.where(n < min_obs, np.nan, beta)
    return pd.Series(beta, index=ret.columns)


def _adx(high, low, close, n=14):
    """标准 ADX(趋向强度), 向量化. 返回 date×code DataFrame."""
    up = high.diff(); dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    pc = close.shift(1)
    tr = (high - low).combine((high - pc).abs(), np.maximum).combine((low - pc).abs(), np.maximum)
    atr = tr.rolling(n).mean()
    pdi = plus_dm.rolling(n).mean() / (atr + 1e-12) * 100
    mdi = minus_dm.rolling(n).mean() / (atr + 1e-12) * 100
    dx = (pdi - mdi).abs() / (pdi + mdi + 1e-12) * 100
    return dx.rolling(n).mean()


# 因子 -> 类型(family), 用于按类型分组对比
FACTOR_FAMILY = {
    "mom_5": "动量", "mom_20": "动量", "mom_60": "动量", "mom_120": "动量",
    "mom_250": "动量", "mom_12_1": "动量",
    "rev_5": "反转", "rev_20": "反转", "rev_60": "反转", "rev_intraday": "反转",
    "bullbear_rev": "牛熊反转",
    "vol_20": "波动", "vol_60": "波动", "ret_skew_60": "波动",
    "ivol_60": "特质波动",
    "amihud_20": "流动性", "dolvol_trend": "流动性",
    "ma_dev_20": "技术面", "ma_dev_60": "技术面", "macd_hist": "技术面",
    "rsi_14": "技术面", "boll_w": "技术面", "adx_14": "技术面",
    "high_52w": "微观结构", "overnight_gap": "微观结构", "intraday_range": "微观结构",
    "downside_vol_60": "微观结构", "drawup_60": "微观结构",
    "vol_ratio": "量价", "vol_price_corr": "量价", "amount_strength": "量价",
}
ALL_FACTOR_NAMES = list(FACTOR_FAMILY.keys())

# ---- zoo 因子接入(候选池刷新): 来自 zoo472 OOS 筛查短名单 ----
# 由 oos_framework/screen_results/zoo472/zoo472_ic_screen.csv 筛选:
#   status==ok & p<0.05 & n_eff_nw>=30, 按修正年化 |ICIR_nw| 降序 Top30,
#   强制纳入 2 个牛熊反转候选 gtja191_159 / qlib158_std5(战略核心, 即便 ICIR 略低).
ZOO_SHORTLIST = [
    "alpha101_012", "qlib158_klow", "gtja191_062", "alpha101_044", "alpha101_013",
    "gtja191_099", "alpha101_067", "alpha101_045", "gtja191_113", "alpha101_016",
    "gtja191_083", "alpha101_055", "gtja191_176", "gtja191_032", "alpha101_015",
    "gtja191_090", "qlib158_corr5", "alpha101_023", "gtja191_038", "gtja191_163",
    "alpha101_025", "qlib158_cord5", "qlib158_min5", "qlib158_kup", "gtja191_137",
    "gtja191_016", "alpha101_050", "qlib158_qtld5", "gtja191_036", "gtja191_080",
    "gtja191_159", "qlib158_std5",
]
ZOO_FACTOR_NAMES = list(ZOO_SHORTLIST)

# ---- PIT 修正的基本面因子(候选池刷新 #2: 与价量正交源) ----
# 由 oos_framework/_build_fund_pit.py 离线预计算(按披露滞后 +45/+120d 前向填充, 杜绝 point-in-time 泄露),
# 存于 Vibe 项目内 _fund_cache/fund_factors_daily_pit.pkl. 运行时从同一路径读.
# 通用理论因子(价值/质量/盈利成长), 非"低波红利"定制复合 -> 过 WFA 滚动 IC 防火墙.
FUND_FACTOR_NAMES = [
    "f_bm", "f_ep", "f_dy",                 # 价值: 账面市值比 / 盈利市值比 / 股息率
    "f_roe", "f_roa", "f_gross", "f_netmargin", "f_lev", "f_current", "f_roic", "f_ocf",  # 质量
    "f_rev_yoy", "f_np_yoy", "f_dnp_yoy", "f_roe_yoy", "f_eps_yoy",  # 盈利/成长
    "f_agt", "f_acc",                       # 新增正交维度: 资产增长(CMA)/应计(Sloan)
]
_FUND_PIT_PATH = (Path(__file__).resolve().parents[2]
                  / "oos_framework" / "screen_results" / "_fund_cache" / "fund_factors_daily_pit.pkl")
_FUND_PIT = None


def _load_fund_pit():
    """读 PIT 日线基本面缓存(一次, 模块级缓存复用). 返回 dict[name->date×code DataFrame 或 Series]."""
    global _FUND_PIT
    if _FUND_PIT is None:
        import pickle as _pk
        with open(_FUND_PIT_PATH, "rb") as fh:
            blob = _pk.load(fh)
        _FUND_PIT = blob["data"]
        print(f"    [fund] 载入 PIT 基本面缓存 {_FUND_PIT_PATH.name} "
              f"({len(_FUND_PIT)} 表)", flush=True)
    return _FUND_PIT


def build_fundamental_factors(wide):
    """PIT 修正的基本面因子: 返回与 build_factors 同格式 dict[name->date×code float32].

    值: 从离线 PIT 缓存(已按披露滞后前向填充的日线比率)取, 与 wide 对齐;
    价值类(f_bm/f_ep/f_dy)用价格 close 当分母, 故必须在分块层面用 w_c 计算.
    失败因子用全 NaN 列占位(同 build_zoo_factors 的 NaN 填充保护), 保证 FUND_FACTOR_NAMES 每个 key 都存在.
    """
    close = wide["close"].astype(np.float32)
    dates, codes = close.index, close.columns
    blank = pd.DataFrame(np.nan, index=dates, columns=codes).astype(np.float32)
    out = {name: blank for name in FUND_FACTOR_NAMES}   # 先全占位, 失败也不缺 key
    try:
        data = _load_fund_pit()
    except Exception as e:
        print(f"    [fund-skip] 缓存载入失败: {type(e).__name__}: {str(e)[:80]} (全 NaN 占位)", flush=True)
        return out
    cclose = close.where(close > 0, np.nan)   # 防除 0 -> inf

    def rg(name):
        """取缓存比率表并 reindex 到当前 wide 的 (dates, codes). 缺失表返回 blank."""
        df = data.get(name)
        if df is None:
            return blank
        return df.reindex(index=dates, columns=codes).astype(np.float32)

    def safe(name, frame):
        try:
            out[name] = frame.reindex(index=dates, columns=codes).astype(np.float32)
        except Exception:
            pass  # 保留占位 blank

    try:
        bps = rg("bps"); eps = rg("eps")
        safe("f_bm", bps / cclose)
        safe("f_ep", eps / cclose)
        # 股息率: 年均股息(Series, code->值) broadcast 到日期后 / close
        div_ps = data.get("_avg_div_ps")
        if div_ps is not None:
            div_ps = pd.Series(div_ps).reindex(codes)
            div_mat = pd.DataFrame(np.tile(div_ps.values, (len(dates), 1)),
                                   index=dates, columns=codes).astype(np.float32)
            safe("f_dy", div_mat / cclose)
        # 质量
        safe("f_roe", rg("roe"))
        safe("f_roa", rg("roa"))
        safe("f_gross", rg("gross_margin"))
        safe("f_netmargin", rg("net_margin"))
        safe("f_lev", -rg("debt_to_asset"))          # 低杠杆=质量(取负使越高越好)
        safe("f_current", rg("current_ratio"))
        safe("f_roic", rg("roic"))
        safe("f_ocf", rg("ocf_to_revenue"))
        # 盈利/成长
        safe("f_rev_yoy", rg("revenue_yoy"))
        safe("f_np_yoy", rg("net_profit_yoy"))
        safe("f_dnp_yoy", rg("deduct_np_yoy"))
        safe("f_roe_yoy", rg("roe_yoy"))
        safe("f_eps_yoy", rg("eps_yoy"))
        # 新增正交维度(需 _build_fund_pit 重跑写入 _fund_cache 才有值; 旧缓存缺列->占位 NaN)
        safe("f_agt", rg("总资产增长率(%)") / 100.0)            # 资产增长(CMA 代理)
        cf_to_ta = rg("资产的经营现金流量回报率(%)") / 100.0
        safe("f_acc", rg("roa") - cf_to_ta)                    # 应计 Sloan = (NI-OCF)/TA = roa - OCF/TA
    except Exception as e:
        print(f"    [fund-skip] 计算异常: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return out


def build_zoo_factors(wide, shortlist=None):
    """用注册表 Alpha.compute 算入选 zoo 因子, 返回与 build_factors 同格式的 dict.

    wide: dict[col->DataFrame](date×code), 与 build_factors 同输入(来自 WFA 分块 w_c).
    派生 returns/vwap/adv 与 screen_zoo_472 完全一致; 行业/roe/accruals/bvps 尽力注入(提升覆盖).
    失败因子(SkipAlpha/RegistryError/异常)静默跳过, 不阻断其它因子; 返回 dict[name->DataFrame].
    """
    if shortlist is None:
        shortlist = ZOO_SHORTLIST
    if not shortlist:
        return {}
    try:
        from src.factors.registry import get_default_registry, SkipAlpha, RegistryError
    except ImportError:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))  # 仓库根
        from src.factors.registry import get_default_registry, SkipAlpha, RegistryError
    close = wide["close"].astype(np.float32)
    panel = dict(wide)
    # 派生列(与 screen_zoo_472 一致: returns/vwap/adv 由 OHLCV 精确变换)
    panel["returns"] = close.pct_change()
    panel["vwap"] = wide["amount"] / wide["amount"].replace(0, np.nan)
    panel["adv"] = wide["amount"].rolling(20).mean()
    # 行业/基本面注入(提升覆盖率; 失败则相关因子自动 SkipAlpha 跳过)
    _inject_zoo_extra(panel, close)
    # sector 为字符串矩阵, 不参与 float32 转换
    panel = {k: (v.astype(np.float32) if k != "sector" else v) for k, v in panel.items()}
    dates, codes = close.index, close.columns
    reg = get_default_registry()
    # 失败时(含 RegistryError: 输出>95% NaN)用全 NaN 列占位, 保证短名单每个 key 都存在于返回 dict.
    # 否则 WFA 分块写盘循环 `fac_c[f]` 会因缺失 key 抛 KeyError.
    # 全 NaN 因子在下游 z-score 后仍是 NaN -> 逐折 IC 选择时排末尾、永不被选中, 安全无害.
    blank = pd.DataFrame(np.nan, index=dates, columns=codes).astype(np.float32)
    out = {}
    for aid in shortlist:
        try:
            fac = reg.compute(aid, panel)
            out[aid] = fac.reindex(index=dates, columns=codes).astype(np.float32)
        except (SkipAlpha, RegistryError, Exception) as e:
            print(f"    [zoo-skip] {aid}: {type(e).__name__}: {str(e)[:80]} (NaN填充, 不参与选择)", flush=True)
            out[aid] = blank
    return out


_ZOO_EXTRA = None


def _inject_zoo_extra(panel, close):
    """注入 sector(CSRC) + roe/accruals/bvps(三张表), 读一次缓存复用(避免分块重复读盘)."""
    global _ZOO_EXTRA
    if _ZOO_EXTRA is None:
        d = {}
        try:
            csrc = pd.read_parquet("/workspace/stock_worm/data/csrc_industry_map.parquet")
            d["sector_map"] = dict(zip(csrc["code"], csrc["csrc_industry"]))
        except Exception:
            d["sector_map"] = {}
        try:
            ff = pd.read_pickle("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
            if "ROE" in ff:
                d["roe"] = ff["ROE"]
        except Exception:
            pass
        try:
            ab = pd.read_pickle("/workspace/stock_worm/data/fundamentals/fund_accrual_bvps_daily.parquet")
            for key in ("accruals", "bvps"):
                if key in ab:
                    d[key] = ab[key]
        except Exception:
            pass
        _ZOO_EXTRA = d
    ext = _ZOO_EXTRA
    if ext.get("sector_map"):
        sec_vec = close.columns.to_series().map(ext["sector_map"]).to_numpy()
        panel["sector"] = pd.DataFrame(np.tile(sec_vec, (close.shape[0], 1)),
                                       index=close.index, columns=close.columns)
    if "roe" in ext:
        panel["roe"] = ext["roe"].reindex(index=close.index, columns=close.columns)
    for key in ("accruals", "bvps"):
        if key in ext:
            panel[key] = ext[key].reindex(index=close.index, columns=close.columns)


def neutralize_factors(factors, ind_map, wide=None, size_proxy=False):
    """横截面行业中性化: 每个因子在每个交易日, 减去所属行业均值(去除行业暴露).

    ind_map: dict/映射 code(9位.SH/.SZ) -> 行业标签(str). 行业是**股票级**(单只股票恒定一行业),
             故中性化 = 每个交易日, 对每只股票减去其所在行业的**横截面均值**.
    wide: 仅 size_proxy=True 时需要(取 amount); 否则不必传.
    size_proxy: 额外回归掉 log(amount) 的截面暴露(流动性/市值代理), 默认关.
    返回: 同结构 dict name -> date×code(中性化后) DataFrame.

    内存策略(关键): 不做 stack(会造 9000×1847 长表撑爆 8G cgroup),
    改用 groupby(axis=1).mean() 直接对宽矩阵算行业均值再广播, 峰值 ~1 个因子宽矩阵.
    """
    try:
        import resource as _rs
        def _rss():
            return _rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss / 1024.0  # MB (Unix)
    except ImportError:  # Windows 无 resource 模块
        def _rss():
            return 0.0
    codes = list(factors.values())[0].columns
    ind_arr = pd.Series([ind_map.get(c, np.nan) for c in codes], index=codes)
    valid = ind_arr.notna().values
    valid_codes = np.array(ind_arr.index[valid].tolist())
    ind_valid = ind_arr[valid].values
    out = {}
    n = len(factors)
    for i, (name, fw) in enumerate(list(factors.items())):
        print(f"    neutralize[{i+1}/{n}] {name} (rss={_rss():.0f}MB)", flush=True)
        F = fw[valid_codes]                            # 仅取有行业的列
        # 向量化行业中性: 每个交易日, 每只股票减去其所属行业横截面均值.
        # 旧实现逐行业 Python 循环(uniq~90行业 × 30因子), 5515只时中性化超 600s 会话上限;
        # 改为按行业分组一次算均值再广播(与旧循环数学等价, 速度 ~10-30x, 已单测验证).
        ind_series = pd.Series(ind_valid, index=valid_codes)
        means = F.T.groupby(ind_series).mean().T       # (dates × 行业) 横截面行业均值
        F_neu = F.values - means.loc[:, ind_valid].values   # 每列减去其行业均值(广播对齐)
        res = fw.astype(np.float32)                   # 全宽表(float32 省内存; 无行业列保留原值)
        res[valid_codes] = F_neu.astype(np.float32)
        out[name] = res
        del F, means, F_neu, res, fw
        del factors[name]   # 逐步释放原始因子, 避免原值+中性值同驻撑爆 8G cgroup
    if size_proxy and wide is not None:
        pass   # 行业中性后若需再剔规模, 可在此对 out 逐因子回归 log(amount); 默认不启用
    print(f"    neutralize: done (rss={_rss():.0f}MB)", flush=True)
    return out


def daily_rank_ic(factor_w, fwd_w):
    """逐日横截面 spearman(factor.rank(), fwd.rank()), 向量化."""
    fr = factor_w.rank(axis=1)
    vr = fwd_w.rank(axis=1)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    vc = vr.sub(vr.mean(axis=1), axis=0)
    num = (fc * vc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (vc ** 2).sum(axis=1))
    return num / den.replace(0, np.nan)


def main():
    t0 = time.time()
    wide = load_wide()
    print(f"面板: {wide['close'].shape[1]} 只 × {wide['close'].shape[0]} 日 "
          f"({wide['close'].index[0].date()}~{wide['close'].index[-1].date()})")

    fwd = wide["close"].pct_change(FWD_HORIZON).shift(-FWD_HORIZON)   # 5日前向收益
    factors = build_factors(wide)
    print(f"因子族: momentum/reversal/volatility/liquidity, 共 {len(factors)} 个")

    # 逐因子 逐年 rank-IC
    years = sorted(set(fwd.index.year))
    mat, mat_pos = {}, {}
    for name, fw in factors.items():
        ic_daily = daily_rank_ic(fw, fwd)
        g = ic_daily.groupby(ic_daily.index.year)
        mat[name] = g.mean()
        mat_pos[name] = g.apply(lambda s: (s > 0).mean())
    ic_mat = pd.DataFrame(mat).T.reindex(columns=years)        # factor × year
    pos_mat = pd.DataFrame(mat_pos).T.reindex(columns=years)

    # 缓存
    pd.to_pickle({"ic": ic_mat, "pos": pos_mat, "factors": list(factors)}, CACHE)

    # 全窗口 IC(用于排序)
    full_ic = ic_mat.mean(axis=1)
    print("\n=== 全窗口 rank-IC(按 5d 前向收益) 排序 ===")
    for name in full_ic.sort_values(ascending=False).index:
        print(f"  {name:<16} 全窗口IC={full_ic[name]:+.4f}  "
              f"2023={ic_mat.loc[name,2023]:+.3f} 2024={ic_mat.loc[name,2024]:+.3f} "
              f"2025={ic_mat.loc[name,2025]:+.3f} 2026={ic_mat.loc[name,2026]:+.3f}")

    # 识别 2023-2026 仍活着(IC>0)的因子
    recent = ic_mat[[y for y in (2023,2024,2025,2026) if y in years]]
    alive_recent = full_ic[(recent > 0).all(axis=1)].sort_values(ascending=False)
    print(f"\n=== 2023-2026 四年 IC 全为正的因子({len(alive_recent)} 个, 跨 regime 稳健) ===")
    for name in alive_recent.index:
        print(f"  {name:<16} 全窗口IC={full_ic[name]:+.4f}")

    # 每年最优因子 + 家族(展示 regime 轮动)
    FAM = {}
    for n in factors:
        if n.startswith("mom"): FAM[n] = "momentum"
        elif n.startswith("rev"): FAM[n] = "reversal"
        elif n in ("vol_20", "vol_60", "ret_skew_60", "ivol_60"): FAM[n] = "volatility"
        else: FAM[n] = "liquidity"
    top_per_year = {}
    for y in years:
        row = ic_mat[y].dropna()
        if row.empty: continue
        best = row.idxmax()
        top_per_year[y] = (best, ic_mat.loc[best, y], FAM.get(best, "?"))
    print("\n=== 每年最优因子(家族) ===")
    for y in sorted(top_per_year):
        bf, icv, fam = top_per_year[y]
        print(f"  {y}: {bf}({fam}) IC={icv:+.3f}")

    # 热力图
    fig, ax = plt.subplots(figsize=(14, 7))
    data = ic_mat.T  # year × factor
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index], fontsize=7)
    ax.set_title("因子 × 年份 rank-IC 热力图 (1489只, 2006-2026, 5d前向收益)")
    ax.set_xlabel("因子"); ax.set_ylabel("年份")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rank-IC")
    fig.tight_layout(); fig.savefig(HEAT, dpi=110); plt.close(fig)
    print(f"\n  热力图: {HEAT}")

    # 报告
    md = ["# 因子动物园 × Regime IC 矩阵（Branch 2 · 库优先）", "",
          f"- 数据: stock_worm 日线面板, {wide['close'].shape[1]} 只 × 2006~2026 (20年)",
          f"- 因子: 从 466 库挑异族(动量/反转/波动/流动性), 按标准公式实现, 共 {len(factors)} 个",
          f"- IC: 逐日截面 rank-IC, 目标=5日前向收益(与回测 5d 持有一致)",
          f"- 注: 库里 academic_rmw/smb/cma/hml(质量/价值)需基本面, 日线面板无 -> 用价格可算异族代替", ""]
    md += ["## 1. 全窗口 rank-IC(按 5d 前向收益)", "",
           "| 因子 | 全窗口IC | 2023 | 2024 | 2025 | 2026 |", "|---|---|---|---|---|---|"]
    for name in full_ic.sort_values(ascending=False).index:
        md.append(f"| {name} | {full_ic[name]:+.4f} | {ic_mat.loc[name,2023]:+.3f} | "
                  f"{ic_mat.loc[name,2024]:+.3f} | {ic_mat.loc[name,2025]:+.3f} | {ic_mat.loc[name,2026]:+.3f} |")
    md += ["", "## 2. 跨 regime 稳健因子(2023-2026 四年 IC 全为正)",
           f"- 共 **{len(alive_recent)}** 个: {', '.join(alive_recent.index)}" if len(alive_recent) else "- 无(所有因子在近年都有翻负年份)",
           "", "## 3. Regime 轮动证据(每年最优因子)", "",
           "| 年份 | 最优因子 | 家族 | 该年IC |",
           "|---|---|---|---|"]
    for y in sorted(top_per_year):
        bf, icv, fam = top_per_year[y]
        md.append(f"| {y} | {bf} | {fam} | {icv:+.3f} |")
    md += ["", "## 4. 结论",
           "- 与现有 7 个微观结构因子(会一起死)不同, 异族因子在不同年份有不同表现 —— 正是'状态→因子'框架的料.",
           "- 下一步: 用本矩阵做**状态选择器**——每个 regime(年/市场状态)只启用该状态下 IC 为正的因子, "
           "由 XGBoost(喂 regime 特征)做组合; 届时 XGBoost 才有真信号可学(回应'XGBoost 还弄不').", ""]
    md += [f"\n---\n*生成于因子动物园, 耗时 {time.time()-t0:.1f}s, 数据 stock_worm 本地缓存*"]
    out = OUT_DIR / "因子动物园_regime_IC矩阵.md"
    out.write_text("\n".join(md), encoding="utf-8")
    ic_mat.to_csv(MAT_CSV)
    print(f"报告: {out}  | 矩阵CSV: {MAT_CSV}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
