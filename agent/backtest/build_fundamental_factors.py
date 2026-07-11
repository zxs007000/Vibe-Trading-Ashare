"""基本面因子构造(质量/成长) —— 任务#61
数据源: akshare stock_financial_abstract(新浪, 可达), 直接含
  - 净资产收益率(ROE)        [常用指标]
  - 营业总收入增长率          [成长能力]  (=营收同比 YoY)
  - 归属母公司净利润增长率    [成长能力]  (=净利同比 YoY)
面板: stock_worm 去生存者偏差面板; 规模代理用 amount(成交额), 因无 mkt_cap 序列.
流程: 爬取(可断点续爬, 每码缓存) -> 长表 -> 前向填充到日线日历 -> 行业+规模中性化 -> 存日线 parquet.
"""
import akshare as ak
import pandas as pd
import numpy as np
import time, sys
from pathlib import Path

ROOT = Path("/workspace/stock_worm/data")
SF_PANEL = ROOT / "ashare_daily_panel_survivorfree.parquet"
ALIVE_PANEL = ROOT / "ashare_daily_panel.parquet"
IND_MAP = ROOT / "csrc_industry_map.parquet"
FUND_RAW = ROOT / "fundamentals" / "fund_raw.parquet"
FUND_DAILY = ROOT / "fundamentals" / "fund_factors_daily.parquet"
FUND_RAW.parent.mkdir(parents=True, exist_ok=True)

# 富集子集(已有主力资金+5m微观结构数据)作为首期验证宇宙
_FF = ROOT / "founder_factors.pkl"
if _FF.exists():
    _ff = pd.read_pickle(_FF)
    CODES = list(_ff[list(_ff.keys())[0]].columns)   # 288 只, 形如 000001.SZ
else:
    # 退化: 用全面板代码
    p = pd.read_parquet(SF_PANEL)
    CODES = sorted(p["code"].unique().tolist())

INDICATORS = {
    "ROE":       ("常用指标", "净资产收益率(ROE)"),
    "rev_yoy":   ("成长能力", "营业总收入增长率"),
    "profit_yoy":("成长能力", "归属母公司净利润增长率"),
}


def _fetch_one(code: str) -> dict:
    sym = code.split(".")[0]
    df = ak.stock_financial_abstract(symbol=sym)
    date_cols = [c for c in df.columns if c not in ("选项", "指标")]
    out = {}
    for key, (opt, ind) in INDICATORS.items():
        row = df[(df["选项"] == opt) & (df["指标"] == ind)]
        if len(row) == 0:
            out[key] = None
            continue
        s = row.iloc[0]
        ser = pd.Series({pd.to_datetime(c, format="%Y%m%d"): s[c] for c in date_cols})
        ser = ser[ser.notna()]
        out[key] = ser
    return out


def crawl(save_every: int = 40):
    """爬取并增量缓存为长表 fund_raw.parquet; 已爬的码自动跳过."""
    done = set()
    records = []
    if FUND_RAW.exists():
        cache = pd.read_parquet(FUND_RAW)
        done = set(cache["code"].unique().tolist())
        records = cache.to_dict("records")
        print(f"[crawl] 续爬: 已有 {len(done)} 码, 总记录 {len(records)}")
    n = 0
    for code in CODES:
        if code in done:
            continue
        try:
            res = _fetch_one(code)
            for key, ser in res.items():
                if ser is None:
                    continue
                for dt, v in ser.items():
                    records.append({"code": code, "report_date": dt, key: float(v)})
        except Exception as e:
            print(f"[crawl] ERR {code}: {repr(e)[:120]}")
        n += 1
        if n % save_every == 0:
            pd.DataFrame(records).to_parquet(FUND_RAW)
            print(f"[crawl] 进度 {n} 码, 已存 {len(records)} 记录")
        time.sleep(0.08)
    pd.DataFrame(records).to_parquet(FUND_RAW)
    print(f"[crawl] 完成: {len(CODES)} 目标码, 实际 {len(done)+n} 处理, {len(records)} 记录 -> {FUND_RAW}")
    return pd.DataFrame(records)


def _load_calendar():
    p = pd.read_parquet(SF_PANEL)
    p["_d"] = pd.to_datetime(p["date"]).dt.normalize()
    alive = pd.read_parquet(ALIVE_PANEL)
    cal = pd.to_datetime(alive["date"]).dt.normalize().unique()
    cal = pd.DatetimeIndex(sorted(cal))
    return cal


def _load_amount(cal):
    p = pd.read_parquet(SF_PANEL)
    p["_d"] = pd.to_datetime(p["date"]).dt.normalize()
    alive = pd.read_parquet(ALIVE_PANEL)
    cal_set = pd.to_datetime(alive["date"]).dt.normalize().unique()
    p = p[p["_d"].isin(cal_set)]
    return p.pivot(index="_d", columns="code", values="amount").reindex(cal)


def build_daily():
    """长表 -> 日线(前向填充) -> 行业+规模中性化 -> 存 fund_factors_daily.parquet.

    返回 dict: name -> date×code 中性化 z-score DataFrame.
    """
    raw = pd.read_parquet(FUND_RAW)
    cal = _load_calendar()
    amount = _load_amount(cal)                       # date×code 成交额(规模代理)
    ind_df = pd.read_parquet(IND_MAP).drop_duplicates("code")   # 行业表去重(避免重复码)
    ind_d = dict(zip(ind_df["code"], ind_df["csrc_industry"]))
    factor_names = list(INDICATORS.keys())

    # 1) 每码前向填充到日线
    daily = {}
    for name in factor_names:
        sub = raw[["code", "report_date", name]].dropna(subset=[name])
        sub = sub.sort_values(["code", "report_date"])
        piv = sub.pivot_table(index="report_date", columns="code", values=name, aggfunc="last")
        piv = piv.reindex(cal).ffill()
        daily[name] = piv

    # 2) 规模分位桶(按当日 amount 分 10 桶)
    #    统一代码轴: 取所有因子列的并集, 避免各因子覆盖码数不同(ROE 1847 / rev_yoy·profit_yoy 1846)
    #    导致 arr 宽度与 bucket/ind_arr 错位 -> IndexError.
    all_cols = set()
    for _n in factor_names:
        all_cols |= set(daily[_n].columns)
    codes = sorted(all_cols)
    sz = amount.reindex(index=cal, columns=codes)
    sz_rank = sz.rank(axis=1, pct=True)
    bucket = np.minimum((sz_rank * 10).fillna(-1).astype(int), 9).values  # (T, n_codes)
    ind_arr = np.array([ind_d.get(c, np.nan) for c in codes])  # 长度严格=len(codes), 避免行业表重复码错位
    ind_uniq = pd.unique(ind_arr[~pd.isna(ind_arr)])

    def demean(arr_t, mask):
        vals = arr_t[mask]
        if np.isfinite(vals).sum() < 5:
            return
        arr_t[mask] = vals - np.nanmean(vals)

    neu = {}
    for name in factor_names:
        m = daily[name].reindex(columns=codes).astype(np.float32)
        arr = m.values.copy()  # (T, n_codes) 避免链式赋值
        for i in range(arr.shape[0]):
            row = arr[i].copy()
            b = bucket[i]
            # 先规模桶 demean
            for bb in range(10):
                mask = (b == bb)
                if not mask.any():
                    continue
                demean(row, mask)
            # 再行业 demean
            for iv in ind_uniq:
                mask = (ind_arr == iv)
                if not mask.any():
                    continue
                demean(row, mask)
            arr[i] = row
        m2 = pd.DataFrame(arr, index=m.index, columns=codes)
        # 截面 z-score
        z = (m2.sub(m2.mean(axis=1), axis=0)).div(m2.std(axis=1).replace(0, np.nan), axis=0)
        neu[name] = z.astype(np.float32)

    pd.to_pickle(neu, FUND_DAILY)
    print(f"[build] 中性化因子已存: {FUND_DAILY}")
    print("[build] 覆盖码数: " + ", ".join(
        f"{n}={int(neu[n].notna().any(axis=0).sum())}" for n in factor_names))
    return neu


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("crawl", "all"):
        crawl()
    if mode in ("build", "all"):
        build_daily()
