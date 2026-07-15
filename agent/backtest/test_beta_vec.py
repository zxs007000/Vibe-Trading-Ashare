"""test_beta_vec.py — 验证向量化 beta 与逐股票 np.cov 循环严格等价.

旧循环语义: 每列 c 用各自 paired 子集 v = rc.notna() & mkt.notna(),
            beta = np.cov(x,y)[0,1] / np.var(x)  (np.cov ddof=1, np.var ddof=0).
向量化必须复刻"逐列子集均值/协方差"语义.
"""
import numpy as np, pandas as pd, time


def beta_loop(ret, mkt, min_obs=250):
    beta = {}
    for c in ret.columns:
        rc = ret[c]; v = rc.notna() & mkt.notna()
        if v.sum() < min_obs:
            beta[c] = np.nan; continue
        x = mkt[v].values; y = rc[v].values
        beta[c] = np.cov(x, y)[0, 1] / np.var(x)
    return pd.Series(beta, index=ret.columns)


def beta_vec(ret, mkt, min_obs=250):
    R = ret.values.astype(np.float64)
    Mc = mkt.values.astype(np.float64)[:, None]
    V = (ret.notna() & mkt.notna().to_numpy()[:, None]).values.astype(np.float64)  # 0/1
    # 关键: IEEE 中 0.0*nan=nan, 必须把无效行值置0再乘(否则污染求和)
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


# ---- 测试1: 合成随机 NaN 模式(复刻真实缺失) ----
rng = np.random.default_rng(0)
dates = pd.date_range("2015-01-01", periods=1200, freq="B")
codes = [f"{i:06d}.SZ" for i in range(400)]
close = pd.DataFrame(10 * (1 + rng.normal(0, 0.02, (len(dates), len(codes)))).cumprod(axis=0),
                     index=dates, columns=codes)
ret = close.pct_change()
mkt = ret.mean(axis=1)
# 注入随机 NaN, 模拟不同上市/停牌
mask = rng.random(ret.shape) < 0.03
ret = ret.mask(mask)
mkt = ret.mean(axis=1)

t0 = time.time(); b1 = beta_loop(ret, mkt); t_loop = time.time() - t0
t0 = time.time(); b2 = beta_vec(ret, mkt);  t_vec = time.time() - t0

both = b1.notna() & b2.notna()
diff = (b1[both] - b2[both]).abs()
print(f"[合成] loop {t_loop:.3f}s  vec {t_vec:.3f}s")
print(f"[合成] 有效列={both.sum()}/{len(b1)}  max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  nan一致={bool((b1.isna()==b2.isna()).all())}")

# ---- 测试2: 真实切片(取全市场面板前 50 只, 近 1500 日) ----
PANEL = "/workspace/stock_worm/data/ashare_daily_panel.parquet"
p = pd.read_parquet(PANEL)
p = p.sort_values(["code", "date"])
close_real = p.pivot(index="date", columns="code", values="close").iloc[-1500:]
codes_r = list(close_real.columns[:50])
close_real = close_real[codes_r]
ret2 = close_real.pct_change()
mkt2 = ret2.mean(axis=1)
b1r = beta_loop(ret2, mkt2)
b2r = beta_vec(ret2, mkt2)
bothr = b1r.notna() & b2r.notna()
diffr = (b1r[bothr] - b2r[bothr]).abs()
print(f"[真实] 有效列={bothr.sum()}/{len(b1r)}  max|diff|={diffr.max():.3e}  mean|diff|={diffr.mean():.3e}  nan一致={bool((b1r.isna()==b2r.isna()).all())}")
print("OLD样例:", b1r.head(5).round(4).to_dict())
print("NEW样例:", b2r.head(5).round(4).to_dict())
