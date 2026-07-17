"""screen_zoo_472.py — 对注册表 472 个 alpha 做全量 IC 筛查 + 牛熊分域.

背景: 用户要"审查 400 多个因子". 真正的因子库是 src/factors/zoo/ 下 472 个 alpha .py
(WorldQuant Alpha101 / GTJA191 / Qlib158 / academic / fundamental / sentiment).
它们用 compute(panel) 接口, panel = dict[字段 -> date×code DataFrame]. 我们的 stock_worm
面板正是这个格式(且是生存者无偏的 1803 只 × 4975 日). 于是:

  1) 加载 stock_worm 面板(OHLCV), 派生 returns/vwap/adv(均由现有数据精确变换).
  2) 遍历注册表全部 alpha: compute(panel); 缺列/缺行业(SkipAlpha)或计算失败(RegistryError)则跳过.
  3) 对每个成功因子算逐日横截面 rank-IC(对 5 日前瞻收益), 得 ICIR + 正态近似 p.
  4) 按市场 regime(牛/熊/震荡, 由 mkt_level 判定)拆分 IC -> 找"牛熊反转"信号:
     熊市 IC 为正(反转有效)且 牛市 IC<=0(动量主导) = 经典牛熊反转特征.
  5) 落盘: 全量 CSV + Markdown 报告(可行/跳过统计、Top|ICIR|、牛熊反转候选).

注意: 本台只做 IC 层筛查(因子预测力 + regime 依赖), 不跑 WFA 回测. 入选短名单后再喂
triple_validation / 接 WFA 管线做样本外确认(需把 alpha ID 接入因子加载层, 下一步).

用法:
  python oos_framework/screen_zoo_472.py --probe 20      # 探针: 前20个, 看可行性+计时
  python oos_framework/screen_zoo_472.py                 # 全量 472
  python oos_framework/screen_zoo_472.py --limit 100     # 仅前100(快筛)
"""
from __future__ import annotations
import sys, time, argparse, math, warnings, gc, signal
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / "agent"), str(REPO / "agent" / "backtest")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.backtest.oos_validation_corrected import load_wide_sf
from agent.backtest.oos_validation import daily_rank_ic
from src.factors.registry import get_default_registry, SkipAlpha, RegistryError
from oos_framework.regime_wfa import _alive_mkt_level, _market_regime_label

OUT = HERE / "screen_results" / "zoo472"
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "zoo472_ic_screen.csv"
REP = OUT / "zoo472_筛查报告.md"

HOLD = 5
BPY = 252 / HOLD

# 单因子 compute 超时(秒). 部分 sentiment 因子会调外部新闻 API(stock_news/em_get)
# 且无超时 -> 永久挂死. 主线程用 SIGALRM 兜底, 超时即标记 error 跳过.
COMPUTE_TIMEOUT = 120


class _ComputeTimeout(BaseException):
    """继承 BaseException: 穿透 stock_news 等内部的 `except Exception` 兜底层,
    确保单因子 compute 超时时能整体上抛、被 run_screen 捕获并跳过(而非被吞掉后继续挂死)."""
    pass


class compute_timeout:
    """基于 SIGALRM 的单线程 compute 超时(本脚本主线程可用)."""

    def __init__(self, seconds: int = COMPUTE_TIMEOUT):
        self.seconds = seconds
        self._old = None

    def _handler(self, signum, frame):
        raise _ComputeTimeout(f"compute 超时 >{self.seconds}s")

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self._old is not None:
            signal.signal(signal.SIGALRM, self._old)
        return False  # 不吞异常


def _norm_two_sided_p(t):
    """正态近似双尾 p(Safe, 无 scipy)."""
    if not (t == t):
        return np.nan
    return 2.0 * math.erfc(abs(t) / math.sqrt(2.0))


def _rank_ic_vr(fac, vr, dates, batch=300):
    """rank-IC = 逐日因子排名 与 fwd排名 的 spearman(=Pearson of ranks).

    vr(前瞻收益排名)在所有因子间不变, 预计算一次复用 -> 比 daily_rank_ic(每次重排 fwd)
    快约 2x. fac 排名一次后按日期分批与 vr 求相关.
    """
    n = len(dates)
    ic = np.full(n, np.nan)
    fr = fac.rank(axis=1)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        fz = fr.iloc[s:e]; vz = vr.iloc[s:e]
        fc = fz.sub(fz.mean(axis=1), axis=0)
        vc = vz.sub(vz.mean(axis=1), axis=0)
        num = (fc * vc).sum(axis=1)
        den = np.sqrt((fc ** 2).sum(axis=1) * (vc ** 2).sum(axis=1))
        ic[s:e] = (num / den.replace(0, np.nan)).values
    return pd.Series(ic, index=dates)


def _flush_csv(rows, csv_path):
    """增量合并写 CSV(去重, 保留最新). 供续跑/防 OOM 丢失进度."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if csv_path.exists():
        try:
            old = pd.read_csv(csv_path)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates("alpha_id", keep="last")
        except Exception:
            pass
    new.to_csv(csv_path, index=False)


def run_screen(ids, reg, panel, fwd, vr, dates, regime, csv_path, probe=False, flush_every=5):
    """逐 alpha 计算 + IC 筛查. 每 flush_every 个增量写 CSV(防 OOM 丢进度). 返回 list[dict]."""
    n = len(ids)
    rows = []
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        rec = {"alpha_id": aid}
        try:
            meta = reg.get(aid).meta
            rec["zoo"] = meta.get("zoo", "")
            rec["theme"] = ",".join(meta.get("theme", []))
            rec["cols"] = ",".join(meta.get("columns_required", []))
            rec["needs_sector"] = bool(meta.get("requires_sector", False))
        except Exception:
            rec["zoo"] = rec["theme"] = rec["cols"] = ""
            rec["needs_sector"] = False
        try:
            with compute_timeout(COMPUTE_TIMEOUT):
                fac = reg.compute(aid, panel)
        except _ComputeTimeout as e:
            rec.update(status="error", reason=f"compute超时(>{COMPUTE_TIMEOUT}s,疑似外部API挂死)",
                       icir=np.nan, p=np.nan)
            rows.append(rec); _maybe_flush(rows, csv_path, flush_every, i, n, t0); continue
        except SkipAlpha as e:
            rec.update(status="skip", reason=str(e)[:120], icir=np.nan, p=np.nan)
            rows.append(rec); _maybe_flush(rows, csv_path, flush_every, i, n, t0); continue
        except RegistryError as e:
            rec.update(status="error", reason=str(e)[:120], icir=np.nan, p=np.nan)
            rows.append(rec); _maybe_flush(rows, csv_path, flush_every, i, n, t0); continue
        except Exception as e:
            rec.update(status="error", reason=f"{type(e).__name__}: {str(e)[:100]}",
                       icir=np.nan, p=np.nan)
            rows.append(rec); _maybe_flush(rows, csv_path, flush_every, i, n, t0); continue
        try:
            ic = _rank_ic_vr(fac, vr, dates).reindex(dates)
            icv = ic.dropna().values
            if len(icv) < 60:
                rec.update(status="no_ic", reason="IC 样本不足", icir=np.nan, p=np.nan)
                rows.append(rec); del fac; gc.collect()
                _maybe_flush(rows, csv_path, flush_every, i, n, t0); continue
            ic_mean = float(icv.mean()); ic_std = float(icv.std())
            icir = ic_mean / (ic_std + 1e-12) * np.sqrt(252)
            tstat = ic_mean / (ic_std / np.sqrt(len(icv)) + 1e-12)
            p = _norm_two_sided_p(tstat)
            # regime 拆分
            ic_s = ic.reindex(regime.index)
            ic_bull = float(ic_s[regime == "bull"].dropna().mean())
            ic_bear = float(ic_s[regime == "bear"].dropna().mean())
            ic_osc = float(ic_s[regime == "osc"].dropna().mean())
            # 牛熊反转特征: 熊市反转有效(IC>0) 且 牛市不反转(IC<=0)
            reversal_in_bear = (ic_bear > 0) and (ic_bull <= 0)
            regime_flip = (ic_bull > 0) != (ic_bear > 0)  # 符号翻转
            rec.update(status="ok", ic_mean=round(ic_mean, 5), ic_std=round(ic_std, 5),
                       icir=round(icir, 3), ic_tstat=round(tstat, 2), p=round(p, 4),
                       ic_bull=round(ic_bull, 5), ic_bear=round(ic_bear, 5),
                       ic_osc=round(ic_osc, 5),
                       reversal_in_bear=int(reversal_in_bear),
                       regime_flip=int(regime_flip),
                       regime_gap=round(ic_bear - ic_bull, 5))
            rows.append(rec)
        except Exception as e:
            rec.update(status="error", reason=f"metric: {type(e).__name__}: {str(e)[:100]}",
                       icir=np.nan, p=np.nan)
            rows.append(rec)
        del fac; gc.collect()
        _maybe_flush(rows, csv_path, flush_every, i, n, t0)
        if probe and i >= 25:
            break
    _flush_csv(rows, csv_path)   # 末尾兜底刷新
    return rows


def _maybe_flush(rows, csv_path, flush_every, i, n, t0):
    if len(rows) >= flush_every:
        _flush_csv(rows, csv_path)
        rows.clear()
        print(f"  进度 {i}/{n}  耗时 {time.time()-t0:.0f}s  已落盘", flush=True)


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0, help="探针: 只跑前 N 个并报告可行性+计时")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个(快筛)")
    ap.add_argument("--stocks", type=int, default=2500,
                    help="截面股票数(按数据完整度取最密的前 N 只, 省内存+提速; 0=全量5515)")
    args = ap.parse_args()

    print("=" * 64)
    print("注册表 472 alpha 全量 IC 筛查 + 牛熊分域")
    print("=" * 64)

    print("\n[1/4] 加载 stock_worm 面板(OHLCV, 生存者无偏)...")
    panel = load_wide_sf()
    # 派生列(均由现有数据精确变换, 提高 alpha 覆盖率)
    close = panel["close"]
    panel["returns"] = close.pct_change()
    panel["vwap"] = panel["amount"] / panel["volume"].replace(0, np.nan)
    panel["adv"] = panel["amount"].rolling(20).mean()
    # 转 float32 省内存(8G cgroup 上限): 面板 9 字段 ~2GB -> ~1GB, 给 alpha 计算留余量
    panel = {k: v.astype(np.float32) for k, v in panel.items()}
    dates = close.index
    # 缩减截面: 按数据完整度取最密的前 N 只(省内存+提速; rank-IC 排序对样本量稳健)
    if args.stocks and args.stocks > 0 and args.stocks < close.shape[1]:
        nn = close.notna().sum(axis=0).sort_values(ascending=False)
        sub = nn.head(args.stocks).index
        panel = {k: v.reindex(columns=sub) for k, v in panel.items()}
        close = panel["close"]
        print(f"  截面缩减: {close.shape[1]}只(最密前{args.stocks}) × {dates[0].date()}~{dates[-1].date()}")
    print(f"  面板字段 {list(panel.keys())}; 全量 {close.shape[1]}只")

    # ---- 注入行业(sector) + 基本面(roe), 复活此前被 SkipAlpha 跳过的基本面/行业因子 ----
    # 数据来源: stock_worm csrc_industry_map(CSRC 71类, 1847码全覆盖) + fund_factors_daily['ROE'].
    # 必须在 float32 转换之后注入: sector 为字符串矩阵, 不能 astype(float32).
    try:
        csrc = pd.read_parquet("/workspace/stock_worm/data/csrc_industry_map.parquet")
        code2ind = dict(zip(csrc["code"], csrc["csrc_industry"]))
        sec_vec = close.columns.to_series().map(code2ind).to_numpy()
        sec_df = pd.DataFrame(
            np.tile(sec_vec, (close.shape[0], 1)),
            index=close.index, columns=close.columns)
        panel["sector"] = sec_df   # 字符串矩阵, 不参与 float32 转换
        print(f"  注入 sector(CSRC): 覆盖 {int(pd.Series(sec_vec).notna().sum())}/{close.shape[1]} 只, "
              f"{len(pd.unique(sec_vec[pd.notna(sec_vec)]))} 类")
    except Exception as e:
        print(f"  [warn] sector 注入失败: {repr(e)[:120]}")
    try:
        ff = pd.read_pickle("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
        roe = ff["ROE"].reindex(index=close.index, columns=close.columns).astype(np.float32)
        panel["roe"] = roe
        print(f"  注入 roe: 覆盖 {int(roe.notna().any(axis=0).sum())}/{close.shape[1]} 只")
    except Exception as e:
        print(f"  [warn] roe 注入失败: {repr(e)[:120]}")
    # 注入 accruals / bvps(三张表真值, 复活 fundamental_quality_accrual / fundamental_value_pb_inv 两个 skip 因子)
    try:
        ab = pd.read_pickle("/workspace/stock_worm/data/fundamentals/fund_accrual_bvps_daily.parquet")
        for key in ("accruals", "bvps"):
            if key in ab:
                ser = ab[key].reindex(index=close.index, columns=close.columns).astype(np.float32)
                panel[key] = ser
                print(f"  注入 {key}: 覆盖 {int(ser.notna().any(axis=0).sum())}/{close.shape[1]} 只")
            else:
                print(f"  [warn] {key} 不在 parquet(构建未完成?)")
    except Exception as e:
        print(f"  [warn] accruals/bvps 注入失败: {repr(e)[:120]}")

    fwd = close.pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    # fwd 排名预计算一次(所有因子共用), 加速 IC
    vr = fwd.rank(axis=1)

    # 市场 regime 标签(复用管线 _alive_mkt_level + _market_regime_label, 与防御门控/ML 分析一致)
    mkt_level = _alive_mkt_level(dates)
    regime = _market_regime_label(mkt_level, dates)
    print(f"  regime: bull={int((regime=='bull').sum())} bear={int((regime=='bear').sum())} "
          f"osc={int((regime=='osc').sum())}")

    print("\n[2/4] 加载注册表(扫描 472 .py)...")
    reg = get_default_registry()
    all_ids = reg.list()
    print(f"  注册表因子数: {len(all_ids)}")
    ids = all_ids[: args.limit] if args.limit else all_ids
    if args.probe:
        ids = ids[: args.probe]
    # 断点续跑: 跳过 CSV 中已完成的 alpha
    done_ids = set()
    if CSV.exists():
        try:
            done_ids = set(pd.read_csv(CSV)["alpha_id"].tolist())
        except Exception:
            done_ids = set()
    if done_ids:
        ids = [i for i in ids if i not in done_ids]
        print(f"  续跑: 已完成 {len(done_ids)} 个, 本次待算 {len(ids)} 个")
    else:
        print(f"  本次筛查: {len(ids)} 个")

    print("\n[3/4] 逐 alpha 计算 + IC 筛查...")
    rows = run_screen(ids, reg, panel, fwd, vr, dates, regime, CSV, probe=bool(args.probe))

    # 报告用全量 CSV(含续跑已完成), 避免漏算
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    skip = df[df["status"] == "skip"]
    err = df[df["status"].isin(["error", "no_ic"])]
    print(f"\n  成功 {len(ok)} / 跳过 {len(skip)} / 失败 {len(err)} (累计)")
    if not skip.empty:
        reasons = Counter()
        for r in skip["reason"]:
            if "sector" in r: reasons["缺行业sector"] += 1
            elif "columns" in r or "required" in r: reasons["缺列(OHLCV外)"] += 1
            elif "extras" in r: reasons["缺extras"] += 1
            else: reasons[r[:40]] += 1
        print("  跳过原因:", dict(reasons))

    print(f"  CSV: {CSV}")

    # 报告
    print("\n[4/4] 生成报告...")
    md = build_report(df, ok, skip, err, len(all_ids), len(ids), dates, t0, args.probe)
    REP.write_text(md, encoding="utf-8")
    print(f"  报告: {REP}  (总耗时 {time.time()-t0:.1f}s)")
    print("=" * 64)


def build_report(df, ok, skip, err, n_total, n_run, dates, t0, probe):
    md = ["# 注册表 472 Alpha 全量 IC 筛查报告（牛熊分域）", "",
          f"- 数据: stock_worm 生存者无偏面板 × {dates[0].date()}~{dates[-1].date()}",
          f"- 注册表总因子: **{n_total}**; 本次筛查: **{n_run}**" + ("(探针)" if probe else ""),
          f"- 方法: 逐 alpha compute(panel) → 逐日横截面 rank-IC(对 {HOLD}d 前瞻收益) → "
          f"ICIR + 正态近似 p; 按牛/熊/震荡 regime 拆 IC",
          f"- 牛熊反转定义: 熊市 IC>0(反转有效) **且** 牛市 IC<=0(动量主导) = regime_flip 或 reversal_in_bear",
          f"- 派生字段: returns/vwap(=amount/volume)/adv(=amount.rolling20) 由 OHLCV 精确变换",
          ""]

    md += ["## 1. 可行性概览", "",
           f"- 成功算出 IC: **{len(ok)}** / 本次 {n_run}",
           f"- 跳过(SkipAlpha, 缺列/行业): **{len(skip)}**",
           f"- 失败/无IC(RegistryError 或 样本不足): **{len(err)}**", ""]

    if not ok.empty:
        ok = ok.copy()
        ok["abs_icir"] = ok["icir"].abs()
        # Top |ICIR|
        top = ok.sort_values("abs_icir", ascending=False).head(30)
        md += ["## 2. Top 30 因子(按 |ICIR| 降序)", "",
               "| rank | alpha_id | zoo | ICIR | p | IC_bull | IC_bear | IC_osc | "
               "regime_flip | reversal_in_bear |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for rk, (_, r) in enumerate(top.iterrows(), 1):
            md.append(f"| {rk} | {r['alpha_id']} | {r['zoo']} | {r['icir']:+.3f} | {r['p']:.3f} | "
                      f"{r['ic_bull']:+.4f} | {r['ic_bear']:+.4f} | {r['ic_osc']:+.4f} | "
                      f"{int(r['regime_flip'])} | {int(r['reversal_in_bear'])} |")
        md += [""]

        # 牛熊反转候选
        rev = ok[(ok["reversal_in_bear"] == 1) & (ok["p"] < 0.05) & (ok["abs_icir"] > 0.3)]
        rev = rev.sort_values("regime_gap", ascending=False)
        md += [f"## 3. 牛熊反转候选(熊市反转有效 + 牛市不反转 + p<0.05 + |ICIR|>0.3): **{len(rev)}** 个", "",
               "| alpha_id | zoo | ICIR | p | IC_bull | IC_bear | IC_osc | regime_gap |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in rev.head(40).iterrows():
            md.append(f"| {r['alpha_id']} | {r['zoo']} | {r['icir']:+.3f} | {r['p']:.3f} | "
                      f"{r['ic_bull']:+.4f} | {r['ic_bear']:+.4f} | {r['ic_osc']:+.4f} | "
                      f"{r['regime_gap']:+.4f} |")
        md += [""]

        # regime 翻转(任意符号翻转)
        flip = ok[(ok["regime_flip"] == 1) & (ok["p"] < 0.05)]
        flip = flip.sort_values("abs_icir", ascending=False)
        md += [f"## 4. Regime 翻转因子(牛/熊 IC 符号相反, p<0.05): **{len(flip)}** 个", "",
               "| alpha_id | zoo | ICIR | p | IC_bull | IC_bear |",
               "|---|---|---|---|---|---|"]
        for _, r in flip.head(30).iterrows():
            md.append(f"| {r['alpha_id']} | {r['zoo']} | {r['icir']:+.3f} | {r['p']:.3f} | "
                      f"{r['ic_bull']:+.4f} | {r['ic_bear']:+.4f} |")
        md += [""]

        # 显著因子总数
        sig = ok[ok["p"] < 0.05]
        md += [f"## 5. 显著性汇总", "",
               f"- 显著因子(p<0.05): **{len(sig)} / {len(ok)}**",
               f"- 显著且 |ICIR|>0.5: **{len(ok[(ok['p']<0.05)&(ok['abs_icir']>0.5)])}**",
               f"- 显著且 |ICIR|>1.0: **{len(ok[(ok['p']<0.05)&(ok['abs_icir']>1.0)])}**", ""]
    else:
        md += ["## 2. 无成功因子(全部跳过/失败, 检查面板字段匹配)", ""]

    md += ["## 6. 下一步", "",
           "- 本台只做 IC 层筛查. 入选短名单(alpha_id)需接入 WFA 因子加载层(把 compute 接入 "
           "`load_engine_inputs_cached` 的 zarr 管线)才能跑样本外回测确认.",
           "- 牛熊反转候选可直接用于增强防御门控层(危机期加权)或 ② 牛熊混合反转因子.",
           "- 跳过因子中【缺行业sector】的需行业/市值面板(当前无)才能算; 其余 OHLCV 类已全部覆盖.", ""]
    md += [f"\n---\n*筛查生成, 耗时 {time.time()-t0:.1f}s*"]
    return "\n".join(md)


if __name__ == "__main__":
    main()
