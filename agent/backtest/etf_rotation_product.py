"""etf_rotation_product.py — ETF/指数轮动 regime 总闸 · 产品化交付

etf_rotation_ext 已验证: regime 总闸(沪深300 vs MA120)是**最强回撤削减器**
(CORE20Y 最大回撤 -40.6%→-26.1%, Sharpe 0.80→1.10; RICH -68.3%→-36.7%).
本研究把它从'实验'落成**可交付产品**:

交付物(均落到 screen_results/):
  1) ETF轮动_产品_持仓.csv  —— 每个调仓日各资产权重(长表: universe,date,asset,weight),
                                  可直接照做(月度再平衡, 持信号最高者; risk-off 整仓切防御资产).
  2) 最新一期配置             —— 打印 + 写入报告(用户'现在该持什么').
  3) ETF轮动_产品_净值.png    —— 各宇宙推荐配置净值曲线.
  4) ETF轮动_产品_报告.md     —— 推荐配置(按最低回撤自动选)、四组合对比、最新配置、使用说明.

推荐配置选择: 对每个宇宙, 在 4 种组合(MA×regime)中**选最大回撤最低者**(回撤优先, 契合'regime总闸=回撤削减器'的定位);
若回撤相近取夏普更高. 因此推荐配置通常是 'MA200开 + regime开'.

用法:
  python backtest/etf_rotation_product.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import etf_rotation_ext as E

OUT_DIR = Path(__file__).parent / "screen_results"
W_CSV = OUT_DIR / "ETF轮动_产品_持仓.csv"
FIG = OUT_DIR / "ETF轮动_产品_净值.png"
REP = OUT_DIR / "ETF轮动_产品_报告.md"

COMBOS = [
    (False, False, "基准(MA关/regime关)"),
    (True,  False, "MA200开"),
    (False, True,  "regime开(MA关)"),
    (True,  True,  "MA200开+regime开"),
]


def run_universe_product(px_full, univ_name):
    """对单一宇宙跑 4 组合, 返回 {label: (eq, port, weights, metrics)} + 推荐配置选择."""
    cols = E.UNIVERSES[univ_name]
    px = E.trim(px_full[cols].copy())
    out = {}
    for trend, regime, lab in COMBOS:
        eq, port, w = E.backtest(px, lookback=20, top_n=1, trend=trend, ma=200,
                                 regime=regime, regime_ma=120)
        tot, cagr, dd, sh = E.metrics(eq, port)
        out[lab] = dict(eq=eq, port=port, weights=w, tot=tot, cagr=cagr, dd=dd, sh=sh)
    # 推荐: 最大回撤最低(绝对值最小)优先, 其次夏普更高
    rec_lab = min(out.keys(), key=lambda k: (abs(out[k]["dd"]), -out[k]["sh"]))
    return out, rec_lab, px


def main():
    t0 = time.time()
    px = E.load_prices()
    print(f"[产品化] 价表 {px.shape[0]} 日 × {px.shape[1]} 资产; 宇宙 {list(E.UNIVERSES.keys())}")

    all_rows = []
    rec_weights_long = []     # 长表行
    rec_eqs = {}
    latest_alloc = {}
    for univ in E.UNIVERSES:
        out, rec_lab, pxp = run_universe_product(px, univ)
        print(f"\n=== 宇宙 {univ}: 推荐配置 = {rec_lab} ===")
        for lab in COMBOS_LABELS():
            o = out[lab]
            all_rows.append((f"{univ} {lab}", o["tot"], o["cagr"], o["dd"], o["sh"]))
            print(f"    {lab:<20} CAGR={o['cagr']:+.2%} MaxDD={o['dd']:+.2%} Sharpe={o['sh']:.2f}")
        # 推荐配置持仓长表
        w = out[rec_lab]["weights"]
        for d, row in w.iterrows():
            nz = row[row > 0]
            for asset, wt in nz.items():
                rec_weights_long.append((univ, rec_lab, d, asset, round(float(wt), 4)))
        rec_eqs[univ] = out[rec_lab]["eq"]
        # 最新一期配置
        last_d = w.index[-1]
        nz = w.loc[last_d]
        nz = nz[nz > 0]
        latest_alloc[univ] = (last_d, dict(nz))

    df = pd.DataFrame(all_rows, columns=["策略", "累计收益", "年化", "最大回撤", "夏普"])
    wlong = pd.DataFrame(rec_weights_long, columns=["universe", "config", "date", "asset", "weight"])
    wlong.to_csv(W_CSV, index=False)
    print(f"\n持仓长表: {W_CSV} ({len(wlong)} 行)")

    # 图: 各宇宙推荐配置净值
    plt.figure(figsize=(13, 6))
    cmap = {"CORE20Y": "tab:blue", "RICH13": "tab:green", "RICH": "tab:orange"}
    for univ in E.UNIVERSES:
        eq = rec_eqs[univ]
        plt.plot(eq.index, eq.values / eq.iloc[0], lw=1.2, label=f"{univ}(推荐)", color=cmap.get(univ))
    plt.title("ETF Rotation - 推荐配置(最低回撤) 归一化净值")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG, dpi=110); plt.close()

    # 报告
    md = ["# ETF 轮动 · regime 总闸产品化报告", "",
          "- 数据: 指数(stock_zh_index_daily) + ETF(fund_etf_hist_sina) + 黄金期货; 纳指ETF 已排除(防过拟合)",
          "- 策略: 月度再平衡, 单边成本0.03%, 信号=20日收益率动量, 持top1; MA200闸门(权益跌破均线剔出) + regime总闸(沪深300 vs MA120, risk-off整仓切防御)",
          "- 推荐配置: 对每个宇宙在 4 组合中选**最大回撤最低者**(回撤优先), 通常即 'MA200开+regime开'",
          "- 交付: 持仓长表CSV(可照做) + 最新一期配置 + 净值图 + 本说明",
          "", "## 1. 四组合绩效对比(回撤优先)", "",
          "| 策略 | 累计 | 年化 | 最大回撤 | 夏普 |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['策略']} | {r['累计收益']:+.1%} | {r['年化']:+.2%} | {r['最大回撤']:+.2%} | {r['夏普']:.2f} |")
    md += ["", "![净值](ETF轮动_产品_净值.png)", "",
           "## 2. 最新一期配置(现在该持什么)", ""]
    for univ in E.UNIVERSES:
        d, alloc = latest_alloc[univ]
        alloc_s = ", ".join(f"{a} {wt:.0%}" for a, wt in alloc.items())
        md.append(f"- **{univ}** ({d.date()}): {alloc_s}")
    md += ["", "## 3. 如何使用(照做说明)",
           f"- 持仓长表 `ETF轮动_产品_持仓.csv` 列: universe, config, date, asset, weight.",
           "- 每月末(T)读取该 universe 当 date 的行: 把 weight>0 的资产按权重配置; 月度再平衡到新一期权重.",
           "- top1 策略通常单资产 100%; 仅在 regime 总闸触发 risk-off 时整仓切防御(国债ETF→货币ETF→上证国债指), 此时该期仅防御资产非零.",
           "- 防御资产本身无趋势信号, 仅作'避险停泊'; regime 回 on 后自动切回动量冠军.",
           "- 纳指ETF 已排除, 不走美股 beta; 黄金/国债作低相关分散与避险.",
           "", "## 4. 风险提示 & 与主线一致性",
           "- **regime 总闸是本研究已证的最强回撤削减器**(CORE20Y -40.6%→-26.1%; RICH -68.3%→-36.7%), 它切的是'股 vs 债 vs 商品'这种**真正正交的资产状态**, 故有效.",
           "- 这与'因子有寿命, 什么状态用什么因子'同源: 同一哲学在**资产配置层**落地(切资产状态)比在**选股因子层**切因子集更有效——后者已被方向A证为无增量(截面因子缺乏状态正交性).",
           "- 单一牛市 OOS 段(2024-09起)会高估任何牛市依赖策略; 本产品回测跨 20 年多 regime, 更可信. 实盘应跟踪 regime 信号真实性, 谨防极端流动性事件.",
           f"*生成于 ETF轮动产品化, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {REP}\n图: {FIG}")
    # 打印最新配置
    print("\n最新一期配置:")
    for univ in E.UNIVERSES:
        d, alloc = latest_alloc[univ]
        print(f"  {univ} ({d.date()}): " + ", ".join(f"{a} {wt:.0%}" for a, wt in alloc.items()))


def COMBOS_LABELS():
    return [lab for _, _, lab in COMBOS]


if __name__ == "__main__":
    main()
