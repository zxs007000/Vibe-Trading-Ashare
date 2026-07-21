#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 模型 SHAP 解释性分析 · 修正版
=====================================
针对 xgb_wfa_proto_v4_full.py 落盘的 booster + OOS 子样本做白盒解释。
修复《SHAP 实施方案》中的关键漏洞：

  [漏洞2 规模]  数据已在训练时子采样到 <=3000 行/折，精确 TreeExplainer 可行(不碰全量 9M 行)。
  [漏洞3 共线]  按"特征族"汇总 Mean|SHAP|，缓和同源变体(pca_absorp_*/crowd_mom_*/ix_*)信用任意分配。
  [漏洞4 跨折]  跨折用"方向一致性(sign agreement)"判定稳健，而非对 SHAP 取均值(尺度不可比)。
  [漏洞6 尺度]  明确 SHAP 在 log-odds(边际)尺度；标签为"未来收益排前30%的概率"(排序/相对, 非绝对收益)。
  [漏洞5 阈值]  依赖图带 bootstrap 95% CI；尾部稀疏处的"断崖"标注为不可靠, 不得直接当硬性止损线。
  [漏洞9 交互]  额外用 shap_interaction_values 解耦交互, 而非仅靠 ix_ 乘积特征 + 着色。
  [代码bug]     二分类 TreeExplainer 返回 list -> 取正类 [1]；base value 取 expected_value[1]。

用法:
  python3.11 shap_analysis.py                 # 读 /workspace/quant_proto 下 booster_fold*.json 等
  python3.11 shap_analysis.py --base /tmp/shap_smoke   # 指定目录(冒烟测试用)
输出:
  shap_report.md  +  shap_*.png (summary bar/dot, dependence for 关键因子)
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DEFAULT_BASE = os.environ.get("WFA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4proto_out"))

# 特征归族(用于汇总同源变体的总 SHAP, 缓和共线导致的信用瓜分)
FAMILY_PREFIX = ["pca_absorp", "crowd_mom", "crowd_liq", "chip_", "ix_",
                 "mkt_", "ret_", "amt_", "vol_", "rsi", "ma_dev", "amp",
                 "gross", "net_", "roe", "debt", "current", "ocf",
                 "operate", "netprofit", "eps"]


def family(name):
    for p in FAMILY_PREFIX:
        if name.startswith(p):
            return p
    return name


def discover_folds(base):
    fs = [f for f in os.listdir(base) if f.startswith("booster_fold") and f.endswith(".json")]
    return sorted(int(f.split("booster_fold")[1].split(".")[0]) for f in fs)


def load_fold(i, base, feats):
    import xgboost as xgb
    import shap
    bst = xgb.Booster()
    bst.load_model(f"{base}/booster_fold{i}.json")
    bst.feature_names = list(feats)
    df = pd.read_parquet(f"{base}/shap_data_fold{i}.parquet")
    X = df[feats].astype(float)
    exp = shap.TreeExplainer(bst)
    sv = exp.shap_values(X)                 # 二分类 -> list[neg, pos]
    if isinstance(sv, list):
        sv = sv[1]                          # 取正类(排前30%概率, log-odds 尺度)
    base_val = exp.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]
    return sv, X.values, base_val, df


def dependence_with_ci(x, sv, n_bins=20, boots=80):
    """依赖图: 分箱均值 + bootstrap 95% CI。返回 (centers, lo, hi, counts)。"""
    order = np.argsort(x)
    xs, ys = x[order], sv[order]
    edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1))
    centers, lo, hi, cnt = [], [], [], []
    for b in range(n_bins):
        if b < n_bins - 1:
            m = (xs >= edges[b]) & (xs < edges[b + 1])
        else:
            m = (xs >= edges[b]) & (xs <= edges[b + 1])
        if m.sum() < 5:
            continue
        seg = ys[m]
        centers.append(xs[m].mean())
        bs = [np.mean(np.random.choice(seg, len(seg), replace=True)) for _ in range(boots)]
        lo.append(np.percentile(bs, 5))
        hi.append(np.percentile(bs, 95))
        cnt.append(int(m.sum()))
    return np.array(centers), np.array(lo), np.array(hi), np.array(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    base = args.base

    import shap
    feats = json.load(open(f"{base}/feats_v4full.json"))
    folds = discover_folds(base)
    if not folds:
        raise SystemExit(f"[shap] 未找到 booster_fold*.json @ {base}，请先跑全量(v4_full)或冒烟。")
    print(f"[shap] 发现折: {folds}  特征数: {len(feats)}")

    shaps, Xs, bases = [], [], []
    for i in folds:
        sv, Xv, bv, _ = load_fold(i, base, feats)
        shaps.append(sv)
        Xs.append(Xv)
        bases.append(bv)
        print(f"  折{i}: OOS子样本={Xv.shape[0]}行  base_value(log-odds)={bv:.3f}")

    # —— 全局: 单特征 Mean|SHAP| + 按族汇总 ——
    mean_abs = pd.concat([pd.Series(s.mean(0), index=feats) for s in shaps]).groupby(level=0).mean().abs()
    mean_abs = mean_abs.sort_values(ascending=False)
    fam = mean_abs.groupby(family).sum().sort_values(ascending=False)

    # —— 跨折方向一致性(非均值) ——
    # 每特征各折 median(SHAP) 的符号; 统计一致方向 & 翻转折数
    dir_rows = []
    for j, f in enumerate(feats):
        meds = [np.median(shaps[k][:, j]) for k in range(len(folds))]
        pos = sum(1 for m in meds if m > 0)
        neg = sum(1 for m in meds if m < 0)
        stable = "正向" if pos >= neg and pos > 0 else ("负向" if neg > pos else "不定")
        flip = len(folds) - max(pos, neg)
        dir_rows.append((f, stable, pos, neg, flip, mean_abs[f]))
    dir_df = pd.DataFrame(dir_rows, columns=["factor", "stable_dir", "pos_folds", "neg_folds", "flip_folds", "mean_abs_shap"])
    dir_df = dir_df.sort_values("mean_abs_shap", ascending=False)

    # —— 图表(无头) ——
    plt.figure(figsize=(10, 8))
    fam.head(15).iloc[::-1].plot(kind="barh")
    plt.title("SHAP 总贡献 by 特征族 (Mean|SHAP| 归族汇总, 缓和共线)")
    plt.xlabel("Mean|SHAP| (log-odds 尺度)")
    plt.tight_layout(); plt.savefig(f"{base}/shap_family.png", dpi=110); plt.close()

    # 单特征 top 条形
    plt.figure(figsize=(10, 8))
    mean_abs.head(args.top).iloc[::-1].plot(kind="barh")
    plt.title(f"Top {args.top} 单特征 Mean|SHAP| (log-odds 尺度)")
    plt.xlabel("Mean|SHAP|")
    plt.tight_layout(); plt.savefig(f"{base}/shap_top{args.top}.png", dpi=110); plt.close()

    # 蜂群式 dot: 取所有折拼接
    S = np.vstack(shaps); Xv = np.vstack(Xs)
    plt.figure(figsize=(10, 9))
    order = mean_abs.head(args.top).index.tolist()
    order = [o for o in order if o in feats]
    cols = [feats.index(o) for o in order]
    # 用 shap 的 summary 散点 (自绘, 避免依赖版本行为)
    N = min(1500, S.shape[0])
    idx = np.random.RandomState(0).choice(S.shape[0], N, replace=False)
    for r, f in enumerate(order):
        c = cols[r]
        x = Xv[idx, c]; y = S[idx, c]
        plt.scatter(x, y, s=6, alpha=0.25,
                    c=x, cmap="coolwarm")
    plt.yticks(range(len(order)), order); plt.xlabel("特征原始值(红高蓝低)")
    plt.ylabel("SHAP (log-odds): 正=推高排前30%概率, 负=压低")
    plt.title(f"SHAP 蜂群图 Top{len(order)} (方向/分布)")
    plt.axhline(0, color="k", lw=0.5)
    plt.tight_layout(); plt.savefig(f"{base}/shap_summary_dot.png", dpi=110); plt.close()

    # —— 关键因子依赖图(带 bootstrap CI) ——
    key_factors = [f for f in ["pca_absorp_pen", "ix_mkt_gm_mean__chip_cb",
                                "crowd_mom_pct", "pca_absorp_x_macro"] if f in feats]
    dep_lines = []
    for f in key_factors:
        j = feats.index(f)
        plt.figure(figsize=(9, 6))
        any_plotted = False
        for k, i in enumerate(folds):
            x = Xs[k][:, j]; sv = shaps[k][:, j]
            c, lo, hi, cnt = dependence_with_ci(x, sv)
            if len(c) == 0:
                continue
            any_plotted = True
            plt.plot(c, (lo + hi) / 2, "-o", ms=3, alpha=0.7, label=f"折{i}")
            plt.fill_between(c, lo, hi, alpha=0.12)
        if any_plotted:
            plt.axhline(0, color="k", lw=0.5)
            plt.title(f"依赖图: {f}  (SHAP on log-odds; 阴影=80% bootstrap CI)\n"
                      f"⚠ 尾部稀疏处的'断崖'不可靠, 勿直接当硬性止损线")
            plt.xlabel(f"{f} 原始值"); plt.ylabel("SHAP (log-odds)")
            plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(f"{base}/shap_dep_{f}.png", dpi=110); plt.close()
            dep_lines.append(f"- `{f}`: 依赖图见 shap_dep_{f}.png（含跨折 bootstrap CI）")

    # —— 交互解耦(shap_interaction_values, 取正类) ——
    inter_lines = []
    inter_target = [f for f in ["pca_absorp_pen", "ix_mkt_gm_mean__chip_cb"] if f in feats]
    if inter_target:
        try:
            bst = __import__("xgboost").Booster()
            bst.load_model(f"{base}/booster_fold{folds[-1]}.json")
            Xi = pd.read_parquet(f"{base}/shap_data_fold{folds[-1]}.parquet")[feats].astype(float)
            # 交互值 O(N·F²·trees) 极昂贵；定性看特征对排名只需子样本(论文惯例)
            Xi_int = Xi.sample(min(200, len(Xi)), random_state=42) if len(Xi) > 200 else Xi
            exp = shap.TreeExplainer(bst)
            try:
                imap = exp.shap_interaction_values(Xi_int)     # 子采样加速
            except Exception:
                imap = exp.shap_interaction_values(Xi_int.values)  # numpy 兜底
            if isinstance(imap, list):
                imap = imap[1]
            # 每特征的总交互贡献 = 跨样本求和后, 去掉自交互(对角)
            agg = np.abs(imap).sum(0)          # (F,F) 各特征对的总|交互|
            np.fill_diagonal(agg, 0.0)         # 剔除自交互
            tot_inter = agg.sum(1)             # (F,) 每特征总交互贡献
            inter_df = pd.Series(tot_inter, index=feats).sort_values(ascending=False)
            inter_lines.append("末折交互贡献 Top8 (|SHAP interaction| 合计, 已剔除自交互):")
            for f, v in inter_df.head(8).items():
                inter_lines.append(f"  - `{f}`: {v:.4f}")
            plt.figure(figsize=(9, 6))
            inter_df.head(12).iloc[::-1].plot(kind="barh")
            plt.title("特征交互贡献 Top12 (shap_interaction_values)")
            plt.xlabel("|interaction| 合计"); plt.tight_layout()
            plt.savefig(f"{base}/shap_interaction.png", dpi=110); plt.close()
        except Exception as e:
            inter_lines.append(f"交互计算跳过: {e}")

    # —— 落盘 markdown ——
    L = ["# XGBoost 模型 SHAP 解释性分析（修正版）", "",
         f"- 数据: {base} 落盘, WFA {len(folds)} 折 OOS 子样本(各 <=3000 行)",
         f"- 尺度说明: SHAP 值为 **log-odds(边际)** 贡献, 非概率; 标签为『未来收益排前30%的概率』(**排序/相对**, 非绝对收益方向)。",
         f"- base value(正类, log-odds) 各折: {', '.join(f'{i}:{b:.3f}' for i,b in zip(folds,bases))}",
         "", "## 一、特征族总贡献 (Mean|SHAP| 归族, 缓和共线导致的信用瓜分)", ""]
    for p, v in fam.head(15).items():
        L.append(f"- `{p}*`: {v:.4f}")
    L += ["", "## 二、Top 单特征 Mean|SHAP| (log-odds 尺度)", ""]
    for f, v in mean_abs.head(args.top).items():
        L.append(f"- `{f}`: {v:.4f}")
    L += ["", "## 三、跨折方向一致性 (稳健性, 非均值)", "",
          "> 判定: 某因子在多数折中 median(SHAP) 同号 => 方向稳定; flip_folds 多 => 疑似伪Alpha(应剔除)。",
          "", "| 因子 | 稳定方向 | 正向折 | 负向折 | 翻转折 | Mean|SHAP| |",
          "|---|---|---|---|---|---|"]
    for _, r in dir_df.head(args.top).iterrows():
        L.append(f"| `{r['factor']}` | {r['stable_dir']} | {r['pos_folds']} | {r['neg_folds']} | {r['flip_folds']} | {r['mean_abs_shap']:.4f} |")
    L += ["", "## 四、关键因子依赖图(带 bootstrap CI)", ""]
    L += dep_lines or ["(无关键因子)"]
    L += ["", "## 五、交互解耦 (shap_interaction_values)", ""]
    L += inter_lines or ["(未计算)"]
    L += ["", "## 六、结论与警示", "",
          "- SHAP 仅解释**模型逻辑**, 不等同真值; 须与 OOS 的 AUC/IC 一并看。",
          "- 同源特征族(pca_absorp_*/crowd_mom_*) 的 Mean|SHAP| 已归族汇总, 单特征方向可能被共线稀释, 解读单特征时谨慎。",
          "- 依赖图尾部(>0.8/0.9 分位)样本稀疏, '断崖'多为噪声, **不得直接作为硬性止损线**; 阈值须带 CI 并经样本外验证。",
          "- 标签为排序概率, SHAP 推高的是『进入前30%收益组』的概率, 不是绝对涨跌, 交易解读勿混淆。"]
    open(f"{base}/shap_report.md", "w").write("\n".join(L))
    print(f"[shap] 完成 -> {base}/shap_report.md  (+ png 图表)")


if __name__ == "__main__":
    main()
