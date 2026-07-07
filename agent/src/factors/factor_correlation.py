"""因子相关性矩阵与去重模块。

本模块计算因子之间的逐对截面相关性，用于检测因子冗余。
即使两个因子来自不同的主题（theme），它们仍可能高度相关
（例如，一个"动量"因子和一个"反转"因子都使用 20 日收益，只是符号相反）。
本模块能发现并可选地剔除这类冗余因子对。

处理流程：
  1. compute_factor_values()  — 批量计算所有因子的值，生成统一数据结构
  2. compute_correlation_matrix()  — 逐对截面平均相关性
  3. find_correlated_pairs()  — 标记超过阈值的高相关因子对
  4. deduplicate()  — 保留 IC 更强的因子，剔除较弱的那个

设计决策：
  - 使用截面相关（cross-sectional）而非时序相关，因为因子选股关注的是
    同一时间点上不同股票的排序是否一致
  - 默认使用 Spearman 秩相关，因为因子值的大小不重要，排序才重要
  - 相关系数按日期平均，得到一个 N×N 汇总矩阵
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_factor_values(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    alpha_ids: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """批量计算指定因子的值。

    参数:
        registry: 因子注册表实例（提供 compute 方法）
        panel: OHLCV 数据面板字典（key 为 "open"/"high"/"low"/"close"/"volume"，
               value 为宽表 DataFrame，index=日期, columns=股票代码）
        alpha_ids: 要计算的因子 ID 列表。为 None 时计算所有已注册因子

    返回值:
        字典，key 为 alpha_id，value 为因子值 DataFrame（index=日期, columns=股票代码）
        计算失败的因子会被静默跳过

    设计说明:
        异常捕获确保单个因子的计算错误不会中断整个批量计算过程
    """
    if alpha_ids is None:
        alpha_ids = registry.list()
    factor_dfs: dict[str, pd.DataFrame] = {}
    for alpha_id in alpha_ids:
        try:
            factor_dfs[alpha_id] = registry.compute(alpha_id, panel)
        except Exception as exc:
            # 静默跳过计算失败的因子，仅记录 debug 日志
            logger.debug("skip %s: %s", alpha_id, exc)
    return factor_dfs


def compute_correlation_matrix(
    factor_dfs: dict[str, pd.DataFrame],
    method: str = "spearman",
) -> pd.DataFrame:
    """计算所有因子之间的逐对截面相关性矩阵。

    算法步骤：
      1. 找到所有因子的共同日期（交集）
      2. 将每个因子的数据堆叠为一维序列（日期×股票），构成宽表
      3. 若使用 Spearman 方法，按日期分组计算秩
      4. 逐日计算截面相关矩阵，累加后取平均值
      5. 返回 N×N 平均相关矩阵

    为什么用截面相关而非时序相关：
      因子选股关心的是"在同一时间点上，两个因子对股票的排序是否一致"。
      如果两个因子的截面排序高度相关，说明它们给出了相似的选股信号，
      同时使用会导致组合集中度风险。

    参数:
        factor_dfs: 字典，key 为 alpha_id，value 为因子值 DataFrame
        method: 相关系数方法，"spearman"（秩相关，默认）或 "pearson"（线性相关）

    返回值:
        N×N 对称 DataFrame，index 和 columns 均为 alpha_id，值为平均相关系数

    设计说明:
        - 每日截面至少需要 5 个有效数据点才计入平均（避免样本量过小）
        - 使用 numpy 矩阵累加以提升性能（避免逐日 DataFrame 拼接）
    """
    # 按字典序排列因子 ID，确保矩阵行列顺序一致
    ids = sorted(factor_dfs.keys())
    n = len(ids)
    if n == 0:
        return pd.DataFrame()
    if n == 1:
        return pd.DataFrame([[1.0]], index=ids, columns=ids)

    # 步骤 1：计算所有因子的共同日期
    common_dates = factor_dfs[ids[0]].index
    for aid in ids[1:]:
        common_dates = common_dates.intersection(factor_dfs[aid].index)
    if common_dates.empty:
        return pd.DataFrame(np.nan, index=ids, columns=ids)

    # 步骤 2：将每个因子堆叠为 (日期, 股票) → 因子值 的一维序列
    # stack() 将 DataFrame 转为 MultiIndex Series (日期, 股票代码)
    stacked: dict[str, pd.Series] = {}
    for aid in ids:
        df = factor_dfs[aid].loc[common_dates]
        stacked[aid] = df.stack()

    # 构建宽表：index=(日期, 股票), columns=因子ID
    wide = pd.DataFrame(stacked)

    # 步骤 3：Spearman 方法需要按日期分组计算秩
    if method == "spearman":
        # groupby(level=0) 按日期分组，rank(method="average") 计算秩
        grouped = wide.groupby(level=0).rank(method="average")
    else:
        grouped = wide

    # 步骤 4：逐日计算截面相关矩阵
    # 使用 numpy 数组累加，避免逐日构建 DataFrame 的开销
    corr_sums = np.zeros((n, n))
    corr_counts = np.zeros((n, n))

    for date in common_dates:
        # 提取当天的截面数据，丢弃 NaN
        day_data = grouped.loc[date].dropna()
        # 截面样本量不足时跳过（阈值 5 只股票）
        if day_data.shape[0] < 5:
            continue
        # 计算当天的 N×N 相关矩阵
        day_corr = day_data.corr()
        present = day_corr.columns.tolist()
        # 将当日相关值累加到 numpy 矩阵
        for i, a in enumerate(ids):
            if a not in present:
                continue
            for j, b in enumerate(ids):
                if b not in present:
                    continue
                val = day_corr.loc[a, b]
                if not np.isnan(val):
                    corr_sums[i, j] += val
                    corr_counts[i, j] += 1

    # 步骤 5：计算平均值
    # errstate(invalid="ignore") 忽略 0/0 产生的 NaN 警告
    with np.errstate(invalid="ignore"):
        avg_corr = np.where(corr_counts > 0, corr_sums / corr_counts, np.nan)

    return pd.DataFrame(avg_corr, index=ids, columns=ids)


def compute_rolling_correlation(
    factor_dfs: dict[str, pd.DataFrame],
    window: int = 60,
    method: str = "spearman",
) -> dict[str, pd.DataFrame]:
    """计算滚动窗口内的逐日相关性矩阵时间序列。

    与 compute_correlation_matrix 不同，本函数不是输出一个平均矩阵，
    而是为每个滚动窗口结束日期输出一个 N×N 相关矩阵。
    可用于可视化两个因子的相关性如何随时间变化。

    算法步骤：
      1. 找到所有因子的共同日期并排序
      2. 对每个窗口（从第 window 天开始），提取窗口内的数据
      3. 在窗口内计算因子之间的相关矩阵
      4. 以日期字符串为 key 存储结果

    参数:
        factor_dfs: 字典，key 为 alpha_id，value 为因子值 DataFrame
        window: 滚动窗口大小（交易日），默认 60 天
        method: "spearman"（秩相关）或 "pearson"（线性相关）

    返回值:
        字典，key 为日期字符串（YYYY-MM-DD），value 为 N×N 相关矩阵 DataFrame

    设计说明:
        - 窗口内样本量不足 5 个时跳过该日期
        - Spearman 方法在窗口内按截面（行）计算秩
    """
    ids = sorted(factor_dfs.keys())
    if len(ids) < 2:
        return {}

    # 找到共同日期并排序
    common_dates = factor_dfs[ids[0]].index
    for aid in ids[1:]:
        common_dates = common_dates.intersection(factor_dfs[aid].index)
    common_dates = common_dates.sort_values()

    result: dict[str, pd.DataFrame] = {}
    # 从第 window 个日期开始滑动
    for i in range(window - 1, len(common_dates)):
        # 提取当前窗口内的日期
        window_dates = common_dates[i - window + 1 : i + 1]
        # 提取窗口内每个因子的数据
        window_factors = {aid: factor_dfs[aid].loc[window_dates] for aid in ids}
        stacked = {}
        for aid in ids:
            df = window_factors[aid]
            if method == "spearman":
                # 按截面（行/日期）计算秩后堆叠
                stacked[aid] = df.rank(axis=1, method="average").stack()
            else:
                stacked[aid] = df.stack()
        wide = pd.DataFrame(stacked)
        # 样本量不足则跳过
        if wide.shape[0] < 5:
            continue
        # 计算窗口内的相关矩阵
        corr = wide.corr()
        # 以日期字符串为 key 存储
        date_key = str(common_dates[i].date())
        result[date_key] = corr

    return result


def find_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """找出相关系数绝对值超过阈值的所有因子对。

    遍历相关矩阵的上三角部分（避免重复和对角线），
    收集所有 |corr| >= threshold 的因子对。

    参数:
        corr_matrix: N×N 相关矩阵（来自 compute_correlation_matrix）
        threshold: 相关性阈值，默认 0.7
                   含义：|corr| >= 0.7 视为高度相关，两个因子存在冗余

    返回值:
        列表，每个元素为 (alpha_id_a, alpha_id_b, correlation) 元组，
        按 |correlation| 降序排列

    设计说明:
        - 只遍历上三角（j > i），避免 (a,b) 和 (b,a) 重复
        - NaN 值视为无效，不参与比较
    """
    ids = corr_matrix.index.tolist()
    pairs: list[tuple[str, str, float]] = []
    # 遍历上三角矩阵
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) >= threshold:
                pairs.append((ids[i], ids[j], float(val)))
    # 按 |相关系数| 降序排列
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def deduplicate(
    corr_matrix: pd.DataFrame,
    ic_scores: dict[str, float],
    threshold: float = 0.7,
) -> tuple[list[str], list[str]]:
    """对高度相关的因子进行去重，保留 IC 绝对值更强的那个。

    采用贪心策略：
      1. 将所有高相关因子对按 |corr| 降序排列
      2. 依次处理每对：若两个因子都还在候选集中，剔除 |IC| 较小的那个
      3. 贪心保证了最高相关性的因子对优先被处理

    为什么用贪心而非最优解：
      因子去重本质上是一个集合覆盖问题（NP-hard），
      贪心策略虽然不保证全局最优，但在实际场景中效果良好，
      且计算复杂度低（O(pairs)），适合大批量因子。

    参数:
        corr_matrix: N×N 相关矩阵
        ic_scores: 字典，key 为 alpha_id，value 为 IC 值（或 |IC|）
        threshold: 相关性阈值，超过此值视为冗余

    返回值:
        (kept, dropped) 元组：
          - kept: 保留的因子 ID 列表（按字典序排列）
          - dropped: 被剔除的因子 ID 列表（按剔除顺序排列）
    """
    # 获取所有高相关因子对，按 |corr| 降序
    pairs = find_correlated_pairs(corr_matrix, threshold)
    # 初始候选集为所有因子
    candidates = set(corr_matrix.index)
    dropped: list[str] = []

    # 贪心遍历：每对中剔除 IC 较弱的那个
    for a, b, _ in pairs:
        # 如果其中某个因子已经被剔除，跳过此对
        if a not in candidates or b not in candidates:
            continue
        # 比较两个因子的 |IC|
        ic_a = abs(ic_scores.get(a, 0.0))
        ic_b = abs(ic_scores.get(b, 0.0))
        # 剔除 IC 较弱的那个（IC 相等时剔除 b）
        to_drop = b if ic_a >= ic_b else a
        candidates.discard(to_drop)
        dropped.append(to_drop)

    # 保留的因子按字典序排列
    kept = sorted(candidates)
    return kept, dropped


def theme_correlation_summary(
    corr_matrix: pd.DataFrame,
    registry: Any,
) -> pd.DataFrame:
    """将因子级相关性聚合为主题级平均相关性矩阵。

    用于评估不同主题之间的分散度：
      - 低跨主题相关 → 跨主题组合能带来真正的分散化收益
      - 高跨主题相关 → 不同主题的因子实际上在捕捉同一个信号

    算法步骤：
      1. 从注册表获取每个因子的主题标签
      2. 将因子按主题分组
      3. 计算每对主题之间所有因子对的 |相关系数| 均值
      4. 输出主题×主题的平均绝对相关矩阵

    参数:
        corr_matrix: N×N 因子级相关矩阵
        registry: 因子注册表实例（用于查询因子的主题标签）

    返回值:
        主题×主题 DataFrame，值为平均绝对相关系数

    设计说明:
        - 使用绝对相关系数（|corr|），因为负相关同样表示信息重叠
        - 对角线上的 a == b 情况被排除（自身相关恒为 1，会拉高均值）
    """
    ids = corr_matrix.index.tolist()
    # 步骤 1 & 2：建立主题 → 因子列表的映射
    theme_map: dict[str, list[str]] = {}
    for aid in ids:
        try:
            alpha = registry.get(aid)
            for theme in alpha.meta.get("theme", []):
                theme_map.setdefault(theme, []).append(aid)
        except Exception:
            continue

    # 步骤 3：计算主题间平均绝对相关
    themes = sorted(theme_map.keys())
    n = len(themes)
    result = np.zeros((n, n))

    for i, t1 in enumerate(themes):
        for j, t2 in enumerate(themes):
            vals = []
            for a in theme_map[t1]:
                for b in theme_map[t2]:
                    # 排除自身相关（a == b 时 corr 恒为 1）
                    if a != b:
                        v = corr_matrix.loc[a, b]
                        if not np.isnan(v):
                            vals.append(abs(v))
            # 取均值，无有效数据时设为 NaN
            result[i, j] = float(np.mean(vals)) if vals else np.nan

    return pd.DataFrame(result, index=themes, columns=themes)
