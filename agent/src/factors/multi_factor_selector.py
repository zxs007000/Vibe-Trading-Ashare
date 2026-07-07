"""多因子选择器 —— 结合主题分散化与逻辑链的因子组合构建。

本模块实现了综合选股哲学：
  1. "孤子 (Soliton)" — 只选当前活跃的因子（经滚动 IC/IR 验证）
  2. "多因子 > 单因子" — 跨主题组合以实现分散化
  3. "逻辑链 (Logical chains)" — 偏好形成因果序列的因子组合
  4. "相关性守卫" — 即使跨主题也要剔除冗余因子

处理流程：
  1. 对所有因子执行滚动 IC 分析 → 找到活跃因子
  2. 在每个主题内按 |滚动 IR| 排名
  3. 应用逻辑链模板对主题组合评分
  4. 交叉检验相关性 → 去重
  5. 输出最终因子组合

设计决策：
  - 逻辑链是预定义的主题序列模板，来自量化投资领域的经典策略组合
  - max_per_theme 限制每个主题最多选 2 个因子，防止单一主题过度集中
  - 复合因子使用百分位秩（pct rank）而非原始值，消除量纲差异
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.factors.factor_correlation import (
    compute_correlation_matrix,
    compute_factor_values,
    deduplicate,
    find_correlated_pairs,
)
from src.factors.rolling_ic import (
    RollingICResult,
    alive_factors_by_theme,
    assess_lifecycle,
    batch_rolling_analysis,
    summary_table,
)

logger = logging.getLogger(__name__)


# ============================================================
# 逻辑链模板（Logical Chain Templates）
# ============================================================
# 每个逻辑链是一个主题序列，代表一种经典的多因子投资逻辑：
#
# value_momentum（价值+动量）：
#   先选低估值股票（value），再确认其上涨趋势（momentum），最后用成交量（volume）验证
#
# quality_growth（质量+成长）：
#   选高质量公司（quality），确认盈利增长（growth），再用动量确认市场认可
#
# low_risk（低风险）：
#   选低波动股票（volatility），确认流动性充足（liquidity），利用短期反转（reversal）入场
#
# smart_money（聪明钱）：
#   追踪微观结构信号（microstructure），用成交量（volume）确认，动量（momentum）跟随
#
# contrarian（逆向）：
#   捕捉过度反应（reversal），结合情绪指标（sentiment），用波动率（volatility）控制风险
#
# fundamental（基本面）：
#   低估值（value）+ 高质量（quality）+ 低杠杆（leverage）的经典基本面组合
LOGICAL_CHAINS: dict[str, list[str]] = {
    "value_momentum": ["value", "momentum", "volume"],
    "quality_growth": ["quality", "growth", "momentum"],
    "low_risk": ["volatility", "liquidity", "reversal"],
    "smart_money": ["microstructure", "volume", "momentum"],
    "contrarian": ["reversal", "sentiment", "volatility"],
    "fundamental": ["value", "quality", "leverage"],
}


@dataclass
class FactorSelection:
    """一轮多因子选择的结果。

    属性:
        selected: 最终选中的因子 ID 列表
        themes_covered: 选中因子所覆盖的主题列表
        logical_chain: 匹配到的最佳逻辑链名称（如 "value_momentum"）
        corr_dropped: 因高相关性被剔除的因子 ID 列表
        dead_factors: 被判定为"已死亡"的因子 ID 列表
        decaying_factors: 被判定为"衰减中"的因子 ID 列表
        details: 选中因子的详细指标 DataFrame
        chain_score: 逻辑链匹配得分（0~1，覆盖的主题比例）
    """

    selected: list[str]
    themes_covered: list[str]
    logical_chain: str | None
    corr_dropped: list[str]
    dead_factors: list[str]
    decaying_factors: list[str]
    details: pd.DataFrame
    chain_score: float


@dataclass
class SelectionConfig:
    """多因子选择器的配置参数。

    属性:
        rolling_window: 滚动窗口大小（交易日），默认 60 天（约一个季度）
        max_per_theme: 每个主题最多选取的因子数，默认 2
                       目的：防止单一主题过度集中
        min_rolling_ir: 最低滚动 IR 要求，默认 0.5
                        含义：IR < 0.5 的因子信号不够稳定，不予选取
        corr_threshold: 相关性去重阈值，默认 0.7
                        含义：|corr| >= 0.7 的因子对被视为冗余
        min_total_factors: 最终组合的最小因子数，默认 3
        max_total_factors: 最终组合的最大因子数，默认 8
                           目的：控制组合复杂度，避免过拟合
        preferred_chain: 偏好的逻辑链名称。为 None 时自动匹配最佳链
    """

    rolling_window: int = 60
    max_per_theme: int = 2
    min_rolling_ir: float = 0.5
    corr_threshold: float = 0.7
    min_total_factors: int = 3
    max_total_factors: int = 8
    preferred_chain: str | None = None


def select_factors(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    config: SelectionConfig | None = None,
) -> FactorSelection:
    """执行完整的多因子选择流程。

    算法步骤：
      1. 批量滚动 IC 分析 → 得到每个因子的生命周期状态和滚动指标
      2. 按主题分组活跃因子 → 每个主题内按 |IR| 排名
      3. 每个主题最多选 max_per_theme 个因子（需满足 min_rolling_ir）
      4. 若候选数超过 max_total_factors → 按全局 |IR| 裁剪
      5. 计算候选因子间的相关矩阵 → 去重（保留 IC 更强的）
      6. 匹配最佳逻辑链
      7. 构建并返回 FactorSelection 结果

    参数:
        registry: 因子注册表实例
        panel: OHLCV 数据面板字典
        return_df: 前瞻收益率 DataFrame
        config: 选择配置参数。为 None 时使用默认配置

    返回值:
        FactorSelection 对象，包含选中的因子组合及其详细信息
    """
    cfg = config or SelectionConfig()

    # 步骤 1：批量滚动 IC 分析
    results = batch_rolling_analysis(
        registry, panel, return_df,
        window=cfg.rolling_window,
    )

    # 步骤 2：按主题分组活跃因子
    alive_by_theme = alive_factors_by_theme(results)
    # 构建 alpha_id → RollingICResult 的快速查找表
    result_map = {r.alpha_id: r for r in results}

    # 记录死亡和衰减中的因子
    dead_ids = [r.alpha_id for r in results if r.current_status == "dead"]
    decaying_ids = [r.alpha_id for r in results if r.current_status == "decaying"]

    # 步骤 3：在每个主题内挑选候选因子
    candidates: list[str] = []
    themes_covered: list[str] = []

    for theme, alpha_ids in sorted(alive_by_theme.items()):
        picked = 0
        for aid in alpha_ids:
            # 每个主题最多选 max_per_theme 个
            if picked >= cfg.max_per_theme:
                break
            r = result_map.get(aid)
            if r is None:
                continue
            # 检查滚动 IR 是否满足最低要求
            ir_val = abs(r.summary.get("last_rolling_ir", 0.0))
            if ir_val < cfg.min_rolling_ir:
                continue
            candidates.append(aid)
            picked += 1
        if picked > 0:
            themes_covered.append(theme)

    # 步骤 4：全局裁剪——若候选数超过上限，按 |IR| 保留前 max_total_factors 个
    if len(candidates) > cfg.max_total_factors:
        candidates = _trim_by_ir(candidates, result_map, cfg.max_total_factors)

    # 步骤 5：相关性去重
    corr_dropped: list[str] = []
    if len(candidates) >= 2:
        # 计算候选因子的值
        factor_dfs = compute_factor_values(registry, panel, candidates)
        if len(factor_dfs) >= 2:
            # 计算相关矩阵
            corr_matrix = compute_correlation_matrix(factor_dfs)
            if not corr_matrix.empty:
                # 提取每个候选因子的 |IC| 作为去重依据
                ic_scores = {
                    aid: abs(result_map[aid].summary.get("last_rolling_ic", 0.0))
                    for aid in candidates
                    if aid in result_map
                }
                # 贪心去重：保留 IC 更强的因子
                kept, dropped = deduplicate(corr_matrix, ic_scores, cfg.corr_threshold)
                corr_dropped = dropped
                candidates = kept

    # 步骤 6：匹配最佳逻辑链
    chain_name, chain_score = _best_logical_chain(themes_covered, cfg.preferred_chain)

    # 步骤 7：构建结果详情 DataFrame
    detail_rows = []
    for aid in candidates:
        r = result_map.get(aid)
        if r:
            detail_rows.append(r.summary)
    details = pd.DataFrame(detail_rows, index=candidates) if detail_rows else pd.DataFrame()

    return FactorSelection(
        selected=candidates,
        themes_covered=themes_covered,
        logical_chain=chain_name,
        corr_dropped=corr_dropped,
        dead_factors=dead_ids,
        decaying_factors=decaying_ids,
        details=details,
        chain_score=chain_score,
    )


def _trim_by_ir(
    candidates: list[str],
    result_map: dict[str, RollingICResult],
    max_n: int,
) -> list[str]:
    """按 |滚动 IR| 降序裁剪候选因子列表，保留前 max_n 个。

    当候选因子数超过 max_total_factors 时使用此函数进行全局排名裁剪。

    参数:
        candidates: 当前候选因子 ID 列表
        result_map: alpha_id → RollingICResult 的映射
        max_n: 最多保留的因子数量

    返回值:
        裁剪后的因子 ID 列表（按 |IR| 降序，最多 max_n 个）
    """
    # 提取每个候选因子的 |last_rolling_ir|
    scored = [
        (aid, abs(result_map[aid].summary.get("last_rolling_ir", 0.0)))
        for aid in candidates
        if aid in result_map
    ]
    # 按 |IR| 降序排列
    scored.sort(key=lambda x: -x[1])
    # 返回前 max_n 个因子 ID
    return [aid for aid, _ in scored[:max_n]]


def _best_logical_chain(
    themes_covered: list[str],
    preferred: str | None = None,
) -> tuple[str | None, float]:
    """找到与当前覆盖主题最匹配的逻辑链。

    匹配逻辑：
      计算每个逻辑链的主题集合与当前覆盖主题的交集比例。
      比例越高，说明当前因子组合越符合该逻辑链的投资逻辑。

    参数:
        themes_covered: 当前选中因子所覆盖的主题列表
        preferred: 偏好的逻辑链名称。若指定且在预定义链中存在，
                   则直接返回该链（即使匹配度不是最高）

    返回值:
        (chain_name, score) 元组：
          - chain_name: 最佳匹配的逻辑链名称，无匹配时为 None
          - score: 匹配得分（0~1），= 交集主题数 / 逻辑链主题总数
    """
    # 如果指定了偏好链且存在，直接返回
    if preferred and preferred in LOGICAL_CHAINS:
        chain_themes = LOGICAL_CHAINS[preferred]
        overlap = len(set(chain_themes) & set(themes_covered))
        score = overlap / len(chain_themes) if chain_themes else 0.0
        return preferred, score

    # 否则遍历所有逻辑链，找匹配度最高的
    best_name: str | None = None
    best_score = 0.0
    for name, chain_themes in LOGICAL_CHAINS.items():
        # 计算交集大小与链主题总数的比值
        overlap = len(set(chain_themes) & set(themes_covered))
        score = overlap / len(chain_themes) if chain_themes else 0.0
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def build_composite_factor(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    selection: FactorSelection,
    weight_method: str = "equal",
) -> pd.DataFrame:
    """将选中的多个因子合成为一个复合因子得分。

    合成步骤：
      1. 计算每个选中因子的值矩阵
      2. 确定权重（等权或 IR 加权）
      3. 将每个因子的截面值转为百分位秩（0~1），消除量纲差异
      4. 按权重加权求和

    为什么用百分位秩而非原始值：
      不同因子的值域可能差异巨大（如 PE 比率 0~200 vs 动量 -0.5~0.5），
      直接加权会导致值域大的因子主导。百分位秩将所有因子映射到 [0,1]，
      实现公平比较。

    参数:
        registry: 因子注册表实例
        panel: OHLCV 数据面板字典
        selection: FactorSelection 对象（来自 select_factors）
        weight_method: 加权方式
            - "equal": 等权重（每个因子 1/N）
            - "ir_weighted": 按 |滚动 IR| 加权，IR 越高的因子权重越大

    返回值:
        复合因子 DataFrame（index=日期, columns=股票代码），值域约 [0, 1]
    """
    if not selection.selected:
        return pd.DataFrame()

    # 步骤 1：批量计算因子值
    factor_dfs = compute_factor_values(registry, panel, selection.selected)
    if not factor_dfs:
        return pd.DataFrame()

    # 步骤 2：确定权重
    if weight_method == "ir_weighted" and not selection.details.empty:
        # IR 加权：权重与 |IR| 成正比
        weights = selection.details["last_rolling_ir"].abs()
        # 归一化使权重之和为 1
        weights = weights / weights.sum()
    else:
        # 等权重：每个因子 1/N
        weights = pd.Series(1.0 / len(factor_dfs), index=list(factor_dfs.keys()))

    # 步骤 3 & 4：转百分位秩后加权
    ranked_dfs = []
    for aid, df in factor_dfs.items():
        # rank(axis=1, pct=True) 按截面计算百分位秩
        ranked = df.rank(axis=1, pct=True)
        w = weights.get(aid, 1.0 / len(factor_dfs))
        ranked_dfs.append(ranked * w)

    # 累加所有加权排名
    composite = ranked_dfs[0]
    for r in ranked_dfs[1:]:
        # add(fill_value=0) 处理 NaN：缺失值视为 0 贡献
        composite = composite.add(r, fill_value=0)

    return composite


def run_periodic_selection(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    rebalance_dates: list[str],
    config: SelectionConfig | None = None,
) -> dict[str, FactorSelection]:
    """在每个再平衡日期运行因子选择（回测选择器）。

    模拟真实交易场景：在每个再平衡日，使用截至该日期的历史数据
    运行完整的因子选择流程，决定下一期使用哪些因子。

    关键设计——避免未来数据泄露：
      对于每个再平衡日期，将面板数据和收益率截断到该日期之前，
      确保选择过程只使用该时间点已知的信息。

    参数:
        registry: 因子注册表实例
        panel: 完整的 OHLCV 数据面板字典
        return_df: 完整的前瞻收益率 DataFrame
        rebalance_dates: 再平衡日期列表（格式：YYYY-MM-DD）
        config: 选择配置参数

    返回值:
        字典，key 为再平衡日期字符串，value 为该日期的 FactorSelection

    设计说明:
        - 若截至某再平衡日期的历史数据不足一个滚动窗口，跳过该日期
        - 选择过程中的异常会被捕获并记录警告，不影响其他日期
    """
    results: dict[str, FactorSelection] = {}
    # 需要 close 数据来截断日期
    close = panel.get("close")
    if close is None:
        return results

    for date_str in rebalance_dates:
        date = pd.Timestamp(date_str)
        # 将面板数据截断到当前再平衡日期（含）
        truncated_panel = {k: v.loc[:date] for k, v in panel.items()}
        truncated_returns = return_df.loc[:date]

        # 检查历史数据是否足够（至少需要一个滚动窗口）
        if truncated_returns.shape[0] < (config.rolling_window if config else 60):
            logger.warning("skip rebalance %s: insufficient history", date_str)
            continue

        try:
            # 在截断数据上运行完整选择流程
            sel = select_factors(registry, truncated_panel, truncated_returns, config)
            results[date_str] = sel
        except Exception as exc:
            logger.warning("selection failed at %s: %s", date_str, exc)

    return results
