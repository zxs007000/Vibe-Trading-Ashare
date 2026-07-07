"""滚动 IC/IR 分析模块 —— 用于因子生命周期追踪。

本模块实现了 "孤子 (soliton)" 哲学：因子是混沌系统中的暂态信号。
通过滚动窗口 IC 和 IR，本模块量化：
  - 因子何时 "活跃"（alive）
  - 因子有效持续时间
  - 因子何时衰减（decay）

每个因子的核心输出：
  - rolling_ic：滑动窗口内的 IC 时间序列
  - rolling_ir：滚动 IC_mean / IC_std（信息比率）
  - lifecycle summary：活跃期、当前状态、衰减日期

设计决策：
  - 默认窗口 60 天（约一个季度），兼顾灵敏度与稳定性
  - min_periods 设为 window // 3，允许窗口初期即输出结果，避免过多 NaN
  - IR 计算中 std 为 0 时用 where 过滤，避免除以零
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.factors.factor_analysis_core import compute_ic_series

logger = logging.getLogger(__name__)


@dataclass
class RollingICResult:
    """单个因子滚动 IC 分析结果的容器。

    属性:
        alpha_id: 因子唯一标识符
        themes: 因子所属的主题标签列表（如 "momentum", "value" 等）
        ic_series: 每日 IC 原始序列（index=日期, values=IC 值）
        rolling_ic_mean: 滚动窗口内的 IC 均值序列
        rolling_ir: 滚动信息比率序列（IC 均值 / IC 标准差）
        rolling_ic_positive_ratio: 滚动窗口内 IC > 0 的天数占比
        current_status: 当前生命周期状态 ("alive" / "decaying" / "dead")
        alive_since: 因子本轮活跃的起始时间
        summary: 汇总指标字典，用于生成表格
    """

    alpha_id: str
    themes: list[str]
    ic_series: pd.Series
    rolling_ic_mean: pd.Series
    rolling_ir: pd.Series
    rolling_ic_positive_ratio: pd.Series
    current_status: str
    alive_since: pd.Timestamp | None
    summary: dict[str, Any]


def compute_rolling_ic(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """计算每日 IC 后取滚动均值。

    算法步骤：
      1. 调用 compute_ic_series 计算逐日截面 IC（因子值与未来收益的秩相关）
      2. 对 IC 序列做滚动窗口均值，平滑短期噪声

    参数:
        factor_df: 因子值矩阵（index=日期, columns=股票代码）
        return_df: 前瞻收益率矩阵（与 factor_df 同构）
        window: 滚动窗口大小，单位：交易日（默认 60 天，约一个季度）

    返回值:
        滚动均值 IC 序列（pd.Series），index 为日期

    设计说明:
        min_periods 设为 max(1, window // 3)，确保窗口填充不足三分之一时
        也能输出结果，减少序列头部的 NaN 空洞。
    """
    # 第一步：计算逐日 IC 序列
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        return pd.Series(dtype=float)
    # 第二步：滚动均值，min_periods 保证初期也有输出
    return ic.rolling(window=window, min_periods=max(1, window // 3)).mean()


def compute_rolling_ir(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """计算滚动信息比率 (IR) = 滚动 IC 均值 / 滚动 IC 标准差。

    IR 衡量因子预测能力的稳定性：
      - IR > 0.5 表示因子在大多数时间内方向正确且稳定
      - IR < 0.2 表示因子信号不稳定或已衰减

    算法步骤：
      1. 计算逐日 IC
      2. 分别计算 IC 的滚动均值和滚动标准差
      3. IR = mean / std，std 为 0 时设为 NaN

    参数:
        factor_df: 因子值矩阵
        return_df: 前瞻收益率矩阵
        window: 滚动窗口大小（交易日）

    返回值:
        滚动 IR 序列（pd.Series）

    设计说明:
        - where(roll_std > 0) 避免除以零
        - replace([inf, -inf], nan) 清理极端值
    """
    # 计算逐日 IC
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        return pd.Series(dtype=float)
    # 滚动均值
    roll_mean = ic.rolling(window=window, min_periods=max(1, window // 3)).mean()
    # 滚动标准差
    roll_std = ic.rolling(window=window, min_periods=max(1, window // 3)).std()
    # IR = mean / std，std 为 0 的位置用 NaN 替代，避免 inf
    ir = roll_mean / roll_std.where(roll_std > 0)
    return ir.replace([np.inf, -np.inf], np.nan)


def compute_rolling_ic_positive_ratio(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """计算滚动窗口内 IC > 0 的天数占比。

    判断标准：
      - 持续 > 0.55：因子具有稳定的正向预测能力
      - 持续 < 0.45：因子已衰减，预测方向不再可靠

    算法步骤：
      1. 计算逐日 IC
      2. 将 IC > 0 的天数标记为 1.0，否则为 0.0
      3. 对标记序列做滚动均值，得到正 IC 占比

    参数:
        factor_df: 因子值矩阵
        return_df: 前瞻收益率矩阵
        window: 滚动窗口大小（交易日）

    返回值:
        滚动 IC 正值占比序列（pd.Series），值域 [0, 1]
    """
    # 计算逐日 IC
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        return pd.Series(dtype=float)
    # 将 IC > 0 转为浮点数 1.0 / 0.0
    positive = (ic > 0).astype(float)
    # 滚动均值即正 IC 占比
    return positive.rolling(window=window, min_periods=max(1, window // 3)).mean()


@dataclass
class LifecycleStatus:
    """因子当前生命周期状态的快照。

    属性:
        alpha_id: 因子标识符
        status: 生命周期状态 ("alive" / "decaying" / "dead")
        alive_since: 本轮活跃的起始日期
        alive_days: 已持续活跃的天数
        last_rolling_ic: 最新一个窗口的滚动 IC 均值
        last_rolling_ir: 最新一个窗口的滚动 IR
        last_ic_positive_ratio: 最新窗口的 IC 正值占比
        peak_rolling_ir: 历史最高滚动 IR 值
        peak_date: 最高 IR 对应的日期
    """

    alpha_id: str
    status: str
    alive_since: pd.Timestamp | None
    alive_days: int | None
    last_rolling_ic: float
    last_rolling_ir: float
    last_ic_positive_ratio: float
    peak_rolling_ir: float
    peak_date: pd.Timestamp | None


# ============================================================
# 生命周期判定阈值（全局常量）
# ============================================================
# 活跃判定：滚动 IC 绝对值 >= 0.015
#   含义：因子与收益至少有微弱的线性关系
_ALIVE_IC_THRESHOLD = 0.015
# 活跃判定：滚动 IR 绝对值 >= 0.2
#   含义：IC 信号的稳定性不能太差
_ALIVE_IR_THRESHOLD = 0.2
# 活跃判定：IC 正值占比 >= 0.45
#   含义：因子至少在接近一半的时间里方向正确
_ALIVE_POS_RATIO_THRESHOLD = 0.45
# 衰减回溯窗口：20 个交易日（约一个月）
#   含义：最近一个月内是否有过活跃迹象，区分"衰减中"和"已死亡"
_DECAY_LOOKBACK = 20


def assess_lifecycle(
    alpha_id: str,
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    window: int = 60,
) -> LifecycleStatus:
    """评估因子当前处于 "活跃"、"衰减中" 还是 "已死亡" 状态。

    分类规则：
      - "alive"（活跃）：滚动 |IC| >= 0.015 且 |IR| >= 0.2 且 IC+ 占比 >= 0.45
      - "decaying"（衰减中）：最近 20 个交易日内曾满足活跃条件，但当前不满足
      - "dead"（已死亡）：最近 window 天内没有任何活跃窗口

    算法步骤：
      1. 计算逐日 IC 序列
      2. 计算滚动 IC 均值、滚动 IR、IC 正值占比
      3. 取最新值判定是否 alive
      4. 若 alive，向前回溯找到活跃起始点
      5. 若 not alive，检查最近 _DECAY_LOOKBACK 天是否有过活跃迹象

    参数:
        alpha_id: 因子标识符
        factor_df: 因子值矩阵
        return_df: 前瞻收益率矩阵
        window: 滚动窗口大小（交易日）

    返回值:
        LifecycleStatus 快照对象
    """
    # 步骤 1：计算逐日 IC
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        # IC 为空时直接判定为 dead
        return LifecycleStatus(
            alpha_id=alpha_id, status="dead", alive_since=None, alive_days=None,
            last_rolling_ic=0.0, last_rolling_ir=0.0, last_ic_positive_ratio=0.0,
            peak_rolling_ir=0.0, peak_date=None,
        )

    # 步骤 2：计算三个滚动指标
    # 滚动 IC 均值
    roll_mean = ic.rolling(window=window, min_periods=max(1, window // 3)).mean()
    # 滚动 IC 标准差
    roll_std = ic.rolling(window=window, min_periods=max(1, window // 3)).std()
    # 滚动 IR = 均值 / 标准差，处理除零和无穷值
    roll_ir = (roll_mean / roll_std.where(roll_std > 0)).replace([np.inf, -np.inf], np.nan)
    # 滚动 IC 正值占比
    pos_ratio = (ic > 0).astype(float).rolling(window=window, min_periods=max(1, window // 3)).mean()

    # 步骤 3：取最新一个时间点的指标值
    last_ic = float(roll_mean.iloc[-1]) if not roll_mean.empty else 0.0
    last_ir = float(roll_ir.iloc[-1]) if not roll_ir.empty else 0.0
    last_pos = float(pos_ratio.iloc[-1]) if not pos_ratio.empty else 0.0
    # 历史峰值 IR 及对应日期
    peak_ir = float(roll_ir.max()) if not roll_ir.empty else 0.0
    peak_date = roll_ir.idxmax() if not roll_ir.empty else None

    # 步骤 4：综合三个条件判定是否活跃
    is_alive = (
        abs(last_ic) >= _ALIVE_IC_THRESHOLD
        and abs(last_ir) >= _ALIVE_IR_THRESHOLD
        and last_pos >= _ALIVE_POS_RATIO_THRESHOLD
    )

    if is_alive:
        # 步骤 5a：活跃 —— 向前回溯找到活跃起始日期
        alive_since = _find_alive_start(roll_mean, roll_ir, pos_ratio, window)
        # 计算已活跃天数
        alive_days = (roll_mean.index[-1] - alive_since).days if alive_since is not None else None
        return LifecycleStatus(
            alpha_id=alpha_id, status="alive", alive_since=alive_since,
            alive_days=alive_days, last_rolling_ic=last_ic, last_rolling_ir=last_ir,
            last_ic_positive_ratio=last_pos, peak_rolling_ir=peak_ir, peak_date=peak_date,
        )

    # 步骤 5b：非活跃 —— 检查最近 _DECAY_LOOKBACK 天是否有过活跃迹象
    recent = roll_mean.iloc[-_DECAY_LOOKBACK:]
    was_alive = (
        (recent.abs() >= _ALIVE_IC_THRESHOLD).any()
        if not recent.empty else False
    )
    # 近期有过活跃迹象 → decaying；否则 → dead
    status = "decaying" if was_alive else "dead"
    return LifecycleStatus(
        alpha_id=alpha_id, status=status, alive_since=None, alive_days=None,
        last_rolling_ic=last_ic, last_rolling_ir=last_ir,
        last_ic_positive_ratio=last_pos, peak_rolling_ir=peak_ir, peak_date=peak_date,
    )


def _find_alive_start(
    roll_mean: pd.Series,
    roll_ir: pd.Series,
    pos_ratio: pd.Series,
    window: int,
) -> pd.Timestamp | None:
    """从序列末尾向前回溯，找到因子首次变为活跃的日期。

    算法逻辑：
      从最后一个时间点开始，逐步向前检查每个时间点是否满足活跃条件。
      当找到第一个不满足活跃条件的时间点时，其下一个时间点即为活跃起始点。
      如果序列从头到尾都满足活跃条件，则返回序列的第一个日期。

    参数:
        roll_mean: 滚动 IC 均值序列
        roll_ir: 滚动 IR 序列
        pos_ratio: IC 正值占比序列
        window: 滚动窗口大小（保留参数，未使用）

    返回值:
        活跃起始日期（pd.Timestamp），若无法确定则返回 None
    """
    if roll_mean.empty:
        return None
    # 从末尾向前遍历
    for i in range(len(roll_mean) - 1, -1, -1):
        # 提取当前时间点的三个指标，NaN 视为 0
        ic_val = abs(float(roll_mean.iloc[i])) if not np.isnan(roll_mean.iloc[i]) else 0.0
        ir_val = abs(float(roll_ir.iloc[i])) if not np.isnan(roll_ir.iloc[i]) else 0.0
        pos_val = float(pos_ratio.iloc[i]) if not np.isnan(pos_ratio.iloc[i]) else 0.0
        # 检查是否满足活跃条件
        is_alive = (
            ic_val >= _ALIVE_IC_THRESHOLD
            and ir_val >= _ALIVE_IR_THRESHOLD
            and pos_val >= _ALIVE_POS_RATIO_THRESHOLD
        )
        if not is_alive:
            # 当前点不活跃，下一个点（如果存在）即为活跃起始点
            if i < len(roll_mean) - 1:
                return roll_mean.index[i + 1]
            return roll_mean.index[i]
    # 整个序列都活跃，返回第一个日期
    return roll_mean.index[0]


def batch_rolling_analysis(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    window: int = 60,
    alpha_ids: list[str] | None = None,
    themes: list[str] | None = None,
    zoo: str | None = None,
) -> list[RollingICResult]:
    """批量对多个因子执行滚动 IC 分析。

    流程：
      1. 确定待分析的因子列表（支持按主题和 zoo 过滤）
      2. 逐个计算因子值、IC 序列、滚动指标
      3. 评估每个因子的生命周期状态
      4. 汇总为 RollingICResult 列表

    参数:
        registry: 因子注册表（Factor Registry 实例）
        panel: OHLCV 数据面板字典（key 为 "open"/"high"/"low"/"close"/"volume"）
        return_df: 前瞻收益率 DataFrame
        window: 滚动窗口大小（交易日）
        alpha_ids: 指定要分析的因子 ID 列表。为 None 时自动从注册表获取全部
        themes: 按主题过滤（如 ["momentum", "value"]）。为 None 时不过滤
        zoo: 按因子库过滤（如 "technical"）。为 None 时不过滤

    返回值:
        RollingICResult 列表，每个成功计算的因子对应一个元素

    设计说明:
        - 计算失败的因子会被静默跳过（记录 debug 日志）
        - 当 themes 包含多个主题时，取各主题因子 ID 的并集
    """
    # 步骤 1：确定待分析的因子列表
    if alpha_ids is None:
        # 若只指定了单个主题，直接按主题过滤
        alpha_ids = registry.list(zoo=zoo, theme=themes[0] if themes and len(themes) == 1 else None)
        if themes and len(themes) > 1:
            # 多主题场景：逐主题查询后取并集
            ids_by_theme = set()
            for t in themes:
                ids_by_theme.update(registry.list(zoo=zoo, theme=t))
            alpha_ids = sorted(ids_by_theme)

    # 步骤 2：逐因子计算
    results: list[RollingICResult] = []
    for alpha_id in alpha_ids:
        try:
            # 计算因子值矩阵
            factor_df = registry.compute(alpha_id, panel)
        except Exception as exc:
            # 计算失败则跳过，不影响其他因子
            logger.debug("skip %s: %s", alpha_id, exc)
            continue

        # 计算逐日 IC
        ic = compute_ic_series(factor_df, return_df)
        if ic.empty:
            continue

        # 步骤 3：计算滚动指标
        # 滚动 IC 均值
        roll_mean = ic.rolling(window=window, min_periods=max(1, window // 3)).mean()
        # 滚动 IC 标准差
        roll_std = ic.rolling(window=window, min_periods=max(1, window // 3)).std()
        # 滚动 IR
        roll_ir = (roll_mean / roll_std.where(roll_std > 0)).replace([np.inf, -np.inf], np.nan)
        # 滚动 IC 正值占比
        pos_ratio = (ic > 0).astype(float).rolling(window=window, min_periods=max(1, window // 3)).mean()

        # 步骤 4：评估生命周期状态
        lifecycle = assess_lifecycle(alpha_id, factor_df, return_df, window)
        # 获取因子元数据（主题标签等）
        alpha = registry.get(alpha_id)
        factor_themes = alpha.meta.get("theme", [])

        # 步骤 5：构造结果对象
        results.append(RollingICResult(
            alpha_id=alpha_id,
            themes=factor_themes,
            ic_series=ic,
            rolling_ic_mean=roll_mean,
            rolling_ir=roll_ir,
            rolling_ic_positive_ratio=pos_ratio,
            current_status=lifecycle.status,
            alive_since=lifecycle.alive_since,
            summary={
                # IC 均值：因子预测能力的总体强度
                "ic_mean": round(float(ic.mean()), 6),
                # IC 标准差：预测能力的波动程度
                "ic_std": round(float(ic.std()), 6),
                # 总体 IR = IC_mean / IC_std
                "ir": round(float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0, 4),
                # 最新滚动 IC
                "last_rolling_ic": round(lifecycle.last_rolling_ic, 6),
                # 最新滚动 IR
                "last_rolling_ir": round(lifecycle.last_rolling_ir, 4),
                # 活跃天数
                "alive_days": lifecycle.alive_days,
                # 当前状态
                "status": lifecycle.status,
                # 主题标签
                "themes": factor_themes,
                # 因子所属库
                "zoo": alpha.zoo,
            },
        ))

    return results


def summary_table(results: list[RollingICResult]) -> pd.DataFrame:
    """将 RollingICResult 列表转换为汇总 DataFrame。

    行 = 因子，列 = 核心指标。按 last_rolling_ir 绝对值降序排列，
    方便快速定位当前最强的因子。

    参数:
        results: RollingICResult 列表

    返回值:
        汇总表（pd.DataFrame），index 为 alpha_id

    设计说明:
        排序时使用 key=abs，因为负 IR 的因子（反向预测）同样有价值
    """
    rows = [r.summary for r in results]
    df = pd.DataFrame(rows, index=[r.alpha_id for r in results])
    # 按 |last_rolling_ir| 降序排列
    df = df.sort_values("last_rolling_ir", ascending=False, key=abs)
    return df


def alive_factors_by_theme(
    results: list[RollingICResult],
) -> dict[str, list[str]]:
    """将当前活跃的因子按主题分组。

    每个主题下的因子按 |last_rolling_ir| 降序排列，
    便于在每个主题中挑选最强的因子。

    参数:
        results: RollingICResult 列表（来自 batch_rolling_analysis）

    返回值:
        字典，key 为主题名称，value 为该主题下活跃因子的 alpha_id 列表，
        按 |last_rolling_ir| 降序排列
    """
    # 筛选状态为 "alive" 的因子
    alive = [r for r in results if r.current_status == "alive"]
    # 按主题分组，同时记录 IR 值用于排序
    by_theme: dict[str, list[tuple[str, float]]] = {}
    for r in alive:
        ir_val = r.summary.get("last_rolling_ir", 0.0)
        for theme in r.themes:
            by_theme.setdefault(theme, []).append((r.alpha_id, abs(ir_val)))
    # 每个主题内按 |IR| 降序排列，主题名按字典序排列
    return {
        theme: [aid for aid, _ in sorted(pairs, key=lambda x: -x[1])]
        for theme, pairs in sorted(by_theme.items())
    }
