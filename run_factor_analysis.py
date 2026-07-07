"""
因子生命周期分析运行脚本 (Factor Lifecycle Analysis Runner)

本脚本使用 astockdata (mootdx) 获取 A 股行情数据，依次执行：
  1. 数据获取：从通达信服务器拉取沪深 300 成分股的 OHLCV 日线数据
  2. 因子注册：加载所有已注册的 alpha 因子
  3. 滚动 IC 分析：评估每个因子的当前活跃度和生命周期状态
  4. 相关性矩阵：计算因子间的截面相关性，发现冗余因子对
  5. 多因子筛选：综合 IC/IR/相关性/逻辑链，输出推荐的因子组合

运行方式：
  python run_factor_analysis.py

输出：
  控制台打印因子寿命看板、活跃因子列表、高相关对、主题相关矩阵、推荐因子组合
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 将 agent 目录加入 Python 路径，以便导入 src 包
sys.path.insert(0, str(Path(__file__).parent / "agent"))

from mootdx.quotes import Quotes
from backtest.loaders.astockdata_loader import DataLoader as AStockDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 沪深 300 样本股票列表
# ============================================================
# 选取具有代表性的 A 股蓝筹股，涵盖消费、金融、科技、医药、能源等行业
# 股票代码格式：XXXXXX.SH（上海）或 XXXXXX.SZ（深圳）
CSI300_SAMPLE = [
    "600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000333.SZ",  # 茅台、五粮液、平安、招行、美的
    "600276.SH", "601166.SH", "000651.SZ", "600030.SH", "601888.SH",  # 恒瑞、兴业、格力、中信、国旅
    "002714.SZ", "600900.SH", "601398.SH", "000001.SZ", "600887.SH",  # 牧原、长电、工行、平安银行、伊利
    "002415.SZ", "601012.SH", "600585.SH", "000568.SZ", "601899.SH",  # 海康、隆基、海螺、泸州老窖、紫金
    "002304.SZ", "600309.SH", "601688.SH", "000002.SZ", "600809.SH",  # 洋河、万华、华泰、万科、汾酒
    "002594.SZ", "601601.SH", "600031.SH", "000725.SZ", "601288.SH",  # 比亚迪、太保、三一、京东方、农行
    "600050.SH", "000063.SZ", "601668.SH", "600104.SH", "002475.SZ",  # 联通、中兴、中建、上汽、立讯
    "601318.SH", "600000.SH", "000776.SZ", "601766.SH", "002230.SZ",  # 平安(重复)、浦发、广发、中车、科大讯飞
    "600048.SH", "002142.SZ", "601628.SH", "600016.SH", "000538.SZ",  # 保利、宁波银行、国寿、民生、云南白药
    "601186.SH", "600547.SH", "002027.SZ", "601138.SH", "600690.SH",  # 中铁、山金、分众、工业富联、海尔
    "002352.SZ", "600010.SH", "601111.SH", "600588.SH", "002049.SZ",  # 顺丰、包钢、国航、用友、紫光
    "600019.SH", "601225.SH", "000100.SZ", "601009.SH", "002001.SZ",  # 宝钢、陕煤、TCL、南京银行、新和成
    "600196.SH", "000568.SZ", "601669.SH", "600176.SH", "002024.SZ",  # 复星、泸州老窖(重复)、电建、玻纤、苏宁
    "601857.SH", "600028.SH", "601985.SH", "002271.SZ", "601155.SH",  # 中石油、中石化、中国核电、东方雨虹、新城
    "600346.SH", "601100.SH", "002241.SZ", "600406.SH", "601919.SH",  # 恒力、华域、歌尔、国电南瑞、中远
    "600703.SH", "601818.SH", "002607.SZ", "600795.SH", "601998.SH",  # 三安、光大、中公、国电电力、中信银行
]


def fetch_ashare_data(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """通过 astockdata (mootdx) 获取 A 股 OHLCV 日线数据。

    参数:
        codes: 股票代码列表（如 ["600519.SH", "000858.SZ"]）
        start_date: 起始日期（格式：YYYY-MM-DD）
        end_date: 结束日期（格式：YYYY-MM-DD）

    返回值:
        字典，key 为股票代码，value 为该股票的 OHLCV DataFrame
        若数据源不可用，返回空字典

    设计说明:
        先检查 loader 是否可用（mootdx 是否正确安装），
        避免在不可用时浪费网络请求时间
    """
    loader = AStockDataLoader()
    # 检查数据源是否可用
    if not loader.is_available():
        logger.error("astockdata not available, check mootdx install")
        return {}
    logger.info("Fetching %d stocks from %s to %s...", len(codes), start_date, end_date)
    # 批量获取日线数据（interval="1D" 表示日频）
    result = loader.fetch(codes, start_date, end_date, interval="1D")
    logger.info("Got data for %d / %d stocks", len(result), len(codes))
    return result


def build_panel(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """将逐股票的 DataFrame 转换为宽表面板格式。

    输入格式：
        {code: DataFrame(columns=[open, high, low, close, volume], index=date)}

    输出格式：
        {"open": DataFrame(index=date, columns=codes),
         "high": DataFrame(...),
         "close": DataFrame(...), ...}

    这种宽表格式是因子计算的标准输入，使得每个因子函数可以方便地
    对所有股票做截面操作。

    参数:
        stock_data: 逐股票的 OHLCV 数据字典

    返回值:
        面板字典，key 为字段名（"open"/"high"/"low"/"close"/"volume"），
        value 为宽表 DataFrame（index=日期, columns=股票代码）
    """
    all_dfs = {}
    for code, df in stock_data.items():
        if df is None or df.empty:
            continue
        # 遍历 open/high/low/close/volume 五个字段
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                continue
            if col not in all_dfs:
                all_dfs[col] = {}
            all_dfs[col][code] = df[col]

    panel = {}
    for col, series_dict in all_dfs.items():
        # 合并为宽表并按日期排序
        panel[col] = pd.DataFrame(series_dict).sort_index()
    return panel


def compute_forward_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """计算次日前瞻收益率。

    前瞻收益率 = (明日收盘价 - 今日收盘价) / 今日收盘价
    这是因子 IC 分析中的标准被解释变量：
      IC = corr(今日因子值, 明日收益率)

    算法：
      pct_change() 计算日收益率 → shift(-1) 将明日收益对齐到今日行

    参数:
        panel: OHLCV 面板字典（必须包含 "close" 键）

    返回值:
        前瞻收益率 DataFrame（index=日期, columns=股票代码）
        最后一行全为 NaN（因为没有"明日"数据）

    注意:
        若面板中缺少 "close" 字段，抛出 ValueError
    """
    close = panel.get("close")
    if close is None:
        raise ValueError("panel missing 'close'")
    # pct_change() 计算日收益率，shift(-1) 向前移一位得到前瞻收益
    return close.pct_change().shift(-1)


def main():
    """主函数：执行完整的因子生命周期分析流程。

    流程概览：
      1. 拉取过去 2 年的 A 股日线数据
      2. 加载因子注册表
      3. 滚动 IC 分析（窗口 60 天）
      4. 因子相关性矩阵 + 主题相关性
      5. 多因子筛选 + 输出推荐组合

    设计决策：
      - 数据区间选择 2 年：足够长以观察因子生命周期，又不至于数据量过大
      - 滚动窗口 60 天：约一个季度，是量化策略中常用的中期评估窗口
      - min_rolling_ir 设为 0.15（低于选择器默认的 0.5）：
        在分析阶段放宽门槛以观察更多候选因子
    """
    # 计算数据区间：过去 2 年到昨天
    # 减 1 天避免当天数据未收盘
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("A-Share Factor Lifecycle Analysis")
    logger.info("=" * 60)

    # ============================================================
    # 第一步：获取行情数据并构建面板
    # ============================================================
    stock_data = fetch_ashare_data(CSI300_SAMPLE, start_date, end_date)
    if not stock_data:
        logger.error("No data fetched, aborting")
        return

    # 将逐股票数据转换为宽表面板
    panel = build_panel(stock_data)
    logger.info("Panel shape: close=%s", panel["close"].shape)

    # 计算前瞻收益率
    return_df = compute_forward_returns(panel)
    logger.info("Return df shape: %s", return_df.shape)

    # 延迟导入因子分析模块（避免在数据获取阶段就加载）
    from src.factors.registry import Registry
    from src.factors.rolling_ic import batch_rolling_analysis, summary_table, alive_factors_by_theme
    from src.factors.factor_correlation import compute_factor_values, compute_correlation_matrix, find_correlated_pairs, theme_correlation_summary
    from src.factors.multi_factor_selector import select_factors, SelectionConfig

    # ============================================================
    # 第二步：加载因子注册表
    # ============================================================
    logger.info("-" * 60)
    logger.info("Step 1: Loading factor registry (all zoos)...")
    logger.info("-" * 60)
    # 初始化因子注册表，自动扫描并加载所有已注册的 alpha 因子
    registry = Registry()
    health = registry.health()
    logger.info("Registry: %d loaded, %d failed", health["loaded"], health["failed"])

    # 优先使用标记为 "equity_cn"（中国 A 股适用）的因子
    # 若无专用因子，退回到取前 50 个通用因子
    cn_alphas = registry.list(universe="equity_cn")
    if not cn_alphas:
        cn_alphas = registry.list()[:50]
        logger.info("No equity_cn factors, using first %d", len(cn_alphas))
    else:
        logger.info("Found %d equity_cn factors", len(cn_alphas))

    # ============================================================
    # 第三步：滚动 IC 分析（核心步骤）
    # ============================================================
    logger.info("-" * 60)
    logger.info("Step 2: Rolling IC analysis (window=60)...")
    logger.info("-" * 60)
    # 对所有候选因子执行滚动 IC 分析
    # window=60 表示使用 60 个交易日的滚动窗口
    rolling_results = batch_rolling_analysis(
        registry, panel, return_df,
        window=60, alpha_ids=cn_alphas,
    )
    logger.info("Computed rolling IC for %d factors", len(rolling_results))

    # 生成因子寿命看板（按 |IR| 降序排列）
    summary = summary_table(rolling_results)
    if not summary.empty:
        logger.info("\n%s", "=" * 60)
        logger.info("TOP 20 FACTORS BY ROLLING IR (因子寿命看板)")
        logger.info("%s", "=" * 60)
        print(summary.head(20).to_string())

    # 按主题分组当前活跃因子
    alive = alive_factors_by_theme(rolling_results)
    logger.info("\n%s", "=" * 60)
    logger.info("ALIVE FACTORS BY THEME (当前活跃因子)")
    logger.info("%s", "=" * 60)
    for theme, aids in sorted(alive.items()):
        logger.info("  %-15s: %d factors → %s", theme, len(aids), aids[:5])

    # ============================================================
    # 第四步：因子相关性矩阵分析
    # ============================================================
    if len(rolling_results) >= 2:
        logger.info("-" * 60)
        logger.info("Step 3: Factor correlation matrix...")
        logger.info("-" * 60)
        # 提取所有成功分析的因子 ID
        factor_ids = [r.alpha_id for r in rolling_results]
        # 批量计算因子值
        factor_dfs = compute_factor_values(registry, panel, factor_ids)
        logger.info("Computed values for %d factors", len(factor_dfs))

        if len(factor_dfs) >= 2:
            # 计算 N×N 截面相关矩阵（默认 Spearman 秩相关）
            corr_matrix = compute_correlation_matrix(factor_dfs)
            logger.info("Correlation matrix shape: %s", corr_matrix.shape)

            # 找出高相关因子对（|corr| >= 0.7）
            high_corr = find_correlated_pairs(corr_matrix, threshold=0.7)
            logger.info("Found %d highly correlated pairs (|corr| > 0.7)", len(high_corr))
            if high_corr:
                logger.info("\n%s", "=" * 60)
                logger.info("HIGHLY CORRELATED PAIRS (需去重)")
                logger.info("%s", "=" * 60)
                # 只显示前 15 对（避免输出过长）
                for a, b, corr in high_corr[:15]:
                    logger.info("  %s <-> %s : %.4f", a, b, corr)

            # 计算主题级相关矩阵（评估跨主题分散度）
            theme_corr = theme_correlation_summary(corr_matrix, registry)
            logger.info("\n%s", "=" * 60)
            logger.info("THEME CORRELATION MATRIX (跨 theme 分散度)")
            logger.info("%s", "=" * 60)
            print(theme_corr.round(3).to_string())

    # ============================================================
    # 第五步：多因子筛选（最终输出）
    # ============================================================
    logger.info("-" * 60)
    logger.info("Step 4: Multi-factor selection...")
    logger.info("-" * 60)
    # 配置选择器参数
    config = SelectionConfig(
        rolling_window=60,       # 滚动窗口 60 天
        max_per_theme=2,         # 每个主题最多 2 个因子
        min_rolling_ir=0.15,     # 最低 IR 门槛 0.15（分析模式下放宽）
        corr_threshold=0.7,      # 相关性去重阈值 0.7
        max_total_factors=8,     # 最终组合最多 8 个因子
    )
    # 执行多因子选择
    selection = select_factors(registry, panel, return_df, config)

    # 输出推荐的因子组合
    logger.info("\n%s", "=" * 60)
    logger.info("SELECTED FACTOR PORTFOLIO (当前推荐因子组合)")
    logger.info("%s", "=" * 60)
    # 显示匹配到的逻辑链及得分
    logger.info("Logical chain: %s (score=%.2f)", selection.logical_chain, selection.chain_score)
    # 显示覆盖的主题
    logger.info("Themes covered: %s", selection.themes_covered)
    # 显示选中的因子列表
    logger.info("Selected %d factors:", len(selection.selected))
    for aid in selection.selected:
        logger.info("  ✓ %s", aid)
    # 显示因相关性被剔除的因子
    if selection.corr_dropped:
        logger.info("Dropped (correlated): %s", selection.corr_dropped)
    # 显示死亡和衰减中的因子数量
    logger.info("Dead factors: %d | Decaying: %d",
                len(selection.dead_factors), len(selection.decaying_factors))

    # 显示选中因子的详细指标
    if not selection.details.empty:
        logger.info("\n%s", "=" * 60)
        logger.info("SELECTION DETAILS")
        logger.info("%s", "=" * 60)
        print(selection.details.to_string())

    logger.info("\n%s", "=" * 60)
    logger.info("Analysis complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
