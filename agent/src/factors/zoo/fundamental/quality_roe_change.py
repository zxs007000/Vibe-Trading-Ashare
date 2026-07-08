"""质量因子：ROE 季度环比变化率。

逻辑：ROE 单季度变化（ROE_t - ROE_t-1）捕捉盈利能力的趋势转折。
传统 ROE 水平值被行业/生命周期"固定"在某个区间，变化率更能反映
边际改善/恶化——尤其在 A 股，ROE 加速上行通常领先股价。

数据源：financial_loader (stock_worm → akshare)，取最近 2 期报告。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    "id": "fundamental_quality_roe_change",
    "nickname": "ROE 季度变化率因子",
    "theme": ["quality"],
    "formula_latex": r"\Delta ROE = \frac{ROE_t - ROE_{t-1}}{|ROE_{t-1}|}",
    "columns_required": ["close"],
    "extras_required": ["roe"],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 0,
    "min_warmup_bars": 0,
    "notes": (
        "ROE 环比变化率，正数=盈利改善加速，负数=盈利恶化。"
        "信号通过 panel['roe'] 注入，已在加载层按报告期前向填充。"
    ),
}


def compute(panel: dict) -> pd.DataFrame:
    """计算 ROE 季度变化率。

    Args:
        panel: 含 "close" 和 extras "roe"（报告期前向填充的 ROE 历史）。

    Returns:
        与 close 同形的 ROE 季度变化率 DataFrame。
    """
    close = panel["close"]
    roe = panel["roe"].astype(float)

    # ROE 季度环比：当期 vs 前一期（shift 约 63 个交易日 ≈ 1 季）
    roe_change = roe.diff(periods=1) / roe.shift(1).abs().replace(0, np.nan)

    # winsorize: clip 到 [-1, 1]（ROE 单季变化不可能超出 100%）
    roe_change = roe_change.clip(-1, 1)

    # 对齐 close 的 index/columns
    result = roe_change.reindex(index=close.index, columns=close.columns)
    return result.fillna(0)
