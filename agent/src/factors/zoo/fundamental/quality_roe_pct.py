"""质量因子：ROE 截面分位（行业中性化）。

逻辑：每期将全部股票的 ROE 做截面排名（0~1 分位），
高分位=行业内盈利能力领先，低分位=落后。

相比于原始 ROE 水平值，截面分位消除了系统性偏差
（如银行天然 ROE 低、消费天然 ROE 高），更公平地横比。

数据源：financial_loader (stock_worm → akshare)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    "id": "fundamental_quality_roe_pct",
    "nickname": "ROE 截面分位因子",
    "theme": ["quality"],
    "formula_latex": r"ROE\_pct = \frac{rank(ROE)}{N-1}",
    "columns_required": ["close"],
    "extras_required": ["roe"],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 0,
    "min_warmup_bars": 0,
    "notes": (
        "ROE 在截面上的百分位排名（0~1）。"
        "消除行业偏差，高分位=行业内盈利领先。"
    ),
}


def compute(panel: dict) -> pd.DataFrame:
    """计算 ROE 截面分位。

    Args:
        panel: 含 "close" 和 extras "roe"。

    Returns:
        与 close 同形的 DataFrame，值范围 [0, 1]。
    """
    close = panel["close"]
    roe = panel["roe"].astype(float)

    # 逐日计算截面排名（0~1 分位）
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for i, dt in enumerate(close.index):
        row = roe.loc[dt].dropna()
        if len(row) < 3:
            continue
        ranks = row.rank(pct=True)  # 0~1
        for col in close.columns:
            if col in ranks.index:
                result.loc[dt, col] = ranks[col]

    return result.fillna(0.5)  # 中性填 0.5
