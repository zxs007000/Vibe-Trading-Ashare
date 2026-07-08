"""价值因子：账面市值比（inverse P/B，基于财务数据源的每股净资产）。

数据源：financial_loader 提供 bvps（每股净资产），与价格比得到 P/B 的倒数，
即「账面市值比」。越高=越便宜（价值越深）。

这把原 value 主题从仅 1 个量价反转因子（vs_ma250）扩到含基本面价值的维度，
与量价反转互补（一个是价格相对年线，一个是价格相对账面价值）。

约定：面板需含 extras['bvps'] 与 columns['close']（同形）。
"""

from __future__ import annotations

import numpy as np

__alpha_meta__ = {
    "id": "fundamental_value_pb_inv",
    "nickname": "账面市值比(反向P/B)",
    "theme": ["value"],
    "formula_latex": r"\frac{bvps_t}{close_t}",
    "columns_required": ["close"],
    "extras_required": ["bvps"],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 0,
    "min_warmup_bars": 0,
    "notes": "bvps/close = 反向P/B。需财务数据源(financial_loader)。越高=越便宜。",
}


def compute(panel: dict) -> "pd.DataFrame":
    close = panel["close"].astype(float)
    bvps = panel["bvps"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = bvps / close.replace(0, np.nan)
    return out
