"""质量因子：应计利润（accruals）反向。

数据源：financial_loader，accruals = (净利润 − 经营现金流量净额) / 股东权益合计。
应计利润越高，说明利润越靠非现金项目堆出来、盈余质量越差；故取负号，
使「数值越高=质量越好」，与 ROE 同方向。

约定：面板需含 extras['accruals']（与 close 同形，报告期前向填充）。
"""

__alpha_meta__ = {
    "id": "fundamental_quality_accrual",
    "nickname": "应计利润反向质量因子",
    "theme": ["quality"],
    "formula_latex": r"-accruals_t",
    "columns_required": ["close"],
    "extras_required": ["accruals"],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 0,
    "min_warmup_bars": 0,
    "notes": "accruals 越高盈余质量越差，故取负。需财务数据源(financial_loader)。",
}


def compute(panel: dict) -> "pd.DataFrame":
    return -panel["accruals"].astype(float)
