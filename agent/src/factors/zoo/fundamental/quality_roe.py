"""质量因子：ROE（净资产收益率）。

数据源：financial_loader（akshare/东方财富，无 token），按报告期前向填充到交易日。
解决战略瓶颈：原 quality 主题仅 3 个因子、value 仅 1 个；本因子把财务质量维度
从 0 扩到真实 ROE 信号。

约定：面板需含 extras['roe']（与 close 同形的 DataFrame，报告期前向填充）。
compute 返回与 close 同形的 ROE 面板（数值越高=质量越好）。
"""

__alpha_meta__ = {
    "id": "fundamental_quality_roe",
    "nickname": "ROE 质量因子",
    "theme": ["quality"],
    "formula_latex": r"roe_t",
    "columns_required": ["close"],
    "extras_required": ["roe"],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 0,
    "min_warmup_bars": 0,
    "notes": "ROE 来自财务数据源(financial_loader)，按报告期前向填充。越高=质量越好。",
}


def compute(panel: dict) -> "pd.DataFrame":
    return panel["roe"].astype(float)
