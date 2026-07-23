#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 数据底座
================================
从 /workspace/stocklake 加载「日期 × 股票」面板，供算子库 / 网格搜索 / 遗传规划 / MCTS 共用。

设计要点（承接 8GB cgroup 实战经验）:
- 单字段面板(date×stock, float32) 在 5448 只全市场下仅 ~42MB，远小于 XGBoost 的 218 维特征矩阵，
  因此因子挖掘阶段可放心加载较大样本（默认 400 只，全市场亦可）。
- 读取采用「逐只读列 → 拼 DataFrame → 按日期对齐」的方式，避免一次性读入整库。
"""
from __future__ import annotations
import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # 抑制 ts_std 在常值窗口产生的 NaN 噪声告警

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = os.path.join(LAKE, "daily")
TURNOVER_DIR = os.path.join(LAKE, "turnover")            # B1: akshare 准确换手率层
FLOAT_SHARES_PANEL = os.path.join(LAKE, "float_shares_panel.parquet")  # 流通股本快照

# 数据湖 daily 层提供的原始基础变量（算子库即围绕它们组合）
BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]
# 默认回测起始（与全市场 WFA 一致）
DEFAULT_START = "2018-10-23"


def list_stocks(limit: int | None = None) -> list[str]:
    """列出数据湖中可用的股票代码（按文件名排序）。"""
    fs = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    codes = [os.path.basename(f)[:6] for f in fs]  # 000001.parquet -> 000001
    if limit:
        codes = codes[:limit]
    return codes


def load_panel(field: str, codes: list[str] | None = None, start: str | None = DEFAULT_START,
               end: str | None = None, dtype=np.float32) -> pd.DataFrame:
    """
    加载某字段的「日期 × 股票」面板。

    参数
    ----
    field : 'open'|'high'|'low'|'close'|'volume'|'amount'
    codes : 股票代码列表；None 表示全部
    start/end : 'YYYY-MM-DD' 字符串过滤
    """
    if field not in BASE_FIELDS:
        raise ValueError(f"field 必须是 {BASE_FIELDS} 之一, 收到 {field!r}")
    if codes is None:
        codes = list_stocks()
    series = {}
    for c in codes:
        f = os.path.join(DAILY, f"{c}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        # 兼容两种存储: date 作列(cloud stocklake) / date 作索引(本地 Claw/stockworm)
        if "date" not in df.columns:
            df = df.reset_index()
        s = df.set_index("date")[field]
        s.index = pd.to_datetime(s.index)  # 统一为 Timestamp, 避免 date/Timestamp 混用
        series[c] = s
    if not series:
        raise RuntimeError("没有任何股票数据被加载, 检查 STOCKLAKE 路径")
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel[panel.index <= pd.Timestamp(end)]
    return panel.astype(dtype)


def load_base_data(codes: list[str] | None = None, fields: list[str] | None = None,
                   start: str | None = DEFAULT_START, end: str | None = None) -> dict[str, pd.DataFrame]:
    """一次性加载多个基础变量面板, 返回 {field: panel}。"""
    fields = fields or BASE_FIELDS
    return {f: load_panel(f, codes=codes, start=start, end=end) for f in fields}


def turnover_available() -> bool:
    """B1 换手率湖是否已构建(至少有部分文件)。"""
    return os.path.isdir(TURNOVER_DIR) and len(os.listdir(TURNOVER_DIR)) > 0


def load_turnover(codes: list[str] | None = None, start: str | None = DEFAULT_START,
                  end: str | None = None, dtype=np.float32) -> pd.DataFrame:
    """
    加载 B1 层的「准确历史换手率」面板 (date×stock, 单位 %)。

    数据来自 akshare stock_zh_a_hist, 自建湖 turnover/<code>.parquet (date 索引 + 单列 turnover)。
    对限售股解禁免疫(交易所披露的真实当日换手率), 与 daily 湖 OHLCV 按交易日对齐后可直接用于
    筹码结构特征。缺失代码的文件会被跳过。
    """
    if not turnover_available():
        raise RuntimeError("turnover 湖不存在, 请先运行 build_turnover_lake.py")
    if codes is None:
        codes = list_stocks()
    series = {}
    for c in codes:
        f = os.path.join(TURNOVER_DIR, f"{c}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        if "date" not in df.columns:
            df = df.reset_index()
        # 兼容 akshare 导出时索引名为 日期 的情况
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
        s = df.set_index("date")["turnover"]
        s.index = pd.to_datetime(s.index)
        series[c] = s
    if not series:
        raise RuntimeError("没有任何换手率数据被加载, 检查 TURNOVER_DIR 路径")
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel[panel.index <= pd.Timestamp(end)]
    return panel.astype(dtype)


def load_float_shares() -> pd.DataFrame | None:
    """
    加载流通股本快照面板 (code 索引, 列 float_shares/total_shares/float_cap)。

    由 build_float_shares.py 用 stockworm company_info 逐股抓取落盘。缺失返回 None。
    仅作横截面规模特征用; 筹码结构特征内部的「逐日流通股本」由换手率反推(见 chip_structure),
    对解禁免疫, 不依赖此快照。
    """
    if not os.path.exists(FLOAT_SHARES_PANEL):
        return None
    df = pd.read_parquet(FLOAT_SHARES_PANEL)
    df.index = df.index.astype(str).str[:6]
    return df


CHIP_DIR = os.path.join(LAKE, "chip_panels")


def chip_lake_available() -> bool:
    """chip_panels 湖(build_chip_lake.py 落盘)是否存在。"""
    return os.path.isdir(CHIP_DIR) and len(glob.glob(os.path.join(CHIP_DIR, "*.parquet"))) > 0


def load_chip_panels_from_lake(codes: list[str] | None = None, start: str | None = DEFAULT_START,
                               end: str | None = None) -> dict[str, pd.DataFrame]:
    """
    从 chip_panels 湖读取筹码结构面板 {field: date×stock}。
    比 build_chip_panels 现算快两个数量级; 湖缺失时抛 RuntimeError。
    """
    if not chip_lake_available():
        raise RuntimeError("chip_panels 湖不存在, 先运行 build_chip_lake.py")
    out = {}
    for f in sorted(glob.glob(os.path.join(CHIP_DIR, "*.parquet"))):
        k = os.path.basename(f)[:-8]
        p = pd.read_parquet(f)
        p.index = pd.to_datetime(p.index)
        if codes is not None:
            cols = [c for c in codes if c in p.columns]
            p = p[cols]
        if start:
            p = p[p.index >= pd.Timestamp(start)]
        if end:
            p = p[p.index <= pd.Timestamp(end)]
        out[k] = p.astype(np.float32)
    return out


def forward_returns(close_panel: pd.DataFrame, horizons=(5, 20, 60)) -> dict[int, pd.DataFrame]:
    """
    计算各 horizon 的向前收益率面板（t 时刻持有 h 日的收益）。
    因子在 t 日的值用于预测该向前收益, 是 IC 计算的标准目标。
    """
    out = {}
    for h in horizons:
        out[h] = close_panel.shift(-h) / close_panel - 1.0
    return out


def derive_variables(data: dict[str, pd.DataFrame], include_chip: bool = True) -> dict[str, pd.DataFrame]:
    """
    从 6 个基础字段派生标准 alpha 变量池(对应报告「将 218 个特征作为原始变量」的思路)。
    返回 {原名: 面板} ∪ {派生名: 面板}, 派生变量包括:
      收益类   : ret_1/5/10/20, log_ret_1
      波动类   : vol_20(已实现波动), amp_1(日振幅), range_20(20日波幅)
      量价类   : vrate_5/20(量比), arate_5(额比), dist_high_20/dist_low_20(距高低点)
      筹码类   : chip_profit_ratio/chip_cost_dev/chip_conc70/chip_conc90/chip_disp/chip_skew
                 (include_chip=True 且 chip_panels 湖存在时自动并入, 按 close 面板对齐)
    这些变量叠加时序/截面算子后, 才能产生有截面区分度的因子(纯价格水位 IC≈0)。
    """
    out = dict(data)
    close, high, low, open_, volume, amount = (
        data["close"], data["high"], data["low"], data["open"], data["volume"], data["amount"])
    r1 = close.pct_change(1)
    out["ret_1"] = r1
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    out["log_ret_1"] = np.log(close / close.shift(1))
    out["vol_20"] = r1.rolling(20, min_periods=10).std()
    out["amp_1"] = (high - low) / close
    out["range_20"] = (high.rolling(20, min_periods=10).max() - low.rolling(20, min_periods=10).min()) / close
    out["vrate_5"] = volume / volume.rolling(5, min_periods=3).mean()
    out["vrate_20"] = volume / volume.rolling(20, min_periods=10).mean()
    out["arate_5"] = amount / amount.rolling(5, min_periods=3).mean()
    out["dist_high_20"] = close / close.rolling(20, min_periods=10).max()
    out["dist_low_20"] = close / close.rolling(20, min_periods=10).min()

    # ---- 筹码结构维度 (用户拍板: chip 进变量池搜因子) ----
    if include_chip and chip_lake_available():
        try:
            chips = load_chip_panels_from_lake(
                codes=list(close.columns),
                start=str(close.index[0].date()), end=str(close.index[-1].date()))
            for k, p in chips.items():
                out[k] = p.reindex(index=close.index, columns=close.columns).astype(np.float32)
        except Exception as e:
            print(f"[derive] chip 湖读取失败, 跳过筹码维度: {e}")
    return out


if __name__ == "__main__":
    codes = list_stocks(400)
    data = load_base_data(codes)
    fwd = forward_returns(data["close"])
    print(f"样本: {len(codes)} 只, 交易日: {len(data['close'])}")
    print(f"面板形状: {data['close'].shape}, 向前收益(20日)非空率: {1-fwd[20].isna().mean().mean():.3f}")
