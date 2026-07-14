"""
数据湖 · 指数层：用 stock-worm(akshare) 拉取大盘指数日线

参考指数(用户指定以上证综指为大盘基准):
    - sh000001  上证综指   (主参考, ABC闸门牛熊判定)
    - sh000300  沪深300    (补充, 宽基基准对照)

输出: index/{code}.parquet  (与 daily 湖同 schema: date索引 + open/close/high/low/volume)
数据源: akshare (stock_zh_index_daily) —— 即 stock-worm 的 index/eastmoney 体系共用源

注: 腾讯 fqkline 端点对 sh000001 返回 param error(不认指数代码),
故走 akshare, 它是 stock-worm 生态内的数据源之一。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "D:/stcok-worm")

LAKE = Path(__file__).parent
INDEX_DIR = LAKE / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# code -> (akshare symbol, 中文名)
INDICES = {
    "sh000001": ("sh000001", "上证综指"),
    "sh000300": ("sh000300", "沪深300"),
}

START_DATE = "1991-01-01"


def fetch_one(code: str, ak_symbol: str, name: str) -> bool:
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        if df is None or df.empty:
            print(f"  [{code}] {name}: 空数据")
            return False

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "close", "high", "low", "volume"]].copy()
        df = df[df.index >= pd.Timestamp(START_DATE)]

        # 与 daily 湖同列顺序, 便于 regime 引擎统一读取
        df = df[["open", "close", "high", "low", "volume"]]
        df.to_parquet(INDEX_DIR / f"{code}.parquet")

        print(f"  [{code}] {name}: 成功 {len(df)} 行, "
              f"{df.index[0].date()} ~ {df.index[-1].date()}, "
              f"末收 {df['close'].iloc[-1]:.2f}")
        return True
    except Exception as e:
        print(f"  [{code}] {name}: 失败 {e!r}")
        return False


def main():
    print(f"指数数据湖目录: {INDEX_DIR}")
    ok = fail = 0
    for code, (ak_symbol, name) in INDICES.items():
        if fetch_one(code, ak_symbol, name):
            ok += 1
        else:
            fail += 1
    print(f"\n完成: 成功 {ok}, 失败 {fail}")


if __name__ == "__main__":
    main()
