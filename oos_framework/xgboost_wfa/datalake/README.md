# 数据湖构建（datalake）

本目录是 **上游数据湖** 的构建脚本——`xgboost_wfa` 管线读取的就是这里产出的 lake。

```
stocklake/
  daily/{code}.parquet                      # index=date, 列: open/high/low/close/volume/amount
  fundamentals/
    income_statement/{code}.parquet        # index=REPORT_DATE
    cash_flow_statement/{code}.parquet     # index=REPORT_DATE
    balance_sheet/{code}.parquet           # index=REPORT_DATE
  BUILD_DONE                                # 三大表齐全标记(驱动全量流水线启动)
```

## 脚本

| 脚本 | 作用 | 数据源 | 备注 |
|---|---|---|---|
| `build_lake.py` | 日线 OHLCV | 东财 | **已 env 化**: `STOCKWORM_LAKE`(默认 `/workspace/stocklake`) |
| `drive_stmt.py` | 利润表/现金流/资产负债表 | 东财 | 写 `BUILD_DONE` 标记; 部分日志路径硬编码 `/workspace`, 本地需改 |
| `build_cnstock_fund.py` | cnstock 基本面(参考) | cnstock | ⚠️ **IP 封禁高风险**, 见下 |
| `run_stmt_loop.sh` | 驱动 drive_stmt 的循环封装 | — | — |

## ⚠️ cnstock 接口 IP 封禁高风险

`data.cnstock.com` 在构建期被**高并发爬取触发整域 403(IP 级)**——
`stock_detail` 与 `fetch_period` **双双返回 403**, 且是整个域名(不是单接口)被封。
表现: 8 并发 ≈200 请求后即 403, 仅抓到 5562 只中的 208 只。

> **结论**: 基本面已**改回东财三大表派生**(已建 95%+), `build_cnstock_fund.py` 仅作参考保留。
> 若确需重跑 cnstock: **务必低并发(≤2)、加限速、遇到 403 立即退避**, 否则会再次被封。
> 限速与封禁退避逻辑见 stock-worm 仓库 `stcok_worm/cnstock.py` 顶部警告与 `_get()`。

## 本地构建提示

- `build_lake.py` 用 `STOCKWORM_LAKE` 环境变量指向上游 lake, 可直接改路径后运行。
- `drive_stmt.py` / `build_cnstock_fund.py` 内有少量**硬编码 `/workspace` 路径**(日志、状态、lake 根),
  本地复现前请把这些路径改成你自己的目录(或设对应环境变量)。
- 不一定要自己建 lake: 只要你的 lake **schema 一致**, 直接给 `xgboost_wfa` 设
  `STOCKLAKE=/your/lake` 即可复用现有管线。
