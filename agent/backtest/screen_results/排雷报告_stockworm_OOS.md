# 排雷报告 · stock_worm 第一优先数据源 + 方正因子 OOS 初验

> 日期：2026-07-09
> 前置：stock_worm 已全量下载（40 文件）并深度阅读（见 `/workspace/stock_worm/FUNCTIONS.md`）

## 一、数据源接驳（已落地）

`stcok_worm` 现在被设为 **A股第一优先数据源**：

| 改动 | 文件 | 说明 |
|------|------|------|
| 包可导入 | `/workspace/stock_worm` → site-packages `.pth` | `import stcok_worm` 全局可用 |
| 回退链置顶 | `backtest/loaders/registry.py` | `FALLBACK_CHAINS["a_share"]` 首位改为 `stock_worm` |
| 自动识别置顶 | `src/market_data.py` | `detect_source("600519.SH")` → `stock_worm`（否则显式 source 短路会绕过回退链）|
| loader 补齐分钟级+amount | `backtest/loaders/stock_worm_loader.py` | 新增 `5m/15m/30m/1h`；保留 `amount` 列；保留完整时间戳 |
| 修复源内截断 bug | `stcok_worm/mootdx_source.py` | 原 `str(datetime)[:10]` 会把 5 分钟 K 线塌缩到同一日索引 → 改为保留完整时间 |

验证（`DataLoader().is_available()`=True，`resolve_loader("a_share").name`="stock_worm"，
`detect_source` 对 `xxxxx.SH/SZ/BJ` 均返回 `stock_worm`）。实测 end-to-end：
- 日线 `600519`：117 行，含 `amount`
- 5 分钟 `600519`：288 行，含 `amount`，索引 `2026-07-08 15:00:00`（时间戳未塌缩）

> 通达信 TCP（端口 7709）在本沙箱**可直连**，因此 5 分钟 `amount` 数据真实可取，
> 这正是方正因子 `clouds_disperse` / `rapids_advance` 此前缺失的数据。

## 二、排雷运行结果（真实 TDX 数据，经 stock_worm）

脚本：`backtest/verify_founder_factors.py`（已改为经 `stcok_worm.mootdx_source` 取 5m + `amount`）
样本：50 只沪深300 成分股，5 分钟 K 线（最近约 3–4 个交易日，mootdx 单次上限 800 根）

| 因子 | N | IC | 方向 |
|------|---|----|------|
| equal_treatment | 50 | +0.7103 | 正向 |
| smart_money | 50 | −0.8101 | 反向 |
| drip_water_stone | 50 | −0.5424 | 反向 |
| moth_to_flame | 50 | +0.5326 | 正向 |
| moderate_risk | 50 | +0.5331 | 正向 |
| scaling_heights | 50 | +0.4340 | 正向 |
| **clouds_disperse** | 50 | **+0.2252** | **正向** |
| wait_rescue | 50 | −0.0110 | 反向 |
| complete_tide | 2 | — | 数据不足（跳过）|
| rapids_advance | — | — | 数据不足（跳过）|

### 关键结论
- **`clouds_disperse` 现用真实 `amount` 计算**（IC=+0.2252，正向），不再依赖 `close*volume` 近似。
  此前缺 `amount` 时该因子只能近似，现已补齐 → 排雷通过（数据链路成立）。
- **`rapids_advance` 在 5 分钟快照下无法验证**：其 `rolling(20)` 日频窗口需 ≥20 个交易日分钟历史，
  而 mootdx 5 分钟仅约 3–4 天 → 因子序列全 NaN，被自动跳过。这是**样本长度限制，非因子错误**。

## 三、已知限制（诚实声明）
1. **样本太短**：mootdx 5 分钟单次仅 ~800 根（≈3–4 交易日），上述 IC 为**单截面初验**，非完整时序 OOS。
   完整 OOS 需 20+ 交易日分钟历史（TDX 需多次翻页拼接，或改用日频 resample）。
2. **IC 方向仅作方向性参考**：小样本下绝对值不稳定，不能直接作为上线依据。

## 四、GitHub 上传状态（待处理）
- 远端：`origin = github.com/zxs007000/Vibe-Trading-Ashare.git`（已确认是本仓库）。
- **当前沙箱 TLS 被黑洞**（github.com → 198.18.0.4，`git ls-remote` 静默失败），**`git push` 无法执行**。
- 已就绪、待推送的改动（本地 `main` 分支，未提交）：
  - `agent/backtest/loaders/registry.py`（stock_worm 置顶）
  - `agent/backtest/loaders/stock_worm_loader.py`（分钟级 + amount）
  - `agent/src/market_data.py`（detect_source 置顶）
  - `agent/backtest/verify_founder_factors.py`（改走 stock_worm + amount）
  - `agent/api_server.py`、`agent/src/preflight.py`（早前 API_SKIP_PREFLIGHT 修复）
  - 新增：`agent/backtest/screen_results/founder_oos_stockworm.txt`（本次 OOS 原始输出）
- **建议**：在具备网络的环境用 PAT 执行 `git add -A && git commit && git push` 即可上传；
  或授权后由本代理在可联网会话中重跑上传。

## 五、下一步
1. 补 `rapids_advance` 的完整 OOS：拼接 ≥20 交易日 5 分钟历史（mootdx `get_kline_history` 翻页），
   或用日频 resample 近似分钟级成交额。
2. 在可联网环境完成 GitHub 推送。
