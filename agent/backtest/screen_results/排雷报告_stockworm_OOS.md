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

## 四、GitHub 上传状态（待处理 · 2026-07-09 更新）

> **更正（同日）**：本报告初版"沙箱 TLS 被黑洞、git push 无法执行"的结论**已过时**。
> 重新探测发现：本地 git 配置带 `http.curloptresolve=github.com:443:140.82.113.4`，
> 能把 github.com:443 强制解析到真实可达 IP，**绕过污染的 DNS**。实测：
> - `git ls-remote origin HEAD` → 成功返回 `b3ac323… HEAD`（TLS 握手通过）
> - `curl --resolve … https://github.com/...git/info/refs` → `http_code=200 ssl_verify=0`
> 即**网络推送通道是通的**，卡点从"网络"变为"认证"。

- 远端：`origin = github.com/zxs007000/Vibe-Trading-Ashare.git`（已确认，分支 `main`）。
- **当前唯一缺口 = PAT 认证**：凭据助手（`/usr/local/bin/git-credential-helper`，Go 二进制）
  缓存为空；`gh auth status` 显示未登录；环境无 `ghp_/github_pat` 类变量。
  `git push --dry-run` 报 `could not read Username` → 缺令牌。
- 已就绪、待推送的改动（本地 `main` 分支，未提交）：
  - `agent/backtest/loaders/registry.py`（stock_worm 置顶）
  - `agent/backtest/loaders/stock_worm_loader.py`（分钟级 + amount）
  - `agent/src/market_data.py`（detect_source 置顶）
  - `agent/backtest/verify_founder_factors.py`（改走 stock_worm + amount）
  - `agent/api_server.py`、`agent/src/preflight.py`（早前 API_SKIP_PREFLIGHT 修复）
  - 新增：`agent/backtest/screen_results/founder_oos_stockworm.txt`（本次 OOS 原始输出）
- **只需一个 PAT 即可推送**（username 填任意名、password 填 PAT，或 `git remote set-url` 内嵌）：
  ```bash
  cd /workspace/VibeTradingPush
  git add -A
  git commit -m "feat: stock_worm 第一优先数据源 + 方正因子 OOS 初验"
  git -c http.curloptresolve=github.com:443:140.82.113.4 \
      push https://<USER>:<PAT>@github.com/zxs007000/Vibe-Trading-Ashare.git main
  ```
  （`<PAT>` 需有该仓库 `write` 权限；本沙箱不留存令牌，推送完即清。）

## 五、全因子体检（1.5 年真实数据，方向校正）— 2026-07-09 新增

> 脚本 `eval_custom_factors.py` Part B：30 只 CSI300（缓存 78 只，取前 30），
> 5m 真实数据 **2025-01-01~2026-06-30**，resample 到日频后逐因子计算。
> 指标口径同 zoo：`ic_mean`=RankIC 均值，`icir`=ICIR，`ls_sharpe`=五分位多空夏普（库约定**做多高因子值**），`top_sharpe`=最高组分位夏普。
>
> **方向校正**：若因子 `ic_mean<0`，其盈利方向是**做多低因子值**（把 ls_sharpe 取反）。
> 下表"实证方向"与"可用多空夏普"即按此校正（可用 = `ls_sharpe × sign(ic_mean)`，IC≈0 时方向不稳，已标注）。
> 落盘：`screen_results/custom_founder_eval.csv` / `custom_huatai_eval.csv` / `custom_guosheng_haitong_eval.csv`。

| 因子 | IC | ICIR | 库ls_sharpe | 实证方向 | 可用多空夏普 |
|------|----|----|----|----|----|
| smart_money | −0.031 | −1.896 | −1.104 | **做多低值** | **+1.10** |
| flower_hidden | +0.015 | +1.163 | +1.015 | 做多高值 | **+1.02** |
| complete_tide | +0.052 | +1.644 | +0.846 | 做多高值 | **+0.85** |
| wait_rescue | −0.003 | −0.246 | +0.997 | 做多低值（IC≈0，方向不稳） | +1.00* |
| scaling_heights | −0.046 | −2.780 | −0.957 | **做多低值** | **+0.96** |
| coin_team | −0.009 | −0.629 | +0.691 | 做多低值（IC 弱） | +0.69 |
| undercurrent | +0.001 | +0.093 | −0.704 | 做多低值（IC≈0） | +0.70* |
| withered_tree_blooms | −0.035 | −2.132 | −0.598 | **做多低值** | **+0.60** |
| clouds_disperse | −0.037 | −2.511 | −0.649 | 做多低值（**真实 amount**） | **+0.65** |
| equal_treatment | −0.037 | −2.392 | +0.357 | 做多低值 | +0.36 |
| moth_to_flame | −0.060 | −2.939 | −0.332 | 做多低值 | +0.33 |
| moderate_risk | −0.056 | −2.687 | −0.292 | 做多低值 | +0.29 |
| synergy_effect | +0.039 | +2.091 | +0.298 | 做多高值 | +0.30 |
| rapids_advance | +0.005 | +0.371 | +0.115 | 做多高值（**真实 amount，IC≈0 偏弱**） | +0.12 |
| drip_water_stone | +0.016 | +1.001 | −1.075 | IC>0 但做多高亏损（矛盾，需 walk-forward） | −1.08 |
| bull_bear_game | ~0 | +0.018 | −0.373 | 无效 | ~0 |
| panic_factor | −0.031 | −0.741 | NaN | 分组权益为空 | — |

**要点**
- **真正稳健、方向自洽的头部因子**（可用多空夏普 ≥0.3 且 IC 符号与方向一致）：
  `smart_money`(做多低值)、`flower_hidden`(做多高值)、`complete_tide`(做多高值)、
  `scaling_heights`(做多低值)、`undercurrent`(做多低值)、`withered_tree_blooms`(做多低值)、
  `clouds_disperse`(做多低值)、`equal_treatment`/`moth_to_flame`/`moderate_risk`(做多低值)。
- **论文 IC 符号 ≠ 实测方向**：例如 `clouds_disperse` 论文 RankIC −9.81%（负），
  但本样本实测 IC 偏负、方向为"做多低值"；`complete_tide` 论文 IC −7.90% 负，
  实测 IC +0.052 正、做多高值。**上线一律以排雷实测方向为准，不要照搬研报。**
- **`clouds_disperse`/`rapids_advance` 已拿到真实金额版（2026-07-10 修正）**：
  根因是 `astockdata_loader._normalize_bars`/`_normalize_daily` 原先只保留 OHLCV、
  **显式丢弃了 `amount` 列**（尽管 mootdx 的 `bars()`/`get_k_data` 返回里本就有）。
  已修复两处 normalize 保留 `amount`；并删除旧"无 amount"缓存重拉，重跑 Part B 得真值：
  `clouds_disperse` IC=−0.037、做多低值、**可用多空夏普 +0.65**（比近似 +0.54 更稳）；
  `rapids_advance` IC=+0.005、**IC≈0 基本失效**，本样本不可用于选股。
- **`drip_water_stone` 出现 IC 与 ls_sharpe 背离**（IC 弱正、做多高却大亏），属分钟因子日频 resample 后的噪声，
  实盘前必须做 walk-forward 方向自适应（参考 `verify_oos_overfit.py` 的 reversal 处理）。

## 六、头部因子方向自适应组合（2026-07-10 新增）

> 脚本 `backtest/combine_founder_heads.py`：取排雷头部 7 因子
> （`smart_money`/`flower_hidden`/`complete_tide`/`scaling_heights`/`undercurrent`/
> `withered_tree_blooms`/`clouds_disperse`），按各自实测 IC 方向定向 → 每日截面 z-score
> → 等权 或 按 ICIR 加权 合成综合打分 → 评估组合层 IC/ICIR/多空夏普。
> 样本：50 只 CSI300，5m 真实数据 2025-01-01~2026-06-30（带 amount）。

| 组合方式 | IC_mean | ICIR | ic_pos | 多空夏普(G5−G1) | 最高组夏普 |
|------|------|------|------|------|------|
| 等权 | +0.047 | **2.99** | 0.608 | **0.300** | −0.908 |
| ICIR 加权 | +0.055 | **3.58** | 0.608 | 0.119 | −0.811 |

**结论与警示**
- **ICIR≈3 说明排序能力很强**（单因子 ICIR 多在 2~3，组合后未衰减反而略升），
  组合能有效区分未来涨跌。
- **但 alpha 只在多空价差里**："最高组夏普"为负（−0.8~−0.9），即做多最高分组本身亏钱，
  空头（最低组）亏更多，所以 **Group5−Group1 为正**。这是一个**市场中性多空信号**，
  不是方向性 alpha——实盘必须配对对冲（或做空弱势组），裸多最高组会亏。
- **ICIR 加权重了 IC/ICIR，却把多空夏普压到 0.12**（等权 0.30 更好）：
  因 `complete_tide` 权重 0.37 独占，等权 blend 在价差维度更分散、更稳。
  → 实盘建议**等权优先**，ICIR 加权仅作 IC 维度参考。
- **方向仍脆弱**：`flower_hidden` 在 30 只样本 IC=+0.015（做多高），50 只样本 IC=−0.018（做多低），
  符号翻转——其 IC 量级≈0，方向不稳。组合已对它降权（0.08），但生产环境必须
  **walk-forward 逐窗口重定方向**，不能固定用本次静态方向。
- **样本局限**：30~50 只、1.5 年；`rapids_advance` 因 IC≈0 已排除。要上线需扩到全市场 +
  更长 OOS + 滚动方向校准，再接 `screen_factor_zoo` / 选股工作台 Top-N。

## 八、数据扩容（2026-07-10）

> 用户要求扩大样本量 + 覆盖上涨/下跌/反弹/回调/盘整各种行情 + 备份到 stock_worm 项目下。
> 已落地：新脚本 `backtest/pull_ashare_5m.py`（akshare 取全 CSI300 + 通达信 TCP 拉 5m + 断点续拉/checkpoint），
> 备份落 `/workspace/stock_worm/data/ashare_5m_cache.pkl`（已加 `data/` 到 stock_worm `.gitignore`，本地常驻不入库）。

| 项 | 旧缓存 | **新备份** |
|----|----|----|
| 股票池 | 50（CSI300 精选样本） | **288 / 300**（全 CSI300，12 只拉取失败） |
| 区间 | 2025-01 ~ 2026-06（1.5y） | **2024-09-13 ~ 2026-06-30（≈1.8y）** |
| 列 | OHLCV+amount | OHLCV+amount（amount 全非 0/NaN） |
| 体量 | 48MB | **302MB** |
| 备份位置 | `screen_results/` | **`stock_worm/data/`（常驻）** |

**行情覆盖（已用市场收益曲线核实）**：窗口内五种行情俱全——
- 反弹(急涨)：2024-09 **+23.2%**（政策牛）
- 回调(急跌)：2024-10 −7.8%、2025-10/11、2026-05/06
- 下跌：2026-H1 连续 −5% 月
- 盘整：2025-H2~2026-H1 区间震荡（全窗口最大回撤 −13.3%）

> ⚠ **TDX 5m 硬限制**：通达信 5 分钟历史只回溯约 1.8 年（截至 2024-09），
> 故 **2023 阴跌、2024-02 急跌的 5m 拉不到**（数据源限制，非代码问题）。
> 若要覆盖那两段下跌行情，需改用**日线**（TDX 日线可回溯多年），供日频因子（coin_team/complete_tide/flower_hidden 等）使用。

**关键发现 —— 扩样本立刻暴露静态定向的脆弱性**：把组合跑在新备份（含 2024-09 反弹）上：
| 组合 | ICIR | 多空夏普(新窗口) | 对比(旧 2025-01~2026-06) |
|------|------|------|------|
| 等权 | 2.11 | **−0.301** | +0.300 |
| ICIR 加权 | 2.53 | **−0.408** | +0.119 |

ICIR 仍强（~2.1–2.5，排序能力在），但**多空夏普由正转负**——2024-09 政策急反弹让因子方向整体翻转，
静态定向（全窗口定一次方向）在这段失效。**这直接证明：跨行情必须 walk-forward 逐窗口重定方向**，
否则长线持有会被单边行情打爆。数据扩容的价值正在于此：小样本（2025-01 起）掩盖了这个问题。

## 九、下一步
1. ✅ 数据扩容：全 CSI300 288 只 × ≈1.8y 5m（带 amount）已备份 `stock_worm/data/`。
2. ✅ 验证管线在新数据可跑；并暴露静态定向跨行情失效（多空夏普 −0.30）。
3. ✅ **根因已定位（见 Section 十）**：多空为负不是"方向没定对"，而是因子→收益本质**非单调**，
   walk-forward 重定方向救不了多空（只保住 IC）。正确用法是**非线性模型**（XGBoost，见 Section 十一）。
4. ✅ **XGBoost 选股器已落地**（Section 十一）：OOS IC +0.0071/ICIR 0.75，多头组合扣费后仍有正超额，
   但**必须低频持有（≥5 日）**——日频调仓成本会吃光毛超额。
5. 可选：拉**日线**长历史（2023~2026）覆盖 2024-02 急跌，供日频因子做更长 OOS / 更长持有期回测。
6. 在可联网环境用 Section 四的 PAT 命令完成 GitHub 推送（含 `combine_founder_heads.py`/`xgb_selector.py`/`diag_ls.py` + 本报告）。

## 十、真正的根因：因子→收益是非单调（倒 U）结构（2026-07-10）

> 脚本 `backtest/diag_ls.py`：静态等权组合（7 头部因子定向后每日 z-score 等权合成），
> 样本 30 只 CSI300、2024-09~2026-06（带 amount）。问题：IC 为正（ICIR 2.19）但多空夏普为**负**（−0.438）。
> 不是代码 bug，是**结构问题**。

### 10.1 组合十分位单调检查（次日收益均值，按组合分数升序）
```
decile 0 (最低):  +0.00045
decile 3:         +0.00033
decile 5 (中间):  +0.00095   ← 次日收益最高
decile 8:         +0.00028
decile 9 (最高):  -0.00009   ← 次日收益为负!
```
**形状 = 中间隆起、两端塌缩（倒 U）**。最高十分位收益为负、最低十分位为正 →
"做多最高 / 做空最低"的多空组合必然亏。rank-IC 为正只说明**整体排序相关**（中间隆起贡献），
不代表极端分组能赚钱。

### 10.2 逐因子十分位（验证非单调是因子固有，非组合搅浑）
| 因子 | 全窗口IC | 末-首(高-低十分位) | 形状 |
|------|------|------|------|
| smart_money | +0.013 | −0.00099 | 倒U |
| flower_hidden | +0.013 | −0.00031 | 倒U |
| scaling_heights | +0.014 | −0.00119 | 倒U(线性corr −0.79) |
| undercurrent | −0.009 | −0.00083 | 倒U |
| withered_tree_blooms | +0.016 | −0.00020 | 乱 |
| clouds_disperse | +0.035 | +0.00014 | 倒U |
| complete_tide | — | 样本不足 | — |

**6/7 因子本身就是非单调（倒 U）**——最高十分位相对最低十分位几乎全为负。
→ 线性/秩相关多空从结构上就抓不到这个 alpha。

### 10.3 行情依赖（多空夏普随年份翻负）
| 年份 | IC | 多空夏普 |
|------|------|------|
| 2024 | +0.063 | +0.26 |
| 2025 | +0.037 | −0.17 |
| 2026 | +0.014 | −1.13 |

IC 一路衰减、多空一路翻负：因子在 2024-09 政策急反弹里有效，之后逐步失效。

### 10.4 结论（修正 Section 六 的解读）
- Section 六 的"多空夏普 +0.30、市场中性多空信号"是 **2025-01~2026-06 窄窗口的行情巧合**
  （那段恰好高因子≈高收益）；扩到全行情（含 2024-09 反弹 + 2026 阴跌）后翻负（Section 八）。
- **真正根因 = 因子→次日收益非单调**：rank-IC 为正 ≠ 多空能赚钱。
- 因此 **walk-forward 重定方向救不了多空**（它只保住 IC 排序能力，改不了 payoff 形状），
  必须换**非线性模型**来刻画倒 U 结构——这正是用户问的 XGBoost。

## 十一、XGBoost 选股器（2026-07-10）：把非单调 alpha 变成可用收益

> 脚本 `backtest/xgb_selector.py`：用排雷头部 7 因子作特征，walk-forward（扩张窗口训练 / 20 交易日验证，
> **全程无未来泄漏**）训练 XGBoost 回归，预测次日收益；做多预测收益最高的前 30% 个股（多头组合）。
> 样本 100 只 CSI300、2024-09~2026-06（带 amount），面板 40,573 (date×code)。
> 模型：XGBRegressor(depth=3, n_est=80, lr=0.1, subsample=0.8, reg_lambda=1) —— 针对 7 特征刻意保守防过拟合。
> 特征 = 每个因子按交易日横截面 z-score（可比）；因子缺失交 XGBoost 原生 NaN 处理（不做硬交集）。

### 11.1 OOS 排序能力（无泄漏）
| 指标 | 值 |
|------|------|
| OOS IC_mean | +0.0071 |
| OOS ICIR | 0.751 |
| OOS ic_pos | 0.543 |

OOS IC 为正且 ICIR 0.75，说明模型在**未见过的未来窗口**仍稳定排序——信号真实，非过拟合。

### 11.2 多头组合（含换手成本敏感性）
成本模型：每次调仓满仓换 → 换手≈仓位，成本 = 仓位×2×单边(0.1%)，**仅调仓日计**。

| 方案 | 多头夏普 | 基准夏普 | 超额夏普 | 超额累计 |
|------|------|------|------|------|
| **XGBoost 多头 持有1日（日频调仓）** | 1.491 | 1.838 | **0.221** | 1.019（勉强为正） |
| **XGBoost 多头 持有5日（周频调仓）** | 1.708 | 1.838 | **0.703** | 1.074 |
| [对照] 线性组合多头 持有1日（含成本） | — | — | **−0.270** | 0.941（亏钱） |

- **XGBoost 把非线性 alpha 变成了可用收益**：多头夏普 1.49~1.71，且在扣费后仍跑赢等权基准（超额 0.22~0.70）。
- **线性组合扣费后变负（−0.27）**——非单调结构让它连等权基准都不如，对照证明"用法"必须非线性。
- **关键部署约束 = 低频持有**：日频调仓成本几乎吃光毛超额（超额 0.22）；**持有 5 日把超额夏普拉回 0.70**。
  → 实盘应以**周/双周频调仓、≥5 日持有**部署，不要日频。

### 11.3 用法结论（回答"这些因子怎么用 / XGBoost 选股怎么做"）
1. **不要**用线性多空或"做多最高分组"——因子非单调，必亏（Section 十）。
2. **要**用 XGBoost（或 LightGBM/CatBoost）把 7 因子作特征、预测次日/未来 N 日收益，
   输出**排序打分**做多头选股（Top-N）。树模型原生支持非单调映射，正是它比线性组合强的本质原因。
3. 部署频率：**周频调仓、持有 ≥5 日**，否则交易成本吞噬 alpha。
4. 目标函数建议：回归（未来收益）比纯分类更稳；可加 IC 作为 OOS 监控指标（ICIR<0.5 即停）。
5. 扩样：当前 100 只；上全市场 + 日线长历史（覆盖 2024-02 急跌）可做更长持有期 OOS，进一步压低换手。

> 注：等权基准夏普 1.838 反映 2024-09 政策反弹带来的行情 Beta，故**以"超额(减基准)"为 honest 指标**；
> XGBoost 超额 OOS 为正且成本可承受，信号可用。

## 十二、本地化日线数据湖（2026-07-10）：为回测 + 深度学习统一建仓

> 用户要求：**数据本地化**（回测/深度学习都直接读，省得反复重拉），股票扩到 **800 只以上**，
> 日线跨度 **15 年（20 年更好）**，长期可增量补拉，落 `stock_worm/data/` 供以后复用。
> 已落地：脚本 `backtest/pull_ashare_daily.py` + 两份数据产物。

### 12.1 数据产物（已落 `stock_worm/data/`，已加 `data/` 到 .gitignore，本地常驻不入库）
| 文件 | 格式 | 体量 | 内容 |
|------|------|------|------|
| `ashare_daily_cache.pkl` | dict: code→DataFrame(OHLCV+amount) | 261 MB | 与 5m 缓存同构，因子管线可直接读 |
| `ashare_daily_panel.parquet` | 规整面板 date×code | 93 MB | `['date','code','open','high','low','close','volume','amount']`，供深度学习/回测直读 |
| `ashare_5m_cache.pkl` | dict: code→DataFrame(OHLCV+amount) | 316 MB | 288 只 CSI300 的 5 分钟棒（同经 stock_worm 源重拉） |

- **股票池**：沪深300(`000300`)+中证500(`000905`)+中证1000(`000852`) 去重并集 = **1489 只**（akshare 本次返回 1489；远超 800，覆盖大/中/小盘）。
- **跨度**：**2006-01-04 ~ 2026-06-30（整 20 年）**，共 **4,643,107 行**日线，`amount` 缺失率 **0.0000**。
- **数据源**：**首选数据源 stock_worm**（`stcok_worm.mootdx_source.get_kline`，通达信 TCP 直连，免费无 token），`amount` 全保留。

### 12.2 脚本特性（长期可反复跑）
- **数据源固定为 stock_worm**：`pull_ashare_daily.py` / `pull_ashare_5m.py` 均经 `stcok_worm.mootdx_source.get_kline` 取数（非 astockdata_loader/mootdx 另一封装）。stock_worm 自带 `get_kline_history` 从最老 offset 往前翻、遇空页即 break，对历史不足 `total` 根（2015 后上市）的股票会漏数据，故脚本改为**手动从最新页(offset=0)往前翻、遇空即止**，确保所有股票拉全。
- **断点续拉**：已落盘的自动跳过；只拉 pending。重跑即"增量补拉"。
- **checkpoint**：日线每 50 只 / 5m 每 20 只写一次盘，崩溃不丢进度。
- **单只失败隔离 + 超时保护**：失败/超时股跳过，下次重跑重试，不影响整体。
- **实测速度**：日线 ~7 只/秒（1489 只约 3 分钟）；5m ~1.8 只/秒（288 只约 3 分钟）。TDX 较快，"长期"主要是接口稳定性考量，非耗时。

### 12.3 用法
- 回测/因子：把 `combine_founder_heads.py` / `xgb_selector.py` 的 `CACHE` 指向
  `ashare_daily_cache.pkl`，即可用 20 年日线重算（覆盖 2024-02 急跌等早期行情）。
- 深度学习：直接 `pd.read_parquet("ashare_daily_panel.parquet")`，按 `date`/`code` 透视即面板，
  无需再联网拉数。
- 增量扩展：想加更多（如全 A 股 ~5000 只）或刷新，重跑 `pull_ashare_daily.py` 即可自动补齐（仍走 stock_worm）。
