# VibeTrading — 项目状态(单页真相源)

> 这份文档取代散落的 run 日志和冗长会话摘要。每次状态更新只改这里,不回放历史。
> 最后更新:2026-07-16

## 0. 一句话定位
Thesis:**因子有寿命,关键是"在什么状态用什么因子",而非找永恒圣杯。** 重心 = regime(状态)分层 + 全市场 OOS 验证。

## 1. 已验证:ML 因子分析(全市场 5515 / XGBoost)
- Part 1 因子重要性在三态(牛/熊/震荡)下**很平** → 这 33 因子池里"状态→因子"依赖弱,给 Thesis 泼冷水。
- Part 2 朴素单 ML 因子(OOS 18-fold):Sharpe +0.046、回撤 -76%(最差,丢分散度)。
- Part 3 ML regime 加权复合(保留 33 因子分散度):Sharpe +0.298、回撤 -69%。
- 结论:ML 增强未超越"冻结 IS 胜者 + 市场级 regime(C)"。

## 2. 已验证:全市场 5515 WFA(公平 IC 标签 + loader/cache 对齐)
> 日志:`oos_framework/screen_results/wfa_run_5515.log`(509s, 峰值 7.0G, exit 0)
> 修复:写盘加 `posix_fadvise(DONTNEED)` 防页缓存撑爆 8G cgroup;loader 删旧错位缓存后重建,与缓存同宇宙(5515)。

| 变体 | Sharpe | 年化 | 回撤 | 通过率 |
|---|---|---|---|---|
| A 无闸 | **+0.618** | +24.2% | -65.80% | 89% |
| B 两档位闸门 | +0.479 | +10.8% | **-70.22%** | 50% |
| C 市场regime | +0.615 | +13.2% | -59.35% | 50% |
| D 因子regime(公平) | +0.507 | +11.6% | -69.92% | 50% |
| E 双regime(公平) | +0.616 | +13.3% | **-59.06%** | 50% |

- 全部 PASS/FAIL 都 FAIL(20 年全样本多头 MaxDD 永远 < -35%,否决规则对本宇宙无信息量;看相对 Sharpe/回撤)。
- **闸门(B)宇宙依赖,非本身坏**:同一闸门设计(ma120+factor_decay,正渐进下限50%)在 **1803 流动大盘最优**(原始 `OOS_WFA验证报告.md`: Sharpe +0.648 / 回撤 -47.8%,优于无闸 +0.625/-70.5%);但在 **5515 全市场反噬**(Sharpe 0.618→0.479、回撤 -65.8%→-70.2%)。两次 run 仅宇宙不同(1803 vs 5515),年份/因子/TOP_K=30%/HOLD=5/闸门逻辑全同。反噬源于小盘长尾上 MA120 穿越 + IC 衰减计数噪声大、误砍仓且躲不掉回撤。
- **市场级 regime(C)/双regime(E)有效**:Sharpe≈无闸(A),回撤 -59%(省 ~6.5pp)。regime 价值=控回撤,非抬 Sharpe。
- **C 不是"最优"**:与 A/E Sharpe 打平;之前硬编码 "C +0.659 / D +0.435" 是错的。C/E 回撤优势真实已验证。因子级 regime(D)全市场弱,印证小样本过拟合。

## 3. 拍板结论(据验证数据修正)
- **因子方式**:市场级 regime(C)或双 regime(E)——保 Sharpe + 控回撤,保留。因子级 regime(D)弃用。
- **闸门**:两档位闸门(B)在 1803 流动大盘验证最优(原始报告),但在 5515 全市场反噬 → **宇宙依赖,不能全市场一刀切**。生产落地需按流动性分层标定(或限流动大盘),而非 blanket 应用到全部 5515。
- WFA 结构:18 fold,train 250d / test 250d,purge 5d,冻结 IS 胜者(ICIR>0)+ 零重学。

## 3.5 防御门控层(已验证 ✅ · Task #79)
> 文件:`oos_framework/defensive_gating.py` · 报告:`oos_framework/screen_results/defensive/防御门控层报告.md` · 图同上 `defensive_equity.png`
> 设计:市场级防御门控,**不动因子选择**(沿用基线 A 活因子集),只在组合层做状态响应。
> 危机检测:mkt_level 跌破 250 日线(-10%) **或** 市场波动 z>2。WFA 测试窗内危机调仓期 ≈236/900(~26%)。
> 危机期动作:① 低波+质量(ivol_60/vol_60/downside_vol_60/ROE/profit_yoy)权重 ×3 抬升;② 反转/小盘(rev_5/20/60/amihud_20/overnight_gap/drawup_60)**保持原权重**(保留 alpha 敞口,不砍);③ 部分降仓至 60%(不归零),空仓吃 4% 防御资产日收益。

| 指标 | 基线 A(无闸) | 防御门控 | 变化 |
|---|---|---|---|
| Sharpe | +0.618 | **+0.648** | **+0.030**(不降反升) |
| 年化 | +24.23% | +20.27% | -3.96%(降仓代价,可接受) |
| 最大回撤 | -65.80% | **-53.94%** | **+11.86pp**(少亏 ~18%) |
| 危机段回撤 | -53.84% | -35.38% | +18.46pp |
| MC 回撤 95%CI | [-87.4%, -44.1%] | [-80.4%, -36.6%] | 上界更低(非偶然) |

- **结论:✅ 回撤显著压低且收益基本不损,可直接并入主流程。** 压回撤来自"降仓 + 低波/质量抬升",而非牺牲已验证的小盘 illiquidity 溢价(amihud ICIR+4.45 显著,危机期权重未砍)。
- 代价:WFA 通过率 89%→67%(危机期吃防御资产、跑输基准,ex_sharpe 0.707→0.415,属预期);年化 -3.96% 在阈值(<5%)内。
- 调参杠杆:危机频率偏高(26%)→若想减收益代价,可下调 CRISIS_POS(0.7)或收窄危机信号(提高 CRISIS_MA_THR / CRISIS_VOL_Z)再测。
- 注意:本门控与 §2 的"两档位闸门(B)"是**两套不同机制**——B 是因子级 IC 衰减+熊市状态机(全市场反噬),本层是市场级危机响应(全市场有效)。二者可叠加,但 B 需先在流动大盘重标定。

## 4. 待办
- [x] **① 防御门控层**:已实现并验证(见 §3.5),建议并入主流程。
- [ ] **② 牛熊混合反转因子**(千问方案):牛=动量分位/熊=反转分位/拐点=60%反转+40%动量,现有数据可做。
- [ ] **③ 低杠杆因子**:⛔阻塞(需资产负债率数据,当前无)。
- [ ] **400+ 因子两段式筛选**:IC 快筛 → top-N → 三方法(triple_validation)+ FDR 确认,harness 已就绪。
- [ ] 闸门修好后,把 C/E 接生产引擎实际组合路径验证(§3 遗留)。
- [ ] 提交已验证工作 + 工程修复。

## 5. 关键文件
| 文件 | 作用 |
|---|---|
| `oos_framework/ml_factor_analysis.py` | ML 重要性 + 增强因子 |
| `oos_framework/regime_wfa.py` | 双 regime 滚动 WFA 引擎(含 fadvise 修复) |
| `oos_framework/oos_validation_corrected.py` | 面板加载 + 因子构建 |
| `oos_framework/screen_results/wfa_run_5515.log` | **验证后的全市场 WFA 日志** |
| `oos_framework/screen_results/_wfa_cache/` | 全市场 zarr 缓存(5515,对齐) |
| `oos_framework/screen_results/_archive/` | 历史散落日志 |
| `oos_framework/triple_validation.py` | **三方法验证台**:传统+WFA+MC自助/置换+可配置A股成本 |
| `oos_framework/screen_results/triple/` | 三方法报告+净值图+运行日志 |
| `oos_framework/defensive_gating.py` | **防御门控层**:危机检测+低波/质量倾斜+部分降仓,治回撤不杀小盘alpha |
| `oos_framework/screen_results/defensive/` | 防御层报告+净值图 |

## 6. 凭据与推送(跨任务延续)
- GitHub token(推送用)持久化于 **仓库外** `/workspace/.vibe_cache/gh_token`(chmod 600,不在 git 版本控制)。
- 推送用一次性 `GIT_ASKPASS` 读该文件,令牌不进 `.git/config`、不落盘到仓库。
- remote: `https://github.com/zxs007000/Vibe-Trading-Ashare.git`(main)。
- ⚠️ token 已出现在聊天记录,建议方便时到 GitHub 吊销轮换。
