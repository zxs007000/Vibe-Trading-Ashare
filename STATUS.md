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

## 3.6 466 因子全量 IC 筛查 + 牛熊反转候选(已完成 ✅ · 本任务)
> 文件:`oos_framework/screen_zoo_472.py`(筛查 harness) · `oos_framework/analyze_bull_bear.py`(牛熊候选提取)
> 数据:stock_worm 生存者无偏面板(1803×4975 日),截面取最密 2500 只;牛熊分域 bull=790/bear=656/osc=3529
> 输出:`oos_framework/screen_results/zoo472/zoo472_ic_screen.csv`(全量 466 行) + `zoo472_筛查报告.md` + `zoo472_牛熊反转候选.md` + `zoo472_bull_bear_candidates.csv`(58 候选去重)
> 方法:逐 alpha `reg.compute(panel)` → 逐日横截面 rank-IC(对 5d 前瞻收益) → ICIR + 正态近似 p;按 bull/bear/osc 拆 IC

- **结果(注入后复核 ✅)**:ok **455** / skip **2**(accruals + bvps,缺三大表原始字段) / error **8**(3 新:alpha101_097 >95%NaN、alpha101_100 超时、roe_pct 超时 + 5 旧) / no_ic 1。**显著(p<0.05) 445 个**,几乎全部强显著。
- **注入复活(用户确认数据源可用,不再追证券之星模块)**:用户澄清证券之星/金融界接口是加在 stockworm 的**财经接口**分类下(并非独立模块),数据湖在加接口后已扩展。本地已有数据源(`fund_factors_daily.parquet` 的 ROE + `csrc_industry_map.parquet` 的 CSRC 71 类)经注入后复活:
  - `panel["sector"]`(CSRC 71类,**覆盖 821/2500 只,33%**)→ 19 个 alpha101 行业因子复活,18 ok + 1 error(097 NaN);
  - `panel["roe"]`(**覆盖 992/2500 只,40%**)→ 3 个 roe 基本面因子复活,2 ok(`quality_roe` ICIR+4.02 / `quality_roe_change` ICIR+1.90) + 1 error(roe_pct 超时)。
  - ⚠️ **覆盖率 caveat**:roe/sector 仅占宇宙 ~33–40%,下游组合需处理"无基本面/行业标签的小盘长尾"——要么只对覆盖子集启用,要么用 `industry_cache` 的申万映射补全。
  - 🔍 **ICIR 虚高排查(已证伪)**:19 个 sector 因子 ICIR 高达 7.3(069)/7.1(087)/5.4(080),但**绝对 IC 仅 ±0.003~0.044(正常区间)**,ICIR 高是 20 年样本里 IC 方向极度一致所致(日频 IC t≈30+),非注入引入的假象。全库 |ICIR|>3 占 337/455,与首轮"几乎全显著"一致。
- **牛熊反转候选**(三档定义,均 p<0.05 且 |ICIR|>=0.5):
  - `regime_flip`(严格符号翻转) **34 个**(原 32,新增 `alpha101_056` bull+0.021/bear−0.005、`alpha101_079` bull+0.009/bear−0.001); `reversal_in_bear`(熊反转+牛不反转) **23 个**; `regime_sensitive`(按 |IC_bear−IC_bull| Top40) 供 regime-gated 选因子。候选去重共 **58 个**(原 56)。
  - **旗帜因子 = `qlib158_std5/10/20/30/60` 波动率族**:牛市 IC 为负(高波动→低收益)、熊市 IC 为正(高波动→高收益),符号干净翻转,|gap| 最高 **0.096**(std30)。这是"牛熊反转"在 466 库里最稳健的样本,直接佐证 thesis"什么状态用什么因子"。
  - 其它翻转:gtja191_189/160/174/159/030(波动率)、`academic_rmw`(质量,牛+0.031/熊−0.029)、alpha101_017/056/079(反转)。
- **修正旧判断**:此前"牛熊反转信号太弱"是只见单一手工反转因子;在 400+ 因子库里,**波动率因子族的牛熊符号翻转非常强且一致**,可作为 ② 牛熊混合反转因子的现成原料。
- 工程修复(本任务攻坚):① `base.py` 的 `ts_rank`/`decay_linear` 用 `sliding_window_view` 展开 `(T,C,n)` 时多个 3D 临时数组撑爆 8G cgroup(EXIT=137)→ **改按列分块(chunk) + 块间 gc**,峰值内存恒定 <1.5G;② `screen_zoo_472.py` 给 `reg.compute` 加 **SIGALRM 120s 超时**(`_ComputeTimeout` 继承 BaseException 以穿透 sentiment 因子内部 `except Exception` 兜底),避免外部新闻 API 永久挂死;③ 续跑用 `python3.11 -u` 直写文件(非 tee 管道,避免块缓冲在超时/被杀时丢输出)。
- **剩余缺口(可选,非阻塞)**:仅 `accruals`/`bvps` 2 因子因缺三大表原始字段仍 skip。用户数据湖已扩展证券之星/金融界(财经接口),若需补齐这 2 个,可扩展 `build_fundamental_factors.py` 从财经接口爬 accruals/bvps。用户已指示"数据源能用就先不找",故本次未追。

## 3.7 数据源升级:stockworm 最新版已并入(2026-07-17)

> 用户指出本地 stockworm 旧 clone 缺证券之星/金融界接口与数据湖构建脚本。已从 GitHub `zxs007000/stock-worm` 拉最新版并入工作目录。

- **操作**:`git clone` 到 `/workspace/stock-worm-latest` → 合并 `stcok_worm/` 代码进 `/workspace/stock_worm/stcok_worm/`(旧代码备份 `stcok_worm.bak`),**`data/` 数据湖完好**(324MB survivorfree 面板等 parquet 未动)。VibeTradingPush 只消费 parquet、不直接 import `stcok_worm`,覆盖安全。
- **证券之星/金融界(JRJ/StockStar)确认存在,按层整合**(非独立文件,故旧版 grep 不到):
  - `tencent.jrj_kline` — 金融界 K线(`gateway.jrj.com/quot-kline`)
  - `signals.dragon_tiger_jrj_daily/summary/stock/branches` — 金融界龙虎榜
  - `news.jrj_news`(金融界快讯) + `news.stockstar_express`(证券之星快讯 `express.stockstar.com`)
- **新增模块(修数据源核心拼图)**:
  - `fundamentals_ext` — **三张表** `income_statement`/`balance_sheet`/`cash_flow_statement` + `financial_indicators`(86 比率) + 分红/解禁 → **直接填补 accruals/bvps 缺口**
  - `lake_build` — 数据湖构建 CLI(`python -m stcok_worm.lake_build --only fin/dividends/unlock/regulatory/industry`)→ 此前缺失的构建脚本就在此
  - `datalake` — 本地湖读取层(需 `STOCKWORM_LAKE` 环境变量指向 lake 根,默认 Windows 路径需改)
  - `industry_map` — 多源行业映射(东财板块优先 + 巨潮兜底,优于当前 csrc 单源)
  - `regulatory` — 监管事件(立案/处罚/问询函,最高性价比负面因子源)
- **冒烟测试通过**:`fundamentals_ext` 三张表对 `000001` 返回 (121×171)/(118×222)/(102×317),接口真实可用。

### 3.7.1 新增宏观数据源模块 `macro`(已推送 GitHub, 2026-07-17)
> 用户需求:爬 GDP / M1M2M0 / A股总市值,用于防御门控「右侧预警」(巴菲特指标=总市值/GDP)。

- **新增文件** `stcok_worm/macro.py`(已 `git commit bfbde9c` 并 `push origin master` 到 `zxs007000/stock-worm`):
  - `gdp_quarterly()` — 名义 GDP(季度,亿元),源国家统计局(akshare 封装)
  - `money_supply_monthly()` — M1/M2/M0(月度,亿元),源国家统计局/央行
  - `total_market_cap_monthly(source='akshare')` — A股总市值(沪+深市价总值,月度)
  - `buffett_input()` — 拼出「总市值/GDP」低频面板,按季末对齐,供右侧预警
- **数据源策略**:主源经 akshare(其本身就是国家统计局/央行/交易所的爬虫);`source=` 预留 `tencent`/`stockstar`/`jrj` 接口(用户点名的证券之星/金融界),直连端点现 404,故先回退 akshare 并打 warning,待上游稳定后接入。
- **数据质量**:中文期次(年/月/季)统一解析为期末日;总市值仅保留沪+深双侧都有值的「完结月份」,剔除当期未完结的 0 值行(已验证 222 行含 0 行数=0)。
- **接驳位置**:`stcok_worm/__init__.py` 已 `from . import macro` 并加入 `__all__`;`/workspace/stock_worm` 工作副本与 GitHub 仓库 `__init__.py` 已同步。下游待接入 `defensive_gating.py`(方案 C:宏观作 regime 调制器,非硬触发)。

## 4. 待办
- [x] **① 防御门控层**:已实现并验证(见 §3.5),建议并入主流程。
- [x] **② 牛熊混合反转因子(原料已就位)**:`qlib158_std*` 波动率族 + `academic_rmw` + alpha101_017/056/079 等 **34 个 regime_flip 因子**已筛出(见 §3.6),可直接构造"牛=动量/熊=反转"混合因子并喂 WFA 确认。
- [ ] **②a 下一步(OOS 确认,用户已批准)**:把候选 alpha 接入 `load_engine_inputs_cached` zarr 因子层 → triple_validation / WFA 做样本外确认(IC 仅样本内,需 OOS 背书)。
- [ ] **③ 低杠杆因子**:🔓**已解锁** — `fundamentals_ext.balance_sheet`(三张表)提供资产负债率/账面市值比,可接入 `build_fundamental_factors.py` 造低杠杆因子。
- [x] **400+ 因子两段式筛选(IC 快筛段已完成)**:466 全量 IC + 牛熊分域已出;top-N → 三方法(triple_validation)+ FDR 确认待做。
- [x] **⑪ 用三张表补 accruals/bvps(数据源就位)**:`fundamentals_ext` 三张表已验证可用(§3.7)→ 写 builder 把 `accruals`/`bvps` 落 `fundamentals/` 湖,重跑 2 个 skip 因子,目标 455→457 ok。
- [ ] **⑫ 跑 lake_build 扩展数据湖**:`python -m stcok_worm.lake_build --only fin`(86 比率)+ `industry`(多源行业映射,替 csrc 单源)+ `regulatory`(监管事件因子源)。需先设 `STOCKWORM_LAKE=/workspace/stock_worm/data`(默认 Windows 路径),并注意 akshare/东财限流(全市场逐只,耗时较长)。
- [ ] 闸门修好后,把 C/E 接生产引擎实际组合路径验证(§3 遗留)。
- [ ] 提交已验证工作 + 工程修复(stockworm 升级 + 筛查结果 + 防御层 + 右侧预警评估)。

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
| `oos_framework/screen_zoo_472.py` | **466 因子全量 IC 筛查 harness**(含 SIGALRM 超时防外部 API 挂死) |
| `oos_framework/analyze_bull_bear.py` | **牛熊反转/高敏感候选提取**(三档定义) |
| `oos_framework/screen_results/zoo472/zoo472_ic_screen.csv` | 全量 466 因子 IC + 牛熊分域结果 |
| `oos_framework/screen_results/zoo472/zoo472_牛熊反转候选.md` | 牛熊反转候选报告(32 flip / 23 bear-rev / 40 sensitive) |
| `agent/src/factors/base.py` | alpha 算子;已修 `ts_rank`/`decay_linear` OOM(按列分块) |

## 6. 凭据与推送(跨任务延续)
- GitHub token(推送用)持久化于 **仓库外** `/workspace/.vibe_cache/gh_token`(chmod 600,不在 git 版本控制)。
- 推送用一次性 `GIT_ASKPASS` 读该文件,令牌不进 `.git/config`、不落盘到仓库。
- remote: `https://github.com/zxs007000/Vibe-Trading-Ashare.git`(main)。
- ⚠️ token 已出现在聊天记录,建议方便时到 GitHub 吊销轮换。
