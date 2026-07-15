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

## 4. 待办
- [ ] 排查/重标定两档位闸门(为何在全市场反噬回撤)。
- [ ] 闸门修好后,把 C/E 接生产引擎实际组合路径验证。
- [ ] 提交已验证工作 + 工程修复(`regime_wfa.py` fadvise、`factor_zoo_daily.py`、`oos_validation_corrected.py`、`ml_factor_analysis.py`、`fair_factor_regime_test.py`、`test_beta_vec.py`)。

## 5. 关键文件
| 文件 | 作用 |
|---|---|
| `oos_framework/ml_factor_analysis.py` | ML 重要性 + 增强因子 |
| `oos_framework/regime_wfa.py` | 双 regime 滚动 WFA 引擎(含 fadvise 修复) |
| `oos_framework/oos_validation_corrected.py` | 面板加载 + 因子构建 |
| `oos_framework/screen_results/wfa_run_5515.log` | **验证后的全市场 WFA 日志** |
| `oos_framework/screen_results/_wfa_cache/` | 全市场 zarr 缓存(5515,对齐) |
| `oos_framework/screen_results/_archive/` | 历史散落日志 |
