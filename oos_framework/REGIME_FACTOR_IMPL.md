# 因子级状态切换实现方案

参考：华安证券译介 Yizhan Shu & John Mulvey, *Dynamic Factor Allocation Leveraging Regime-Switching Signals* (JPM, 2024)

---

## 核心差异：从"市场级"到"因子级"状态

| | 现有 anti 闸门 | 新方案 |
|---|---|---|
| 状态判定对象 | 上证综指（一个维度） | 每个因子自己的主动收益（N 维） |
| 状态含义 | 大盘牛/熊 → 调仓位 | 某因子自身的牛/熊 → 调该因子权重 |
| 切换方式 | 硬切换（换全集） | 软切换（各因子独立加权） |
| 仓位管理 | 由闸门统一调 | 保留 anti 闸门做仓位 + 新做因子权重 |

核心洞察：**一个因子自己的"熊市"可能发生在大盘牛市里**（如 2024 年基本面因子在牛市后半段失效），反之亦然（如 2018 大熊市里低波动/质量因子仍是"牛市"）。

---

## Phase 1: 构建因子级绩效序列（~2 小时）

### 目的
为每个因子产出日频"主动收益"时间序列（该因子当天的多空/多头收益），作为后续状态分析的特征。

### 方法
对每个因子，每日计算因子值在横截面的多空收益：

```
fwd_ret = 未来 HOLD 日收益率（已有 compute_forward_returns）
factor_score = 当日因子 z-score（已有）

每日:
  long_stocks  = factor_score top 20% 的股票
  short_stocks = factor_score bottom 20% 的股票
  factor_pnl[t] = mean(fwd_ret[long_stocks]) - mean(fwd_ret[short_stocks])
```

产出：`factor_pnl.parquet`，shape = (dates × n_factors)，每个 cell 是该因子当日的多空收益。

### 已有基础
- `compute_forward_returns(close, horizon)` → 已有
- 因子全面板 `all_factors` → WFA 脚本已有
- 只需要对每个因子做分组统计

### 脚本
`build_factor_pnl.py`：
```python
def build_factor_pnl(all_factors, fwd, top_pct=0.2):
    """为每个因子产出日频多空收益序列"""
    dates = sorted(all_factors[list(all_factors.keys())[0]].index)
    pnl = {}
    for name, fv in all_factors.items():
        pnl_series = []
        for d in dates:
            if d not in fwd.index: continue
            score = fv.loc[d].dropna()
            if len(score) < 10: continue
            n = max(1, int(len(score) * top_pct))
            long_idx = score.nlargest(n).index
            short_idx = score.nsmallest(n).index
            ret_long = fwd.loc[d, long_idx].mean()
            ret_short = fwd.loc[d, short_idx].mean()
            pnl_series.append((d, ret_long - ret_short))
        pnl[name] = pd.DataFrame(pnl_series, columns=['date','pnl']).set_index('date')['pnl']
    return pd.DataFrame(pnl)
```

输出文件：`oos_framework/screen_results/factor_pnl_full.parquet`

---

## Phase 2: 因子级 regime 诊断（~1 小时）

### 目的
基于因子绩效序列，判定每个因子历史上什么时候是"牛市"、什么时候是"熊市"。

### 方法（简化版 SJM）

原文用 SJM 多维聚类，我们用简化版：**每个因子单独用滚动指标判牛熊**，不跨因子共享状态。

每个因子每日计算以下特征（窗口可按需调整）：

```
特征集（每因子，日频）：
  收益类:
    - pnl_ewma_21   (21日 EWMA)
    - pnl_ewma_63   (63日 EWMA)
    - pnl_sharpe_63 (63日滚动 Sharpe = mean/std)
    - pnl_cum_63    (63日累计)
  风险类:
    - pnl_dd_63     (63日最大回撤)
    - pnl_vol_63    (63日标准差)

判熊规则（每因子独立）:
  is_bear = (pnl_cum_63 < 0) & (pnl_sharpe_63 < 0)  # 持续亏 + Sharpe 负
  is_bull = (pnl_cum_63 > 0) & (pnl_sharpe_63 > 0.2) # 持续赚 + Sharpe 稳
  其余 = neutral  # 震荡/过渡
```

用日频特征避免年频样本不足问题（~4000 天 vs 16 年），每状态有足够 IC 观测点。

### 产出
- 日频三态标签 DataFrame：`factor_regime.csv`，shape = (dates × n_factors)，值 ∈ {1=bull, 0=neutral, -1=bear}
- 状态分布统计：每个因子 bull/bear/neutral 占比

### 脚本
`diagnose_factor_regime.py`（可独立运行，不依赖回测引擎）

---

## Phase 3: regime 条件因子权重 + WFA（~3 小时）

### 3.1 因子权重调整

不改选股逻辑（仍用 TOP_K），改**信号合成权重**。

现有流程：
```python
signal = 冻结因子等权平均 z-score  # 所有因子的权重相同
```

改为 regime 加权：
```python
for each frozen factor f:
    if factor_regime[f] == bull:
        weight[f] = 1.0    # 全权重
    elif factor_regime[f] == bear:
        weight[f] = 0.3    # 降至 30%（不完全关掉，防 regime 判错）
    else:  # neutral
        weight[f] = 0.7    # 中等
signal = weighted_average(z_scores, weights)  # 加权平均
```

**关键设计：0.3 底而不是 0**——因子级 regime 判定没有市场级闸门那么稳（因子绩效序列噪声更大），留底防误判。

### 3.2 与 anti 闸门的协同

```
最终仓位 = anti 闸门仓位 × 信号强度

其中:
  anti 闸门仓位 = 牛市 100% / 熊市 max(50%, 信号 × 75%)
  信号强度 = regime 加权合成（新）
```

两层独立：
- anti 闸门管"买多少"（仓位），按大盘牛熊
- 因子 regime 管"买什么"（选股信号内部各因子贡献），按各自牛熊

### 3.3 WFA 验证

修改 `wfa_fullmarket_alldelisted.py` 的 `build_signal` 函数，接入因子 regime 权重：

```python
def build_signal_weighted(all_factors, ev, frozen_set, factor_regime, current_date, weight_src="is"):
    """regime 加权的信号合成"""
    weights = {}
    for name in frozen_set:
        regime = factor_regime[name].loc[:current_date].iloc[-1]  # 最近一日的 regime
        if regime == 1:   # bull
            weights[name] = 1.0
        elif regime == -1: # bear
            weights[name] = 0.3
        else:             # neutral
            weights[name] = 0.7
    
    # 使用 IS_IC 或 IS_ICIR 作为 base weight，再乘 regime modifier
    for name in frozen_set:
        base = ev[name]["is_ic"]
        weights[name] *= max(0.01, base)
    
    # 加权平均
    total_w = sum(weights.values())
    signal = sum(z * (w / total_w) for (name, z), w in ...)
    return signal
```

跑两组对照：
- A: anti 闸门 + regime 加权信号（新）
- B: anti 闸门 + 等权信号（现有基线）

用 `wfa_fullmarket_alldelisted.py` 跑，对比 Sharpe、gate drag、逐折差异。

---

## Phase 4（可选）：Black-Litterman 因子级动态配置

完整复现论文思路，适合因子数较少时（如按大类分组：价值/动量/质量/成长/低波 5 个大类因子）：

1. 每折：
   - 对每个因子大类计算其"主动收益"（该类因子组合 - 等权全A）
   - 用 SJM 简化版判定每个大类的牛/熊状态
   - 牛市的预期收益 = 历史上该大类在牛市状态的平均月收益
   - 熊市的预期收益 = 历史上该大类在熊市状态的平均月收益
2. Black-Litterman：
   - 先验 = 等权配置
   - 观点 = 各类因子的预期收益
   - 后验收益 → 均值方差优化（纯多头 + 满仓）
3. 产出每个大类的动态权重

此阶段更复杂，建议 Phase 3 跑出正信号后再投入。

---

## 文件清单

| 文件 | 用途 | Phase |
|---|---|---|
| `build_factor_pnl.py` | 构建日频因子绩效序列 | 1 |
| `screen_results/factor_pnl_full.parquet` | 因子绩效数据 | 1 |
| `diagnose_factor_regime.py` | 因子级 regime 诊断 | 2 |
| `screen_results/factor_regime.csv` | 因子日频三态标签 | 2 |
| `oos_engine.py`（改动） | `build_signal` 支持 regime 加权 | 3 |
| `wfa_fullmarket_alldelisted.py`（改动） | 接入因子 regime + 双管线对照 | 3 |

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 因子绩效序列噪声大（日频多空收益不稳定） | 用 63 日滚动统计平滑；不对单个因子做 regime，按大类聚合 |
| 因子协同效应丢失（等权天然分散，加权可能集中） | regime modifier 用 0.3~1.0 窄区间，不极端关停 |
| 额外复杂度，过拟合 | Phase 2 先诊断 regime 标签是否有区分度（bull vs bear 期绩效差 > 2σ），没区分度就不进 Phase 3 |
| 因子级 regime 与市场级闸门信号冲突 | 两层独立：仓位 = anti × 信号，各自用各自的 regime 信息 |

---

## 与论文的关键差异（诚实标注）

| | 论文 | 本方案 |
|---|---|---|
| 资产 | 7 个现成因子 ETF 指数 | 5000+ 个股组成因子（更底层） |
| 状态模型 | SJM（多维聚类 + 跳惩罚） | 简化滚动规则（绩效 + Sharpe） |
| 组合优化 | Black-Litterman + MVO | regime 加权合成信号 + 仓位闸门 |
| 因子相关性 | 低（ETF 层面） | 高（个股层面，因子间重叠） |
| 换手 | ~500%/年（指数级） | 取决于 TOP_K 和 HOLD 参数 |

本方案的精神一致（因子级状态 → 动态配置），实现更适合 A 股个股因子场景。
