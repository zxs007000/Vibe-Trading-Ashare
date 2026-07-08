# 因子主题清单 (Theme Inventory)

> 基于 `agent/src/factors/zoo/` 下 460 个因子,按 theme 重新分类。
> 标注了"直接因子"(官方theme标签)和"替代因子"(标注偏差但回测行为接近)。
> 所有因子均为价格代理因子(PRICE PROXY),非财报表因子。

## 总览

| Theme | 直接因子 | 替代因子 | 合计可用 | 评价 |
|---|---|---|---|---|
| **volume** | 146 | — | 146 | ★★★★★ 最充足 |
| **momentum** | 144 | — | 144 | ★★★★★ |
| **reversal** | 65 | — | 65 | ★★★★ |
| **volatility** | 55 | — | 55 | ★★★★ |
| **microstructure** | 21 | — | 21 | ★★★ |
| **quality** | 3 | 55(低波→质量代理) | 58 | ★★★ 直接因子少 |
| **liquidity** | 3 | 146(量→流动性代理) | 149 | ★★★ 直接因子少 |
| **value** | 1 | 7+(长窗反转→价值代理) | 8+ | ★★ 最需补充 |

---

## value (1 直接 + 7+ 替代)

### 直接因子
| ID | Zoo | Nickname | Formula | 说明 |
|---|---|---|---|---|
| `academic_hml` | academic | HML — value via inverse 252d return | `-(close_t-close_{t-252})/close_{t-252}` | Fama-French HML,本质=252日反转 |

> **核心洞察**:HML 就是 -252日收益率。所以 zoo 里**任何窗口≥60天的反转因子**都可以是 value 的替代品——它们在捕捉相同的"长期均值回归"效应。

### 替代因子(长窗反转,窗口≥60天)
| ID | Zoo | 原Theme | 窗口 | 为什么适合 value |
|---|---|---|---|---|
| `alpha101_071` | alpha101 | reversal | 180d | 含180日adv,极长窗反转 |
| `gtja191_103` | gtja191 | reversal | 100d | `(20-lowday(low,20))/20*100` —— 距低点距离 |
| `gtja191_108` | gtja191 | reversal | 120d | 120日VWAP相关性反转 |
| `gtja191_118` | gtja191 | reversal | 100d | 上下影线比率,捕捉长期极端 |
| `qlib158_cntd60` | qlib158 | reversal | 60d | 涨跌天数差 |
| `qlib158_cntn60` | qlib158 | reversal | 60d | 60日下跌天数占比 |
| `qlib158_cntp60` | qlib158 | reversal | 60d | 60日上涨天数占比 |
| *(更多长窗反转因子可扩展)* | | | | |

### 当前逻辑链使用情况
- `value_momentum` → 节点1 value
- `value_qlowvol` → 节点1 value
- `value_stable` → 节点1 value

> ⚠️ 三条链共用一个直接因子+少量替代因子。value 是整个逻辑链体系的瓶颈节点。建议扩建:回测验证上述替代因子的 IC 表现,补充到 value 池。

---

## quality (3 直接 + 55 替代)

### 直接因子(academic)
| ID | Zoo | Nickname | Formula | 说明 |
|---|---|---|---|---|
| `academic_cma` | academic | CMA — investment via inverse volume growth | `-Δ60 log(mean(vol,60)+1)` | Fama-French CMA,代理=量缩→质量高 |
| `academic_rmw` | academic | RMW — quality via inverse 60d vol | `-std(ret, 60)` | Fama-French RMW,代理=低波→盈利强 |
| `academic_smb` | academic | SMB — small via inverse dollar-volume | `-log(mean(vol·close,60)+1)` | Fama-French SMB,代理=小市值 |

> **核心洞察**:academic 的 quality 因子全是价格代理(量缩/低波/小盘),没有真正的 ROE/现金流因子。但在 A 股环境下,低波动率本身就是一项重要的质量信号——高波动股更多是投机标的。

### 替代因子(低波动→质量代理,55 个 volatility 因子)
| 代表性因子 | Zoo | 特点 |
|---|---|---|
| `academic_retskew` | academic | 收益偏度(质量股通常左偏低) |
| `alpha101_001` | alpha101 | Kakushadze #1 |
| `alpha101_018` | alpha101 | Kakushadze #18 |
| *(完整 55 个 volatility 因子可回测筛选)* | | |

### 当前逻辑链使用情况
- `value_qlowvol` → 节点2 quality + 节点3 volatility
- `quality_momentum` → 节点1 quality

---

## liquidity (3 直接 + 146 替代)

### 直接因子
| ID | Zoo | Nickname | Formula |
|---|---|---|---|
| `academic_illiq` | academic | Amihud 2002 illiquidity | `mean(|r_t|/(close_t·volume_t), 21)` |
| `gtja191_132` | gtja191 | — | `mean(amount, 20)` |
| `gtja191_144` | gtja191 | — | see body |

### 替代因子(volume → 流动性代理,146 个)
所有标 volume 的因子本质上都在衡量流动性——成交量是流动性的直接测度。

> 精选 volume 因子(有公式/有衰减周期)建议优先使用。完整 146 个因子可在回测中按 IC 排序筛选。

### 当前逻辑链使用情况
- `liq_momentum` → 节点1 liquidity
- `value_stable` → 节点3 liquidity

---

## microstructure (21 个)

| Zoo | 数量 | 代表性因子 |
|---|---|---|
| gtja191 | 2 | alpha_111, alpha_171 |
| qlib158 | 19 | cord5/10/20/30/60, corr5/10/20/30/60, klen/klow/klow2/kmid/kmid2/ksft/ksft2/kup/kup2 |

> 21 个 microstructure 因子充足。cord 系列是订单相关性, k 系列是 k 线形态。

### 当前逻辑链使用情况
- `micro_reversal` → 节点1 microstructure

---

## momentum (144 个)

| Zoo | 数量 | 典型窗口 |
|---|---|---|
| academic | 3 (carhart_mom, high52w, mkt_rf) | 12m-1m / 52w |
| alpha101 | ~50 | 多窗口 |
| gtja191 | ~70 | 多窗口 |
| qlib158 | ~20 | 多窗口 |

> 最充足的因子库。中短期(20-60d)和长期(60d+)都有大量覆盖。

---

## reversal (65 个)

| Zoo | 数量 |
|---|---|
| academic | 1 (strev) |
| alpha101 | ~25 |
| gtja191 | ~25 |
| qlib158 | ~14 |

> reversal 因子覆盖 5-180d 窗口。短窗 reversal(5-20d)用于交易型(vol_reversal/micro_reversal),长窗 reversal(60d+)可用作 value 替代。

---

## volatility (55 个)

全部可作 quality 替代。低波动=质量信号 在 A 股环境下是有效的。

---

## volume (146 个)

全部可作 liquidity 替代。同时是 momentum 链的确认信号节点。

---

## 因子质量评级体系

| 星级 | 标准 |
|---|---|
| ★★★ | 有 nickname + formula_latex + decay_horizon |
| ★★ | 有 formula_latex 或 nickname 之一 |
| ★ | 只有 theme 标签,无命名/无公式 |
