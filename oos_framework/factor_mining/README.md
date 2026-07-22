# factor_mining · 因子挖掘进阶工具箱（报告第四章实现）

对应《量化交易技术架构与策略优化综合报告》第四章「因子挖掘进阶方向」的完整可运行实现。
在 `xgboost_wfa` 给出的 WFA 验证框架之上，本包把第四章提出的 **4 个因子挖掘方向** 全部工程化落地，
共用同一套「数据湖面板 → 算子库 → Rank-IC/ICIR 评估」底座，可直接复现、可直接接入实战。

> 数据底座：`/workspace/stocklake` 的 `daily/` 层（5448 只 A 股，2018-07 起，日频开高低收量额）。
> 微观结构（Level-2/Tick）需另行接入，本包提供合成演示与可直接复用的特征函数。

---

## 1. 共用底座

| 模块 | 职责 |
|---|---|
| `base_data.py` | 从数据湖加载「日期×股票」面板；`derive_variables()` 把 6 个基础字段派生为 19 个 alpha 变量（收益/波动/量价/距高低点）；`forward_returns()` 计算多 horizon 向前收益 |
| `operators.py` | 算子库（时序 TS / 截面 CS / 二元 BIN）+ 因子表达式 DSL（嵌套元组）+ 求值/随机生成/序列化；`ts_mean`/`ts_rank` 走 Numba JIT |
| `evaluate.py` | 因子评估核心指标：**Rank-IC**（Numba 逐日横截面 Spearman）、**ICIR**（IC 均值/标准差）、**换手率**（横截面 Rank 变化）；这正是报告第四章强调的因子筛选指标 |

### 因子表达式 DSL
```python
('var', name)            # 变量叶, 如 ('var','ret_5')
('const', value)         # 常数
('ts', op, child, w)     # 时序算子, 如 ('ts','ts_rank',('var','arate_5'),5)
('cs', op, child)        # 截面算子, 如 ('cs','cs_rank',...)
('bin', op, c1, c2)      # 二元算子, 如 ('bin','sub',c1,c2)
```

---

## 2. 四大方向与报告第四章映射

### 方向 1 · 算子 + 变量网格搜索  →  报告 4.1
在「变量 × 时序算子 × 窗口 ×（可选）截面算子」的高维网格上枚举因子，逐一对齐 WFA 目标的 IC/ICIR，
输出 Top 候选。时序均值/排名走 Numba；评估从「非空率」升级为「Rank-IC / ICIR」。

```python
from factor_mining import list_stocks, load_base_data, derive_variables, forward_returns, grid_search
codes = list_stocks(300)
data  = derive_variables(load_base_data(codes))
fwd   = forward_returns(data["close"])
top   = grid_search(data, fwd, top_k=15)   # 正 IC 候选, 按 ICIR 降序
```

### 方向 2 · 遗传规划因子工厂（NSGA-II 多目标）  →  报告 4.2
自动生成 Alpha 表达式树，通过变异/交叉/选择进化。**不依赖 DEAP**（零额外安装），用 sklearn 随机数
实现种群与 tournament 选择，引入多目标惩罚：最大化 ICIR + 最小化换手（对应代码示例 `weights=(1.0,-1.0)`），
输出 **Pareto 前沿**。适应度复用 `evaluate` 的真实 Rank-IC / 换手率，并对退化因子（零预测力 / 近常数）下修。

```python
from factor_mining import evolve
res = evolve(data, fwd, pop_size=40, generations=20, seed=42)
print(res["pareto"])   # Pareto 前沿: [{expr, icir20, turnover, ic20}, ...]
```

### 方向 3 · LLM + MCTS 公式化挖掘  →  报告 4.3
把因子表达式当成一棵树，用**蒙特卡洛树搜索（MCTS）**在「变量 × 算子 × 窗口」组合空间中寻优；
每一步既可由**本地启发式 proposer** 生成（默认，无需任何 API key），也可挂接 **LLM（OpenAI 兼容接口）**
对当前最优候选做「改写建议」，与本地搜索融合，形成「LLM 提议 → 回测反馈 → 再提议」闭环。
LLM 不直接算数，只出主意；好坏由 `evaluate` 的真实 Rank-IC / 换手率裁决。

```python
from factor_mining import MCTSAgent
from factor_mining.llm_mcts import _make_llm_client
agent = MCTSAgent(data, fwd, max_depth=2, llm=_make_llm_client(), seed=7)
cands = agent.search(iterations=200)         # 本地启发式(无 key 时)
# 启用 LLM: 设置环境变量 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 后自动接入
#   OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://... OPENAI_MODEL=gpt-4o-mini python run_mining.py
added = agent.llm_refine([c["expr"] for c in cands])   # LLM 反馈闭环新增候选
```

### 方向 4 · 微观结构挖掘（Level-2 / Tick）  →  报告 4.4
`microstructure_features(tick)` 是**真实可运行**的微观结构特征函数（输入逐笔成交面板
`[ts, price, volume, side]`，输出日频特征）；`demo_microstructure()` 用合成 tick 跑通整条流水线。
接入真实 Level-2 时，只需替换 `demo` 的 tick 输入，特征函数无需改动。

特征：成交量自相关（知情交易持续度）、单笔成交回归截距（大单占比代理）、量价轨迹欧氏距离（量价背离）、
实现价差（流动性成本）、买卖方向不平衡（主力净流入）。

```python
from factor_mining import microstructure_features, demo_microstructure
feats = microstructure_features(real_tick_df)   # 真实数据
demo  = demo_microstructure(n_stocks=6)          # 合成演示
```

---

## 3. 一键编排

`run_mining.py` 跑通全部四个方向并产出结果摘要：

```bash
cd oos_framework
python factor_mining/run_mining.py                  # 默认 150 只快速验收
python factor_mining/run_mining.py --stocks 400     # 更大样本
python factor_mining/run_mining.py --out results.md # 写入 markdown
```

---

## 4. 最近一次运行结果（150 只样本，2018-10-23 ~ 2026-07-20）

> 完整表格见 `FACTOR_MINING_RESULTS.md`。以下为各方向 Top 代表（20 日收益为预测目标）：

| 方向 | 代表因子 | IC(20d) | ICIR(20d) |
|---|---|---|---|
| 1 网格搜索 | `ts_mean(dist_high_20,20)` | +0.023 | +0.14 |
| 2 遗传规划 | `vrate_5 div ts_pct(vrate_5,20) sub ts_max(amount div high,10)` | +0.082 | +0.57 |
| 3 LLM+MCTS | `vol_20 div ret_1 mul vrate_5` | +0.040 | +0.32 |
| 4 微观结构 | 5 维 tick 特征（合成演示，需接入真实 Level-2） | — | — |

**读图要点**：方向 2/3 的复合因子 ICIR 显著高于方向 1 的简单算子（GP/MCTS 能组合出方向 1 网格覆盖不到的
高阶交互），但仍需回到 `xgboost_wfa` 做 Walk-Forward 验证、剔除过拟合，才能真正「走向实战」。

---

## 5. 与代码示例文档的关系

本包在用户提供的《因子挖掘进阶方向：Python 实现代码示例》基础上做了工程化改造：
- 输入从「单变量 1D 数组」升级为「日期×股票面板」，截面算子天然生效；
- 评估指标从「非空率」升级为「Rank-IC / ICIR / 换手率」；
- 遗传规划去除 DEAP 依赖、改为 NSGA-II 式 Pareto 排序；
- LLM 钩子从「仅 OpenAI」改为「本地启发式默认 + 可选 LLM」，无 key 也能完整复现；
- MCTS 在表达式树上搜索（而非字符串拼接），并修复了叶子识别/动作去重等工程坑。
