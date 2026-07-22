#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 方向3: LLM + MCTS 公式化因子挖掘
================================================
对应报告第四章 4.3「大模型驱动的公式挖掘」与代码示例文档第 3 节(LLMMiningAgent)。

核心思想: 把因子表达式当成一棵树, 用蒙特卡洛树搜索(MCTS)在「变量 × 算子 × 窗口」的
组合空间中找高 ICIR / 低换手的因子; 每一步既可由本地启发式 proposer 生成, 也可挂接 LLM
(OpenAI 兼容接口) 提出「下一步应该怎么改公式」。LLM 不直接算数, 只出主意; 真正的好坏由
evaluate 模块的真实 Rank-IC / 换手率裁决 —— 形成「LLM 提议 → 回测反馈 → 再提议」的闭环。

工程化改造(承接代码示例文档):
  - 默认 **本地启发式 proposer**(无需任何 API key), 用 UCB 引导的 MCTS 在表达式树上搜索;
  - 可选 **LLM 钩子**: 设置环境变量 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 后,
    自动用 LLM 对当前最优候选做「改写建议」, 与本地搜索融合; 无 key 时静默降级为本地模式;
  - 反馈信号就是 evaluate_factor 给出的 icir20 / 换手, 回灌给 LLM 让其「越改越准」。
"""
from __future__ import annotations
import os
import math
import time
import numpy as np
from .operators import (evaluate_expr, random_expr, expr_to_str, all_leaves,
                        TS_OPS, CS_OPS, WINDOWS)
from .evaluate import evaluate_factor, turnover, factor_valid_ratio


# ---------------------------------------------------------------------------
# LLM 钩子(可选): OpenAI 兼容接口, 无 key 时返回 None -> 全程本地搜索
# ---------------------------------------------------------------------------
def _make_llm_client():
    """若环境变量齐全则返回一个 chat 调用闭包, 否则返回 None。"""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI  # 仅在需要时 import, 缺包不影响本地模式
    except Exception:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key, base_url=base_url)
    return lambda prompt: _llm_chat(client, model, prompt)


def _llm_chat(client, model, prompt: str) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "你是量化因子研究员, 只输出一个合法的因子表达式字符串, "
                            "格式如 ts_rank(arate_5,5) 或 bin_sub(ts_mean(ret_5,10), cs_rank(vol_20))。"
                            "不要解释, 不要加引号。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# 表达式树上的 MCTS
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("expr", "parent", "children", "visits", "value", "is_complete", "used")

    def __init__(self, expr, parent=None):
        self.expr = expr
        self.parent = parent
        self.children = []
        self.used = set()        # 已展开过的动作, 保证每个动作只生成一个子节点
        self.visits = 0
        self.value = 0.0          # 累计 reward(icir) 之和, 用于 UCB
        self.is_complete = False  # 是否已是完整因子(无未展开叶)


def _child_nodes(expr):
    """返回表达式的直接子树操作数（不含算子名/窗口等元信息）。"""
    if not isinstance(expr, tuple):
        return []
    t = expr[0]
    if t in ("ts", "cs"):
        return [expr[2]]          # 子表达式
    if t == "bin":
        return [expr[2], expr[3]]  # 两个子表达式
    return []


def _is_complete(expr, max_depth) -> bool:
    """完整表达式: 没有任何「仍可展开的变量叶子」（所有变量叶已达最大深度）。"""
    return len(_expandable_leaves(expr, max_depth)) == 0


def _expandable_leaves(expr, max_depth):
    """返回还可被算子包裹的变量叶子(DSL 中表示为 ('var', name) 节点)及其当前深度。

    注意:
    - 算子名('ts_mean' 等)和窗口(整数)是元信息, 不算叶子;
    - 只递归进入真正的子表达式操作数(_child_nodes), 避免把算子名误当变量;
    - 叶子统一用 name 字符串标识(如 'vol_20'), 供 _apply_action 精确替换。
    """
    leaves = []
    def _walk(e, depth):
        if isinstance(e, tuple) and e and e[0] == "var":
            if depth < max_depth:
                leaves.append((e[1], depth))   # 记录变量名
            return
        if isinstance(e, tuple) and e and e[0] == "const":
            return  # 常数不可展开
        if isinstance(e, tuple):
            for c in _child_nodes(e):
                _walk(c, depth + 1)
        # 极少数裸字符串叶子(防御性)
        elif depth < max_depth:
            leaves.append((e, depth))
    _walk(expr, 0)
    return leaves


def _apply_action(expr, leaf, action, rng, windows, max_depth):
    """对一个变量叶子('var', leaf)施加算子动作, 返回新表达式。

    action 形如 ('ts', op, w) | ('cs', op) | ('bin', op, other_leaf)
    leaf 为变量名字符串(如 'vol_20'); 只替换类型为 'var' 且名字匹配的节点。
    """
    def _replace(e):
        if isinstance(e, tuple):
            if e[0] == "var" and e[1] == leaf:
                if action[0] == "ts":
                    return ("ts", action[1], ("var", leaf), action[2])
                if action[0] == "cs":
                    return ("cs", action[1], ("var", leaf))
                if action[0] == "bin":
                    _, op, other = action
                    return ("bin", op, ("var", leaf), ("var", other))
            return tuple([e[0]] + [_replace(c) for c in e[1:]])
        return e
    return _replace(expr)


class MCTSAgent:
    """
    LLM + MCTS 因子挖掘代理。

    参数
    ----
    data      : {field: 面板} (含派生变量)
    fwd_dict  : forward_returns(close) 输出
    vars      : 可用变量名列表(默认 data 中除 close 外全部)
    max_depth : 表达式树最大深度(防止组合爆炸)
    llm       : None(默认本地) 或 _make_llm_client() 返回的闭包
    horizons  : 评估用的收益窗口, 默认 (20,)
    """

    def __init__(self, data, fwd_dict, vars=None, max_depth=3, llm=None,
                 horizons=(20,), seed=42):
        self.data = data
        self.fwd = fwd_dict
        self.vars = vars or [v for v in data if v != "close"]
        self.max_depth = max_depth
        self.llm = llm
        self.horizons = horizons
        self.rng = np.random.default_rng(seed)
        self.windows = WINDOWS
        self.ts_ops = list(TS_OPS.keys())
        self.cs_ops = list(CS_OPS.keys())
        # 评估缓存: 避免同表达式重复算 IC
        self._cache = {}

    # -- 评估 ----------------------------------------------------------------
    def _evaluate(self, expr):
        """返回 icir20(奖励), ic20, turnover; 无效/低覆盖/退化(零预测力)给惩罚。"""
        key = expr_to_str(expr)
        if key in self._cache:
            return self._cache[key]
        try:
            factor = evaluate_expr(expr, self.data)
            valid = factor_valid_ratio(factor)
            if valid < 0.8:
                r = (-1.0, 0.0, 100.0)
            else:
                ev = evaluate_factor(factor, self.fwd, self.horizons)[20]
                ic, icir, to = float(ev["ic"]), float(ev["icir"]), float(turnover(factor))
                # 退化过滤: |IC| 过小(零预测力) 或 换手≈0(近常数) 视为无效
                if abs(ic) < 0.004 or to < 0.001:
                    r = (-1.0, ic, 100.0)
                else:
                    r = (icir, ic, to)
        except Exception:
            r = (-1.0, 0.0, 100.0)
        self._cache[key] = r
        return r

    # -- 动作生成 ------------------------------------------------------------
    def _legal_actions(self, expr):
        leaves = _expandable_leaves(expr, self.max_depth)
        actions = []
        for (leaf, _) in leaves:
            for op in self.ts_ops:
                for w in self.windows:
                    actions.append(("ts", op, w, leaf))
            for op in self.cs_ops:
                actions.append(("cs", op, leaf))
        # bin 动作: 仅在已有 >=2 个变量可见时允许(组合两个变量)
        if leaves:
            for (leaf, _) in leaves:
                for other in self.vars:
                    if other != leaf:
                        actions.append(("bin", "bin_sub", other, leaf))
                        actions.append(("bin", "bin_div", other, leaf))
        return actions

    def _child_expr(self, expr, action):
        kind = action[0]
        if kind == "ts":
            op, w, leaf = action[1], action[2], action[3]
            return _apply_action(expr, leaf, ("ts", op, w), self.rng, self.windows, self.max_depth)
        if kind == "cs":
            op, leaf = action[1], action[2]
            return _apply_action(expr, leaf, ("cs", op), self.rng, self.windows, self.max_depth)
        # bin
        _, op, other, leaf = action
        return _apply_action(expr, leaf, ("bin", op, other), self.rng, self.windows, self.max_depth)

    # -- MCTS 单步 -----------------------------------------------------------
    def _rollout_reward(self, expr):
        """从当前表达式随机补全到完整, 返回其 icir 作为模拟奖励。"""
        e = expr
        guard = 0
        while not _is_complete(e, self.max_depth) and guard < self.max_depth + 2:
            acts = self._legal_actions(e)
            if not acts:
                break
            a = acts[self.rng.integers(0, len(acts))]
            e = self._child_expr(e, a)
            guard += 1
        return self._evaluate(e)[0]

    def _select(self, node, c=1.4):
        best, best_u = None, -1e9
        for ch in node.children:
            if ch.visits == 0:
                u = 1e9
            else:
                u = ch.value / ch.visits + c * math.sqrt(math.log(node.visits) / ch.visits)
            if u > best_u:
                best_u, best = u, ch
        return best

    def _expand(self, node):
        """只展开一个「尚未尝试过」的合法动作, 保证节点系统化扩展(非重复)。"""
        acts = [a for a in self._legal_actions(node.expr) if a not in node.used]
        if not acts:
            node.is_complete = True
            return None
        a = acts[self.rng.integers(0, len(acts))]
        node.used.add(a)
        child_expr = self._child_expr(node.expr, a)
        child = _Node(child_expr, parent=node)
        child.is_complete = _is_complete(child_expr, self.max_depth)
        node.children.append(child)
        return child

    def search(self, iterations=200, root=None, verbose=True):
        """运行 MCTS, 返回按 icir 降序的 Top 候选列表。"""
        if root is None:
            # 根: 随机起点(单变量), 让搜索有方向
            root = _Node(random_expr(self.vars, 1, self.rng))
        t0 = time.time()
        for it in range(iterations):
            # 1) selection: 沿 UCB 下降, 记录路径
            path = [root]
            node = root
            while node.children and not node.is_complete:
                nxt = self._select(node)
                if nxt is None:
                    break
                node = nxt
                path.append(node)
            # 2) expansion: 若选中节点已「完成」(无可展开叶子), 回溯到最近可扩展祖先
            if node.is_complete:
                for anc in reversed(path[:-1]):
                    if not anc.is_complete:
                        node = anc
                        break
            if not node.is_complete:
                child = self._expand(node)
                if child is not None:
                    node = child
            # 3) simulation: 从当前节点补全到完整表达式, 用真实 icir 作奖励
            reward = self._rollout_reward(node.expr)
            # 4) backprop
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value += max(reward, -1.0)
                cur = cur.parent
            if verbose and (it + 1) % max(1, iterations // 8) == 0:
                print(f"  [MCTS] iter {it+1:>3}/{iterations}  耗时 {time.time()-t0:.1f}s  "
                      f"缓存命中 {len(self._cache)}")
        # 收集所有访问过的完整/非平凡表达式, 按真实 icir 排序
        candidates = self._collect(root)
        candidates.sort(key=lambda c: c["icir"], reverse=True)
        return candidates

    def _collect(self, root):
        out = []
        stack = [root]
        while stack:
            n = stack.pop()
            icir, ic, to = self._evaluate(n.expr)
            out.append({"expr": expr_to_str(n.expr), "expr_tuple": n.expr, "icir": round(icir, 3),
                        "ic20": round(ic, 4), "turnover": round(to, 3),
                        "visits": n.visits})
            stack.extend(n.children)
        # 去重(同一表达式可能多次出现)
        seen, uniq = set(), []
        for c in out:
            if c["expr"] in seen:
                continue
            seen.add(c["expr"])
            uniq.append(c)
        return uniq

    # -- LLM 反馈闭环 --------------------------------------------------------
    def llm_refine(self, top_exprs, rounds=3):
        """用 LLM 对当前 Top 候选做改写(无 key 则跳过)。返回 (是否启用, 新增候选列表)。"""
        if self.llm is None:
            return False, []
        added = []
        for expr in top_exprs[:3]:
            icir, ic, to = self._evaluate(_parse(expr))
            prompt = (f"现有因子 {expr} 在 A 股样本上 icir20={icir:.2f}, ic20={ic:.3f}, "
                      f"换手率={to:.3f}。请提出一个更优的改写版本(可加入新的时序/截面算子或变量), "
                      f"仅输出表达式。可用变量: {', '.join(self.vars)}。")
            suggestion = self.llm(prompt)
            p = _parse(suggestion) if suggestion else None
            if p is not None:
                r_icir, r_ic, r_to = self._evaluate(p)
                added.append({"expr": expr_to_str(p), "icir": round(r_icir, 3),
                              "ic20": round(r_ic, 4), "turnover": round(r_to, 3),
                              "via": "llm"})
        return True, added


# ---------------------------------------------------------------------------
# 极简表达式解析(供 LLM 文本回灌): 支持 ts_/cs_/bin_ 前缀的函数式写法
# ---------------------------------------------------------------------------
def _parse(s):
    """把 'ts_rank(arate_5,5)' 这类字符串解析为本包的嵌套元组表达式。失败返回 None。"""
    import re
    s = s.strip()
    if not s:
        return None
    try:
        # 用递归下降解析
        idx = [0]
        toks = s.replace(" ", "")
        def peek():
            return toks[idx[0]] if idx[0] < len(toks) else ""
        def parse_expr():
            # 读取函数名
            m = re.match(r"[a-zA-Z_]+", toks[idx[0]:])
            if not m:
                return None
            name = m.group(0)
            idx[0] += len(name)
            if peek() != "(":
                # 纯变量
                if name in ("var",):
                    return None
                return ("var", name)
            idx[0] += 1  # consume '('
            args = []
            while peek() and peek() != ")":
                if peek() == "(":
                    args.append(parse_expr())
                else:
                    # 读 token(变量名或数字)
                    mm = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", toks[idx[0]:])
                    if not mm:
                        return None
                    tok = mm.group(0)
                    idx[0] += len(tok)
                    args.append(("var", tok) if not tok.isdigit() else ("const", float(tok)))
                if peek() == ",":
                    idx[0] += 1
            if peek() == ")":
                idx[0] += 1
            return _normalize(name, args)
        return parse_expr()
    except Exception:
        return None


def _normalize(name, args):
    """把函数名 + 参数统一映射到本包算子节点。"""
    if name in TS_OPS:
        # ts_<op>(child, w)
        return ("ts", name, args[0], int(args[1][1]) if len(args) > 1 else 20)
    if name in CS_OPS:
        return ("cs", name, args[0])
    if name.startswith("bin_") or name in ("bin_sub", "bin_div", "bin_mul", "bin_corr"):
        op = name[4:] if name.startswith("bin_") else name
        return ("bin", op, args[0], args[1])
    return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from .base_data import load_base_data, forward_returns, list_stocks
    from . import derive_variables
    codes = list_stocks(120)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"])
    agent = MCTSAgent(data, fwd, max_depth=2, llm=_make_llm_client(), seed=7)
    print(f"LLM 钩子: {'已启用' if agent.llm else '未配置(本地启发式)'}")
    cands = agent.search(iterations=120, verbose=True)
    print(f"\nMCTS 产出 {len(cands)} 个候选, Top5:")
    for c in cands[:5]:
        print(f"  {c['expr']:<46} icir={c['icir']:+.2f} ic20={c['ic20']:+.3f} to={c['turnover']:.3f}")
    if agent.llm:
        ok, added = agent.llm_refine([c["expr"] for c in cands])
        for a in added:
            print(f"  [LLM] {a['expr']:<46} icir={a['icir']:+.2f}")
