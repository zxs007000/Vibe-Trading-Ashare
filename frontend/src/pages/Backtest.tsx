import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import type { LucideIcon } from "lucide-react";
import {
  Landmark,
  LineChart,
  Anchor,
  Bitcoin,
  Boxes,
  Coins,
  Globe,
  ShieldCheck,
  Activity,
  Repeat,
  Ghost,
  Database,
  ArrowRight,
  Cpu,
} from "lucide-react";
import { echarts } from "@/lib/echarts";

/* ────────────────────────────────────────────────────────────
 * 数据：7 大回测引擎 + 3 大统计检验 + 影子账户
 * 内容与 agent/backtest/engines 及 validation.py 一一对应
 * ──────────────────────────────────────────────────────────── */

interface MethodCard {
  id: string;
  icon: LucideIcon;
  nameZh: string;
  nameEn: string;
  tags: string[];
  intro: string;
  scenario: string;
  pros: string[];
  cons: string[];
  backend: string;
  accent: string;
  /** 雷达图五维：规则保真度 / 市场覆盖 / 稳健性 / 过拟合防护 / 计算成本 */
  dims?: number[];
}

const ENGINES: MethodCard[] = [
  {
    id: "a-share",
    icon: Landmark,
    nameZh: "A 股引擎",
    nameEn: "China A-Share",
    tags: ["T+1", "涨跌停", "无卖空"],
    intro:
      "内置 A 股交易规则——T+1 回转限制、±10% 涨跌停板（ST 股 ±5%）、买入次日才可卖出、不支持裸卖空。最贴近实盘的 A 股执行约束。",
    scenario: "沪深主板 / 创业板 / 科创板的单边多头、打板、均线与因子选股策略回测。",
    pros: ["规则保真度高，真实反映涨跌停封板与流动性枯竭", "与 astockdata 等 A 股数据源无缝衔接"],
    cons: ["不支持融券卖空（需借券通道）", "北交所需切换 akshare / tushare 源"],
    backend: "ChinaAEngine · engines/china_a.py",
    accent: "#ff6b00",
    dims: [95, 70, 65, 50, 30],
  },
  {
    id: "us",
    icon: LineChart,
    nameZh: "美股引擎",
    nameEn: "US Equities",
    tags: ["T+0", "盘前盘后", "做空"],
    intro:
      "适配美股规则，支持 T+0 日内交易、盘前（Pre-market）与盘后（After-hours）流动性建模、可卖空，含 Reg SHO 与 PDT 约束提示。",
    scenario: "美股个股 / ETF 的日内、动量、财报事件与多空中性策略。",
    pros: ["覆盖盘前盘后薄流动性与滑点", "原生支持多空双向"],
    cons: ["需海外数据源（yfinance / finnhub 等）", "盘前盘后成交假设对流动性敏感"],
    backend: "GlobalEquityEngine · engines/global_equity.py",
    accent: "#00d4ff",
    dims: [90, 65, 70, 55, 40],
  },
  {
    id: "hk",
    icon: Anchor,
    nameZh: "港股引擎",
    nameEn: "HK Equities",
    tags: ["互联互通", "汇率折算", "印花税"],
    intro:
      "复用 GlobalEquityEngine，针对港股加载互联互通（北向 / 南向）汇率折算、动态印花税与半日市安排，支持 T+2 交收与卖空（须可借货）。",
    scenario: "港股通标的、红筹 / 蓝筹与高股息策略，以及 A/H 溢价套利。",
    pros: ["汇率与税费动态计入净值", "与美股共享引擎，跨市场一致"],
    cons: ["卖空受借货与限价规则约束", "碎股与半日市需特殊处理"],
    backend: "GlobalEquityEngine · HK market hooks",
    accent: "#00e676",
    dims: [88, 60, 68, 55, 42],
  },
  {
    id: "crypto",
    icon: Bitcoin,
    nameZh: "加密货币引擎",
    nameEn: "Crypto",
    tags: ["永续", "资金费", "强平"],
    intro:
      "建模加密永续合约的资金费率（funding）、标记价格强平、CEX / DEX 价差与链上确认延迟，支持高杠杆与多空双向。",
    scenario: "永续合约网格、资金费套利、CEX / DEX 价差与趋势策略。",
    pros: ["资金费与强平逻辑完整", "可模拟杠杆爆仓路径"],
    cons: ["极端行情下滑点与插针难精确", "需交易所数据源（ccxt / okx）"],
    backend: "CryptoEngine · engines/crypto.py",
    accent: "#ff2d6a",
    dims: [82, 55, 60, 50, 50],
  },
  {
    id: "futures",
    icon: Boxes,
    nameZh: "期货引擎",
    nameEn: "Futures",
    tags: ["主力滚动", "展期损耗", "保证金"],
    intro:
      "基于 FuturesBaseEngine，处理主力合约换月滚动、展期（roll）损耗、合约乘数与保证金占用，覆盖 CFFEX / SHFE / DCE / ZCE / INE 及 CME / ICE / Eurex。",
    scenario: "商品 / 金融期货的展期、跨期套利与 CTA 趋势策略。",
    pros: ["展期损耗与乘数精确", "跨交易所统一建模"],
    cons: ["主力合约切换规则多样化", "夜盘与节假日需对齐"],
    backend: "ChinaFuturesEngine + GlobalFuturesEngine",
    accent: "#b388ff",
    dims: [85, 75, 66, 55, 55],
  },
  {
    id: "forex",
    icon: Coins,
    nameZh: "外汇引擎",
    nameEn: "Forex",
    tags: ["点差", "隔夜 swap", "高杠杆"],
    intro:
      "建模外汇现货 / CFD 的动态 ECN 点差采样、隔夜掉期（swap）利息与高杠杆保证金，支持多货币对与对冲。",
    scenario: "主要货币对做市、套利与宏观方向性策略。",
    pros: ["点差与隔夜费动态计入", "高杠杆路径清晰"],
    cons: ["重要数据发布时流动性骤变", "需报价数据源"],
    backend: "ForexEngine · engines/forex.py",
    accent: "#ffd000",
    dims: [80, 60, 62, 50, 45],
  },
  {
    id: "composite",
    icon: Globe,
    nameZh: "跨市场组合回测",
    nameEn: "Composite",
    tags: ["共享资金池", "跨市场", "风险预算"],
    intro:
      "CompositeEngine 以共享资金池跨市场调度子引擎，按风险预算在 A 股 / 港股 / 美股 / 加密 / 期货 / 外汇间分配仓位，统一核算组合净值与回撤。",
    scenario: "多资产配置、全球宏观与跨市场套利组合。",
    pros: ["跨市场一致执行与组合层归因", "统一回撤 / 相关性视角"],
    cons: ["跨时区结算与汇率对冲复杂", "耦合度高，调试成本大"],
    backend: "CompositeEngine · engines/composite.py",
    accent: "#4d9fff",
    dims: [78, 95, 80, 60, 70],
  },
];

const VALIDATIONS: MethodCard[] = [
  {
    id: "monte-carlo",
    icon: Activity,
    nameZh: "蒙特卡洛模拟",
    nameEn: "Monte Carlo",
    tags: ["p 值", "收益分布"],
    intro:
      "对交易序列做 N 次随机置换 / 重采样，估计策略夏普与最大回撤在随机基线下的 p 值，判断超额收益是否显著。",
    scenario: "检验“这条曲线是不是运气”。",
    pros: ["不依赖正态假设", "直观给出显著性水平"],
    cons: ["仅检验统计量分布，不还原真实路径依赖"],
    backend: "validation.monte_carlo_test",
    accent: "#00e676",
  },
  {
    id: "bootstrap",
    icon: ShieldCheck,
    nameZh: "Bootstrap 置信区间",
    nameEn: "Bootstrap CI",
    tags: ["置信区间", "夏普"],
    intro:
      "对权益曲线做 Bootstrap 重采样，给出夏普比率的置信区间，量化策略表现的估计不确定性。",
    scenario: "报告夏普的可靠性，而非单点估计。",
    pros: ["直接量化估计误差", "易解释、易沟通"],
    cons: ["假设收益独立同分布，对自相关序列偏乐观"],
    backend: "validation.bootstrap_sharpe_ci",
    accent: "#00d4ff",
  },
  {
    id: "walk-forward",
    icon: Repeat,
    nameZh: "滚动前向验证",
    nameEn: "Walk-Forward",
    tags: ["样本外", "滚动窗口"],
    intro:
      "将样本切分为滚动的训练 / 测试窗口，在每个窗口外样本上验证策略一致性，缓解过拟合与前视偏差。",
    scenario: "参数敏感型策略的稳健性检验。",
    pros: ["最贴近真实部署的样本外表现", "可测参数稳定性"],
    cons: ["窗口切分与再训练成本高", "短样本下窗口数受限"],
    backend: "validation.walk_forward_analysis",
    accent: "#ff6b00",
  },
];

/* ────────────────────────────────────────────────────────────
 * 雷达图：7 大引擎 × 五维能力对比
 * ──────────────────────────────────────────────────────────── */

function CapabilityRadar({ engines }: { engines: MethodCard[] }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  const option = useMemo(() => {
    const indicators = [
      { name: t("backtestLab.dimRule"), max: 100 },
      { name: t("backtestLab.dimMarket"), max: 100 },
      { name: t("backtestLab.dimRobust"), max: 100 },
      { name: t("backtestLab.dimGuard"), max: 100 },
      { name: t("backtestLab.dimCost"), max: 100 },
    ];
    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "item", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0" } },
      legend: {
        type: "scroll",
        bottom: 0,
        textStyle: { color: "#5a5a80", fontSize: 11 },
        pageTextStyle: { color: "#5a5a80" },
        pageIconColor: "#00d4ff",
      },
      radar: {
        center: ["50%", "48%"],
        radius: "66%",
        indicator: indicators,
        axisName: { color: "#d8d8f0", fontSize: 12 },
        splitNumber: 4,
        axisLine: { lineStyle: { color: "#1e1e3a" } },
        splitLine: { lineStyle: { color: "#1e1e3a" } },
        splitArea: { areaStyle: { color: ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.04)"] } },
      },
      series: [
        {
          type: "radar",
          data: engines.map((e) => ({
            name: e.nameZh,
            value: e.dims ?? [0, 0, 0, 0, 0],
            symbolSize: 4,
            lineStyle: { color: e.accent, width: 2 },
            itemStyle: { color: e.accent },
            areaStyle: { color: e.accent, opacity: 0.08 },
          })),
        },
      ],
    };
  }, [engines, t]);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption(option);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
    };
  }, [option]);

  return <div ref={ref} className="h-[420px] w-full" />;
}

/* ────────────────────────────────────────────────────────────
 * 卡片
 * ──────────────────────────────────────────────────────────── */

function MethodCardView({ m, index }: { m: MethodCard; index?: number }) {
  const { t } = useTranslation();
  const Icon = m.icon;
  return (
    <article
      className="group relative flex flex-col rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-4 transition-colors hover:border-[color:var(--accent)]"
      style={{ ["--accent" as string]: m.accent }}
    >
      <div
        className="absolute inset-x-0 top-0 h-[2px] rounded-t-xl opacity-70"
        style={{ background: m.accent }}
      />
      <header className="flex items-start gap-3">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
          style={{ background: `${m.accent}1a`, color: m.accent }}
        >
          <Icon className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {index !== undefined && (
              <span className="font-mono text-xs text-[var(--dash-dim)]">{String(index).padStart(2, "0")}</span>
            )}
            <h3 className="truncate text-base font-semibold text-[var(--dash-text-bright)]">{m.nameZh}</h3>
          </div>
          <p className="text-[11px] uppercase tracking-wide text-[var(--dash-dim)]">{m.nameEn}</p>
        </div>
      </header>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {m.tags.map((tag) => (
          <span
            key={tag}
            className="rounded-full border px-2 py-0.5 text-[10px] font-medium"
            style={{ borderColor: `${m.accent}40`, color: m.accent, background: `${m.accent}12` }}
          >
            {tag}
          </span>
        ))}
      </div>

      <p className="mt-3 text-[13px] leading-relaxed text-[var(--dash-text)]">{m.intro}</p>

      <div className="mt-3 rounded-lg bg-[var(--dash-surface-2)] p-2.5">
        <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--dash-dim)]">
          {t("backtestLab.scenarioLabel")}
        </p>
        <p className="mt-1 text-[12px] leading-relaxed text-[var(--dash-text)]">{m.scenario}</p>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--dash-accent-green)]">
            {t("backtestLab.prosLabel")}
          </p>
          <ul className="mt-1 space-y-1">
            {m.pros.map((p) => (
              <li key={p} className="flex gap-1 text-[12px] leading-snug text-[var(--dash-text)]">
                <span className="text-[var(--dash-accent-green)]">+</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--dash-accent-pink)]">
            {t("backtestLab.consLabel")}
          </p>
          <ul className="mt-1 space-y-1">
            {m.cons.map((c) => (
              <li key={c} className="flex gap-1 text-[12px] leading-snug text-[var(--dash-text)]">
                <span className="text-[var(--dash-accent-pink)]">−</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <footer className="mt-3 flex items-center gap-1.5 border-t border-[var(--dash-border)] pt-2.5">
        <Cpu className="h-3.5 w-3.5 text-[var(--dash-dim)]" />
        <span className="truncate font-mono text-[10px] text-[var(--dash-dim)]">{m.backend}</span>
      </footer>
    </article>
  );
}

/* ────────────────────────────────────────────────────────────
 * 页面
 * ──────────────────────────────────────────────────────────── */

export function Backtest() {
  const { t } = useTranslation();

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-5 md:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-[var(--dash-text-bright)]">
            <Database className="h-5 w-5 text-[var(--dash-accent-cyan)]" />
            {t("backtestLab.title")}
          </h1>
          <p className="mt-1 text-sm text-[var(--dash-dim)]">{t("backtestLab.subtitle")}</p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--dash-accent-green)]/40 bg-[var(--dash-accent-green)]/10 px-3 py-1 text-xs font-medium text-[var(--dash-accent-green)]">
          <Database className="h-3.5 w-3.5" />
          {t("backtestLab.dataSourcePreferred")}
        </span>
      </div>

      {/* 数据源面板 */}
      <div className="mt-4 flex flex-col gap-3 rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-4 sm:flex-row sm:items-center">
        <div className="flex items-center gap-2 text-sm font-medium text-[var(--dash-text-bright)]">
          <Database className="h-4 w-4 text-[var(--dash-accent-cyan)]" />
          {t("backtestLab.dataSourceTitle")}
        </div>
        <p className="flex-1 text-[13px] leading-relaxed text-[var(--dash-text)]">{t("backtestLab.dataSourceDesc")}</p>
        <code className="rounded-md border border-[var(--dash-border)] bg-[var(--dash-surface-2)] px-2 py-1 text-[10px] text-[var(--dash-dim)]">
          {t("backtestLab.dataSourceChain")}
        </code>
      </div>

      {/* 7 大引擎 */}
      <section className="mt-8">
        <div className="mb-3">
          <h2 className="text-lg font-semibold text-[var(--dash-text-bright)]">{t("backtestLab.enginesTitle")}</h2>
          <p className="text-sm text-[var(--dash-dim)]">{t("backtestLab.enginesSubtitle")}</p>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {ENGINES.map((m, i) => (
            <MethodCardView key={m.id} m={m} index={i + 1} />
          ))}
        </div>
      </section>

      {/* 能力对比雷达 */}
      <section className="mt-8">
        <div className="mb-3">
          <h2 className="text-lg font-semibold text-[var(--dash-text-bright)]">{t("backtestLab.comparisonTitle")}</h2>
          <p className="text-sm text-[var(--dash-dim)]">{t("backtestLab.comparisonSubtitle")}</p>
        </div>
        <div className="rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-3">
          <CapabilityRadar engines={ENGINES} />
        </div>
      </section>

      {/* 统计检验 */}
      <section className="mt-8">
        <div className="mb-3">
          <h2 className="text-lg font-semibold text-[var(--dash-text-bright)]">{t("backtestLab.validationTitle")}</h2>
          <p className="text-sm text-[var(--dash-dim)]">{t("backtestLab.validationSubtitle")}</p>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {VALIDATIONS.map((m) => (
            <MethodCardView key={m.id} m={m} />
          ))}
        </div>
      </section>

      {/* 影子账户横幅 */}
      <section className="mt-8">
        <div className="relative overflow-hidden rounded-xl border border-[var(--dash-accent-cyan)]/40 bg-gradient-to-r from-[var(--dash-surface)] to-[var(--dash-surface-2)] p-5">
          <div
            className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full opacity-20 blur-2xl"
            style={{ background: "var(--dash-accent-cyan)" }}
          />
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-[var(--dash-accent-cyan)]/15 text-[var(--dash-accent-cyan)]">
              <Ghost className="h-6 w-6" />
            </div>
            <div className="flex-1">
              <h3 className="text-base font-semibold text-[var(--dash-text-bright)]">{t("backtestLab.shadowTitle")}</h3>
              <p className="mt-1 text-[13px] leading-relaxed text-[var(--dash-text)]">{t("backtestLab.shadowDesc")}</p>
            </div>
            <Link
              to="/agent"
              className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-[var(--dash-accent-cyan)] px-4 py-2 text-sm font-medium text-[#04121a] transition-opacity hover:opacity-90"
            >
              {t("backtestLab.ctaButton")}
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mt-8 mb-4 rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-5 text-center">
        <h3 className="text-base font-semibold text-[var(--dash-text-bright)]">{t("backtestLab.ctaTitle")}</h3>
        <p className="mx-auto mt-1 max-w-xl text-[13px] text-[var(--dash-dim)]">{t("backtestLab.ctaDesc")}</p>
        <div className="mt-4 flex justify-center gap-3">
          <Link
            to="/agent"
            className="inline-flex items-center gap-1.5 rounded-md bg-[var(--dash-accent-orange)] px-4 py-2 text-sm font-medium text-[#1a0d00] transition-opacity hover:opacity-90"
          >
            {t("backtestLab.ctaButton")}
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/reports"
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--dash-border)] bg-[var(--dash-surface-2)] px-4 py-2 text-sm font-medium text-[var(--dash-text)] transition-colors hover:border-[var(--dash-accent-cyan)]"
          >
            {t("backtestLab.ctaButton2")}
          </Link>
        </div>
      </section>
    </div>
  );
}
