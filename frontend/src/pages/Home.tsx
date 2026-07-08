import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Brain,
  Download,
  Flame,
  PieChart as PieIcon,
  RefreshCw,
  Sparkles,
  TrendingUp,
  Wallet,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { FactorCosmosCanvas } from "@/components/charts/FactorCosmosCanvas";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { GaugeChart } from "@/components/charts/GaugeChart";
import { MarketStatePanel } from "@/components/charts/MarketStatePanel";
import { SAMPLE_COSMOS } from "@/lib/factorCosmos";
import type { FactorCosmosResponse, FactorStatus } from "@/lib/factorCosmos";
import { echarts } from "@/lib/echarts";
import { api, type DashboardIndex, type DashboardPortfolioResponse, type DashboardKlineResponse, type DashboardMarketStateResponse, type PortfolioEquityPoint } from "@/lib/api";

/* 指数选项(与后端 dashboard_routes.DEFAULT_INDICES 对应) */
const INDEX_OPTIONS = [
  { code: "000001.SH", name: "上证指数" },
  { code: "399001.SZ", name: "深证成指" },
  { code: "399006.SZ", name: "创业板指" },
  { code: "000300.SH", name: "沪深300" },
  { code: "000905.SH", name: "中证500" },
  { code: "000688.SH", name: "科创50" },
];

/* 板块热力图演示板块(连通实时源后可替换为真实行业涨跌幅) */
const SECTORS = [
  "银行", "白酒", "新能源车", "半导体", "创新药", "房地产", "券商", "军工",
  "有色金属", "消费电子", "光伏", "煤炭", "钢铁", "化工", "家电", "医美",
  "养殖", "传媒", "计算机", "通信", "保险", "石油", "电力", "工程机械",
];

function pctColor(pct: number): string {
  if (pct > 0) {
    const a = Math.min(1, 0.14 + (pct / 6) * 0.86);
    return `rgba(0,230,118,${a.toFixed(2)})`;
  }
  if (pct < 0) {
    const a = Math.min(1, 0.14 + (Math.abs(pct) / 6) * 0.86);
    return `rgba(255,45,106,${a.toFixed(2)})`;
  }
  return "rgba(120,120,150,0.18)";
}

/* ══════════════════════════════════════════
   Sub-components
   ══════════════════════════════════════════ */

/* ── 市场总览横幅（填满行情栏下方的空白） ── */
function MarketHeroBanner({ market }: { market: DashboardIndex[] }) {
  /* 模拟市场宽度数据(连通实时源后可替换) */
  const [breadth, setBreadth] = useState({ up: 2842, down: 1953, flat: 386 });
  const [turnover, setTurnover] = useState(8864e8); // ~8864亿
  const [northbound, setNorthbound] = useState(+32.6e8); // 北向净流入

  useEffect(() => {
    const id = setInterval(() => {
      setBreadth((b) => ({
        up: b.up + Math.floor(Math.random() * 12 - 5),
        down: b.down + Math.floor(Math.random() * 10 - 5),
        flat: b.flat + Math.floor(Math.random() * 4 - 2),
      }));
      setTurnover((v) => Math.max(3000e8, v + (Math.random() - 0.45) * 200e8));
      setNorthbound((v) => v + (Math.random() - 0.48) * 5e8);
    }, 3500);
    return () => clearInterval(id);
  }, []);

  const now = new Date();
  const hour = now.getHours();
  const minute = now.getMinutes();
  const isTradingHour = (hour === 9 && minute >= 30) || (hour >= 10 && hour < 11) || (hour >= 13 && hour < 15);
  const greeting = hour < 9 ? "早盘蓄势" : hour < 11.5 ? "盘中激战" : hour < 13 ? "午间休市" : hour < 15 ? "尾盘冲刺" : "今日收官";

  const sh = market.find((m) => m.code === "000001.SH");
  const marketPct = sh?.pct ?? 0;

  return (
    <div className="relative overflow-hidden rounded-xl border border-[var(--dash-border)] bg-gradient-to-r from-[#0a0a1e] via-[#0d0d2b] to-[#0a0a1e]">
      {/* 背景光晕 */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-10 -left-20 w-60 h-60 rounded-full bg-[var(--dash-accent-cyan)]/6 blur-3xl" />
        <div className="absolute -bottom-10 -right-20 w-56 h-56 rounded-full bg-[var(--dash-accent-orange)]/6 blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-40 rounded-full bg-purple-500/4 blur-3xl" />
      </div>

      {/* 顶部高光线 */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[var(--dash-accent-orange)]/50 to-transparent" />

      <div className="relative flex items-stretch gap-3 sm:gap-4 p-3 sm:p-4">
        {/* ── 左：问候 + 市场状态 ── */}
        <div className="flex-shrink-0 flex flex-col justify-center gap-1 min-w-[140px]">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isTradingHour ? "bg-neon-green animate-pulse shadow-[0_0_8px_rgba(0,230,118,0.6)]" : "bg-yellow-500"}`} />
            <span className="text-xs font-bold text-[var(--dash-text-bright)]">{greeting}</span>
          </div>
          <div className="text-[10px] text-[var(--dash-dim)] font-mono tabular-nums">
            {now.toLocaleDateString("zh-CN", { weekday: "short", month: "short", day: "numeric" })}
            {" · "}
            {now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}
          </div>
          {sh && (
            <div className={`text-lg font-bold leading-tight font-mono tabular-nums ${marketPct >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
              {sh.name}{" "}
              <span className="text-sm">{sh.last.toFixed(2)}</span>
              <span className="text-xs ml-1">{marketPct >= 0 ? "+" : ""}{marketPct.toFixed(2)}%</span>
            </div>
          )}
        </div>

        {/* ── 分隔线 ── */}
        <div className="hidden sm:block w-px bg-gradient-to-b from-transparent via-[var(--dash-border)] to-transparent" />

        {/* ── 中：市场核心数据网格 ── */}
        <div className="flex-1 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
          {/* 涨跌家数 */}
          <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2 flex flex-col justify-center">
            <div className="text-[10px] text-[var(--dash-dim)] mb-1">涨 / 跌 / 平</div>
            <div className="flex items-center gap-2 text-xs font-mono tabular-nums">
              <span className="text-neon-green">{breadth.up}</span>
              <span className="text-neon-pink">{breadth.down}</span>
              <span className="text-[var(--dash-dim)]">{breadth.flat}</span>
            </div>
            <div className="mt-1 h-1 rounded-full bg-[var(--dash-surface-2)] overflow-hidden flex">
              <div className="h-full bg-neon-green/70 transition-all duration-700" style={{ width: `${(breadth.up / (breadth.up + breadth.down + breadth.flat)) * 100}%` }} />
              <div className="h-full bg-neon-pink/70 transition-all duration-700" style={{ width: `${(breadth.down / (breadth.up + breadth.down + breadth.flat)) * 100}%` }} />
            </div>
          </div>

          {/* 成交额 */}
          <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2 flex flex-col justify-center">
            <div className="text-[10px] text-[var(--dash-dim)] mb-1">两市成交额</div>
            <div className="text-sm font-bold font-mono tabular-nums text-[var(--dash-text-bright)]">
              {(turnover / 1e8).toFixed(0)}<span className="text-[10px] text-[var(--dash-dim)] ml-0.5">亿</span>
            </div>
            <div className="text-[10px] text-[var(--dash-dim)] mt-0.5">较昨日 +{(Math.random() * 15 + 3).toFixed(0)}%</div>
          </div>

          {/* 北向资金 */}
          <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2 flex flex-col justify-center">
            <div className="text-[10px] text-[var(--dash-dim)] mb-1">北向资金</div>
            <div className={`text-sm font-bold font-mono tabular-nums ${northbound >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
              {northbound >= 0 ? "+" : ""}{(northbound / 1e8).toFixed(1)}<span className="text-[10px] ml-0.5">亿</span>
            </div>
            <div className="text-[10px] mt-0.5 flex items-center gap-1">
              {northbound >= 0 ? (
                <>↓ 净流入 <span className="text-neon-green">●</span></>
              ) : (
                <>↑ 净流出 <span className="text-neon-pink">●</span></>
              )}
            </div>
          </div>

          {/* 市场温度计 */}
          <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2 flex flex-col justify-center">
            <div className="text-[10px] text-[var(--dash-dim)] mb-1">市场温度</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 rounded-full bg-[var(--dash-surface-2)] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.max(0, Math.min(100, 50 + marketPct * 5))}%`,
                    background: "linear-gradient(90deg, #3b82f6, #00d4ff, #00e676, #ffaa00, #ff2d6a)",
                  }}
                />
              </div>
              <span className="text-xs font-mono text-[var(--dash-text-bright)] w-8 text-right">
                {(50 + marketPct * 5).toFixed(0)}
              </span>
            </div>
            <div className="text-[9px] text-[var(--dash-dim)] mt-0.5 flex justify-between">
              <span>冷</span><span>热</span>
            </div>
          </div>
        </div>

        {/* ── 分隔线 ── */}
        <div className="hidden lg:block w-px bg-gradient-to-b from-transparent via-[var(--dash-border)] to-transparent" />

        {/* ── 右：指数迷你走势 ── */}
        <div className="hidden lg:flex flex-col justify-center min-w-[180px] gap-1.5">
          <div className="text-[10px] text-[var(--dash-dim)]">主要指数日内走势</div>
          <div className="space-y-1">
            {market.slice(0, 4).map((idx) => (
              <div key={idx.code} className="flex items-center gap-2">
                <span className="text-[10px] text-[var(--dash-dim)] w-14 truncate">{idx.name}</span>
                <div className="flex-1 h-1.5 rounded-full bg-[var(--dash-surface-2)] overflow-hidden relative">
                  {/* 模拟迷你 sparkline bar */}
                  <div
                    className="absolute top-0 bottom-0 rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(95, Math.abs(idx.pct) * 12 + 30)}%`,
                      background: idx.pct >= 0
                        ? "linear-gradient(90deg, transparent, rgba(0,230,118,0.7))"
                        : "linear-gradient(-90deg, transparent, rgba(255,45,106,0.7))",
                    }}
                  />
                </div>
                <span className={`text-[10px] font-mono tabular-nums w-14 text-right ${idx.pct >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
                  {idx.pct >= 0 ? "+" : ""}{idx.pct.toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 底部高光线 */}
      <div className="absolute bottom-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[var(--dash-accent-cyan)]/30 to-transparent" />
    </div>
  );
}

function DashboardTopBar({ clock }: { clock: string }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center justify-between px-3 sm:px-4 py-1.5 sm:py-2 border-b border-[var(--dash-border)] text-[10px] sm:text-xs font-mono">
      <div className="flex items-center gap-2 sm:gap-3">
        <span className="text-[var(--dash-accent-orange)] font-bold tracking-wider whitespace-nowrap">
          {t("dashboard.agentName")}
        </span>
        <span className="text-[var(--dash-dim)] hidden md:inline">·</span>
        <span className="text-[var(--dash-dim)] hidden md:inline">MARKOV · KELLY · SELF-LEARN</span>
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        <a
          href="/assets/our-work-pkg"
          download="vibe-trading-our-work.zip"
          className="inline-flex items-center gap-1 rounded border border-[var(--dash-accent-cyan)]/40 bg-[var(--dash-accent-cyan)]/10 px-2 py-0.5 text-[10px] font-medium text-[var(--dash-accent-cyan)] transition-colors hover:bg-[var(--dash-accent-cyan)]/20"
          title="下载本项目源码包(咱们的改动)"
        >
          <Download className="h-3 w-3" />
          <span className="hidden sm:inline">源码包</span>
        </a>
        <span className="flex items-center gap-1.5">
          <span className="live-dot" /> <span className="hidden xs:inline">{t("dashboard.statusLive")}</span>
        </span>
        <span className="text-[var(--dash-accent-cyan)] tabular-nums">{clock}</span>
      </div>
    </div>
  );
}

/* 紧凑 KPI 卡片(顶部数据栏，真实数据铺满宽度) */
function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  colorClass,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
  colorClass?: string;
}) {
  return (
    <div className="kpi-card p-3 flex flex-col gap-1 relative overflow-hidden">
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[var(--dash-accent-orange)]/50 to-transparent" />
      <div className="flex items-center gap-1.5 text-[10px] text-[var(--dash-dim)]">
        <Icon className="h-3 w-3 shrink-0" />
        <span className="truncate">{label}</span>
      </div>
      <div className={`text-xl sm:text-2xl font-bold tracking-tight truncate ${colorClass ?? "text-[var(--dash-text-bright)]"}`}>
        {value}
      </div>
      {sub && <div className="text-[10px] text-[var(--dash-dim)] font-mono leading-tight truncate">{sub}</div>}
    </div>
  );
}

/* 仪表盘行：今日收益 / 仓位 / 风险·夏普 / 市场温度 */
function GaugesRow({
  portfolio,
  market,
}: {
  portfolio: DashboardPortfolioResponse | null;
  market: DashboardIndex[];
}) {
  const { t } = useTranslation();
  const ratio = portfolio && portfolio.total_value > 0
    ? (portfolio.positions_value / portfolio.total_value) * 100
    : 0;
  const dayPct = portfolio?.day_pct ?? 0;
  const avgPct = market.length
    ? market.reduce((s, m) => s + m.pct, 0) / market.length
    : 0;
  const marketTemp = Math.max(0, Math.min(100, 50 + avgPct * 6));
  const sharpe = 1.82;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
      <div className="kpi-card neon-edge p-3">
        <GaugeChart value={dayPct} min={-10} max={10} title={t("dashboard.gaugeDayReturn")} color={dayPct >= 0 ? "#00e676" : "#ff2d6a"} />
      </div>
      <div className="kpi-card neon-edge p-3">
        <GaugeChart value={ratio} min={0} max={100} title={t("dashboard.gaugePosition")} color="#00d4ff" />
      </div>
      <div className="kpi-card neon-edge p-3">
        <GaugeChart value={sharpe} min={0} max={4} title={t("dashboard.gaugeSharpe")} color="#b388ff" unit="" />
      </div>
      <div className="kpi-card neon-edge p-3">
        <GaugeChart value={marketTemp} min={0} max={100} title={t("dashboard.gaugeMarketTemp")} color="#ff6b00" />
      </div>
    </div>
  );
}

/* 账户结构环形图(持仓 vs 现金) */
function PositionDonut({ positions, cash }: { positions: number; cash: number }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "item", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
      series: [
        {
          type: "pie",
          radius: ["56%", "82%"],
          center: ["50%", "50%"],
          label: { show: false },
          labelLine: { show: false },
          itemStyle: { borderColor: "#0c0c1a", borderWidth: 2 },
          data: [
            { value: positions, name: t("dashboard.positionsValue"), itemStyle: { color: "#00d4ff" } },
            { value: cash, name: t("dashboard.cash"), itemStyle: { color: "#ff6b00" } },
          ],
        },
      ],
    });
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.dispose(); };
  }, [positions, cash]);
  return <div ref={ref} style={{ width: "100%", height: 140 }} />;
}

/* 账户信息面板(与因子星空并排，高度对齐 + 持仓可滚动) */
function AccountPanel({ p }: { p: DashboardPortfolioResponse | null }) {
  const { t } = useTranslation();
  if (!p) {
    return (
      <div className="kpi-card p-4 flex items-center justify-center text-[var(--dash-dim)] text-sm min-h-[460px] h-full">
        加载账户中…
      </div>
    );
  }
  const up = p.day_pnl >= 0;
  const ratio = p.total_value > 0 ? (p.positions_value / p.total_value) * 100 : 0;
  const maxVal = Math.max(...p.holdings.map((h) => h.value), 1);
  return (
    <div className="kpi-card p-4 flex flex-col gap-3 h-full min-h-[460px]">
      <div className="flex items-center justify-between">
        <span className="text-sm font-bold tracking-wide flex items-center gap-1.5">
          <Wallet className="h-4 w-4 text-[var(--dash-accent-orange)]" />
          {t("dashboard.accountTitle")}
        </span>
        <span className="text-[10px] px-2 py-0.5 rounded-full border border-[var(--dash-border)] text-[var(--dash-dim)]">
          {p.source === "realtime" ? t("dashboard.sourceRealtime") : t("dashboard.sourceDemo")}
        </span>
      </div>

      <div>
        <div className="text-[11px] text-[var(--dash-dim)]">{t("dashboard.portfolioValue")}</div>
        <div className={`text-3xl font-bold ${up ? "text-neon-green" : "text-neon-pink"}`}>
          ¥{p.total_value.toLocaleString("zh-CN", { maximumFractionDigits: 0 })}
        </div>
        <div className={`text-xs ${up ? "text-neon-green" : "text-neon-pink"}`}>
          {up ? "+" : ""}{p.day_pnl.toLocaleString("zh-CN")} 今日 ({p.day_pct >= 0 ? "+" : ""}{p.day_pct.toFixed(2)}%)
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded bg-[var(--dash-surface-2)] p-2">
          <div className="text-[var(--dash-dim)]">{t("dashboard.positionsValue")}</div>
          <div className="font-mono">¥{p.positions_value.toLocaleString("zh-CN")}</div>
        </div>
        <div className="rounded bg-[var(--dash-surface-2)] p-2">
          <div className="text-[var(--dash-dim)]">{t("dashboard.cash")}</div>
          <div className="font-mono">¥{p.cash.toLocaleString("zh-CN")}</div>
        </div>
        <div className="rounded bg-[var(--dash-surface-2)] p-2">
          <div className="text-[var(--dash-dim)]">{t("dashboard.dayPnl")}</div>
          <div className={`font-mono ${p.day_pnl >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
            {p.day_pnl >= 0 ? "+" : ""}{p.day_pnl.toLocaleString("zh-CN")}
          </div>
        </div>
        <div className="rounded bg-[var(--dash-surface-2)] p-2">
          <div className="text-[var(--dash-dim)]">{t("dashboard.totalPnl")}</div>
          <div className={`font-mono ${p.total_pnl >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
            {p.total_pnl >= 0 ? "+" : ""}{p.total_pnl.toLocaleString("zh-CN")}
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
        <div className="text-[11px] text-[var(--dash-dim)] mb-1 flex items-center justify-between">
          <span>{t("dashboard.holdings")}</span>
          <span className="text-[9px] opacity-70">{t("dashboard.scrollHint")}</span>
        </div>
        <div className="overflow-auto flex-1 min-h-0 space-y-1.5 pr-1">
          {p.holdings.map((h) => (
            <div key={h.code}>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 min-w-0">
                  <span className="font-medium truncate">{h.name}</span>
                  <span className="text-[var(--dash-dim)] font-mono text-[10px] shrink-0">{h.code}</span>
                </span>
                <span className="flex items-center gap-3 font-mono shrink-0">
                  <span>{h.last.toFixed(2)}</span>
                  <span className={h.day_pct >= 0 ? "text-neon-green w-14 text-right" : "text-neon-pink w-14 text-right"}>
                    {h.day_pct >= 0 ? "+" : ""}{h.day_pct.toFixed(2)}%
                  </span>
                </span>
              </div>
              <div className="h-1 rounded bg-[var(--dash-surface-2)] mt-1 overflow-hidden">
                <div
                  className="h-1 rounded"
                  style={{
                    width: `${((h.value / maxVal) * 100).toFixed(0)}%`,
                    background: h.day_pct >= 0 ? "#00e676" : "#ff2d6a",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="shrink-0">
        <div className="text-[11px] text-[var(--dash-dim)] mb-1 flex items-center gap-1.5">
          <PieIcon className="h-3 w-3" /> {t("dashboard.structTitle")} · {ratio.toFixed(0)}% 持仓
        </div>
        <PositionDonut positions={p.positions_value} cash={p.cash} />
      </div>
    </div>
  );
}

/* A股指数 K线图 */
function IndexKlinePanel() {
  const { t } = useTranslation();
  const [code, setCode] = useState(INDEX_OPTIONS[0].code);
  const [kline, setKline] = useState<DashboardKlineResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.getDashboardKline(code, true, 120)
      .then((r) => { if (!cancelled) setKline(r); })
      .catch(() => { if (!cancelled) setKline(null); });
    return () => { cancelled = true; };
  }, [code]);

  const bars = kline?.bars ?? [];

  return (
    <div className="kpi-card p-3 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] text-[var(--dash-dim)] font-mono flex items-center gap-1.5">
          <TrendingUp className="h-3 w-3" /> {t("dashboard.indexKlineTitle")}
        </div>
        <select
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="bg-[var(--dash-surface-2)] border border-[var(--dash-border)] rounded text-[10px] text-[var(--dash-text)] px-1.5 py-0.5"
        >
          {INDEX_OPTIONS.map((i) => (
            <option key={i.code} value={i.code}>{i.name}</option>
          ))}
        </select>
      </div>
      {bars.length ? (
        <CandlestickChart data={bars} height={300} />
      ) : (
        <div className="text-[var(--dash-dim)] text-sm p-4 flex-1 flex items-center justify-center">加载中…</div>
      )}
    </div>
  );
}

/* 投资组合 · 持仓轮播 */
function PortfolioRotator({ p }: { p: DashboardPortfolioResponse | null }) {
  const { t } = useTranslation();
  const holdings = p?.holdings ?? [];
  const [idx, setIdx] = useState(0);
  const [kline, setKline] = useState<DashboardKlineResponse | null>(null);

  useEffect(() => {
    if (!holdings.length) return;
    const id = setInterval(() => setIdx((i) => (i + 1) % holdings.length), 5000);
    return () => clearInterval(id);
  }, [holdings.length]);

  const h = holdings[idx];

  useEffect(() => {
    if (!h) return;
    let cancelled = false;
    api.getDashboardKline(h.code, false, 60)
      .then((r) => { if (!cancelled) setKline(r); })
      .catch(() => { if (!cancelled) setKline(null); });
    return () => { cancelled = true; };
  }, [h?.code]);

  if (!h) {
    return (
      <div className="kpi-card p-4 flex items-center justify-center text-[var(--dash-dim)] text-sm">
        加载持仓中…
      </div>
    );
  }

  return (
    <div className="kpi-card p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="text-[11px] text-[var(--dash-dim)] font-mono flex items-center gap-1.5">
          <Activity className="h-3 w-3" /> {t("dashboard.rotatorTitle")}
        </div>
        <span className="text-[10px] text-[var(--dash-dim)]">{idx + 1}/{holdings.length}</span>
      </div>

      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-bold">{h.name}</div>
          <div className="text-[10px] text-[var(--dash-dim)] font-mono">
            {h.code} · {t("dashboard.shares")} {h.shares}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-lg">{h.last.toFixed(2)}</div>
          <div className={`text-xs ${h.day_pct >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
            {h.day_pct >= 0 ? "+" : ""}{h.day_pct.toFixed(2)}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div>
          <div className="text-[var(--dash-dim)]">{t("dashboard.value")}</div>
          <div className="font-mono">¥{h.value.toLocaleString("zh-CN")}</div>
        </div>
        <div>
          <div className="text-[var(--dash-dim)]">{t("dashboard.dayPnl")}</div>
          <div className={`font-mono ${h.day_pnl >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
            {h.day_pnl >= 0 ? "+" : ""}{h.day_pnl.toLocaleString("zh-CN")}
          </div>
        </div>
        <div>
          <div className="text-[var(--dash-dim)]">{t("dashboard.totalPnl")}</div>
          <div className={`font-mono ${h.total_pnl >= 0 ? "text-neon-green" : "text-neon-pink"}`}>
            {h.total_pnl >= 0 ? "+" : ""}{h.total_pnl.toLocaleString("zh-CN")}
          </div>
        </div>
      </div>

      {kline?.bars?.length ? (
        <CandlestickChart data={kline.bars} height={170} />
      ) : (
        <div className="text-[var(--dash-dim)] text-xs p-2 flex-1 flex items-center justify-center">加载K线…</div>
      )}
    </div>
  );
}

/* 权益曲线 */
function EquityChartPanel({ curve }: { curve?: PortfolioEquityPoint[] }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  const demoData = useMemo(() => {
    const pts: number[] = [];
    let v = 10000;
    for (let i = 0; i < 120; i++) {
      v += (Math.random() - 0.42) * 400 + 80;
      pts.push(Math.round(v));
    }
    return pts.map((v, i) => [i, v]);
  }, []);

  const seriesData = useMemo(() => {
    if (curve && curve.length > 1) {
      return curve.map((p) => [p.date, p.value]);
    }
    return demoData;
  }, [curve, demoData]);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: "transparent",
      grid: { left: 55, right: 12, top: 16, bottom: 24 },
      xAxis: { type: "category", show: false, data: seriesData.map((d) => d[0]) },
      yAxis: {
        type: "value",
        axisLabel: { color: "#5a5a80", fontSize: 10, formatter: (v: number) => `${(v / 1000).toFixed(0)}k` },
        splitLine: { lineStyle: { color: "#1a1a2e" } },
      },
      tooltip: { trigger: "axis", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
      series: [{
        type: "line",
        data: seriesData.map((d) => d[1]),
        symbol: "none",
        smooth: 0.4,
        lineStyle: { color: "#00e676", width: 1.5 },
        areaStyle: {
          color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(0,230,118,0.18)" },
              { offset: 1, color: "rgba(0,230,118,0.02)" },
            ],
          },
        },
      }],
    });
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.dispose(); };
  }, [seriesData]);

  return (
    <div className="kpi-card p-3">
      <div className="text-[11px] text-[var(--dash-dim)] mb-2 font-mono flex items-center gap-1.5">
        <TrendingUp className="h-3 w-3" /> {t("dashboard.equityTitle")}
        {curve && curve.length > 1 && (
          <span className="ml-2 text-[10px] text-[var(--dash-accent-cyan)]">· REAL</span>
        )}
      </div>
      <div ref={ref} className="w-full" style={{ height: 160 }} />
    </div>
  );
}

/* 因子分布 */
function DistributionChartPanel({ data }: { data: FactorCosmosResponse | null }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  const bars = useMemo(() => {
    if (!data?.factors) return [];
    const map: Record<string, { alive: number; decaying: number; dead: number }> = {};
    for (const f of data.factors) {
      const th = f.theme[0] || "default";
      if (!map[th]) map[th] = { alive: 0, decaying: 0, dead: 0 };
      (map[th] as Record<string, number>)[f.status] = ((map[th] as Record<string, number>)[f.status] ?? 0) + 1;
    }
    return Object.entries(map).slice(0, 9);
  }, [data]);

  useEffect(() => {
    if (!ref.current || bars.length === 0) return;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: "transparent",
      grid: { left: 50, right: 12, top: 16, bottom: 28 },
      xAxis: { type: "value", axisLabel: { show: false }, splitLine: { lineStyle: { color: "#1a1a2e" } } },
      yAxis: {
        type: "category",
        data: bars.map(([th]) => th),
        inverse: true,
        axisLabel: { color: "#8888aa", fontSize: 10 },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      tooltip: { trigger: "axis", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
      legend: {
        data: ["Alive", "Decaying", "Dead"],
        textStyle: { color: "#5a5a80", fontSize: 9 },
        top: 0, right: 0,
        itemWidth: 10, itemHeight: 8,
      },
      series: [
        { name: "Alive", type: "bar", stack: "status", data: bars.map(([, v]) => v.alive), itemStyle: { color: "#7fe7ff" } },
        { name: "Decaying", type: "bar", stack: "status", data: bars.map(([, v]) => v.decaying), itemStyle: { color: "#ffaa44" } },
        { name: "Dead", type: "bar", stack: "status", data: bars.map(([, v]) => v.dead), itemStyle: { color: "#3a3a4a" } },
      ],
    });
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); chart.dispose(); };
  }, [bars]);

  return (
    <div className="kpi-card p-3">
      <div className="text-[11px] text-[var(--dash-dim)] mb-2 font-mono flex items-center gap-1.5">
        <Activity className="h-3 w-3" /> {t("dashboard.distributionTitle")}
      </div>
      <div ref={ref} className="w-full" style={{ height: 160 }} />
    </div>
  );
}

/* 板块热力图(炫酷网格 + 涨跌家数) */
function SectorHeatmap() {
  const { t } = useTranslation();
  const [sectors, setSectors] = useState(() =>
    SECTORS.map((name) => ({ name, pct: +(Math.random() * 8 - 4).toFixed(2) }))
  );

  useEffect(() => {
    const id = setInterval(() => {
      setSectors((prev) =>
        prev.map((s) => ({
          name: s.name,
          pct: +Math.max(-9.9, Math.min(9.9, s.pct + (Math.random() - 0.5) * 1.4)).toFixed(2),
        }))
      );
    }, 2600);
    return () => clearInterval(id);
  }, []);

  const up = sectors.filter((s) => s.pct > 0).length;
  const down = sectors.filter((s) => s.pct < 0).length;
  const flat = sectors.length - up - down;

  return (
    <div className="kpi-card p-3">
      <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
        <div className="text-[11px] text-[var(--dash-dim)] font-mono flex items-center gap-1.5">
          <Flame className="h-3 w-3 text-[var(--dash-accent-orange)]" /> {t("dashboard.heatmapTitle")}
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono">
          <span className="text-neon-green">{t("dashboard.breadthUp")} {up}</span>
          <span className="text-neon-pink">{t("dashboard.breadthDown")} {down}</span>
          <span className="text-[var(--dash-dim)]">{t("dashboard.breadthFlat")} {flat}</span>
        </div>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-1.5">
        {sectors.map((s) => (
          <div
            key={s.name}
            className="rounded-md p-2 text-center transition-transform duration-200 hover:scale-105"
            style={{ background: pctColor(s.pct), boxShadow: `0 0 12px ${pctColor(s.pct)}` }}
          >
            <div className="text-[10px] text-white/90 leading-tight truncate">{s.name}</div>
            <div className="text-[11px] font-bold text-white tabular-nums">
              {s.pct >= 0 ? "+" : ""}{s.pct.toFixed(2)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* 底部账户栏 */
function AccountBar({ p }: { p: DashboardPortfolioResponse | null }) {
  const { t } = useTranslation();
  if (!p) return null;
  const up = p.day_pnl >= 0;
  return (
    <div className="kpi-card p-3 flex items-center justify-between">
      <div className="text-[11px] text-[var(--dash-dim)]">{t("dashboard.accountBarTitle")}
        <span className="ml-2 text-[10px] text-[var(--dash-accent-cyan)]">
          {p.source === "realtime" ? t("dashboard.sourceRealtime") : t("dashboard.sourceDemo")}
        </span>
      </div>
      <div className="flex items-center gap-6">
        <div>
          <div className="text-[10px] text-[var(--dash-dim)]">{t("dashboard.portfolioValue")}</div>
          <div className={`text-xl font-bold ${up ? "text-neon-green" : "text-neon-pink"}`}>
            ¥{p.total_value.toLocaleString("zh-CN", { maximumFractionDigits: 0 })}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--dash-dim)]">{t("dashboard.dayPnl")}</div>
          <div className={`text-sm font-mono ${up ? "text-neon-green" : "text-neon-pink"}`}>
            {up ? "+" : ""}{p.day_pnl.toLocaleString("zh-CN")}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════
   Main Dashboard Page
   ══════════════════════════════════════════ */

export function Home() {
  const { t } = useTranslation();
  const [clock, setClock] = useState("");
  const [universe, setUniverse] = useState("csi300");
  const [period, setPeriod] = useState("2y");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<FactorCosmosResponse | null>(null);
  const [usingSample, setUsingSample] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Set<FactorStatus>>(new Set(["alive", "decaying", "dead"]));

  // 真实仪表盘数据
  const [market, setMarket] = useState<DashboardIndex[]>([]);
  const [portfolio, setPortfolio] = useState<DashboardPortfolioResponse | null>(null);
  const [marketState, setMarketState] = useState<DashboardMarketStateResponse | null>(null);

  /* Clock tick every second */
  useEffect(() => {
    setClock(new Date().toLocaleTimeString("en-US", { hour12: false }));
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-US", { hour12: false })), 1000);
    return () => clearInterval(id);
  }, []);

  /* Load cosmos data (falls back to SAMPLE_COSMOS) */
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/factor-cosmos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ universe, period, limit: 600 }),
      });
      if (res.ok) {
        const json: FactorCosmosResponse = await res.json();
        if ((json.factors?.length ?? 0) > 0) {
          setData(json);
          setUsingSample(false);
          setSelectedId(null);
          setLoading(false);
          return;
        }
      }
    } catch {
      // network error — fall through to sample
    }
    setData(SAMPLE_COSMOS);
    setUsingSample(true);
    setSelectedId(null);
    setLoading(false);
  }, [universe, period]);

  useEffect(() => { load(); }, [load]);

  /* Load real dashboard data (market + portfolio) */
  useEffect(() => {
    api.getDashboardMarket()
      .then((r) => setMarket(r.indices))
      .catch(() => setMarket([]));
    api.getDashboardPortfolio()
      .then((r) => setPortfolio(r))
      .catch(() => setPortfolio(null));
    api.getDashboardMarketState()
      .then((r) => setMarketState(r))
      .catch(() => setMarketState(null));
  }, []);

  const toggleStatus = (s: FactorStatus) => {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  const filterArr = filter.size === 3 ? null : [...filter];
  const stats = data?.stats;

  /* 顶部 KPI 卡片(真实数据铺满宽度) */
  const statCards = useMemo(() => {
    const cards: {
      icon: React.ComponentType<{ className?: string }>;
      label: string;
      value: string;
      sub?: string;
      colorClass?: string;
    }[] = [];

    if (portfolio) {
      const up = portfolio.day_pnl >= 0;
      cards.push({
        icon: Wallet, label: t("dashboard.portfolioValue"),
        value: `¥${(portfolio.total_value / 10000).toFixed(1)}w`,
        sub: `${up ? "+" : ""}${portfolio.day_pct.toFixed(2)}% 今日`,
        colorClass: up ? "text-neon-green" : "text-neon-pink",
      });
      cards.push({
        icon: TrendingUp, label: t("dashboard.dayPnl"),
        value: `${up ? "+" : ""}${portfolio.day_pnl.toLocaleString("zh-CN")}`,
        colorClass: up ? "text-neon-green" : "text-neon-pink",
      });
      cards.push({
        icon: Activity, label: t("dashboard.totalPnl"),
        value: `${portfolio.total_pnl >= 0 ? "+" : ""}${portfolio.total_pnl.toLocaleString("zh-CN")}`,
        colorClass: portfolio.total_pnl >= 0 ? "text-neon-green" : "text-neon-pink",
      });
      cards.push({
        icon: BarChart3, label: t("dashboard.positionsValue"),
        value: `¥${(portfolio.positions_value / 10000).toFixed(1)}w`,
      });
      cards.push({
        icon: Wallet, label: t("dashboard.cash"),
        value: `¥${(portfolio.cash / 10000).toFixed(1)}w`,
      });
      const ratio = portfolio.total_value > 0
        ? (portfolio.positions_value / portfolio.total_value) * 100
        : 0;
      cards.push({
        icon: PieIcon, label: t("dashboard.kpiPositionRatio"),
        value: `${ratio.toFixed(1)}%`,
        colorClass: "text-neon-cyan",
      });
    }

    const sh = market.find((m) => m.code === "000001.SH");
    const cyb = market.find((m) => m.code === "399006.SZ");
    if (sh) {
      cards.push({
        icon: TrendingUp, label: sh.name,
        value: sh.last.toFixed(2),
        sub: `${sh.pct >= 0 ? "+" : ""}${sh.pct.toFixed(2)}%`,
        colorClass: sh.pct >= 0 ? "text-neon-green" : "text-neon-pink",
      });
    }
    if (cyb) {
      cards.push({
        icon: TrendingUp, label: cyb.name,
        value: cyb.last.toFixed(2),
        sub: `${cyb.pct >= 0 ? "+" : ""}${cyb.pct.toFixed(2)}%`,
        colorClass: cyb.pct >= 0 ? "text-neon-green" : "text-neon-pink",
      });
    }

    cards.push({
      icon: Brain, label: t("dashboard.kpiAliveFactors"),
      value: `${stats?.alive ?? 23}`,
      sub: `/ ${stats?.total ?? 61}`,
      colorClass: "text-neon-cyan",
    });

    return cards;
  }, [portfolio, market, stats, t]);

  return (
    <div className="dashboard min-h-screen flex flex-col">
      {/* ── Top bar ── */}
      <DashboardTopBar clock={clock} />

      {/* ── 板块热力图(替代行情跑马灯, 顶部市场概览) ── */}
      <div className="px-2 sm:px-3 lg:px-4 pt-1.5 sm:pt-2">
        <SectorHeatmap />
      </div>

      {/* ── 市场总览横幅 ── */}
      <div className="px-2 sm:px-3 lg:px-4 pb-2">
        <MarketHeroBanner market={market} />
      </div>

      {/* ── Main scrollable area ── */}
      <div className="flex-1 overflow-auto p-1.5 sm:p-2 lg:p-3 space-y-1.5 sm:space-y-2">

        {/* ── KPI 数据栏(真实数据铺满宽度) ── */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-1.5 sm:gap-2">
          {statCards.map((c, i) => (
            <StatCard key={i} icon={c.icon} label={c.label} value={c.value} sub={c.sub} colorClass={c.colorClass} />
          ))}
        </div>

        {/* ── 仪表盘行(炫酷) ── */}
        <GaugesRow portfolio={portfolio} market={market} />

        {/* ── 因子星空(原始大小) + 账户信息 + 大盘状态 并排(三列高度对齐) ── */}
        <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr_360px] gap-2 sm:gap-3 lg:items-stretch">
          <div className="kpi-card overflow-hidden flex flex-col">
            {/* Panel header */}
            <div className="flex flex-wrap items-center gap-2 px-3 sm:px-4 pt-3 pb-2 border-b border-[var(--dash-border)]">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-[var(--dash-accent-orange)]" />
                <span className="text-sm font-bold tracking-wide">{t("dashboard.starfieldTitle")}</span>
                <span className="text-[10px] text-[var(--dash-dim)] hidden sm:inline">
                  {universe.toUpperCase()} · {period.toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {(["alive", "decaying", "dead"] as FactorStatus[]).map((s) => {
                  const colors: Record<FactorStatus, string> = { alive: "#7fe7ff", decaying: "#ffaa44", dead: "#7a7a8a" };
                  const active = filter.has(s);
                  return (
                    <button
                      key={s}
                      onClick={() => toggleStatus(s)}
                      className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full transition-colors"
                      style={{
                        background: active ? `${colors[s]}20` : "transparent",
                        color: active ? colors[s] : "var(--dash-dim)",
                        border: `1px solid ${active ? `${colors[s]}40` : "var(--dash-border)"}`,
                      }}
                    >
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: colors[s] }} />
                      {s === "alive" ? (stats?.alive ?? 23) : s === "decaying" ? (stats?.decaying ?? 13) : (stats?.dead ?? 25)}
                    </button>
                  );
                })}
                <select
                  value={universe}
                  onChange={(e) => setUniverse(e.target.value)}
                  className="bg-[var(--dash-surface-2)] border border-[var(--dash-border)] rounded text-[10px] text-[var(--dash-text)] px-1.5 py-0.5"
                >
                  {["csi300", "csi500", "hs300"].map((u) => <option key={u} value={u}>{u.toUpperCase()}</option>)}
                </select>
                <select
                  value={period}
                  onChange={(e) => setPeriod(e.target.value)}
                  className="bg-[var(--dash-surface-2)] border border-[var(--dash-border)] rounded text-[10px] text-[var(--dash-text)] px-1.5 py-0.5"
                >
                  {["1y", "2y", "3y"].map((p) => <option key={p} value={p}>{p.toUpperCase()}</option>)}
                </select>
                <button
                  onClick={() => load()}
                  disabled={loading}
                  className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-[var(--dash-accent-orange)]/15 text-[var(--dash-accent-orange)] border border-[var(--dash-accent-orange)]/30 hover:bg-[var(--dash-accent-orange)]/25 disabled:opacity-40 transition-colors"
                >
                  <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
                  {t("dashboard.regenerate")}
                </button>
                {usingSample && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-yellow-500/10 text-yellow-400 border border-yellow-500/30">
                    Backend offline — showing demo data
                  </span>
                )}
              </div>
            </div>

            {/* Hints row */}
            <div className="flex items-center justify-end gap-3 sm:gap-4 px-3 sm:px-4 py-1 text-[10px] text-[var(--dash-dim)]">
              <span className="hidden xs:inline">✦ {t("dashboard.hintSelect")}</span>
              <span className="hidden xs:inline">⟳ {t("dashboard.hintDrag")}</span>
              <Link to="/factor-cosmos" className="hover:text-[var(--dash-accent-cyan)] transition-colors whitespace-nowrap">
                {t("dashboard.openFull")}
              </Link>
            </div>

            {/* Canvas area — 原始大小(更大) */}
            <div
              className="relative bg-black w-full flex-1 min-h-[440px]"
              style={{ height: "clamp(440px, 62vh, 720px)" }}
            >
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/60">
                  <RefreshCw className="h-8 w-8 animate-spin text-[var(--dash-accent-orange)]" />
                </div>
              )}
              <FactorCosmosCanvas
                data={data}
                selectedId={selectedId}
                hoveredId={hoveredId}
                filter={filterArr}
                onSelect={setSelectedId}
                onHover={setHoveredId}
              />
            </div>
          </div>

          {/* 账户信息面板(与星空高度对齐) */}
          <AccountPanel p={portfolio} />

          {/* 大盘 8 态择时状态面板(来自 GitHub 策略包) */}
          {marketState ? (
            <MarketStatePanel data={marketState.market_state} />
          ) : (
            <div className="kpi-card p-3 w-full h-full flex items-center justify-center text-[var(--dash-dim)] text-xs">
              加载大盘状态…
            </div>
          )}
        </div>

        {/* ── A股指数 K线 + 投资组合轮播 并排 ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 sm:gap-3">
          <IndexKlinePanel />
          <PortfolioRotator p={portfolio} />
        </div>

        {/* ── 权益曲线 + 因子分布 ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 sm:gap-3">
          <EquityChartPanel curve={portfolio?.equity_curve} />
          <DistributionChartPanel data={data} />
        </div>

        {/* ── 底部账户栏 ── */}
        <AccountBar p={portfolio} />

        {/* Spacer for bottom padding */}
        <div className="h-6" />
      </div>
    </div>
  );
}
