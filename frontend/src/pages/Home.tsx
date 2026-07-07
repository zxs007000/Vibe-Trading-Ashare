import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Brain,
  Download,
  RefreshCw,
  Sparkles,
  Target,
  TrendingUp,
  Zap,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { FactorCosmosCanvas } from "@/components/charts/FactorCosmosCanvas";
import { SAMPLE_COSMOS } from "@/lib/factorCosmos";
import type { FactorCosmosResponse, FactorStatus } from "@/lib/factorCosmos";
import { echarts } from "@/lib/echarts";

/* ─── Demo KPI data (TODO: replace with backend real metrics) ─── */
const DEMO_KPI = [
  {
    key: "biggestWin" as const,
    value: "×125",
    sub: "ENTRY $62 → $3,742",
    icon: Zap,
    colorClass: "text-neon-orange",
  },
  {
    key: "totalPnl" as const,
    value: "$233,432",
    sub: "ALL-TIME · 95 DAYS · +2,146×",
    icon: TrendingUp,
    colorClass: "text-neon-green",
  },
  {
    key: "sharpe" as const,
    value: "4.70",
    sub: "APY 50% · HOLD 5M 25S",
    icon: BarChart3,
    colorClass: "text-neon-cyan",
  },
  {
    key: "winRate" as const,
    value: "47.0%",
    sub: "15,891 TRADES · AVG R/R 3.92",
    icon: Target,
    colorClass: "text-neon-pink",
  },
];

const DEMO_TICKER = [
  { sym: "BTC", price: "62,955", chg: "+0.23%", up: true },
  { sym: "ETH", price: "3,373", chg: "+0.41%", up: true },
  { sym: "SOL", price: "142.32", chg: "-2.14%", up: false },
  { sym: "XRP", price: "0.52", chg: "+1.08%", up: true },
  { sym: "CSI300", price: "3,891", chg: "+0.15%", up: true },
];

const UNIVERSES = ["csi300", "csi500", "hs300"];
const PERIODS = ["1y", "2y", "3y"];

/* ══════════════════════════════════════════
   Sub-components (defined inside module scope
   so they share imports without prop-drilling)
   ══════════════════════════════════════════ */

function KPICard({
  icon: Icon,
  value,
  sub,
  colorClass,
}: {
  icon: React.ComponentType<{ className?: string }>;
  value: string;
  sub: string;
  colorClass: string;
}) {
  return (
    <div className="kpi-card p-2.5 sm:p-4 flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs text-[var(--dash-dim)]">
        <Icon className="h-3.5 w-3.5" />
      </div>
      <div className={`text-2xl sm:text-3xl font-bold tracking-tight ${colorClass}`}>
        {value}
      </div>
      <div className="text-[11px] text-[var(--dash-dim)] leading-tight font-mono">
        {sub}
      </div>
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

function TickerBar() {
  return (
    <div className="ticker-wrap border-b border-[var(--dash-border)] bg-[var(--dash-surface)] px-4 py-1.5 hidden xl:block overflow-hidden">
      <div className="ticker-inner inline-flex gap-5 sm:gap-6 text-[10px] sm:text-[11px] font-mono whitespace-nowrap">
        {/* duplicate for seamless loop */}
        {[...DEMO_TICKER, ...DEMO_TICKER].map((t, i) => (
          <span key={i} className="flex items-center gap-1 shrink-0">
            <span className="text-[var(--dash-dim)]">{t.sym}</span>
            <span className="text-[var(--dash-text)]">${t.price}</span>
            <span className={t.up ? "text-neon-green" : "text-neon-pink"}>
              {t.up ? "↑" : "↓"}{t.chg}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

function EquityChartPanel({ compact = false }: { compact?: boolean }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  // Generate demo equity curve (smooth upward with volatility)
  const demoData = useMemo(() => {
    const pts: number[] = [];
    let v = 10000;
    for (let i = 0; i < 120; i++) {
      v += (Math.random() - 0.42) * 400 + 80; // slight positive drift
      pts.push(Math.round(v));
    }
    return pts.map((v, i) => [i, v]);
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: "transparent",
      grid: { left: 45, right: 12, top: 16, bottom: 24 },
      xAxis: { type: "category", show: false },
      yAxis: {
        type: "value",
        axisLabel: { color: "#5a5a80", fontSize: 10, formatter: (v: number) => `${(v / 1000).toFixed(0)}k` },
        splitLine: { lineStyle: { color: "#1a1a2e" } },
      },
      tooltip: { trigger: "axis", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
      series: [{
        type: "line",
        data: demoData,
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
  }, [demoData]);

  return (
    <div className="kpi-card p-3">
      <div className="text-[11px] text-[var(--dash-dim)] mb-2 font-mono flex items-center gap-1.5">
        <TrendingUp className="h-3 w-3" /> {t("dashboard.equityTitle")}
      </div>
      <div ref={ref} className="w-full" style={{ height: compact ? 100 : 140 }} />
    </div>
  );
}

function DistributionChartPanel({ data, compact = false }: { data: FactorCosmosResponse | null; compact?: boolean }) {
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
        {
          name: "Alive", type: "bar", stack: "status",
          data: bars.map(([, v]) => v.alive),
          itemStyle: { color: "#7fe7ff" },
        },
        {
          name: "Decaying", type: "bar", stack: "status",
          data: bars.map(([, v]) => v.decaying),
          itemStyle: { color: "#ffaa44" },
        },
        {
          name: "Dead", type: "bar", stack: "status",
          data: bars.map(([, v]) => v.dead),
          itemStyle: { color: "#3a3a4a" },
        },
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
      <div ref={ref} className="w-full" style={{ height: compact ? 100 : 140 }} />
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
      // Dynamic import to avoid bundling api when not needed at runtime here.
      // Inline POST call keeps the dependency light.
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

  const toggleStatus = (s: FactorStatus) => {
    setFilter(prev => {
      const next = new Set(prev);
      next.has(s) ? next.delete(s) : next.add(s);
      return next;
    });
  };

  const filterArr = filter.size === 3 ? null : [...filter];
  const stats = data?.stats;

  return (
    <div className="dashboard min-h-screen flex flex-col">
      {/* ── Top bar ── */}
      <DashboardTopBar clock={clock} />

      {/* ── Ticker ── */}
      <TickerBar />

      {/* ── Main scrollable area ── */}
      <div className="flex-1 overflow-auto p-2 sm:p-3 lg:p-4 space-y-2 sm:space-y-3">

        {/* ── KPI Grid ── */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-1.5 sm:gap-2 lg:gap-3">
          {DEMO_KPI.map(kpi => (
            <KPICard
              key={kpi.key}
              icon={kpi.icon}
              value={kpi.value}
              sub={kpi.sub}
              colorClass={kpi.colorClass}
            />
          ))}
          {/* Extra KPI: factor counts from cosmos data */}
          <div className="kpi-card p-2.5 sm:p-4 flex flex-col gap-1">
            <div className="flex items-center gap-2 text-xs text-[var(--dash-dim)]">
              <Brain className="h-3.5 w-3.5" />
            </div>
            <div className="text-2xl sm:text-3xl font-bold tracking-tight text-neon-cyan">
              {stats?.alive ?? 23}
            </div>
            <div className="text-[11px] text-[var(--dash-dim)] leading-tight font-mono">
              {t("dashboard.kpiAliveFactors")} / {t("dashboard.kpiTotalFactors")} {stats?.total ?? 61}
            </div>
          </div>
        </div>

        {/* ── Starfield Panel ── */}
        <div className="kpi-card overflow-hidden">
          {/* Panel header */}
          <div className="flex flex-wrap items-center gap-2 px-3 sm:px-4 pt-3 pb-2 border-b border-[var(--dash-border)]">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-[var(--dash-accent-orange)]" />
              <span className="text-sm font-bold tracking-wide">{t("dashboard.starfieldTitle")}</span>
              <span className="text-[10px] text-[var(--dash-dim)] hidden sm:inline">
                {universe.toUpperCase()} · {period.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {/* Status chips */}
              {(["alive", "decaying", "dead"] as FactorStatus[]).map(s => {
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
              {/* Controls */}
              <select
                value={universe}
                onChange={e => setUniverse(e.target.value)}
                className="bg-[var(--dash-surface-2)] border border-[var(--dash-border)] rounded text-[10px] text-[var(--dash-text)] px-1.5 py-0.5"
              >
                {UNIVERSES.map(u => <option key={u} value={u}>{u.toUpperCase()}</option>)}
              </select>
              <select
                value={period}
                onChange={e => setPeriod(e.target.value)}
                className="bg-[var(--dash-surface-2)] border border-[var(--dash-border)] rounded text-[10px] text-[var(--dash-text)] px-1.5 py-0.5"
              >
                {PERIODS.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
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
              Open full view →
            </Link>
          </div>

          {/* Canvas area — must have explicit height so getBoundingClientRect returns non-zero */}
          <div
            className="relative bg-black w-full"
            style={{ height: "clamp(320px, 50vh, 520px)" }}
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

        {/* ── Bottom charts ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 lg:gap-3">
          <EquityChartPanel compact />
          <DistributionChartPanel data={data} compact />
        </div>

        {/* Spacer for bottom padding */}
        <div className="h-6" />
      </div>
    </div>
  );
}
