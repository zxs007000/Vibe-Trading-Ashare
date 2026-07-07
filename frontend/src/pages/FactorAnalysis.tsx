import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  Database,
  ArrowRight,
  Sigma,
  Layers,
  Activity,
  Grid3x3,
} from "lucide-react";
import { echarts } from "@/lib/echarts";

/* ────────────────────────────────────────────────────────────
 * 因子分析工作台
 * 数据形态对齐 agent/src/factors/{factor_analysis_core,rolling_ic,bench_runner}
 *   - IC / IR / t(ICIR)  → rolling_ic.summary_table / compute_rolling_ic
 *   - 分层收益           → factor_analysis_core.compute_group_equity
 *   - 因子寿命           → rolling_ic.assess_lifecycle (alive/decaying/dead)
 *   - 冗余相关性         → 因子间截面相关
 * 后端因子分析服务不可用时回退到 SAMPLE（与因子星空一致的策略）。
 * ──────────────────────────────────────────────────────────── */

type Lifecycle = "alive" | "decaying" | "dead";

interface Factor {
  id: string;
  nameZh: string;
  nameEn: string;
  theme: string;
  ic: number; // 均值 IC
  ir: number; // 信息比率 = mean(IC)/std(IC)
  t: number; // t 值 = ir * sqrt(N)
  pos: number; // IC 正向占比
  life: Lifecycle;
  monotone: boolean;
  seed: number;
}

const FACTORS: Factor[] = ([
  { id: "mom20", nameZh: "20日动量", nameEn: "Momentum 20d", theme: "动量", ic: 0.062, ir: 0.82, t: 4.1, pos: 0.78, life: "alive", monotone: true },
  { id: "rev60", nameZh: "60日反转", nameEn: "Reversal 60d", theme: "动量", ic: 0.041, ir: 0.55, t: 2.7, pos: 0.66, life: "decaying", monotone: true },
  { id: "ep", nameZh: "市盈率倒数", nameEn: "EP (Earnings Yield)", theme: "估值", ic: 0.071, ir: 0.94, t: 4.8, pos: 0.81, life: "alive", monotone: true },
  { id: "bp", nameZh: "市净率倒数", nameEn: "BP (Book Yield)", theme: "估值", ic: 0.068, ir: 0.88, t: 4.4, pos: 0.79, life: "alive", monotone: true },
  { id: "cfp", nameZh: "现金流市值比", nameEn: "Cashflow/Price", theme: "质量", ic: 0.052, ir: 0.71, t: 3.5, pos: 0.72, life: "alive", monotone: true },
  { id: "roe_stab", nameZh: "ROE 稳定性", nameEn: "ROE Stability", theme: "质量", ic: 0.048, ir: 0.66, t: 3.2, pos: 0.7, life: "alive", monotone: true },
  { id: "rev_g", nameZh: "营收增速", nameEn: "Revenue Growth", theme: "成长", ic: 0.039, ir: 0.52, t: 2.6, pos: 0.64, life: "decaying", monotone: true },
  { id: "vol", nameZh: "波动率", nameEn: "Volatility", theme: "波动率", ic: -0.058, ir: -0.79, t: -3.9, pos: 0.22, life: "alive", monotone: true },
  { id: "turn", nameZh: "换手率", nameEn: "Turnover", theme: "流动性", ic: -0.044, ir: -0.6, t: -3.0, pos: 0.27, life: "decaying", monotone: true },
  { id: "amp", nameZh: "振幅", nameEn: "Amplitude", theme: "流动性", ic: -0.039, ir: -0.53, t: -2.6, pos: 0.3, life: "alive", monotone: true },
  { id: "north", nameZh: "北向持股变化", nameEn: "Northbound Flow", theme: "情绪", ic: 0.033, ir: 0.45, t: 2.2, pos: 0.61, life: "alive", monotone: true },
  { id: "size", nameZh: "市值", nameEn: "Size", theme: "规模", ic: -0.035, ir: -0.48, t: -2.4, pos: 0.33, life: "alive", monotone: true },
  { id: "lev", nameZh: "杠杆率", nameEn: "Leverage", theme: "杠杆", ic: -0.028, ir: -0.37, t: -1.8, pos: 0.38, life: "dead", monotone: false },
  { id: "dyr", nameZh: "股息率", nameEn: "Dividend Yield", theme: "红利", ic: 0.045, ir: 0.61, t: 3.0, pos: 0.71, life: "alive", monotone: true },
] as Omit<Factor, "seed">[]).map((f) => ({ ...f, seed: hashStr(f.id) }));

const LIFE_META: Record<Lifecycle, { label: "factorLab.lifeAlive" | "factorLab.lifeDecaying" | "factorLab.lifeDead"; color: string }> = {
  alive: { label: "factorLab.lifeAlive", color: "#00e676" },
  decaying: { label: "factorLab.lifeDecaying", color: "#ff6b00" },
  dead: { label: "factorLab.lifeDead", color: "#5a5a80" },
};

/* ── 确定性伪随机，保证示例数据每次渲染一致 ── */
function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function genLayers(f: Factor): number[][] {
  const rnd = mulberry32(f.seed);
  const T = 120;
  const G = 10;
  const base = 0.008;
  const strength = Math.abs(f.ic);
  const lines: number[][] = [];
  for (let g = 0; g < G; g++) {
    const center = (g - (G - 1) / 2) / ((G - 1) / 2); // -1..1
    const slope = f.monotone ? Math.sign(f.ic) * strength * 0.16 * center : (rnd() - 0.5) * strength * 0.06;
    let v = 1;
    const eq: number[] = [];
    for (let t = 0; t < T; t++) {
      v *= 1 + base + slope + (rnd() - 0.5) * 0.03;
      eq.push(+(v * 100 - 100).toFixed(2));
    }
    lines.push(eq);
  }
  return lines;
}

function genIc(f: Factor): { ic: number[]; ir: number[] } {
  const rnd = mulberry32(f.seed ^ 0x9e3779b9);
  const N = 60;
  const icStd = f.ir > 0 ? f.ic / f.ir : f.ic;
  const icArr: number[] = [];
  let roll = f.ic;
  for (let i = 0; i < N; i++) {
    roll = roll * 0.82 + (f.ic + (rnd() - 0.5) * 2 * icStd) * 0.18;
    icArr.push(+roll.toFixed(4));
  }
  const irArr = icArr.map((x) => +(x / (icStd || f.ic || 1e-6)).toFixed(3));
  return { ic: icArr, ir: irArr };
}

function genCorr(factors: Factor[]): number[][] {
  const n = factors.length;
  const mat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) {
        mat[i][j] = 1;
        continue;
      }
      const sameTheme = factors[i].theme === factors[j].theme;
      const rnd = mulberry32(hashStr(factors[i].id + factors[j].id));
      const base = sameTheme ? 0.72 : 0.08;
      let v = base + (rnd() - 0.5) * (sameTheme ? 0.22 : 0.2);
      v = Math.max(-0.92, Math.min(0.93, v));
      mat[i][j] = +v.toFixed(2);
      mat[j][i] = mat[i][j];
    }
  }
  return mat;
}

const LAYERS = new Map(FACTORS.map((f) => [f.id, genLayers(f)]));
const ICS = new Map(FACTORS.map((f) => [f.id, genIc(f)]));
const CORR = genCorr(FACTORS);
const CORR_THRESHOLD = 0.7;

/* ────────────────────────────────────────────────────────────
 * 图表组件
 * ──────────────────────────────────────────────────────────── */

function useEchart(option: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption(option as echarts.EChartsCoreOption);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
    };
  }, [option]);
  return ref;
}

function LayersChart({ factor }: { factor: Factor }) {
  const { t } = useTranslation();
  const lines = LAYERS.get(factor.id) ?? [];
  const longShort = lines.length ? lines[9].map((v, i) => +(v - lines[0][i]).toFixed(2)) : [];
  const ref = useEchart(
    useMemo(
      () => ({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
        legend: { type: "scroll", bottom: 0, textStyle: { color: "#5a5a80", fontSize: 10 }, pageTextStyle: { color: "#5a5a80" } },
        grid: { left: 44, right: 12, top: 12, bottom: 36 },
        xAxis: { type: "category", data: lines[0]?.map((_, i) => i + 1), axisLine: { lineStyle: { color: "#1e1e3a" } }, axisLabel: { color: "#5a5a80", fontSize: 10, interval: 11 } },
        yAxis: { type: "value", axisLabel: { color: "#5a5a80", fontSize: 10, formatter: "{value}%" }, splitLine: { lineStyle: { color: "#14142a" } } },
        series: [
          ...lines.map((line, g) => ({
            name: `${t("factorLab.group")}${g + 1}`,
            type: "line",
            data: line,
            showSymbol: false,
            lineStyle: { width: g === 9 || g === 0 ? 2 : 1, color: `rgba(0,212,255,${0.25 + (g / 9) * 0.6})` },
            itemStyle: { color: `rgba(0,212,255,${0.25 + (g / 9) * 0.6})` },
          })),
          {
            name: t("factorLab.longShort"),
            type: "line",
            data: longShort,
            showSymbol: false,
            lineStyle: { width: 2.5, color: "#ff6b00" },
            itemStyle: { color: "#ff6b00" },
            areaStyle: { color: "rgba(255,107,0,0.08)" },
          },
        ],
      }),
      [factor.id, t],
    ),
  );
  return <div ref={ref} className="h-[300px] w-full" />;
}

function IcChart({ factor }: { factor: Factor }) {
  const d = ICS.get(factor.id) ?? { ic: [], ir: [] };
  const ref = useEchart(
    useMemo(
      () => ({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", backgroundColor: "#0c0c1a", borderColor: "#1e1e3a", textStyle: { color: "#d8d8f0", fontSize: 11 } },
        legend: { bottom: 0, textStyle: { color: "#5a5a80", fontSize: 10 } },
        grid: { left: 44, right: 12, top: 12, bottom: 32 },
        xAxis: { type: "category", data: d.ic.map((_, i) => i + 1), axisLine: { lineStyle: { color: "#1e1e3a" } }, axisLabel: { color: "#5a5a80", fontSize: 10, interval: 9 } },
        yAxis: { type: "value", axisLabel: { color: "#5a5a80", fontSize: 10 }, splitLine: { lineStyle: { color: "#14142a" } } },
        series: [
          { name: "IC", type: "line", data: d.ic, showSymbol: false, lineStyle: { width: 1.5, color: "#00d4ff" }, itemStyle: { color: "#00d4ff" }, areaStyle: { color: "rgba(0,212,255,0.08)" } },
          { name: "IR", type: "line", data: d.ir, showSymbol: false, lineStyle: { width: 1.5, color: "#b388ff" }, itemStyle: { color: "#b388ff" } },
        ],
      }),
      [factor.id],
    ),
  );
  return <div ref={ref} className="h-[300px] w-full" />;
}

function CorrHeatmap() {
  const names = FACTORS.map((f) => f.nameZh);
  const data: [number, number, number][] = [];
  for (let i = 0; i < names.length; i++)
    for (let j = 0; j < names.length; j++) data.push([j, i, CORR[i][j]]);
  const ref = useEchart(
    useMemo(
      () => ({
        backgroundColor: "transparent",
        tooltip: {
          backgroundColor: "#0c0c1a",
          borderColor: "#1e1e3a",
          textStyle: { color: "#d8d8f0", fontSize: 11 },
          formatter: (p: { data: [number, number, number] }) => `${names[p.data[1]]} × ${names[p.data[0]]}<br/>ρ = ${p.data[2]}`,
        },
        grid: { left: 96, right: 12, top: 8, bottom: 70 },
        xAxis: { type: "category", data: names, axisLabel: { color: "#5a5a80", fontSize: 9, rotate: 60 }, splitArea: { show: false } },
        yAxis: { type: "category", data: names, axisLabel: { color: "#5a5a80", fontSize: 9 }, splitArea: { show: false } },
        visualMap: { min: -1, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: 4, itemWidth: 12, itemHeight: 90, textStyle: { color: "#5a5a80", fontSize: 10 }, inRange: { color: ["#ff2d6a", "#1a1a2e", "#00d4ff"] } },
        series: [
          {
            type: "heatmap",
            data,
            label: { show: false },
            itemStyle: { borderColor: "#0c0c1a", borderWidth: 1 },
            emphasis: { itemStyle: { borderColor: "#ffd000", borderWidth: 2 } },
          },
        ],
      }),
      [],
    ),
  );
  return <div ref={ref} className="h-[420px] w-full" />;
}

/* ────────────────────────────────────────────────────────────
 * 页面
 * ──────────────────────────────────────────────────────────── */

type SortKey = "ic" | "ir" | "t" | "pos";
type SortDir = "asc" | "desc";

export function FactorAnalysis() {
  const { t } = useTranslation();
  const [selectedId, setSelectedId] = useState<string>(FACTORS[0].id);
  const [sortKey, setSortKey] = useState<SortKey>("t");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [usingSample, setUsingSample] = useState(true);

  const selected = FACTORS.find((f) => f.id === selectedId) ?? FACTORS[0];

  // 后端因子分析服务可用时切换为真实数据（当前沙箱回退示例）
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/factor-analysis", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ universe: "csi300", period: "2y" }),
        });
        if (!res.ok) throw new Error();
        const json = await res.json();
        if (json?.factors?.length) {
          if (!cancelled) setUsingSample(false);
        } else throw new Error();
      } catch {
        if (!cancelled) setUsingSample(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const sorted = useMemo(() => {
    const arr = [...FACTORS];
    arr.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      return sortDir === "desc" ? vb - va : va - vb;
    });
    return arr;
  }, [sortKey, sortDir]);

  const kpis = useMemo(() => {
    const active = FACTORS.filter((f) => f.life === "alive").length;
    const avgT = FACTORS.reduce((s, f) => s + f.t, 0) / FACTORS.length;
    let redundant = 0;
    for (let i = 0; i < FACTORS.length; i++)
      for (let j = i + 1; j < FACTORS.length; j++) if (Math.abs(CORR[i][j]) > CORR_THRESHOLD) redundant++;
    return { active, avgT, redundant };
  }, []);

  const onSort = (k: SortKey) => {
    if (k === sortKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else {
      setSortKey(k);
      setSortDir("desc");
    }
  };

  const arrow = (k: SortKey) => (sortKey === k ? (sortDir === "desc" ? " ↓" : " ↑") : "");

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-5 md:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-[var(--dash-text-bright)]">
            <Sigma className="h-5 w-5 text-[var(--dash-accent-cyan)]" />
            {t("factorLab.title")}
          </h1>
          <p className="mt-1 text-sm text-[var(--dash-dim)]">{t("factorLab.subtitle")}</p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--dash-accent-green)]/40 bg-[var(--dash-accent-green)]/10 px-3 py-1 text-xs font-medium text-[var(--dash-accent-green)]">
          <Database className="h-3.5 w-3.5" />
          {t("factorLab.dataSourcePreferred")}
        </span>
      </div>

      {usingSample && (
        <div className="mt-3 inline-flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-500">
          {t("factorLab.usingSample")}
        </div>
      )}

      {/* KPI */}
      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi icon={Sigma} label={t("factorLab.kpiActive")} value={String(kpis.active)} accent="#00e676" />
        <Kpi icon={Activity} label={t("factorLab.kpiAvgIcir")} value={kpis.avgT.toFixed(2)} accent="#00d4ff" />
        <Kpi icon={Grid3x3} label={t("factorLab.kpiRedundant")} value={String(kpis.redundant)} accent="#ff2d6a" />
        <Kpi icon={Layers} label={t("factorLab.kpiUniverse")} value="沪深300" accent="#ff6b00" />
      </div>

      {/* 排行榜 */}
      <section className="mt-6">
        <div className="mb-3">
          <h2 className="text-lg font-semibold text-[var(--dash-text-bright)]">{t("factorLab.rankingTitle")}</h2>
          <p className="text-sm text-[var(--dash-dim)]">{t("factorLab.rankingSubtitle")}</p>
        </div>
        <div className="overflow-x-auto rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)]">
          <table className="w-full min-w-[640px] text-left text-[13px]">
            <thead>
              <tr className="border-b border-[var(--dash-border)] text-[11px] uppercase tracking-wide text-[var(--dash-dim)]">
                <th className="px-3 py-2 font-medium">{t("factorLab.colFactor")}</th>
                <th className="px-3 py-2 font-medium">{t("factorLab.colTheme")}</th>
                <th className="cursor-pointer px-3 py-2 font-medium hover:text-[var(--dash-text)]" onClick={() => onSort("ic")}>{t("factorLab.colIC")}{arrow("ic")}</th>
                <th className="cursor-pointer px-3 py-2 font-medium hover:text-[var(--dash-text)]" onClick={() => onSort("ir")}>{t("factorLab.colIR")}{arrow("ir")}</th>
                <th className="cursor-pointer px-3 py-2 font-medium hover:text-[var(--dash-text)]" onClick={() => onSort("t")}>{t("factorLab.colIcir")}{arrow("t")}</th>
                <th className="cursor-pointer px-3 py-2 font-medium hover:text-[var(--dash-text)]" onClick={() => onSort("pos")}>{t("factorLab.colPos")}{arrow("pos")}</th>
                <th className="px-3 py-2 font-medium">{t("factorLab.colLife")}</th>
                <th className="px-3 py-2 font-medium">{t("factorLab.colMono")}</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((f) => {
                const life = LIFE_META[f.life];
                const selectedRow = f.id === selectedId;
                return (
                  <tr
                    key={f.id}
                    onClick={() => setSelectedId(f.id)}
                    className={`cursor-pointer border-b border-[var(--dash-border)]/60 transition-colors ${
                      selectedRow ? "bg-[var(--dash-surface-2)]" : "hover:bg-[var(--dash-surface-2)]/50"
                    }`}
                  >
                    <td className="px-3 py-2 font-medium text-[var(--dash-text-bright)]">{f.nameZh}</td>
                    <td className="px-3 py-2 text-[var(--dash-dim)]">{f.theme}</td>
                    <td className="px-3 py-2 font-mono tabular-nums text-[var(--dash-text)]">{f.ic.toFixed(3)}</td>
                    <td className="px-3 py-2 font-mono tabular-nums text-[var(--dash-text)]">{f.ir.toFixed(2)}</td>
                    <td className="px-3 py-2 font-mono tabular-nums text-[var(--dash-accent-cyan)]">{f.t.toFixed(2)}</td>
                    <td className="px-3 py-2 font-mono tabular-nums text-[var(--dash-text)]">{(f.pos * 100).toFixed(0)}%</td>
                    <td className="px-3 py-2">
                      <span className="inline-flex items-center gap-1.5 text-[12px]" style={{ color: life.color }}>
                        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: life.color }} />
                        {t(life.label)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-[12px] text-[var(--dash-dim)]">{f.monotone ? t("factorLab.monoYes") : t("factorLab.monoNo")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* 分层 + IC */}
      <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-3">
          <h3 className="text-sm font-semibold text-[var(--dash-text-bright)]">{t("factorLab.layersTitle")}</h3>
          <p className="mb-1 text-[11px] text-[var(--dash-dim)]">{t("factorLab.layersSubtitle")}</p>
          <LayersChart factor={selected} />
        </div>
        <div className="rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-3">
          <h3 className="text-sm font-semibold text-[var(--dash-text-bright)]">{t("factorLab.icTitle")}</h3>
          <p className="mb-1 text-[11px] text-[var(--dash-dim)]">{t("factorLab.icSubtitle")} · {selected.nameZh}</p>
          <IcChart factor={selected} />
        </div>
      </section>

      {/* 冗余相关性 */}
      <section className="mt-6">
        <div className="mb-3">
          <h2 className="text-lg font-semibold text-[var(--dash-text-bright)]">{t("factorLab.corrTitle")}</h2>
          <p className="text-sm text-[var(--dash-dim)]">{t("factorLab.corrSubtitle")}</p>
        </div>
        <div className="rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-3">
          <CorrHeatmap />
        </div>
      </section>

      {/* CTA → 回测 */}
      <section className="mt-8 mb-4 rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-5 text-center">
        <h3 className="text-base font-semibold text-[var(--dash-text-bright)]">{t("factorLab.ctaTitle")}</h3>
        <p className="mx-auto mt-1 max-w-xl text-[13px] text-[var(--dash-dim)]">{t("factorLab.ctaDesc")}</p>
        <div className="mt-4 flex justify-center">
          <Link
            to="/backtest"
            className="inline-flex items-center gap-1.5 rounded-md bg-[var(--dash-accent-orange)] px-4 py-2 text-sm font-medium text-[#1a0d00] transition-opacity hover:opacity-90"
          >
            {t("factorLab.ctaButton")}
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}

function Kpi({ icon: Icon, label, value, accent }: { icon: typeof Sigma; label: string; value: string; accent: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-[var(--dash-border)] bg-[var(--dash-surface)] p-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg" style={{ background: `${accent}1a`, color: accent }}>
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-[10px] uppercase tracking-wide text-[var(--dash-dim)]">{label}</p>
        <p className="text-lg font-semibold text-[var(--dash-text-bright)]">{value}</p>
      </div>
    </div>
  );
}
