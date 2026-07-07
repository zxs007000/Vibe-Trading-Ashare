import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  RefreshCw,
  Sparkles,
  Loader2,
  AlertTriangle,
  MousePointer2,
  ZoomIn,
  Hand,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  SAMPLE_COSMOS,
  tKey,
  type FactorCosmosResponse,
  type CosmosFactor,
  type FactorStatus,
} from "@/lib/factorCosmos";
import { FactorCosmosCanvas } from "@/components/charts/FactorCosmosCanvas";
import { FactorCosmosLegend } from "@/components/charts/FactorCosmosLegend";
import { FactorCosmosDetail } from "@/components/charts/FactorCosmosDetail";

const UNIVERSES = ["csi300", "csi500", "hs300", "sp500", "sz50"] as const;
const PERIODS = ["1m", "6m", "1y", "2y", "3y"] as const;

const STATUS_STAT = {
  alive: { label: "factorCosmos.statAlive" as const, color: "#7fe7ff" },
  decaying: { label: "factorCosmos.statDecaying" as const, color: "#ffaa44" },
  dead: { label: "factorCosmos.statDead" as const, color: "#7a7a8a" },
} satisfies Record<FactorStatus, { label: string; color: string }>;

export function FactorCosmos() {
  const { t } = useTranslation();
  const [universe, setUniverse] = useState<string>("csi300");
  const [period, setPeriod] = useState<string>("2y");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usingSample, setUsingSample] = useState(false);
  const [data, setData] = useState<FactorCosmosResponse | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Set<FactorStatus>>(
    new Set<FactorStatus>(["alive", "decaying", "dead"]),
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getFactorCosmos({ universe, period, limit: 600 });
      if (!res?.factors?.length) throw new Error("empty");
      setData(res);
      setUsingSample(false);
      setSelectedId(null);
    } catch {
      // 后端不可用(无国内 IP / 未启动)→ 回退示例数据,保证星空可演示
      setData(SAMPLE_COSMOS);
      setUsingSample(true);
    } finally {
      setLoading(false);
    }
  }, [universe, period]);

  useEffect(() => {
    load();
  }, [load]);

  const toggleStatus = useCallback((s: FactorStatus) => {
    setFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }, []);

  const { stats, themeCounts, filterArr } = useMemo(() => {
    const d = data;
    const stats = d?.stats ?? { total: 0, alive: 0, decaying: 0, dead: 0 };
    const themeCounts: Record<string, { alive: number; total: number }> = {};
    if (d) {
      for (const f of d.factors) {
        const tc = (themeCounts[f.theme[0]] ||= { alive: 0, total: 0 });
        tc.total += 1;
        if (f.status === "alive") tc.alive += 1;
      }
    }
    return { stats, themeCounts, filterArr: [...filter] };
  }, [data, filter]);

  const { selected, related, inChain } = useMemo(() => {
    if (!data) return { selected: null, related: [] as CosmosFactor[], inChain: false };
    const sel = data.factors.find((f) => f.id === selectedId) ?? null;
    const relSet = new Set<string>();
    if (selectedId) {
      for (const c of data.correlations) {
        if (c.a === selectedId) relSet.add(c.b);
        if (c.b === selectedId) relSet.add(c.a);
      }
    }
    const related = data.factors.filter((f) => relSet.has(f.id));
    const chainThemes = new Set(data.logical_chain?.themes_covered ?? []);
    const inChain = !!sel && sel.theme.some((th) => chainThemes.has(th));
    return { selected: sel, related, inChain };
  }, [data, selectedId]);

  const hints = [
    { icon: MousePointer2, label: t("factorCosmos.hintSelect") },
    { icon: Hand, label: t("factorCosmos.hintDrag") },
    { icon: ZoomIn, label: t("factorCosmos.hintZoom") },
  ];

  return (
    <div className="mx-auto w-full max-w-[1440px] px-4 py-5 md:px-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
            <Sparkles className="h-5 w-5 text-primary" />
            {t("factorCosmos.title")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">{t("factorCosmos.subtitle")}</p>
        </div>
        <div className="flex items-center gap-2">
          {(["alive", "decaying", "dead"] as FactorStatus[]).map((s) => (
            <div
              key={s}
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2 py-1"
            >
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: STATUS_STAT[s].color }} />
              <span className="text-[11px] text-muted-foreground">{t(STATUS_STAT[s].label)}</span>
              <span className="text-xs font-mono tabular-nums">
                {s === "alive" ? stats.alive : s === "decaying" ? stats.decaying : stats.dead}
              </span>
            </div>
          ))}
          <div className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2 py-1">
            <span className="text-[11px] text-muted-foreground">{t("factorCosmos.statTotal")}</span>
            <span className="text-xs font-mono tabular-nums">{stats.total}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card p-3">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          {t("factorCosmos.universe")}
          <select
            value={universe}
            onChange={(e) => setUniverse(e.target.value)}
            className="rounded-md border border-border bg-background px-2 py-1.5 text-xs text-foreground outline-none"
          >
            {UNIVERSES.map((u) => (
              <option key={u} value={u}>{tKey(t, "factorCosmos.universe_" + u)}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          {t("factorCosmos.period")}
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="rounded-md border border-border bg-background px-2 py-1.5 text-xs text-foreground outline-none"
          >
            {PERIODS.map((p) => (
              <option key={p} value={p}>{tKey(t, "factorCosmos.period_" + p)}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          {loading ? t("factorCosmos.regenerating") : t("factorCosmos.regenerate")}
        </button>
        {usingSample && (
          <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-500">
            {t("factorCosmos.usingSample")}
          </span>
        )}
        <div className="ml-auto flex items-center gap-3 text-[11px] text-muted-foreground/70">
          {hints.map((h) => (
            <span key={h.label} className="inline-flex items-center gap-1">
              <h.icon className="h-3.5 w-3.5" />
              {h.label}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <div className="relative overflow-hidden rounded-xl border border-border bg-black/40 h-[62vh] min-h-[480px]">
          {loading && !data ? (
            <div className="flex h-full items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <span className="ml-2 text-sm text-muted-foreground">{t("factorCosmos.loading")}</span>
            </div>
          ) : error ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <AlertTriangle className="h-6 w-6 text-danger mb-2" />
              <p className="text-sm text-foreground">{t("factorCosmos.errorTitle")}</p>
              <p className="mt-1 text-xs text-muted-foreground">{error}</p>
            </div>
          ) : (
            <FactorCosmosCanvas
              data={data}
              selectedId={selectedId}
              hoveredId={hoveredId}
              filter={filterArr.length === 3 ? null : filterArr}
              onSelect={setSelectedId}
              onHover={setHoveredId}
            />
          )}
          {!loading && data && (
            <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/40 px-2 py-1 text-[10px] text-muted-foreground/80">
              {tKey(t, "factorCosmos.universe_" + universe)} · {tKey(t, "factorCosmos.period_" + period)}
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-3">
            <FactorCosmosLegend
              filter={filter}
              onToggle={toggleStatus}
              themes={data?.themes ?? []}
              themeCounts={themeCounts}
              logicalChain={data?.logical_chain ?? null}
            />
          </div>
          <div className="rounded-xl border border-border bg-card p-3">
            <h3 className="mb-2 text-xs font-medium text-muted-foreground">
              {t("factorCosmos.detailTitle")}
            </h3>
            <FactorCosmosDetail
              factor={selected}
              related={related}
              inChain={inChain}
              onPick={(id) => setSelectedId(id)}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
