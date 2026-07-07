import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, AlertTriangle, Calculator } from "lucide-react";
import { getMetricLabel } from "@/lib/formatters";
import type { BacktestMetrics, EquityPoint } from "@/lib/api";

interface Props {
  metrics?: BacktestMetrics | null;
  equityCurve?: EquityPoint[] | null;
}

interface Recompute {
  totalReturn: number;
  annual: number;
  maxDd: number;
  n: number;
  first: number;
  last: number;
}

function recompute(equity: number[]): Recompute | null {
  if (!equity || equity.length < 2) return null;
  const first = equity[0];
  const last = equity[equity.length - 1];
  if (!first || !Number.isFinite(first) || !Number.isFinite(last)) return null;
  const totalReturn = last / first - 1;
  let peak = first;
  let mdd = 0;
  for (const v of equity) {
    if (v > peak) peak = v;
    if (peak > 0) mdd = Math.min(mdd, v / peak - 1);
  }
  const n = equity.length - 1; // 期数
  // 几何年化:假设权益曲线为交易日频,252 个交易日/年
  const annual = n > 0 && first > 0 ? Math.pow(last / first, 252 / n) - 1 : 0;
  return { totalReturn, annual, maxDd: mdd, n, first, last };
}

const TOL = 0.02; // 相对容差 2%

function consistent(a: number, b: number): boolean {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  if (a === b) return true;
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-9);
  return Math.abs(a - b) / denom <= TOL;
}

function Row({
  label,
  reported,
  computed,
  fmt,
}: {
  label: string;
  reported: number | null;
  computed: number | null;
  fmt: (v: number) => string;
}) {
  const ok = reported != null && computed != null ? consistent(reported, computed) : null;
  return (
    <div className="grid grid-cols-[1.4fr_1fr_1fr_auto] items-center gap-2 border-b border-border/50 py-2 text-xs last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-mono tabular-nums">
        {reported != null ? fmt(reported) : <span className="text-muted-foreground/50">—</span>}
      </span>
      <span className="text-right font-mono tabular-nums text-foreground">
        {computed != null ? fmt(computed) : <span className="text-muted-foreground/50">—</span>}
      </span>
      <span className="flex justify-end">
        {ok == null ? (
          <span className="text-[10px] text-muted-foreground/60">n/a</span>
        ) : ok ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-500" />
        ) : (
          <AlertTriangle className="h-4 w-4 text-amber-500" />
        )}
      </span>
    </div>
  );
}

export function BacktestMetricsPanel({ metrics, equityCurve }: Props) {
  const { t } = useTranslation();

  const rc = useMemo(() => {
    const eq = (equityCurve || [])
      .map((p) => Number(p.equity))
      .filter((v) => Number.isFinite(v));
    return recompute(eq);
  }, [equityCurve]);

  if (!rc && !metrics) {
    return (
      <div className="rounded-xl border border-border/60 bg-muted/20 p-3 text-xs text-muted-foreground">
        {t("backtest.noData")}
      </div>
    );
  }

  const window =
    equityCurve && equityCurve.length > 1
      ? `${equityCurve[0].time} → ${equityCurve[equityCurve.length - 1].time}`
      : null;

  const pct = (v: number) => (v * 100).toFixed(2) + "%";
  const num = (v: number) => v.toFixed(4);

  return (
    <div className="rounded-xl border border-border/60 bg-card p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Calculator className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold">{t("backtest.accuracy")}</h3>
        <span className="ml-auto text-[10px] text-muted-foreground">
          {window ? `${window} · ${rc?.n ?? "?"} ${t("backtest.periods")}` : ""}
        </span>
      </div>

      <div className="grid grid-cols-[1.4fr_1fr_1fr_auto] gap-2 text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>{t("backtest.metric")}</span>
        <span className="text-right">{t("backtest.reported")}</span>
        <span className="text-right">{t("backtest.fromCurve")}</span>
        <span className="text-right">{t("backtest.check")}</span>
      </div>

      <Row
        label={getMetricLabel("total_return")}
        reported={metrics?.total_return ?? null}
        computed={rc ? rc.totalReturn : null}
        fmt={pct}
      />
      <Row
        label={getMetricLabel("annual_return")}
        reported={metrics?.annual_return ?? null}
        computed={rc ? rc.annual : null}
        fmt={pct}
      />
      <Row
        label={getMetricLabel("max_drawdown")}
        reported={metrics?.max_drawdown ?? null}
        computed={rc ? rc.maxDd : null}
        fmt={pct}
      />
      <Row
        label={getMetricLabel("final_value")}
        reported={metrics?.final_value ?? null}
        computed={rc ? rc.last : null}
        fmt={num}
      />
      <Row
        label={getMetricLabel("trade_count")}
        reported={metrics?.trade_count ?? null}
        computed={null}
        fmt={(v) => String(v)}
      />

      <p className="text-[11px] leading-relaxed text-muted-foreground border-t border-border/50 pt-2">
        {t("backtest.methodology")}
      </p>
    </div>
  );
}
