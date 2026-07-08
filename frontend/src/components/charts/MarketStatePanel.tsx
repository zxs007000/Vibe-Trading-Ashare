import { Activity, AlertTriangle, TrendingUp } from "lucide-react";
import type { MarketStateData } from "@/lib/api";

/* ─── State colour map ─── */

const STATE_COLORS: Record<string, string> = {
  "单边上涨": "#00e676",
  "单边下跌": "#ff1744",
  "冲高回落": "#ff6b00",
  "反弹": "#00b0ff",
  "震荡上行": "#b2ff59",
  "震荡下行": "#ff9100",
  "宽幅震荡": "#ffea00",
  "窄幅盘整": "#9e9e9e",
};

/* ─── Match helpers ─── */

const MATCH_STYLES: Record<
  MarketStateData["recommended"][number]["match"],
  { border: string; bg: string; marker: string }
> = {
  preferred: {
    border: "border-green-500/40",
    bg: "bg-green-500/10",
    marker: "✓",
  },
  neutral: {
    border: "border-[var(--dash-border)]",
    bg: "bg-transparent",
    marker: "–",
  },
  avoid: {
    border: "border-red-500/40",
    bg: "bg-red-500/10",
    marker: "✗",
  },
};

/* ─── Sub-components ─── */

function SnapshotRow({
  label,
  value,
  unit = "",
}: {
  label: string;
  value: number;
  unit?: string;
}) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-[10px] text-[var(--dash-dim)]">{label}</span>
      <span className="text-[11px] font-mono tabular-nums text-[var(--dash-text)]">
        {value.toLocaleString()}
        {unit}
      </span>
    </div>
  );
}

function SnapshotPct({
  label,
  pct,
}: {
  label: string;
  pct: number;
}) {
  const color = pct >= 0 ? "var(--dash-accent-green)" : "var(--dash-accent-pink)";
  const sign = pct >= 0 ? "+" : "";
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-[10px] text-[var(--dash-dim)]">{label}</span>
      <span
        className="text-[11px] font-mono tabular-nums"
        style={{ color }}
      >
        {sign}
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

/* ══════════════════════════════════════════
   Main component
   ══════════════════════════════════════════ */

export function MarketStatePanel({ data }: { data: MarketStateData }) {
  const stateColor = STATE_COLORS[data.label_zh] ?? "#9e9e9e";

  const grouped = {
    preferred: data.recommended.filter((r) => r.match === "preferred"),
    neutral: data.recommended.filter((r) => r.match === "neutral"),
    avoid: data.recommended.filter((r) => r.match === "avoid"),
  };

  const sectionLabel: Record<string, string> = {
    preferred: "Preferred",
    neutral: "Neutral",
    avoid: "Avoid",
  };

  return (
    <div className="kpi-card p-3 w-full h-full flex flex-col">
      {/* ── Header: State + confidence ── */}
      <div className="flex items-center gap-2 mb-2">
        <TrendingUp
          className="h-4 w-4"
          style={{ color: stateColor }}
        />
        <span
          className="text-xs font-bold uppercase tracking-wider"
          style={{ color: stateColor }}
        >
          {data.label_zh}
        </span>
        <span className="flex-1" />
        <span className="text-[10px] text-[var(--dash-dim)] font-mono">
          {data.state}
        </span>
      </div>

      {/* ── Confidence bar + duration ── */}
      <div className="flex items-center gap-2 mb-3">
        <Activity className="h-3 w-3 text-[var(--dash-dim)]" />
        <div className="flex-1 h-1.5 rounded-full bg-[var(--dash-border)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.round(data.confidence * 100)}%`,
              background: `linear-gradient(90deg, ${stateColor}80, ${stateColor})`,
            }}
          />
        </div>
        <span className="text-[10px] font-mono text-[var(--dash-text)]">
          {(data.confidence * 100).toFixed(0)}%
        </span>
        <span className="text-[10px] text-[var(--dash-dim)]">
          {data.duration_days}d
        </span>
      </div>

      {/* ── Key indicator snapshot ── */}
      <div className="border-t border-[var(--dash-border)] pt-2 mb-2">
        <div className="text-[10px] text-[var(--dash-dim)] mb-1.5 font-mono uppercase tracking-wider">
          Market Snapshot
        </div>
        <div className="space-y-0">
          <SnapshotRow label="Close" value={data.snapshot.close} />
          <SnapshotRow label="MA60" value={data.snapshot.ma60} />
          <SnapshotRow label="MA250" value={data.snapshot.ma250} />
          <SnapshotRow label="Vol20" value={data.snapshot.vol20_annual_pct} unit="%" />
          <SnapshotPct label="Ret 20d" pct={data.snapshot.ret20_pct} />
          <SnapshotPct label="Ret 5d" pct={data.snapshot.ret5_pct} />
        </div>
      </div>

      {/* ── Recommended logic chains ── */}
      <div className="border-t border-[var(--dash-border)] pt-2 mb-2">
        <div className="text-[10px] text-[var(--dash-dim)] mb-1.5 font-mono uppercase tracking-wider">
          Logic Chains
        </div>

        {(["preferred", "neutral", "avoid"] as const).map((match) => {
          const chains = grouped[match];
          if (chains.length === 0) return null;
          const style = MATCH_STYLES[match];

          return (
            <div key={match} className="mb-1.5">
              <div className="text-[9px] text-[var(--dash-dim)] mb-1 font-mono uppercase">
                {sectionLabel[match]}
              </div>
              {chains.map((chain) => (
                <div
                  key={chain.chain_id}
                  className={`flex items-center gap-1.5 rounded border px-2 py-1 mb-1 text-[11px] transition-colors ${style.border} ${style.bg}`}
                >
                  <span className="text-[10px] w-3 text-center font-mono">
                    {style.marker}
                  </span>
                  <span className="flex-1 text-[var(--dash-text)] truncate">
                    {chain.name_zh}
                  </span>
                  <span className="text-[10px] font-mono text-[var(--dash-dim)]">
                    {chain.score.toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
          );
        })}

        {/* Avoid warning when active */}
        {grouped.avoid.length > 0 && (
          <div className="flex items-center gap-1 mt-1 text-[10px] text-red-400/70">
            <AlertTriangle className="h-3 w-3" />
            <span>{grouped.avoid.length} chain(s) suppressed</span>
          </div>
        )}
      </div>

      {/* ── Recent state transitions ── */}
      <div className="border-t border-[var(--dash-border)] pt-2 mt-auto">
        <div className="text-[10px] text-[var(--dash-dim)] mb-1.5 font-mono uppercase tracking-wider">
          Recent Transitions
        </div>
        <div className="space-y-1">
          {data.recent_transitions.map((t, i) => {
            const c = STATE_COLORS[t.label_zh] ?? "#9e9e9e";
            const isLatest = i === data.recent_transitions.length - 1;
            return (
              <div key={`${t.date}-${i}`} className="flex items-center gap-2 relative">
                {/* Timeline dot + line */}
                <div className="flex flex-col items-center self-stretch">
                  <div
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{
                      background: c,
                      boxShadow: isLatest ? `0 0 6px ${c}80` : undefined,
                    }}
                  />
                  {i < data.recent_transitions.length - 1 && (
                    <div className="w-px flex-1 bg-[var(--dash-border)] my-1" />
                  )}
                </div>
                <div className="flex-1 flex items-center justify-between pb-1">
                  <span
                    className="text-[11px]"
                    style={{ color: c }}
                  >
                    {t.label_zh}
                  </span>
                  <span className="text-[10px] font-mono text-[var(--dash-dim)]">
                    {t.date.slice(5)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
