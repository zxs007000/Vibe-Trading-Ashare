import { useTranslation } from "react-i18next";
import { Sparkles, Star, Circle, Dot } from "lucide-react";
import {
  THEME_COLORS,
  themeColor,
  tKey,
  type FactorStatus,
  type CosmosLogicalChain,
} from "@/lib/factorCosmos";
import { cn } from "@/lib/utils";

interface Props {
  filter: Set<FactorStatus>;
  onToggle: (s: FactorStatus) => void;
  themes: string[];
  themeCounts: Record<string, { alive: number; total: number }>;
  logicalChain: CosmosLogicalChain | null;
}

const STATUS_ORDER: FactorStatus[] = ["alive", "decaying", "dead"];

const STATUS_STYLE = {
  alive: { dot: "#7fe7ff", label: "factorCosmos.statusAlive" as const },
  decaying: { dot: "#ffaa44", label: "factorCosmos.statusDecaying" as const },
  dead: { dot: "#3a3a4a", label: "factorCosmos.statusDead" as const },
} satisfies Record<FactorStatus, { dot: string; label: string }>;

export function FactorCosmosLegend({
  filter,
  onToggle,
  themes,
  themeCounts,
  logicalChain,
}: Props) {
  const { t } = useTranslation();

  return (
    <div className="space-y-4">
      {/* Status filter */}
      <section>
        <h3 className="text-xs font-medium text-muted-foreground mb-2">
          {t("factorCosmos.filterTitle")}
        </h3>
        <div className="flex flex-col gap-1.5">
          {STATUS_ORDER.map((s) => {
            const active = filter.has(s);
            return (
              <button
                key={s}
                type="button"
                onClick={() => onToggle(s)}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-1.5 text-xs border transition-colors text-left",
                  active
                    ? "border-primary/40 bg-primary/5 text-foreground"
                    : "border-transparent bg-muted/30 text-muted-foreground hover:bg-muted/50",
                )}
                aria-pressed={active}
              >
                {s === "alive" ? (
                  <Sparkles className="h-3.5 w-3.5 shrink-0" style={{ color: STATUS_STYLE[s].dot }} />
                ) : s === "decaying" ? (
                  <Star className="h-3.5 w-3.5 shrink-0" style={{ color: STATUS_STYLE[s].dot }} />
                ) : (
                  <Circle className="h-3.5 w-3.5 shrink-0" style={{ color: STATUS_STYLE[s].dot }} />
                )}
                <span className="flex-1">{t(STATUS_STYLE[s].label)}</span>
                <Dot
                  className={cn(
                    "h-4 w-4 shrink-0 transition-opacity",
                    active ? "opacity-100 text-primary" : "opacity-30",
                  )}
                />
              </button>
            );
          })}
        </div>
      </section>

      {/* Theme palette */}
      <section>
        <h3 className="text-xs font-medium text-muted-foreground mb-2">
          {t("factorCosmos.themeLegend")}
        </h3>
        <div className="flex flex-col gap-1">
          {themes.map((th) => {
            const c = themeCounts[th] ?? { alive: 0, total: 0 };
            return (
              <div
                key={th}
                className="flex items-center gap-2 text-xs text-muted-foreground px-1 py-0.5"
              >
                <span
                  className="h-2.5 w-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: themeColor(th) }}
                />
                <span className="flex-1 truncate">{tKey(t, "alphaZoo.themes." + th, th)}</span>
                {c.alive > 0 && (
                  <span className="text-[10px] font-mono text-emerald-500/90">
                    {c.alive}·
                  </span>
                )}
                <span className="text-[10px] font-mono tabular-nums">{c.total}</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* Logical chain */}
      {logicalChain && logicalChain.themes_covered.length > 0 && (
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            {t("factorCosmos.logicalChainTitle")}
          </h3>
          <div className="border rounded-lg p-3 bg-gradient-to-br from-primary/5 to-transparent">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium capitalize">
                {logicalChain.name ?? t("factorCosmos.unnamedChain")}
              </span>
              <span className="text-xs font-mono tabular-nums text-primary">
                {t("factorCosmos.chainScore")} {(logicalChain.score * 100).toFixed(0)}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {logicalChain.themes_covered.map((th) => (
                <span
                  key={th}
                  className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium"
                  style={{
                    backgroundColor: (THEME_COLORS[th] ?? THEME_COLORS.default) + "22",
                    color: themeColor(th),
                  }}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: themeColor(th) }}
                  />
                  {tKey(t, "alphaZoo.themes." + th, th)}
                </span>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground/80">
              {t("factorCosmos.logicalChainHint")}
            </p>
          </div>
        </section>
      )}
    </div>
  );
}
