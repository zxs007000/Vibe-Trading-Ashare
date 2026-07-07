import { useTranslation } from "react-i18next";
import { ExternalLink, Link2, Sparkles } from "lucide-react";
import {
  themeColor,
  tKey,
  type CosmosFactor,
  type FactorStatus,
} from "@/lib/factorCosmos";
import { cn } from "@/lib/utils";

interface Props {
  factor: CosmosFactor | null;
  related: CosmosFactor[];
  inChain: boolean;
  onPick: (id: string) => void;
}

const STATUS_STYLE = {
  alive: { label: "factorCosmos.statusAlive" as const, color: "#7fe7ff" },
  decaying: { label: "factorCosmos.statusDecaying" as const, color: "#ffaa44" },
  dead: { label: "factorCosmos.statusDead" as const, color: "#7a7a8a" },
} satisfies Record<FactorStatus, { label: string; color: string }>;

function fmt(n: number, d = 4): string {
  return n.toFixed(d);
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums text-foreground truncate text-right">{children}</span>
    </div>
  );
}

export function FactorCosmosDetail({ factor, related, inChain, onPick }: Props) {
  const { t } = useTranslation();

  if (!factor) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-4 text-center">
        <Sparkles className="h-6 w-6 text-muted-foreground/40 mb-2" />
        <p className="text-xs text-muted-foreground/80">{t("factorCosmos.detailNone")}</p>
      </div>
    );
  }

  const st = STATUS_STYLE[factor.status];

  return (
    <div className="space-y-3">
      {/* 头部 */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 rounded-full shrink-0"
            style={{ backgroundColor: themeColor(factor.theme[0]) }}
          />
          <h3 className="font-mono text-sm font-medium truncate" title={factor.id}>
            {factor.id}
          </h3>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{ backgroundColor: st.color + "22", color: st.color }}
          >
            {t(st.label)}
          </span>
          {factor.selected && (
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/15 px-2 py-0.5 text-[10px] font-medium text-primary">
              <Sparkles className="h-3 w-3" />
              {t("factorCosmos.selected")}
            </span>
          )}
          {inChain && (
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/15 px-2 py-0.5 text-[10px] font-medium text-primary">
              <Link2 className="h-3 w-3" />
              {t("factorCosmos.inChain")}
            </span>
          )}
        </div>
      </div>

      {/* 主题标签 */}
      <div className="flex flex-wrap gap-1">
        {factor.theme.map((th) => (
          <span
            key={th}
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{ backgroundColor: themeColor(th) + "22", color: themeColor(th) }}
          >
            {tKey(t, "alphaZoo.themes." + th, th)}
          </span>
        ))}
      </div>

      {/* 指标 */}
      <div className="divide-y divide-border/60">
        <Row label={t("factorCosmos.fIr")}>{fmt(factor.ir)}</Row>
        <Row label={t("factorCosmos.fIc")}>{fmt(factor.ic_mean, 6)}</Row>
        <Row label={t("factorCosmos.fAliveDays")}>
          {factor.alive_days != null ? factor.alive_days : "—"}
        </Row>
        <Row label={t("factorCosmos.fZoo")}>{factor.zoo ?? "—"}</Row>
      </div>

      {/* 相关因子 */}
      {related.length > 0 && (
        <div>
          <h4 className="text-[11px] font-medium text-muted-foreground mb-1.5">
            {t("factorCosmos.fRelated")}
          </h4>
          <div className="flex flex-col gap-1 max-h-40 overflow-auto">
            {related.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => onPick(r.id)}
                className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-muted/60 transition-colors text-left"
              >
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ backgroundColor: themeColor(r.theme[0]) }}
                />
                <span
                  className={cn(
                    "flex-1 truncate font-mono",
                    r.status === "dead" ? "text-muted-foreground/60" : "text-foreground",
                  )}
                >
                  {r.id}
                </span>
                <span className="text-[10px] text-muted-foreground/70">IR {fmt(r.ir, 2)}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 在 Alpha Zoo 打开 */}
      <a
        href={`/alpha-zoo/${encodeURIComponent(factor.id)}`}
        className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
      >
        <ExternalLink className="h-3.5 w-3.5" />
        {t("factorCosmos.openAlpha")}
      </a>
    </div>
  );
}
