// 因子星空图(Factor Cosmos)前端类型定义与示例数据
//
// 后端 POST /factor-cosmos 返回的数据结构。即使后端不可用(沙箱无国内 IP),
// 前端也会回退到 SAMPLE_COSMOS 渲染星空,便于演示。

import type { TFunction } from "i18next";

export type FactorStatus = "alive" | "decaying" | "dead";

// i18n 的类型是字面量键并集,动态拼接的键(如 "factorCosmos.universe_" + u)
// 无法通过编译期类型检查,故用此辅助函数把键降为宽泛类型以绕过严格校验。
export function tKey(t: TFunction, key: string, defaultValue?: string): string {
  // 动态键无法匹配 i18n 的模板字面量重载,这里把 t 降到宽泛函数签名再调用。
  const fn = t as unknown as (...args: unknown[]) => string;
  return defaultValue !== undefined
    ? fn(key, { defaultValue })
    : fn(key);
}

export interface CosmosFactor {
  id: string;
  theme: string[];
  status: FactorStatus;
  /** rolling IR,|IR| 越大星越亮 */
  ir: number;
  /** rolling IC 均值 */
  ic_mean: number;
  /** 活跃天数(仅 alive 有值) */
  alive_days?: number | null;
  /** 是否在被推荐的因子组合中 */
  selected: boolean;
  zoo: string;
}

export interface CosmosLogicalChain {
  name: string | null;
  score: number;
  themes_covered: string[];
}

export interface CosmosCorrelation {
  a: string;
  b: string;
  /** 相关系数,绝对值越大越相关 */
  r: number;
}

export interface CosmosStats {
  total: number;
  alive: number;
  decaying: number;
  dead: number;
}

export interface FactorCosmosResponse {
  status: string;
  generated_at?: string;
  universe: string;
  period: string;
  factors: CosmosFactor[];
  logical_chain: CosmosLogicalChain;
  correlations: CosmosCorrelation[];
  themes: string[];
  stats: CosmosStats;
}

export const FACTOR_COSMOS_ENDPOINT = "/factor-cosmos";

// ---------------------------------------------------------------------------
// Theme → 星体颜色(暗空主题,匹配 Gravia 截图)
// ---------------------------------------------------------------------------

export const THEME_COLORS: Record<string, string> = {
  momentum: "#ff00aa", // 品红
  value: "#00ccff", // 青蓝
  quality: "#44ff88", // 翠绿
  volatility: "#ffaa00", // 琥珀
  microstructure: "#cc66ff", // 紫罗兰
  volume: "#66aaff", // 天蓝
  reversal: "#ff6644", // 橙红
  sentiment: "#ffee66", // 金黄
  growth: "#66ffcc", // 薄荷
  leverage: "#aa88ff", // 薰衣草
  liquidity: "#88ddff", // 浅蓝
  default: "#9aa0c0", // 未知主题灰紫
};

export function themeColor(theme: string | undefined): string {
  if (!theme) return THEME_COLORS.default;
  return THEME_COLORS[theme] ?? THEME_COLORS.default;
}

/** 返回因子主色:优先第一个 theme 的颜色 */
export function factorColor(f: CosmosFactor): string {
  return themeColor(f.theme[0]);
}

// ---------------------------------------------------------------------------
// 示例数据:后端不可用时回退渲染(约 60 颗星,含全部主题/状态)
// ---------------------------------------------------------------------------

export const SAMPLE_COSMOS: FactorCosmosResponse = (() => {
  const themes = Object.keys(THEME_COLORS).filter((t) => t !== "default");
  const zoos = ["qlib158", "gtja191", "alpha101", "academic"];
  const factors: CosmosFactor[] = [];

  let seed = 42;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  let n = 0;
  for (const theme of themes) {
    const count = 4 + Math.floor(rnd() * 4); // 每主题 4-7 颗
    for (let i = 0; i < count; i++) {
      const roll = rnd();
      const status: FactorStatus =
        roll > 0.62 ? "alive" : roll > 0.32 ? "decaying" : "dead";
      const ir =
        status === "alive"
          ? 0.15 + rnd() * 0.55
          : status === "decaying"
            ? 0.05 + rnd() * 0.15
            : 0.0;
      factors.push({
        id: `${zoos[n % zoos.length]}_${theme}_${i}`,
        theme: [theme],
        status,
        ir: Number(ir.toFixed(4)),
        ic_mean: Number((ir * 0.07).toFixed(4)),
        alive_days: status === "alive" ? Math.floor(10 + rnd() * 60) : null,
        selected: status === "alive" && rnd() > 0.4,
        zoo: zoos[n % zoos.length],
      });
      n++;
    }
  }

  // 构造相关对(同主题内部 + 跨主题少量)
  const correlations: CosmosCorrelation[] = [];
  for (let t = 0; t < themes.length; t++) {
    const group = factors.filter((f) => f.theme[0] === themes[t]);
    for (let a = 0; a < group.length; a++) {
      for (let b = a + 1; b < group.length; b++) {
        if (rnd() > 0.4) {
          correlations.push({
            a: group[a].id,
            b: group[b].id,
            r: Number((0.7 + rnd() * 0.28).toFixed(4)),
          });
        }
      }
    }
  }

  const stats: CosmosStats = {
    total: factors.length,
    alive: factors.filter((f) => f.status === "alive").length,
    decaying: factors.filter((f) => f.status === "decaying").length,
    dead: factors.filter((f) => f.status === "dead").length,
  };

  return {
    status: "ok",
    generated_at: new Date().toISOString(),
    universe: "csi300",
    period: "2y",
    factors,
    logical_chain: {
      name: "smart_money",
      score: 0.67,
      themes_covered: ["microstructure", "volume", "momentum"],
    },
    correlations,
    themes,
    stats,
  };
})();
