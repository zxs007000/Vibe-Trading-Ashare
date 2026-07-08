import { useEffect, useRef } from "react";
import { echarts } from "@/lib/echarts";

interface GaugeChartProps {
  value: number;
  min?: number;
  max?: number;
  title: string;
  unit?: string;
  color?: string;
  height?: number;
}

/**
 * Neon radial gauge used for the dashboard "仪表" row
 * (今日收益 / 仓位占比 / 风险·夏普 / 市场温度).
 */
export function GaugeChart({
  value,
  min = 0,
  max = 100,
  title,
  unit = "%",
  color = "#00e676",
  height = 150,
}: GaugeChartProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: "transparent",
      series: [
        {
          type: "gauge",
          min,
          max,
          startAngle: 210,
          endAngle: -30,
          radius: "94%",
          center: ["50%", "60%"],
          progress: { show: true, width: 9, itemStyle: { color } },
          axisLine: { lineStyle: { width: 9, color: [[1, "rgba(255,255,255,0.07)"]] } },
          axisTick: { show: false },
          splitLine: { length: 9, distance: 4, lineStyle: { color: "#2a2a45", width: 1 } },
          axisLabel: { show: false },
          pointer: { width: 4, length: "58%", itemStyle: { color } },
          anchor: { show: true, size: 8, itemStyle: { color, borderColor: color } },
          title: { show: false },
          detail: {
            valueAnimation: true,
            fontSize: 20,
            fontWeight: "bolder",
            color: "#f0f0ff",
            offsetCenter: [0, "36%"],
            formatter: (v: number) => `${v.toFixed(1)}${unit}`,
          },
          data: [{ value }],
        },
      ],
    });
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
    };
  }, [value, min, max, color, unit]);

  return (
    <div className="flex flex-col items-center">
      <div ref={ref} style={{ width: "100%", height }} />
      <div className="-mt-1 text-[11px] text-[var(--dash-dim)] font-mono text-center">{title}</div>
    </div>
  );
}
