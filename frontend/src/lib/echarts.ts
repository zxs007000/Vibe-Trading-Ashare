import * as echarts from "echarts/core";
import {
  CandlestickChart,
  LineChart,
  BarChart,
  HeatmapChart,
  RadarChart,
  GaugeChart,
} from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkPointComponent,
  ToolboxComponent,
  MarkLineComponent,
  MarkAreaComponent,
  VisualMapComponent,
  RadarComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  CandlestickChart, LineChart, BarChart, HeatmapChart, RadarChart, GaugeChart,
  GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, MarkPointComponent,
  ToolboxComponent, MarkLineComponent, MarkAreaComponent,
  VisualMapComponent, RadarComponent,
  CanvasRenderer,
]);

export const CHART_GROUP = "quant-charts";

let _connected = false;

export function connectCharts() {
  if (!_connected) {
    echarts.connect(CHART_GROUP);
    _connected = true;
  }
}

export { echarts };
