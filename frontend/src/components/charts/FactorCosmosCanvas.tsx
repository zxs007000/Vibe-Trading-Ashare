import { useEffect, useRef } from "react";
import type {
  FactorCosmosResponse,
  CosmosFactor,
  FactorStatus,
} from "@/lib/factorCosmos";
import { themeColor } from "@/lib/factorCosmos";

// ===========================================================================
// Factor Cosmos — Canvas 2D 星空渲染引擎
//
// 渲染逻辑:
//   - 暗空背景 + 静态星尘
//   - 力导向布局(斥力 + 相关引力 + 主题聚簇)把因子摆成星空
//   - alive 星:亮色发光晕,大小 ∝ |IR|,脉动
//   - decaying 星:暖橙弱发光,慢脉动
//   - dead 星:极暗灰点
//   - 选中/悬停因子:相关因子(浅色星)高亮外环
//   - 逻辑链:贝塞尔光通道 + 流动粒子
//   - 相关对:低透明度虚线
//
// 交互:悬停(highlight)、点击(选中)、拖拽星体、滚轮缩放、空白拖拽平移
// ===========================================================================

interface StarNode {
  f: CosmosFactor;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
  radius: number;
  phase: number;
  fixed: boolean;
}

interface Props {
  data: FactorCosmosResponse | null;
  selectedId: string | null;
  hoveredId: string | null;
  filter: FactorStatus[] | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
}

const STATUS_ALPHA: Record<FactorStatus, number> = {
  alive: 1,
  decaying: 0.85,
  dead: 0.32,
};

export function FactorCosmosCanvas({
  data,
  selectedId,
  hoveredId,
  filter,
  onSelect,
  onHover,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const nodesRef = useRef<StarNode[]>([]);
  const nodeByIdRef = useRef<Map<string, StarNode>>(new Map());
  const corrRef = useRef<{ a: string; b: string; r: number }[]>([]);
  const dustRef = useRef<{ x: number; y: number; r: number; a: number }[]>([]);
  const sizeRef = useRef({ w: 0, h: 0 });
  const zoomRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{ id: string | null; dx: number; dy: number } | null>(null);
  const settleRef = useRef(0);
  const relatedRef = useRef<Set<string>>(new Set());

  // 用 ref 保存最新 props,供动画循环读取
  const selRef = useRef(selectedId);
  const hovRef = useRef(hoveredId);
  const filterRef = useRef(filter);
  const onSelectRef = useRef(onSelect);
  const onHoverRef = useRef(onHover);
  selRef.current = selectedId;
  hovRef.current = hoveredId;
  filterRef.current = filter;
  onSelectRef.current = onSelect;
  onHoverRef.current = onHover;

  // ---- 构建节点(数据变化时) ----
  useEffect(() => {
    if (!data) {
      nodesRef.current = [];
      nodeByIdRef.current = new Map();
      corrRef.current = [];
      dustRef.current = [];
      return;
    }
    const { w, h } = sizeRef.current;
    const cx = w / 2 || 400;
    const cy = h / 2 || 300;
    const themeList = data.themes.length ? data.themes : ["default"];
    const themeAngle: Record<string, number> = {};
    themeList.forEach((t, i) => {
      themeAngle[t] = (i / themeList.length) * Math.PI * 2;
    });

    const nodes: StarNode[] = data.factors.map((f, i) => {
      const t0 = f.theme[0] ?? "default";
      const ang = themeAngle[t0] ?? Math.random() * Math.PI * 2;
      const ringR = Math.min(w, h) * 0.32 + (i % 7) * 6;
      const a = ang + (Math.random() - 0.5) * 0.5;
      const radius =
        f.status === "alive"
          ? 3 + Math.min(Math.abs(f.ir) * 12, 10)
          : f.status === "decaying"
            ? 3.5
            : 3;
      return {
        f,
        x: cx + Math.cos(a) * ringR,
        y: cy + Math.sin(a) * ringR,
        vx: 0,
        vy: 0,
        color: themeColor(f.theme[0]),
        radius,
        phase: Math.random() * Math.PI * 2,
        fixed: false,
      };
    });

    const byId = new Map<string, StarNode>();
    nodes.forEach((n) => byId.set(n.f.id, n));
    nodeByIdRef.current = byId;
    nodesRef.current = nodes;
    corrRef.current = data.correlations;

    // 背景星尘(世界坐标,散布在较大区域)
    const dust: { x: number; y: number; r: number; a: number }[] = [];
    const R = Math.max(w, h) * 1.2;
    for (let i = 0; i < 260; i++) {
      dust.push({
        x: cx + (Math.random() - 0.5) * R,
        y: cy + (Math.random() - 0.5) * R,
        r: Math.random() * 1.2 + 0.2,
        a: Math.random() * 0.04 + 0.01,
      });
    }
    dustRef.current = dust;
    settleRef.current = 320; // 重新布局帧数
  }, [data]);

  // 选中/悬停变化时,更新相关因子集合(浅色星)
  useEffect(() => {
    const focus = hoveredId ?? selectedId;
    const set = new Set<string>();
    if (focus && data) {
      for (const c of data.correlations) {
        if (c.a === focus) set.add(c.b);
        if (c.b === focus) set.add(c.a);
      }
    }
    relatedRef.current = set;
  }, [hoveredId, selectedId, data]);

  // ---- 主渲染循环 ----
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let t0 = performance.now();

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      sizeRef.current = { w: rect.width, h: rect.height };
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    const stepPhysics = () => {
      const nodes = nodesRef.current;
      const n = nodes.length;
      if (n === 0) return;
      const { w, h } = sizeRef.current;

      // 主题质心
      const cent: Record<string, { x: number; y: number; n: number }> = {};
      for (const nd of nodes) {
        for (const th of nd.f.theme) {
          const c = (cent[th] ||= { x: 0, y: 0, n: 0 });
          c.x += nd.x;
          c.y += nd.y;
          c.n += 1;
        }
      }
      for (const k in cent) {
        cent[k].x /= cent[k].n;
        cent[k].y /= cent[k].n;
      }

      // 斥力(O(N^2),N≤600 可接受)
      const REP = 1400;
      for (let i = 0; i < n; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < n; j++) {
          const b = nodes[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) d2 = 1;
          if (d2 < 22000) {
            const d = Math.sqrt(d2);
            const f = REP / d2;
            const fx = (dx / d) * f;
            const fy = (dy / d) * f;
            if (!a.fixed) {
              a.vx += fx;
              a.vy += fy;
            }
            if (!b.fixed) {
              b.vx -= fx;
              b.vy -= fy;
            }
          }
        }
      }

      // 相关引力(弹簧,目标 ~90px)
      const byId = nodeByIdRef.current;
      for (const c of corrRef.current) {
        if (Math.abs(c.r) < 0.5) continue;
        const na = byId.get(c.a);
        const nb = byId.get(c.b);
        if (!na || !nb) continue;
        const dx = nb.x - na.x;
        const dy = nb.y - na.y;
        const d = Math.hypot(dx, dy) || 1;
        const target = 90;
        const f = (d - target) * 0.01 * Math.abs(c.r);
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        if (!na.fixed) {
          na.vx += fx;
          na.vy += fy;
        }
        if (!nb.fixed) {
          nb.vx -= fx;
          nb.vy -= fy;
        }
      }

      // 主题聚簇 + 边界 + 积分
      const margin = 24;
      for (const nd of nodes) {
        // 主题质心吸引
        let tx = 0;
        let ty = 0;
        let tn = 0;
        for (const th of nd.f.theme) {
          const c = cent[th];
          if (c) {
            tx += c.x;
            ty += c.y;
            tn++;
          }
        }
        if (tn > 0) {
          nd.vx += (tx / tn - nd.x) * 0.002;
          nd.vy += (ty / tn - nd.y) * 0.002;
        }
        nd.vx *= 0.85;
        nd.vy *= 0.85;
        if (!nd.fixed) {
          nd.x += nd.vx;
          nd.y += nd.vy;
        }
        // 边界
        if (nd.x < margin) {
          nd.x = margin;
          nd.vx *= -0.5;
        }
        if (nd.x > w - margin) {
          nd.x = w - margin;
          nd.vx *= -0.5;
        }
        if (nd.y < margin) {
          nd.y = margin;
          nd.vy *= -0.5;
        }
        if (nd.y > h - margin) {
          nd.y = h - margin;
          nd.vy *= -0.5;
        }
      }
    };

    const draw = (now: number) => {
      const t = (now - t0) / 1000;
      const { w, h } = sizeRef.current;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      // 物理步进(布局收敛期)
      if (settleRef.current > 0) {
        stepPhysics();
        settleRef.current -= 1;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // 背景
      const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.75);
      g.addColorStop(0, "#0a0a14");
      g.addColorStop(1, "#000007");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // 视图变换(缩放 + 平移)
      ctx.translate(offsetRef.current.x, offsetRef.current.y);
      ctx.scale(zoomRef.current, zoomRef.current);

      // 背景星尘
      ctx.globalCompositeOperation = "lighter";
      for (const d of dustRef.current) {
        ctx.fillStyle = `rgba(180,200,255,${d.a})`;
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";

      const nodes = nodesRef.current;
      const sel = selRef.current;
      const hov = hovRef.current;
      const related = relatedRef.current;
      const filt = filterRef.current;

      // 逻辑链光通道
      drawLogicalChains(ctx, data, nodeByIdRef.current, t);

      // 相关虚线
      ctx.globalCompositeOperation = "source-over";
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 0.6;
      for (const c of corrRef.current) {
        if (Math.abs(c.r) < 0.5) continue;
        const na = nodeByIdRef.current.get(c.a);
        const nb = nodeByIdRef.current.get(c.b);
        if (!na || !nb) continue;
        const isHi =
          (sel && (c.a === sel || c.b === sel)) ||
          (hov && (c.a === hov || c.b === hov));
        ctx.strokeStyle = isHi
          ? `rgba(120,170,255,${0.5 * Math.abs(c.r)})`
          : `rgba(90,120,200,${0.12 * Math.abs(c.r)})`;
        ctx.beginPath();
        ctx.moveTo(na.x, na.y);
        ctx.lineTo(nb.x, nb.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // 星体
      for (const nd of nodes) {
        const status = nd.f.status;
        if (filt && !filt.includes(status)) continue;
        const isSel = nd.f.id === sel;
        const isHov = nd.f.id === hov;
        const isRel = related.has(nd.f.id) && (hov || sel);
        const base = STATUS_ALPHA[status];
        const pulse =
          status === "alive"
            ? 0.72 + 0.28 * Math.sin(t * 2.2 + nd.phase)
            : status === "decaying"
              ? 0.5 + 0.3 * Math.sin(t * 1.1 + nd.phase)
              : 1;
        const r = nd.radius * (isHov ? 1.35 : 1);

        // 发光晕(lighter)
        if (status !== "dead") {
          ctx.globalCompositeOperation = "lighter";
          const glowR = r * (status === "alive" ? 3.2 : 2.0) * pulse;
          const gg = ctx.createRadialGradient(nd.x, nd.y, 0, nd.x, nd.y, glowR);
          const col = nd.color;
          gg.addColorStop(0, hexA(col, status === "alive" ? 0.55 : 0.32));
          gg.addColorStop(1, hexA(col, 0));
          ctx.fillStyle = gg;
          ctx.beginPath();
          ctx.arc(nd.x, nd.y, glowR, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalCompositeOperation = "source-over";
        }

        // 核心
        ctx.globalAlpha = base * (isSel ? 1 : 1);
        ctx.fillStyle = status === "dead" ? "#3a3a4a" : nd.color;
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
        ctx.fill();

        // 选中环
        if (isSel) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(nd.x, nd.y, r + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
        // 相关高亮外环(浅色星)
        if (isRel) {
          ctx.strokeStyle = "rgba(120,170,255,0.9)";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(nd.x, nd.y, r + 3, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // 悬停标签
      const hovNode = hov ? nodeByIdRef.current.get(hov) : null;
      if (hovNode) {
        const label = hovNode.f.id;
        ctx.font = "11px ui-monospace, monospace";
        const tw = ctx.measureText(label).width;
        const lx = hovNode.x + hovNode.radius + 6;
        const ly = hovNode.y - hovNode.radius - 6;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(lx - 3, ly - 12, tw + 6, 16);
        ctx.fillStyle = "#dfe3ff";
        ctx.fillText(label, lx, ly);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    // ---- 交互 ----
    const screenToWorld = (sx: number, sy: number) => {
      return {
        x: (sx - offsetRef.current.x) / zoomRef.current,
        y: (sy - offsetRef.current.y) / zoomRef.current,
      };
    };
    const hitTest = (sx: number, sy: number): StarNode | null => {
      const wpt = screenToWorld(sx, sy);
      const nodes = nodesRef.current;
      for (let i = nodes.length - 1; i >= 0; i--) {
        const nd = nodes[i];
        const rr = nd.radius + 6;
        if ((nd.x - wpt.x) ** 2 + (nd.y - wpt.y) ** 2 <= rr * rr) return nd;
      }
      return null;
    };

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      if (dragRef.current) {
        const wpt = screenToWorld(sx, sy);
        if (dragRef.current.id) {
          const nd = nodeByIdRef.current.get(dragRef.current.id);
          if (nd) {
            nd.x = wpt.x - dragRef.current.dx;
            nd.y = wpt.y - dragRef.current.dy;
            nd.fixed = true;
          }
        } else {
          // 平移
          offsetRef.current.x += e.movementX;
          offsetRef.current.y += e.movementY;
        }
        return;
      }
      const hit = hitTest(sx, sy);
      onHoverRef.current?.(hit ? hit.f.id : null);
    };
    const onDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const hit = hitTest(sx, sy);
      if (hit) {
        dragRef.current = { id: hit.f.id, dx: 0, dy: 0 };
        const wpt = screenToWorld(sx, sy);
        dragRef.current.dx = wpt.x - hit.x;
        dragRef.current.dy = wpt.y - hit.y;
      } else {
        dragRef.current = { id: null, dx: 0, dy: 0 };
      }
    };
    const onUp = (_e: MouseEvent) => {
      if (dragRef.current && dragRef.current.id) {
        const nd = nodeByIdRef.current.get(dragRef.current.id);
        if (nd) nd.fixed = false;
        if (!_e.shiftKey) onSelectRef.current?.(dragRef.current.id);
      } else if (dragRef.current && dragRef.current.id === null) {
        // 空白处点击 → 取消选中(但拖拽平移不取消)
        onSelectRef.current?.(null);
      }
      dragRef.current = null;
    };
    const onLeave = () => onHoverRef.current?.(null);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newZoom = Math.max(0.4, Math.min(3, zoomRef.current * factor));
      const wx = (sx - offsetRef.current.x) / zoomRef.current;
      const wy = (sy - offsetRef.current.y) / zoomRef.current;
      offsetRef.current.x = sx - wx * newZoom;
      offsetRef.current.y = sy - wy * newZoom;
      zoomRef.current = newZoom;
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", onLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    const onVis = () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
      } else {
        raf = requestAnimationFrame(draw);
      }
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas ref={canvasRef} className="block w-full h-full cursor-grab active:cursor-grabbing" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 逻辑链光通道绘制
// ---------------------------------------------------------------------------

function drawLogicalChains(
  ctx: CanvasRenderingContext2D,
  data: FactorCosmosResponse | null,
  byId: Map<string, StarNode>,
  t: number,
) {
  if (!data || !data.logical_chain?.themes_covered?.length) return;
  const chain = data.logical_chain;
  const themes = chain.themes_covered;

  // 为每个主题找代表因子:优先 selected & alive
  const reps: (StarNode | null)[] = themes.map((th) => {
    const cands = [...byId.values()].filter((n) => n.f.theme.includes(th));
    if (!cands.length) return null;
    return (
      cands.find((n) => n.f.selected && n.f.status === "alive") ||
      cands.find((n) => n.f.status === "alive") ||
      cands[0]
    );
  });

  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < reps.length - 1; i++) {
    const a = reps[i];
    const b = reps[i + 1];
    if (!a || !b) continue;
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2 + (a.y - b.y) * 0.18;
    // 渐变发光曲线
    const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
    grad.addColorStop(0, hexA(a.color, 0.5));
    grad.addColorStop(1, hexA(b.color, 0.5));
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo(mx, my, b.x, b.y);
    ctx.stroke();
    // 流动粒子
    const N = 5;
    for (let k = 0; k < N; k++) {
      const p = ((t * 0.25 + k / N) % 1);
      const px = quad(a.x, mx, b.x, p);
      const py = quad(a.y, my, b.y, p);
      ctx.fillStyle = hexA(a.color, 0.8 * (1 - Math.abs(p - 0.5) * 2));
      ctx.beginPath();
      ctx.arc(px, py, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalCompositeOperation = "source-over";
}

function quad(p0: number, p1: number, p2: number, t: number): number {
  const mt = 1 - t;
  return mt * mt * p0 + 2 * mt * t * p1 + t * t * p2;
}

// hex (#rrggbb) → rgba 字符串,带 alpha
function hexA(hex: string, a: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
