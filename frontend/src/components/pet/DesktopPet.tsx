import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { X, Minus, MessageSquare, Send, RotateCcw, Sparkles, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

interface PetMsg {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  tool?: string;
}

const POS_KEY = "vibe-deskpet-pos";
const SID_KEY = "vibe-deskpet-sid";

function loadPos(): { x: number; y: number } {
  try {
    const v = JSON.parse(localStorage.getItem(POS_KEY) || "");
    if (v && typeof v.x === "number" && typeof v.y === "number") return v;
  } catch {
    /* ignore */
  }
  return { x: window.innerWidth - 96, y: window.innerHeight - 120 };
}

export function DesktopPet() {
  const { t, i18n } = useTranslation();
  const [pos, setPos] = useState<{ x: number; y: number }>(() =>
    typeof window === "undefined" ? { x: 0, y: 0 } : loadPos(),
  );
  const [open, setOpen] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const [messages, setMessages] = useState<PetMsg[]>([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sid, setSid] = useState<string | null>(() =>
    typeof window === "undefined" ? null : localStorage.getItem(SID_KEY),
  );

  const dragging = useRef(false);
  const dragOff = useRef({ dx: 0, dy: 0 });
  const esRef = useRef<EventSource | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const prevOpen = useRef(false);
  const greetKey = i18n.language === "zh-CN" ? "deskpet.greetingZh" : "deskpet.greeting";

  // 自动滚动到底部
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, open, minimized]);

  // 仅在「面板打开的瞬间」加载历史会话; 发送过程中 setSid 不再触发重载,
  // 避免把乐观添加的 [用户, 助手占位] 覆盖成服务端 [用户], 导致 SSE text_delta 找不到占位而“无回应”。
  useEffect(() => {
    const justOpened = open && !prevOpen.current;
    prevOpen.current = open;
    if (!justOpened || !sid) return;
    api
      .getSessionMessages(sid)
      .then((msgs) => {
        if (!Array.isArray(msgs) || msgs.length === 0) return;
        setMessages(
          msgs.map((m) => ({
            id: m.message_id,
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content ?? "",
          })),
        );
      })
      .catch(() => {});
  }, [open, sid]);

  const closeEs = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const ensureSession = useCallback(async (): Promise<string> => {
    if (sid) return sid;
    const s = await api.createSession("Vibe Pet");
    localStorage.setItem(SID_KEY, s.session_id);
    setSid(s.session_id);
    return s.session_id;
  }, [sid]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || thinking) return;
    setInput("");
    setError(null);
    const myMsg: PetMsg = { id: "u" + Date.now(), role: "user", content: text };
    const asstId = "a" + Date.now();
    setMessages((m) => [...m, myMsg, { id: asstId, role: "assistant", content: "" }]);
    setThinking(true);

    try {
      const sessionId = await ensureSession();
      await api.sendMessage(sessionId, text);
      closeEs();
      const es = new EventSource(api.sseUrl(sessionId));
      esRef.current = es;
      let acc = "";
      es.addEventListener("text_delta", (e) => {
        try {
          const d = JSON.parse((e as MessageEvent).data);
          acc += String(d.delta ?? "");
          setMessages((m) => m.map((x) => (x.id === asstId ? { ...x, content: acc } : x)));
        } catch {
          /* ignore */
        }
      });
      es.addEventListener("tool_call", (e) => {
        try {
          const d = JSON.parse((e as MessageEvent).data);
          const tool = String(d.tool || "");
          setMessages((m) =>
            m.map((x) =>
              x.id === asstId && !x.tool ? { ...x, tool: tool || x.tool } : x,
            ),
          );
        } catch {
          /* ignore */
        }
      });
      const finish = () => {
        closeEs();
        setThinking(false);
      };
      es.addEventListener("attempt.completed", finish);
      es.addEventListener("attempt.failed", () => {
        setError(t("deskpet.error"));
        finish();
      });
      es.addEventListener("done", finish);
      es.onerror = () => {
        // 连接关闭(含正常结束)→ 收尾
        setThinking(false);
        closeEs();
      };
    } catch {
      setError(t("deskpet.error"));
      setThinking(false);
      setMessages((m) =>
        m.map((x) => (x.id === asstId && !x.content ? { ...x, content: t("deskpet.error") } : x)),
      );
    }
  }, [input, thinking, sid, ensureSession, closeEs, t]);

  const newChat = useCallback(async () => {
    closeEs();
    setMessages([]);
    setError(null);
    try {
      const s = await api.createSession("Vibe Pet");
      localStorage.setItem(SID_KEY, s.session_id);
      setSid(s.session_id);
    } catch {
      setError(t("deskpet.error"));
    }
  }, [closeEs, t]);

  // 拖拽
  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    dragOff.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    const x = Math.max(12, Math.min(window.innerWidth - 80, e.clientX - dragOff.current.dx));
    const y = Math.max(12, Math.min(window.innerHeight - 80, e.clientY - dragOff.current.dy));
    setPos({ x, y });
  };
  const onPointerUp = () => {
    dragging.current = false;
    localStorage.setItem(POS_KEY, JSON.stringify(pos));
  };

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const bubble = thinking
    ? t("deskpet.thinking")
    : !open && messages.length === 0
      ? t(greetKey)
      : !open && lastAssistant
        ? lastAssistant.content.slice(0, 48) + (lastAssistant.content.length > 48 ? "…" : "")
        : null;

  return (
    <>
      <style>{`
        @keyframes dp-breathe { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-3px) scale(1.03)} }
        @keyframes dp-blink { 0%,96%,100%{transform:scaleY(1)} 98%{transform:scaleY(0.1)} }
        @keyframes dp-ring { 0%{transform:scale(0.9);opacity:.7} 100%{transform:scale(1.5);opacity:0} }
        .dp-breathe{animation:dp-breathe 3.2s ease-in-out infinite}
        .dp-eye{transform-origin:center;animation:dp-blink 5s infinite}
        .dp-think .dp-ring{animation:dp-ring 1.4s ease-out infinite}
      `}</style>

      <div
        className="fixed z-[55] select-none"
        style={{ left: pos.x, top: pos.y, touchAction: "none" }}
      >
        {/* 气泡 */}
        {bubble && !open && (
          <div className="absolute bottom-16 right-0 mb-1 max-w-[230px] rounded-2xl rounded-br-sm bg-popover px-3 py-2 text-xs text-foreground shadow-lg ring-1 ring-black/5">
            {bubble}
          </div>
        )}

        {/* 聊天面板 */}
        {open && (
          <div
            className={cn(
              "absolute bottom-20 right-0 flex w-[360px] max-w-[92vw] flex-col rounded-2xl border border-border bg-card shadow-2xl ring-1 ring-black/10",
              minimized ? "h-[52px]" : "h-[460px] max-h-[70vh]",
            )}
          >
            {/* 头部 */}
            <div className="flex items-center gap-2 rounded-t-2xl border-b border-border bg-primary/10 px-3 py-2">
              <Bot className="h-4 w-4 text-primary" />
              <span className="flex-1 text-sm font-medium">{t("deskpet.title")}</span>
              <button
                onClick={newChat}
                title={t("deskpet.newSession")}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => setMinimized((v) => !v)}
                title={t("deskpet.minimize")}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                {minimized ? <MessageSquare className="h-3.5 w-3.5" /> : <Minus className="h-3.5 w-3.5" />}
              </button>
              <button
                onClick={() => setOpen(false)}
                title={t("deskpet.close")}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {!minimized && (
              <>
                {/* 消息列表 */}
                <div ref={listRef} className="flex-1 space-y-2 overflow-auto p-3">
                  {messages.length === 0 && (
                    <p className="text-xs text-muted-foreground">{t(greetKey)}</p>
                  )}
                  {messages.map((m) => (
                    <div
                      key={m.id}
                      className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}
                    >
                      <div
                        className={cn(
                          "max-w-[85%] rounded-2xl px-3 py-2 text-xs leading-relaxed",
                          m.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted text-foreground",
                        )}
                      >
                        {m.role === "assistant" && m.tool && (
                          <span className="mb-1 inline-flex items-center gap-1 text-[10px] text-primary">
                            <Sparkles className="h-3 w-3" />
                            {m.tool}
                          </span>
                        )}
                        {m.content || (thinking ? t("deskpet.thinking") : "")}
                      </div>
                    </div>
                  ))}
                  {error && <p className="text-xs text-danger">{error}</p>}
                </div>

                {/* 输入 */}
                <div className="flex items-center gap-2 border-t border-border p-2">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    placeholder={t("deskpet.placeholder")}
                    className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-xs text-foreground outline-none focus:border-primary"
                  />
                  <button
                    onClick={send}
                    disabled={thinking || !input.trim()}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground disabled:opacity-50"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* 吉祥物 */}
        <div
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onClick={() => (dragging.current ? null : setOpen((v) => !v))}
          className={cn(
            "relative flex h-16 w-16 cursor-grab items-center justify-center active:cursor-grabbing",
            thinking && "dp-think",
          )}
          title={t("deskpet.hintClick")}
        >
          <span className="dp-ring absolute inset-2 rounded-full bg-primary/30" />
          <span className="dp-breathe absolute inset-1 rounded-full bg-gradient-to-br from-primary to-fuchsia-500 shadow-lg" />
          {/* 眼睛 */}
          <span className="absolute left-5 top-6 h-2 w-2 rounded-full bg-white dp-eye" />
          <span className="absolute right-5 top-6 h-2 w-2 rounded-full bg-white dp-eye" />
          {/* 腮红 / 嘴 */}
          <span className="absolute left-4 top-9 h-1.5 w-1.5 rounded-full bg-fuchsia-300/70" />
          <span className="absolute right-4 top-9 h-1.5 w-1.5 rounded-full bg-fuchsia-300/70" />
          {thinking && (
            <span className="absolute -top-1 right-1 flex gap-0.5 text-[10px] text-white">
              <span className="animate-bounce">·</span>
              <span className="animate-bounce" style={{ animationDelay: "0.15s" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "0.3s" }}>·</span>
            </span>
          )}
        </div>
      </div>
    </>
  );
}
