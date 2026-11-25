import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../server/agent.ts";

const examples = ["提醒我买牛奶", "看看我的待办"];

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const history = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (history.current) history.current.scrollTop = history.current.scrollHeight;
  }, [messages, pending, error]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || pending) return;
    const next: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(next);
    setDraft("");
    setPending(true);
    setError("");
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next }),
      });
      const body = await response.json();
      if (!response.ok || typeof body.reply !== "string")
        throw new Error(body.error || "服务暂时不可用。");
      setMessages([...next, { role: "assistant", content: body.reply }]);
    } catch (cause) {
      setMessages(messages);
      setDraft(text);
      setError(cause instanceof Error ? `发送失败：${cause.message}` : "发送失败，请重试。");
    } finally {
      setPending(false);
      requestAnimationFrame(() => input.current?.focus());
    }
  }

  return (
    <div className="app">
      <header>
        <img src="/logo.png" alt="" />
        <strong>TokeNow</strong>
        <span>小明智能体</span>
      </header>
      <main>
        <div className="scroll" ref={history}>
          {messages.length === 0 ? (
            <section className="welcome">
              <img src="/logo.png" alt="" />
              <h1>小明智能体</h1>
              <p>今天有什么待办？</p>
              <div className="examples">
                {examples.map((item) => (
                  <button key={item} onClick={() => void send(item)}>
                    {item}
                    <span aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="thread" role="log" aria-label="对话记录" aria-live="polite">
              {messages.map((message, i) => (
                <div key={i} className={`message ${message.role}`}>
                  <small>{message.role === "user" ? "你" : "小明智能体"}</small>
                  <div>{message.content}</div>
                </div>
              ))}
              {pending && <p role="status">正在回复…</p>}
            </div>
          )}
        </div>
        <div className="compose">
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void send(draft);
            }}
          >
            <textarea
              aria-label="消息"
              ref={input}
              placeholder="输入消息…"
              rows={2}
              value={draft}
              readOnly={pending}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing &&
                  event.nativeEvent.keyCode !== 229
                ) {
                  event.preventDefault();
                  void send(draft);
                }
              }}
            />
            <button className="send" disabled={pending || !draft.trim()} aria-label="发送消息">
              ↑
            </button>
          </form>
          <small>Enter 发送 · Shift + Enter 换行</small>
        </div>
      </main>
    </div>
  );
}
