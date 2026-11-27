import { useEffect, useRef } from "react";
import type { ChatMessage } from "../shared/chat.ts";

const examplePrompts = ["提醒我买牛奶", "看看我的待办"];

interface ChatHistoryProps {
  messages: ChatMessage[];
  pending: boolean;
  visible?: boolean;
  onSelectPrompt: (prompt: string) => void;
}

export function ChatHistory({
  messages,
  pending,
  visible = true,
  onSelectPrompt,
}: ChatHistoryProps) {
  const viewport = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (viewport.current && (visible || viewport.current.scrollHeight > 0)) {
      viewport.current.scrollTop = viewport.current.scrollHeight;
    }
  }, [messages, pending, visible]);

  return (
    <div className="scroll" ref={viewport}>
      {messages.length === 0 ? (
        <section className="welcome">
          <img src="/logo.png" alt="" />
          <p className="welcome-title">小明智能体</p>
          <p>今天有什么待办？</p>
          <div className="examples">
            {examplePrompts.map((prompt) => (
              <button key={prompt} type="button" onClick={() => onSelectPrompt(prompt)}>
                {prompt}
                <span aria-hidden="true">↗</span>
              </button>
            ))}
          </div>
        </section>
      ) : (
        <div className="thread" role="log" aria-label="对话记录" aria-live="polite">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <small>{message.role === "user" ? "你" : "小明智能体"}</small>
              <div>{message.content}</div>
            </div>
          ))}
          {pending && <p role="status">正在回复…</p>}
        </div>
      )}
    </div>
  );
}
