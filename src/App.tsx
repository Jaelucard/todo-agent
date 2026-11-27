import { useRef, useState, type KeyboardEvent } from "react";
import type { ChatMessage } from "../shared/chat.ts";
import {
  ChatRequestRejected,
  requestReply,
  requestTaskList,
  type TaskStatus,
} from "./chat-client.ts";
import { ChatHistory } from "./ChatHistory.tsx";

type Recovery =
  | { status: "available" | "checking" | "failed" }
  | { status: "loaded"; tasks: TaskStatus[] };

function isSendShortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
  // 中文输入法会用 Enter 确认选词，部分浏览器此时只报告键码 229。
  const composing = event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229;
  return event.key === "Enter" && !event.shiftKey && !composing;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [recovery, setRecovery] = useState<Recovery>();
  const input = useRef<HTMLTextAreaElement>(null);
  const checking = recovery?.status === "checking";

  async function send(text: string) {
    const content = text.trim();
    if (!content || pending || checking) return;
    const outgoing: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(outgoing);
    setDraft("");
    setPending(true);
    setError("");
    setRecovery(undefined);
    try {
      const answer = await requestReply(outgoing);
      setMessages([...outgoing, { role: "assistant", content: answer }]);
    } catch (cause) {
      setMessages(messages);
      if (cause instanceof ChatRequestRejected) {
        setDraft(text);
        setError(`发送失败：${cause.message}`);
      } else {
        setRecovery({ status: "available" });
      }
    } finally {
      setPending(false);
      requestAnimationFrame(() => input.current?.focus());
    }
  }

  async function checkTasks() {
    if (checking) return;
    setRecovery({ status: "checking" });
    try {
      const tasks = await requestTaskList();
      setRecovery({ status: "loaded", tasks });
    } catch {
      setRecovery({ status: "failed" });
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
        <ChatHistory messages={messages} pending={pending} onSelectPrompt={send} />
        <div className="compose">
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {recovery && (
            <div>
              <p className="error" role="alert">
                发送结果未知，原操作可能仍在处理。请先查看当前任务，再决定是否重新发送。
              </p>
              <button type="button" disabled={checking} onClick={() => void checkTasks()}>
                查看当前任务
              </button>
              {checking && <p role="status">正在读取任务…</p>}
              {recovery.status === "failed" && <p role="status">读取任务清单失败。</p>}
              {recovery.status === "loaded" &&
                (recovery.tasks.length ? (
                  <ul
                    aria-label="当前任务"
                    style={{ maxHeight: "12rem", overflowY: "auto", overflowWrap: "anywhere" }}
                  >
                    {recovery.tasks.map((task) => (
                      <li key={task.id}>
                        {task.title}（{task.done ? "已完成" : "待完成"}）
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p role="status">当前没有待办。</p>
                ))}
            </div>
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
                if (isSendShortcut(event)) {
                  event.preventDefault();
                  void send(draft);
                }
              }}
            />
            <button
              className="send"
              disabled={pending || checking || !draft.trim()}
              aria-label="发送消息"
            >
              ↑
            </button>
          </form>
          <small>Enter 发送 · Shift + Enter 换行</small>
        </div>
      </main>
    </div>
  );
}
