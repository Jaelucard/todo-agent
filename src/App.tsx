import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { ChatMessage } from "../shared/chat.ts";
import { ChatRequestRejected, requestReply, requestTaskList } from "./chat-client.ts";
import { ChatHistory } from "./ChatHistory.tsx";
import { TaskDashboard, type DashboardState } from "./TaskDashboard.tsx";

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
  const [unknownOutcome, setUnknownOutcome] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardState>({ tasks: null, status: "loading" });
  const [mobileView, setMobileView] = useState<"chat" | "tasks">("chat");
  const input = useRef<HTMLTextAreaElement>(null);
  const sending = useRef(false);
  const latestRead = useRef(0);

  const refreshTasks = useCallback(async () => {
    const revision = ++latestRead.current;
    setDashboard((previous) => ({ ...previous, status: "loading" }));
    try {
      const tasks = await requestTaskList();
      if (latestRead.current === revision) setDashboard({ tasks, status: "ready" });
    } catch {
      if (latestRead.current === revision) {
        setDashboard((previous) => ({ ...previous, status: "error" }));
      }
    }
  }, []);

  useEffect(() => {
    void refreshTasks();
    return () => {
      latestRead.current += 1;
    };
  }, [refreshTasks]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || sending.current) return;
    sending.current = true;
    const outgoing: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(outgoing);
    setDraft("");
    setPending(true);
    setError("");
    setUnknownOutcome(false);
    try {
      const answer = await requestReply(outgoing);
      setMessages([...outgoing, { role: "assistant", content: answer }]);
    } catch (cause) {
      setMessages(messages);
      if (cause instanceof ChatRequestRejected) {
        setDraft(text);
        setError(`发送失败：${cause.message}`);
      } else {
        setUnknownOutcome(true);
      }
    } finally {
      sending.current = false;
      setPending(false);
      void refreshTasks();
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
      <nav className="mobile-views" aria-label="工作台视图">
        <button
          type="button"
          aria-pressed={mobileView === "chat"}
          aria-controls="workspace-chat"
          onClick={() => setMobileView("chat")}
        >
          对话
        </button>
        <button
          type="button"
          aria-pressed={mobileView === "tasks"}
          aria-controls="workspace-tasks"
          onClick={() => setMobileView("tasks")}
        >
          任务
        </button>
      </nav>
      <main className="workspace" data-view={mobileView}>
        <section className="chat pane" id="workspace-chat" aria-labelledby="chat-heading">
          <div className="pane-heading">
            <h2 id="chat-heading">对话</h2>
          </div>
          <ChatHistory messages={messages} pending={pending} onSelectPrompt={send} />
          <div className="compose">
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            {unknownOutcome && (
              <div className="recovery">
                <p className="error" role="alert">
                  发送结果未知，原操作可能仍在处理。请先查看当前任务，再决定是否重新发送。
                </p>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setMobileView("tasks");
                    void refreshTasks();
                    requestAnimationFrame(() => document.getElementById("tasks-heading")?.focus());
                  }}
                >
                  查看当前任务
                </button>
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
              <button className="send" disabled={pending || !draft.trim()} aria-label="发送消息">
                ↑
              </button>
            </form>
            <small>Enter 发送 · Shift + Enter 换行</small>
          </div>
        </section>
        <TaskDashboard state={dashboard} onRefresh={() => void refreshTasks()} />
      </main>
    </div>
  );
}
