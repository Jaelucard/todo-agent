import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../shared/chat.ts";
import { ChatRequestRejected, requestReply, requestTaskList } from "./chat-client.ts";
import { ChatHistory } from "./ChatHistory.tsx";
import { TaskDashboard, type DashboardState } from "./TaskDashboard.tsx";
import { isSendShortcut } from "./task-view.ts";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [failedDraft, setFailedDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [unknownOutcome, setUnknownOutcome] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardState>({ tasks: null, status: "loading" });
  const [mobileView, setMobileView] = useState<"chat" | "tasks">("chat");
  const [chatHasUpdate, setChatHasUpdate] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const sending = useRef(false);
  const latestRead = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);
  const mobileViewRef = useRef<"chat" | "tasks">("chat");

  const selectMobileView = useCallback((view: "chat" | "tasks") => {
    mobileViewRef.current = view;
    setMobileView(view);
    if (view === "chat") setChatHasUpdate(false);
  }, []);

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
      activeRequest.current?.abort();
    };
  }, [refreshTasks]);

  function cancelPending() {
    activeRequest.current?.abort();
  }

  async function send(text: string) {
    const content = text.trim();
    if (!content || sending.current) return;
    sending.current = true;
    const controller = new AbortController();
    activeRequest.current = controller;
    const outgoing: ChatMessage[] = [...messages, { role: "user", content }];
    selectMobileView("chat");
    setMessages(outgoing);
    setDraft("");
    setFailedDraft("");
    setPending(true);
    setError("");
    setUnknownOutcome(false);
    let shouldFocusInput = true;
    try {
      const answer = await requestReply(outgoing, controller.signal);
      setMessages([...outgoing, { role: "assistant", content: answer }]);
    } catch (cause) {
      setMessages(messages);
      if (cause instanceof ChatRequestRejected) {
        setDraft(text);
        setError(`发送失败：${cause.message}`);
      } else {
        shouldFocusInput = false;
        setFailedDraft(content);
        setUnknownOutcome(true);
      }
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null;
      sending.current = false;
      setPending(false);
      if (mobileViewRef.current !== "chat") setChatHasUpdate(true);
      void refreshTasks();
      if (shouldFocusInput) requestAnimationFrame(() => input.current?.focus());
    }
  }

  const pendingTaskCount = dashboard.tasks
    ? dashboard.tasks.filter((task) => !task.done).length
    : null;

  return (
    <div className="app">
      <header>
        <img src="/logo.png" alt="" />
        <h1 className="brand-title">
          <strong>TokeNow</strong>
          <span>小明智能体</span>
        </h1>
      </header>
      <nav className="mobile-views" aria-label="工作台视图">
        <button
          type="button"
          aria-pressed={mobileView === "chat"}
          aria-controls="workspace-chat"
          onClick={() => selectMobileView("chat")}
        >
          <span>对话</span>
          {chatHasUpdate && (
            <>
              <span className="tab-dot" aria-hidden="true" />
              <span className="sr-only">（有新消息）</span>
            </>
          )}
        </button>
        <button
          type="button"
          aria-pressed={mobileView === "tasks"}
          aria-controls="workspace-tasks"
          onClick={() => selectMobileView("tasks")}
        >
          {pendingTaskCount === null ? "任务" : `任务 (${pendingTaskCount})`}
        </button>
      </nav>
      <main className="workspace" data-view={mobileView}>
        <section className="chat pane" id="workspace-chat" aria-labelledby="chat-heading">
          <div className="pane-heading">
            <h2 id="chat-heading">对话</h2>
          </div>
          <ChatHistory
            messages={messages}
            pending={pending}
            visible={mobileView === "chat"}
            onSelectPrompt={send}
          />
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
                {failedDraft && (
                  <div className="recovery-draft">
                    <span className="recovery-draft-label">未确认发送的内容：</span>
                    <p className="recovery-draft-text">{failedDraft}</p>
                  </div>
                )}
                <div className="recovery-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      selectMobileView("tasks");
                      void refreshTasks();
                      requestAnimationFrame(() =>
                        document.getElementById("tasks-heading")?.focus(),
                      );
                    }}
                  >
                    查看当前任务
                  </button>
                  {failedDraft && (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => {
                        setDraft(failedDraft);
                        setUnknownOutcome(false);
                        setFailedDraft("");
                        requestAnimationFrame(() => input.current?.focus());
                      }}
                    >
                      填入输入框
                    </button>
                  )}
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      setUnknownOutcome(false);
                      setFailedDraft("");
                    }}
                  >
                    关闭提示
                  </button>
                </div>
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
              {pending && (
                <button
                  type="button"
                  className="secondary-button cancel-send"
                  onClick={cancelPending}
                >
                  取消
                </button>
              )}
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
