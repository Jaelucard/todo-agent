import type { ChatMessage } from "../shared/chat.ts";

export class ChatRequestRejected extends Error {
  override name = "ChatRequestRejected";
}

export interface TaskStatus {
  id: string;
  title: string;
  done: boolean;
}

export async function requestReply(messages: ChatMessage[]): Promise<string> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  const body: unknown = await response.json();
  if (typeof body === "object" && body !== null) {
    if (response.ok && "reply" in body && typeof body.reply === "string" && body.reply.trim())
      return body.reply;
    if (
      (response.status === 400 || response.status === 413) &&
      "error" in body &&
      typeof body.error === "string" &&
      body.error.trim()
    ) {
      throw new ChatRequestRejected(body.error);
    }
  }
  throw new Error("没有收到有效的聊天响应。");
}

function isTaskStatus(value: unknown): value is TaskStatus {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    typeof value.id === "string" &&
    "title" in value &&
    typeof value.title === "string" &&
    "done" in value &&
    typeof value.done === "boolean"
  );
}

export async function requestTaskList(): Promise<TaskStatus[]> {
  const response = await fetch("/api/tasks", {
    method: "GET",
    cache: "no-store",
    signal: AbortSignal.timeout(5_000),
  });
  if (response.ok) {
    const body: unknown = await response.json();
    if (Array.isArray(body) && body.every(isTaskStatus)) return body;
  }
  throw new Error("读取任务清单失败。");
}
