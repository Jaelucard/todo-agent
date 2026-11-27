import type { ChatMessage } from "../shared/chat.ts";
import { readTaskMetadata, type TaskMetadata } from "../shared/task-metadata.ts";

export class ChatRequestRejected extends Error {
  override name = "ChatRequestRejected";
}

export interface TaskStatus extends TaskMetadata {
  id: string;
  title: string;
  done: boolean;
}

export async function requestReply(messages: ChatMessage[], signal?: AbortSignal): Promise<string> {
  const timeoutSignal = AbortSignal.timeout(25_000);
  const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal: requestSignal,
  });
  const body: unknown = await response.json();
  if (typeof body === "object" && body !== null) {
    if (response.ok && "reply" in body && typeof body.reply === "string" && body.reply.trim()) {
      return body.reply;
    }
    const confirmedRejection =
      response.status === 400 ||
      response.status === 413 ||
      (response.status === 503 && "mutated" in body && body.mutated === false);
    if (
      confirmedRejection &&
      "error" in body &&
      typeof body.error === "string" &&
      body.error.trim()
    ) {
      throw new ChatRequestRejected(body.error);
    }
  }
  throw new Error("没有收到有效的聊天响应。");
}

function isTaskStatus(value: unknown): value is Pick<TaskStatus, "id" | "title" | "done"> {
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
    if (Array.isArray(body) && body.every(isTaskStatus)) {
      return body.map((task) => ({
        id: task.id,
        title: task.title,
        done: task.done,
        ...readTaskMetadata(task),
      }));
    }
  }
  throw new Error("读取任务清单失败。");
}
