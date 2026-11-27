import { resolve } from "node:path";
import express, { type ErrorRequestHandler } from "express";
import { chatRequestSchema, type ChatMessage } from "../shared/chat.ts";
import type { TaskStore } from "./task-store.ts";
import { taskRouter } from "./tasks.ts";

export class ChatServiceUnavailableError extends Error {
  override name = "ChatServiceUnavailableError";
  readonly status = 503;
  readonly code: "config_error" | "model_error";
  readonly mutated = false;

  constructor(code: "config_error" | "model_error", message: string) {
    super(message);
    this.code = code;
  }
}

interface AppDependencies {
  reply(messages: ChatMessage[], signal?: AbortSignal): Promise<string>;
  tasks: TaskStore;
}

export function createApp({ reply, tasks }: AppDependencies) {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.post("/api/chat", async (req, res) => {
    const input = chatRequestSchema.safeParse(req.body);
    if (!input.success) {
      return res.status(400).json({ error: "请提供对话记录，并以用户消息结尾。" });
    }
    const controller = new AbortController();
    const onClose = () => {
      if (!res.writableEnded) controller.abort();
    };
    res.on("close", onClose);
    try {
      const answer = await reply(input.data.messages, controller.signal);
      if (!res.writableEnded && !controller.signal.aborted) {
        res.json({ reply: answer });
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      throw error;
    } finally {
      res.off("close", onClose);
    }
  });

  app.use("/api/tasks", taskRouter(tasks));
  app.use("/api", (_req, res) => res.status(404).json({ error: "接口不存在。" }));
  app.use(handleApiError);

  app.use(express.static(resolve("public")));
  app.get("/assets/chat.js", (_req, res) => res.sendFile("chat.js", { root: resolve("dist") }));
  return app;
}

const handleApiError: ErrorRequestHandler = (error, _req, res, _next) => {
  if (res.headersSent || res.writableEnded) return;
  if (error instanceof ChatServiceUnavailableError) {
    return res.status(503).json({
      error: error.message,
      code: error.code,
      mutated: false,
    });
  }
  if (error.type === "entity.parse.failed") {
    return res.status(400).json({ error: "请求需要使用有效的 JSON。" });
  }
  if (error.type === "entity.too.large") {
    return res.status(413).json({ error: "对话太长，请刷新后开始新的对话。" });
  }
  const errorName = error instanceof Error ? error.name : "UnknownError";
  console.error(`[app] unexpected_error error=${errorName}`);
  res.status(500).json({ error: "服务暂时无法处理请求，请稍后重试。" });
};
