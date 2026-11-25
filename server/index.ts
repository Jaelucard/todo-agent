import express from "express";
import type { ErrorRequestHandler } from "express";
import { resolve } from "node:path";
import { z } from "zod";
import { reply } from "./agent.ts";
import { taskRouter } from "./tasks.ts";

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use("/api/tasks", taskRouter());

const chatRequest = z.object({
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().trim().min(1),
      }),
    )
    .min(1),
});

app.post("/api/chat", async (req, res) => {
  const input = chatRequest.safeParse(req.body);
  if (!input.success || input.data.messages.at(-1)?.role !== "user") {
    return res.status(400).json({ error: "请提供对话记录，并以用户消息结尾。" });
  }
  res.json({ reply: await reply(input.data.messages) });
});

app.use("/api", (_req, res) => res.status(404).json({ error: "接口不存在。" }));
const handleError: ErrorRequestHandler = (error, _req, res, _next) => {
  if (error.type === "entity.parse.failed")
    return res.status(400).json({ error: "请求需要使用有效的 JSON。" });
  if (error.type === "entity.too.large")
    return res.status(413).json({ error: "对话太长，请刷新后开始新的对话。" });
  console.error(error);
  res.status(500).json({ error: "服务暂时无法处理请求，请稍后重试。" });
};
app.use(handleError);

if (process.argv.includes("--dev")) {
  const { createServer } = await import("vite");
  const vite = await createServer({ server: { middlewareMode: true }, appType: "spa" });
  app.use(vite.middlewares);
} else {
  app.use(express.static(resolve("dist")));
}
const port = Number(process.env.PORT || 3000);
app.listen(port, "127.0.0.1", () => console.log(`小明智能体已启动：http://127.0.0.1:${port}`));
