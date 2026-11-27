import { join } from "node:path";
import { createAgent } from "./agent.ts";
import { ChatServiceUnavailableError, createApp } from "./app.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore, type TaskStore } from "./task-store.ts";

export function createTaskAgentApp(tasks: TaskStore, provider?: ModelProvider) {
  const agent = createAgent(tasks, provider);
  return createApp({
    tasks,
    async reply(messages, signal) {
      const turn = await agent.runTurn(messages, signal);
      if (
        turn.outcomes.length === 0 &&
        (turn.failure === "config_error" || turn.failure === "model_error")
      ) {
        const message =
          turn.failure === "config_error"
            ? "模型服务未正确配置，请检查 .env 中的 DEEPSEEK_API_KEY 后重试。本轮未修改任务。"
            : "模型服务暂时无法处理请求，请稍后重试。本轮未修改任务。";
        throw new ChatServiceUnavailableError(turn.failure, message);
      }
      return turn.reply;
    },
  });
}

if (import.meta.main) {
  const tasks = createJsonTaskStore(join("data", "tasks.json"));
  const port = Number(process.env.PORT || 3000);
  createTaskAgentApp(tasks).listen(port, "127.0.0.1", () => {
    console.log(`小明智能体已启动：http://127.0.0.1:${port}`);
  });
}
