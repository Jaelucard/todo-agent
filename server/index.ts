import { join } from "node:path";
import { createAgent } from "./agent.ts";
import { createApp } from "./app.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore, type TaskStore } from "./task-store.ts";

export function createTaskAgentApp(tasks: TaskStore, provider?: ModelProvider) {
  const agent = createAgent(tasks, provider);
  return createApp({
    tasks,
    async reply(messages) {
      return (await agent.runTurn(messages)).reply;
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
