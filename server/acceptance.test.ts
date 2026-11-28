import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ChatMessage } from "../shared/chat.ts";
import { createTaskAgentApp } from "./index.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

function toolCall(id: string, name: string, args: object) {
  return { id, type: "function", function: { name, arguments: JSON.stringify(args) } };
}

test("the official six-turn conversation preserves task identity and exact saved state", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-acceptance-"));
  const filename = join(directory, "tasks.json");
  const tasks = createJsonTaskStore(filename);
  const history: ChatMessage[] = [];
  let script: ReturnType<typeof toolCall>[] = [];
  const requests: Parameters<ModelProvider>[0][] = [];
  const app = createTaskAgentApp(tasks, async (request) => {
    const call = script[requests.length];
    requests.push(request);
    return call
      ? {
          finish_reason: "tool_calls",
          message: { role: "assistant", content: null, tool_calls: [call] },
        }
      : { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(async () => {
    try {
      await server[Symbol.asyncDispose]();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const url = `http://127.0.0.1:${address.port}`;

  async function readApiTasks() {
    const response = await fetch(url + "/api/tasks");
    assert.equal(response.status, 200);
    return response.json();
  }

  async function turn(
    input: string,
    calls: ReturnType<typeof toolCall>[] = [],
    terminalOnWrite = false,
  ) {
    script = calls;
    requests.length = 0;
    const messages: ChatMessage[] = [...history, { role: "user", content: input }];
    const response = await fetch(url + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    assert.equal(response.status, 200);
    const { reply } = await response.json();
    assert.equal(typeof reply, "string");
    if (calls.length) {
      assert.equal(requests.length, calls.length + (terminalOnWrite ? 0 : 1));
      const first = requests[0].messages;
      assert.equal(first[0].role, "system");
      assert.deepEqual(first.slice(1), messages);
    } else {
      assert.equal(requests.length, 0, "This action only needs the local task store");
    }
    const results = calls.slice(0, requests.length - 1).map((call, index) => {
      const [assistant, result] = requests[index + 1].messages.slice(-2);
      assert.deepEqual(assistant, { role: "assistant", content: null, tool_calls: [call] });
      assert.ok(result.role === "tool" && result.tool_call_id === call.id);
      assert.ok(typeof result.content === "string");
      return JSON.parse(result.content);
    });
    history.push(...messages.slice(-1), { role: "assistant", content: reply });
    return { reply, results };
  }

  async function saved() {
    const bytes = await readFile(filename, "utf8");
    const state = JSON.parse(bytes);
    assert.deepEqual(await readApiTasks(), state);
    return { bytes, state };
  }

  assert.deepEqual(await readApiTasks(), []);
  await assert.rejects(readFile(filename), { code: "ENOENT" });

  const createMilk = await turn("帮我记一下买牛奶", [
    toolCall("create-milk", "create_task", { title: "买牛奶" }),
  ]);
  const { state: milkState } = await saved();
  assert.equal(milkState.length, 1);
  assert.match(milkState[0].id, /^[0-9a-f-]{36}$/u);
  const milk = { id: milkState[0].id, title: "买牛奶", done: false };
  assert.deepEqual(milkState, [milk]);
  assert.deepEqual(createMilk.results, [{ ok: true, tool: "create_task", task: milk }]);
  assert.match(createMilk.reply, /已添加.*买牛奶/u);

  const createParcel = await turn("再记一条取快递");
  const { state: both, bytes: bothBytes } = await saved();
  assert.equal(both.length, 2);
  assert.match(both[1].id, /^[0-9a-f-]{36}$/u);
  assert.notEqual(both[1].id, milk.id);
  const parcel = { id: both[1].id, title: "取快递", done: false };
  assert.deepEqual(both, [milk, parcel]);
  assert.match(createParcel.reply, /已添加.*取快递/u);

  const list = await turn("我有哪些任务？");
  assert.deepEqual(await saved(), { state: [milk, parcel], bytes: bothBytes });
  assert.equal(list.reply, "当前清单：\n- 买牛奶（待完成）\n- 取快递（待完成）");

  const complete = await turn(
    "牛奶买好了",
    [
      toolCall("lookup-milk", "list_tasks", {}),
      toolCall("write-milk", "complete_task", { candidateIds: [milk.id] }),
    ],
    true,
  );
  const completedMilk = { ...milk, done: true };
  assert.deepEqual((await saved()).state, [completedMilk, parcel]);
  assert.deepEqual(complete.results, [{ ok: true, tool: "list_tasks", tasks: [milk, parcel] }]);
  assert.match(complete.reply, /已完成.*买牛奶/u);

  const remove = await turn("取快递这条删掉吧");
  const { state: finalState, bytes: finalBytes } = await saved();
  assert.deepEqual(finalState, [completedMilk]);
  assert.match(remove.reply, /已删除.*取快递/u);

  const finalList = await turn("再给我看看清单");
  assert.deepEqual(await saved(), { state: [completedMilk], bytes: finalBytes });
  assert.equal(finalList.reply, "当前清单：\n- 买牛奶（已完成）");
});
