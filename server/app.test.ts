import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import type { ChatMessage } from "../shared/chat.ts";
import { createApp } from "./app.ts";
import { createTaskAgentApp } from "./index.ts";
import { createJsonTaskStore } from "./task-store.ts";

async function startTestApp(t: TestContext) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  const conversations: ChatMessage[][] = [];
  const app = createApp({
    tasks: createJsonTaskStore(filename),
    async reply(messages) {
      conversations.push(messages);
      return "收到";
    },
  });

  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;

  async function request(method: string, path: string, body?: unknown) {
    const response = await fetch(`http://127.0.0.1:${port}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return {
      status: response.status,
      body: response.status === 204 ? null : await response.json(),
    };
  }

  return { request, conversations, filename };
}

test("任务 API 可以创建、列出、修改和删除任务", async (t) => {
  const { request } = await startTestApp(t);
  const collection = "/api/tasks";
  assert.deepEqual(await request("GET", collection), { status: 200, body: [] });

  const created = await request("POST", collection, { title: " 买牛奶 " });
  assert.equal(created.status, 201);
  const task = created.body;
  assert.equal(typeof task.id, "string");
  assert.deepEqual(task, { id: task.id, title: "买牛奶", done: false });

  const item = collection + "/" + task.id;
  const completed = await request("PATCH", item, { done: true });
  assert.deepEqual(completed, { status: 200, body: { ...task, done: true } });
  const renamed = await request("PATCH", item, { title: "买面包" });
  assert.deepEqual(renamed, { status: 200, body: { ...task, title: "买面包", done: true } });
  assert.deepEqual((await request("GET", collection)).body, [renamed.body]);

  assert.deepEqual(await request("DELETE", item), { status: 204, body: null });
  assert.deepEqual((await request("GET", collection)).body, []);
});

test("任务 API 拒绝无效输入并区分不存在的任务", async (t) => {
  const { request } = await startTestApp(t);
  assert.equal((await request("POST", "/api/tasks", { title: " " })).status, 400);
  assert.equal((await request("PATCH", "/api/tasks/missing", {})).status, 400);
  assert.equal((await request("PATCH", "/api/tasks/missing", { done: true })).status, 404);
  assert.equal((await request("DELETE", "/api/tasks/missing")).status, 404);
});

test("更新已有 JSON 清单时保留其他任务和扩展字段", async (t) => {
  const { request, filename } = await startTestApp(t);
  const first = { id: "a", title: "买牛奶", done: false, priority: 1 };
  const second = { id: "b", title: "取快递", done: false };
  await writeFile(filename, JSON.stringify([first, second]));

  assert.equal((await request("PATCH", "/api/tasks/a", { done: true })).status, 200);
  assert.deepEqual(createJsonTaskStore(filename).list(), [{ ...first, done: true }, second]);
});

test("任务 API 可以保存优先级和工作量，部分更新保留其他字段", async (t) => {
  const { request, filename } = await startTestApp(t);
  const legacy = { id: "untouched", title: "买牛奶", done: true };
  await writeFile(filename, JSON.stringify([legacy]));
  const collection = "/api/tasks";
  const created = await request("POST", collection, {
    title: " 写报告 ",
    priority: "high",
    workload: "medium",
  });
  assert.equal(created.status, 201);
  assert.deepEqual(created.body, {
    id: created.body.id,
    title: "写报告",
    done: false,
    priority: "high",
    workload: "medium",
  });
  const item = collection + "/" + created.body.id;
  assert.deepEqual(await request("PATCH", item, { workload: "large" }), {
    status: 200,
    body: { ...created.body, workload: "large" },
  });
  const expected = [legacy, { ...created.body, workload: "large" }];
  assert.deepEqual((await request("GET", collection)).body, expected);
  assert.deepEqual(createJsonTaskStore(filename).list(), expected);
  for (const invalid of [{ priority: null }, { workload: "huge" }, { priority: 1 }]) {
    assert.equal((await request("PATCH", item, invalid)).status, 400);
    assert.equal((await request("POST", collection, { title: "坏输入", ...invalid })).status, 400);
  }
  assert.deepEqual((await request("GET", collection)).body, expected);
});

test("聊天 API 只将有效的完整对话传给智能体，并返回其回复", async (t) => {
  const { request, conversations } = await startTestApp(t);
  const messages = [
    { role: "user", content: "你好" },
    { role: "assistant", content: "你好！" },
    { role: "user", content: "今天有哪些待办？" },
  ];
  const valid = await request("POST", "/api/chat", { messages });
  assert.deepEqual(valid, { status: 200, body: { reply: "收到" } });

  const invalid = await request("POST", "/api/chat", { messages: messages.slice(0, 2) });
  assert.equal(invalid.status, 400);
  assert.deepEqual(conversations, [messages]);
});

test("HTTP 路由严格校验异常 JSON、超大负载、未知接口、Unicode 与原型键", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-http-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  const app = createApp({
    tasks: createJsonTaskStore(filename),
    async reply() {
      return "收到";
    },
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;
  const baseUrl = `http://127.0.0.1:${port}`;

  // Malformed JSON -> 400
  const malformed = await fetch(`${baseUrl}/api/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: '{"title":',
  });
  assert.equal(malformed.status, 400);

  // >1MB payload -> 413
  const oversized = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "a".repeat(1_100_000) }] }),
  });
  assert.equal(oversized.status, 413);

  // Unknown /api/* -> 404
  const notFound = await fetch(`${baseUrl}/api/nonexistent`);
  assert.equal(notFound.status, 404);

  // Invalid /api/chat payloads: role:"system", empty array, whitespace-only content -> 400
  for (const payload of [
    { messages: [] },
    { messages: [{ role: "system", content: "hi" }] },
    { messages: [{ role: "user", content: "   " }] },
  ]) {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    assert.equal(res.status, 400);
  }

  // Invalid task payloads: 201 chars, invisible zero-width-only, extra unknown key, raw __proto__ -> 400
  for (const rawBody of [
    JSON.stringify({ title: "长".repeat(201) }),
    JSON.stringify({ title: "\u200B\u200C\uFEFF" }),
    JSON.stringify({ title: "买牛奶", extra: true }),
    '{"title":"买牛奶","__proto__":{"polluted":true}}',
  ]) {
    const res = await fetch(`${baseUrl}/api/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: rawBody,
    });
    assert.equal(res.status, 400, rawBody);
  }
  assert.equal("polluted" in Object.prototype, false);

  // CANON-13: Bidi RLO stripped on POST/PATCH while valid ZWJ emoji is preserved intact.
  const created = await fetch(`${baseUrl}/api/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "👨‍👩‍👧‍👦 \u202E买牛奶\u2066" }),
  });
  assert.equal(created.status, 201);
  const createdTask = await created.json();
  assert.equal(createdTask.title, "👨‍👩‍👧‍👦 买牛奶");

  const invalidPatch = await fetch(`${baseUrl}/api/tasks/${createdTask.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "\u200B\u200C\uFEFF" }),
  });
  assert.equal(invalidPatch.status, 400);

  // Unhandled storage error -> 500 with sanitized error logging (no stack/message leak in body).
  await writeFile(filename, "{broken-json");
  const stderrMock = t.mock.method(console, "error", () => {});
  const serverErr = await fetch(`${baseUrl}/api/tasks`);
  assert.equal(serverErr.status, 500);
  const errBody = await serverErr.json();
  assert.equal(errBody.error, "服务暂时无法处理请求，请稍后重试。");
  assert.equal(stderrMock.mock.callCount(), 1);
});

test("静态资源 /assets/chat.js 在以点开头的工作目录下仍返回 200", async (t) => {
  const originalCwd = process.cwd();
  const baseDir = await mkdtemp(join(tmpdir(), "todo-agent-dot-"));
  const dotWorkDir = join(baseDir, ".dot-workspace");
  await mkdir(join(dotWorkDir, "dist"), { recursive: true });
  await writeFile(join(dotWorkDir, "dist", "chat.js"), 'console.log("ok");\n', "utf8");
  t.after(async () => {
    process.chdir(originalCwd);
    await rm(baseDir, { recursive: true, force: true });
  });

  process.chdir(dotWorkDir);
  const app = createApp({
    tasks: createJsonTaskStore(join(baseDir, "tasks.json")),
    async reply() {
      return "收到";
    },
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;

  const res = await fetch(`http://127.0.0.1:${port}/assets/chat.js`);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /console\.log\("ok"\)/);
});

test("POST /api/chat 在客户端断开时触发 AbortSignal 传播", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-abort-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  let abortedPromiseResolve!: () => void;
  const abortedPromise = new Promise<void>((resolve) => {
    abortedPromiseResolve = resolve;
  });
  let startedResolve!: () => void;
  const startedPromise = new Promise<void>((resolve) => {
    startedResolve = resolve;
  });
  let observedSignal: AbortSignal | undefined;

  const app = createApp({
    tasks: createJsonTaskStore(join(directory, "tasks.json")),
    async reply(_messages, signal) {
      observedSignal = signal;
      signal?.addEventListener("abort", () => abortedPromiseResolve(), { once: true });
      startedResolve();
      await abortedPromise;
      return "cancelled";
    },
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;

  const clientAbort = new AbortController();
  const pendingFetch = fetch(`http://127.0.0.1:${port}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "慢请求" }] }),
    signal: clientAbort.signal,
  });
  await startedPromise;
  clientAbort.abort();
  await assert.rejects(pendingFetch);
  await abortedPromise;
  assert.equal(observedSignal?.aborted, true);
});

test("冷重启持久化与已删除数据目录自动恢复生效", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-restart-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const dataDir = join(directory, "data");
  const filename = join(dataDir, "tasks.json");

  // Instance #1 creates a task via chat.
  {
    const tasks = createJsonTaskStore(filename);
    const app = createTaskAgentApp(tasks, async (request) => {
      if (request.messages.at(-1)?.role === "tool") {
        return { finish_reason: "stop", message: { role: "assistant", content: "已添加。" } };
      }
      return {
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "c1",
              type: "function",
              function: {
                name: "create_task",
                arguments: '{"title":"跨重启任务","priority":"high"}',
              },
            },
          ],
        },
      };
    });
    const server = app.listen(0, "127.0.0.1");
    await once(server, "listening");
    const { port } = server.address() as AddressInfo;
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: "新增跨重启任务，优先级高" }],
        }),
      });
      assert.equal(res.status, 200);
    } finally {
      await server[Symbol.asyncDispose]();
    }
  }

  // Instance #2 boots cold on the same tasks.json and answers a 1-turn list query.
  {
    const tasks = createJsonTaskStore(filename);
    const app = createTaskAgentApp(tasks, async () => {
      throw new Error("direct list should read persisted tasks without model call");
    });
    const server = app.listen(0, "127.0.0.1");
    await once(server, "listening");
    const { port } = server.address() as AddressInfo;
    try {
      const chatRes = await fetch(`http://127.0.0.1:${port}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "我有哪些任务？" }] }),
      });
      assert.equal(chatRes.status, 200);
      const { reply } = await chatRes.json();
      assert.match(reply, /跨重启任务（待完成 · 优先级：高）/);

      // Remove data/ directory at runtime and verify POST /api/tasks recreates it cleanly.
      await rm(dataDir, { recursive: true, force: true });
      const recreate = await fetch(`http://127.0.0.1:${port}/api/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "目录重建后任务" }),
      });
      assert.equal(recreate.status, 201);
      assert.deepEqual(
        tasks.list().map((task) => task.title),
        ["目录重建后任务"],
      );
    } finally {
      await server[Symbol.asyncDispose]();
    }
  }
});

test("单实例 53 个并发 REST 操作不丢失任何更新", async (t) => {
  const { request, filename } = await startTestApp(t);
  // Phase 1: 20 parallel POST creates (20 ops).
  const created = await Promise.all(
    Array.from({ length: 20 }, (_, i) =>
      request("POST", "/api/tasks", { title: `并发任务-${i + 1}`, priority: "medium" }),
    ),
  );
  assert.equal(
    created.every((res) => res.status === 201),
    true,
  );
  const ids = created.map((res) => res.body.id as string);
  assert.equal(new Set(ids).size, 20);

  // Phase 2: 33 parallel mixed ops (10 PATCH done:true + 10 PATCH workload:large + 5 DELETE + 8 GET = 33 ops; total = 53 ops).
  const mixed = await Promise.all([
    ...ids.slice(0, 10).map((id) => request("PATCH", `/api/tasks/${id}`, { done: true })),
    ...ids.slice(10, 15).map((id) => request("PATCH", `/api/tasks/${id}`, { workload: "large" })),
    ...ids.slice(0, 5).map((id) => request("PATCH", `/api/tasks/${id}`, { priority: "high" })),
    ...ids.slice(15, 20).map((id) => request("DELETE", `/api/tasks/${id}`)),
    ...Array.from({ length: 8 }, () => request("GET", "/api/tasks")),
  ]);
  assert.equal(
    mixed.every((res) => res.status === 200 || res.status === 204),
    true,
  );

  const finalTasks = createJsonTaskStore(filename).list();
  assert.equal(finalTasks.length, 15);
  assert.equal(finalTasks.filter((task) => task.done).length, 10);
  assert.equal(finalTasks.filter((task) => task.priority === "high").length, 5);
  assert.equal(finalTasks.filter((task) => task.workload === "large").length, 5);
});
