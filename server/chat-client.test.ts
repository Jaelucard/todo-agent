import assert from "node:assert/strict";
import { test } from "node:test";
import * as client from "../src/chat-client.ts";

const messages = [{ role: "user", content: "把买牛奶删掉" }] as const;

test("chat distinguishes confirmed validation rejection from an unknown outcome", async (t) => {
  const cases = [
    { response: Response.json({ error: "请提供对话记录。" }, { status: 400 }), rejected: true },
    { response: Response.json({ error: "对话太长。" }, { status: 413 }), rejected: true },
    {
      response: Response.json(
        {
          error: "模型服务未正确配置，请检查 .env 中的 DEEPSEEK_API_KEY 后重试。本轮未修改任务。",
          code: "config_error",
          mutated: false,
        },
        { status: 503 },
      ),
      rejected: true,
    },
    {
      response: Response.json(
        {
          error: "模型服务暂时无法处理请求，请稍后重试。本轮未修改任务。",
          code: "model_error",
          mutated: false,
        },
        { status: 503 },
      ),
      rejected: true,
    },
    { response: Response.json({ error: "代理网关 503。" }, { status: 503 }), rejected: false },
    {
      response: Response.json({ error: "部分变更后失败。", mutated: true }, { status: 503 }),
      rejected: false,
    },
    { response: Response.json({ error: "服务暂时不可用。" }, { status: 500 }), rejected: false },
    { response: Response.json({ error: "代理错误。" }, { status: 502 }), rejected: false },
    { response: new Response("not JSON", { status: 400 }), rejected: false },
    { response: Response.json({ reply: "" }), rejected: false },
    { response: new Error("connection closed after request"), rejected: false },
    { response: new DOMException("chat request timed out", "TimeoutError"), rejected: false },
  ];
  let next = 0;
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    const result = cases[next++].response;
    if (result instanceof Error) throw result;
    return result;
  });

  for (const { rejected } of cases) {
    await assert.rejects(client.requestReply([...messages]), (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(error.name === "ChatRequestRejected", rejected);
      return true;
    });
  }
  assert.equal(fetch.mock.callCount(), cases.length);
  for (const request of fetch.mock.calls) {
    assert.equal(request.arguments[0], "/api/chat");
    assert.equal(request.arguments[1]?.method, "POST");
    assert.ok(request.arguments[1]?.signal instanceof AbortSignal);
  }
});

test("chat propagates caller AbortSignal and treats user cancellation as unknown outcome", async (t) => {
  const controller = new AbortController();
  let observedSignal: AbortSignal | undefined;
  t.mock.method(globalThis, "fetch", async (_url: unknown, init?: RequestInit) => {
    observedSignal = init?.signal ?? undefined;
    controller.abort();
    assert.equal(observedSignal?.aborted, true);
    throw new DOMException("The operation was aborted.", "AbortError");
  });

  await assert.rejects(client.requestReply([...messages], controller.signal), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.notEqual(error.name, "ChatRequestRejected");
    return true;
  });
  assert.ok(observedSignal instanceof AbortSignal);
  assert.equal(observedSignal.aborted, true);
});

test("the recovery check reads current task statuses without sending another chat request", async (t) => {
  const tasks = [
    { id: "done", title: "<b>买牛奶</b>", done: true },
    { id: "pending", title: "取快递", done: false },
  ];
  const responses = [
    Response.json(tasks),
    Response.json([]),
    Response.json([{ id: "invalid", title: "取快递", done: "yes" }]),
    Response.json({ error: "暂不可用。" }, { status: 503 }),
  ];
  let next = 0;
  const fetch = t.mock.method(globalThis, "fetch", async () => responses[next++]);

  assert.deepEqual(await client.requestTaskList(), tasks);
  assert.deepEqual(await client.requestTaskList(), []);
  await assert.rejects(client.requestTaskList());
  await assert.rejects(client.requestTaskList());
  assert.equal(fetch.mock.callCount(), responses.length);
  for (const request of fetch.mock.calls) {
    const [url, init] = request.arguments;
    assert.equal(url, "/api/tasks");
    assert.equal(init?.method, "GET");
    assert.equal(init?.cache, "no-store");
    assert.ok(init?.signal instanceof AbortSignal);
  }
});

test("dashboard reads show supported metadata and tolerate legacy extension values", async (t) => {
  const report = { id: "report", title: "写报告", done: false };
  const milk = { id: "milk", title: "买牛奶", done: true };
  t.mock.method(globalThis, "fetch", async () =>
    Response.json([
      { ...report, priority: "high", workload: "large" },
      { ...milk, priority: 1, workload: "small", legacyNote: "保留" },
    ]),
  );
  assert.deepEqual(await client.requestTaskList(), [
    { ...report, priority: "high", workload: "large" },
    { ...milk, workload: "small" },
  ]);
});
