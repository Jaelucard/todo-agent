import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { APIConnectionTimeoutError } from "openai";
import { createTaskAgentApp } from "./index.ts";
import {
  buildModelRequest,
  getModelClient,
  isModelTimeout,
  MODEL_TIMEOUT_MS,
  model,
  ModelConfigError,
} from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

test("server/model.ts throws ModelConfigError on missing key, caches OpenAI client, and builds strict payloads", (t) => {
  const originalKey = process.env.DEEPSEEK_API_KEY;
  t.after(() => {
    if (originalKey === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = originalKey;
  });

  delete process.env.DEEPSEEK_API_KEY;
  assert.throws(() => getModelClient(), ModelConfigError);

  process.env.DEEPSEEK_API_KEY = "   ";
  assert.throws(() => getModelClient(), ModelConfigError);

  // Offline client instantiation & caching (0 network calls).
  process.env.DEEPSEEK_API_KEY = "sk-test-offline-cache-key-1";
  const first = getModelClient();
  const second = getModelClient();
  assert.equal(first, second, "identical (apiKey, baseURL) should reuse cached OpenAI client");

  process.env.DEEPSEEK_API_KEY = "sk-test-offline-cache-key-2";
  const third = getModelClient();
  assert.notEqual(third, first, "changing DEEPSEEK_API_KEY should construct a fresh OpenAI client");

  const payload = buildModelRequest([{ role: "user", content: "你好" }], [], "auto");
  assert.deepEqual(payload, {
    model,
    messages: [{ role: "user", content: "你好" }],
    tools: [],
    tool_choice: "auto",
    reasoning_effort: "none",
    max_tokens: 1024,
    stream: false,
  });

  assert.equal(MODEL_TIMEOUT_MS, 20_000);
  assert.equal(isModelTimeout(new APIConnectionTimeoutError()), true);
  assert.equal(isModelTimeout(new Error("generic")), false);
});

test("unset DEEPSEEK_API_KEY returns HTTP 503 config_error with sanitized stderr while /api/tasks stays 200", async (t) => {
  const originalKey = process.env.DEEPSEEK_API_KEY;
  delete process.env.DEEPSEEK_API_KEY;
  const dir = await mkdtemp(join(tmpdir(), "todo-agent-model-503-"));
  const tasks = createJsonTaskStore(join(dir, "tasks.json"));
  const logged: string[] = [];
  t.mock.method(console, "error", (...args: unknown[]) => {
    logged.push(args.map(String).join(" "));
  });
  const app = createTaskAgentApp(tasks);
  const server = app.listen(0, "127.0.0.1");
  t.after(async () => {
    try {
      await server[Symbol.asyncDispose]();
    } finally {
      await rm(dir, { recursive: true, force: true });
      if (originalKey === undefined) delete process.env.DEEPSEEK_API_KEY;
      else process.env.DEEPSEEK_API_KEY = originalKey;
    }
  });
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;
  const baseUrl = `http://127.0.0.1:${port}`;

  const chatRes = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "帮我记一下测试无密钥故障" }] }),
  });
  assert.equal(chatRes.status, 503);
  const chatBody = await chatRes.json();
  assert.deepEqual(chatBody, {
    error: "模型服务未正确配置，请检查 .env 中的 DEEPSEEK_API_KEY 后重试。本轮未修改任务。",
    code: "config_error",
    mutated: false,
  });
  assert.equal(logged.length, 1);
  assert.equal(
    logged[0],
    "[agent] model_failure category=config_error status=none error=ModelConfigError",
  );

  // REST /api/tasks remains fully operational.
  const restRes = await fetch(`${baseUrl}/api/tasks`);
  assert.equal(restRes.status, 200);
  assert.deepEqual(await restRes.json(), []);
});

test("upstream model_error returns HTTP 503 with zero secret leakage, while partial-save returns HTTP 200", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "todo-agent-model-err-"));
  const tasks = createJsonTaskStore(join(dir, "tasks.json"));
  const logged: string[] = [];
  t.mock.method(console, "error", (...args: unknown[]) => {
    logged.push(args.map(String).join(" "));
  });

  let mode: "zero_mutation_error" | "partial_save_error" = "zero_mutation_error";
  let round = 0;
  const app = createTaskAgentApp(tasks, async () => {
    round++;
    if (mode === "partial_save_error" && round === 1) {
      return {
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "c1",
              type: "function",
              function: { name: "create_task", arguments: '{"title":"买牛奶"}' },
            },
          ],
        },
      };
    }
    const err = Object.assign(new Error("Upstream 429 Bearer sk-topsecret1234567890"), {
      name: "RateLimitError",
      status: 429,
    });
    throw err;
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(async () => {
    try {
      await server[Symbol.asyncDispose]();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;
  const baseUrl = `http://127.0.0.1:${port}`;

  // Case 1: Zero mutations -> HTTP 503 { error, code: "model_error", mutated: false }
  const failed = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "帮我记一下买牛奶" }] }),
  });
  assert.equal(failed.status, 503);
  const failedBody = await failed.json();
  assert.deepEqual(failedBody, {
    error: "模型服务暂时无法处理请求，请稍后重试。本轮未修改任务。",
    code: "model_error",
    mutated: false,
  });
  assert.equal(logged.length, 1);
  assert.equal(
    logged[0],
    "[agent] model_failure category=model_error status=429 error=RateLimitError",
  );
  assert.doesNotMatch(logged[0], /sk-topsecret|Bearer/);

  // Case 2: Partial save (outcomes.length > 0 before upstream failure) -> HTTP 200 reporting saved mutation
  mode = "partial_save_error";
  round = 0;
  const partial = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "帮我记一下买牛奶" }] }),
  });
  assert.equal(partial.status, 200);
  const partialBody = await partial.json();
  assert.match(partialBody.reply, /已添加待办「买牛奶」。/);
  assert.match(partialBody.reply, /以上变更已保存，其余操作已停止。/);
  assert.doesNotMatch(partialBody.reply, /sk-topsecret/);
  assert.equal(tasks.list().length, 1);
});

test("client disconnect aborts before write (0 mutations) and preserves committed writes if aborted after write", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "todo-agent-abort-e2e-"));
  const tasks = createJsonTaskStore(join(dir, "tasks.json"));
  t.mock.method(console, "error", () => {});

  // Sub-scenario A: Abort before any write executes.
  let waitStartedResolve!: () => void;
  const waitStarted = new Promise<void>((resolve) => {
    waitStartedResolve = resolve;
  });
  let providerAbortedResolve!: () => void;
  const providerAborted = new Promise<void>((resolve) => {
    providerAbortedResolve = resolve;
  });
  const appBefore = createTaskAgentApp(tasks, async (_request, signal) => {
    signal.addEventListener("abort", () => providerAbortedResolve(), { once: true });
    waitStartedResolve();
    return new Promise(() => {});
  });
  const serverA = appBefore.listen(0, "127.0.0.1");
  t.after(() => serverA[Symbol.asyncDispose]());
  await once(serverA, "listening");
  const portA = (serverA.address() as AddressInfo).port;

  const abortA = new AbortController();
  const reqA = fetch(`http://127.0.0.1:${portA}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "帮我记一下未写入任务" }] }),
    signal: abortA.signal,
  });
  await waitStarted;
  abortA.abort();
  await assert.rejects(reqA);
  await providerAborted;
  assert.deepEqual(tasks.list(), []);

  // Sub-scenario B: Abort AFTER round-1 create_task commits -> disk keeps the saved task.
  let round2StartedResolve!: () => void;
  const round2Started = new Promise<void>((resolve) => {
    round2StartedResolve = resolve;
  });
  let round2AbortedResolve!: () => void;
  const round2Aborted = new Promise<void>((resolve) => {
    round2AbortedResolve = resolve;
  });
  let callCount = 0;
  const appAfter = createTaskAgentApp(tasks, async (_request, signal) => {
    if (++callCount === 1) {
      return {
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "c-after",
              type: "function",
              function: { name: "create_task", arguments: '{"title":"已提交任务"}' },
            },
          ],
        },
      };
    }
    signal.addEventListener("abort", () => round2AbortedResolve(), { once: true });
    round2StartedResolve();
    return new Promise(() => {});
  });
  const serverB = appAfter.listen(0, "127.0.0.1");
  t.after(async () => {
    try {
      await serverB[Symbol.asyncDispose]();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
  await once(serverB, "listening");
  const portB = (serverB.address() as AddressInfo).port;

  const abortB = new AbortController();
  const reqB = fetch(`http://127.0.0.1:${portB}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "帮我记一下已提交任务" }] }),
    signal: abortB.signal,
  });
  await round2Started;
  abortB.abort();
  await assert.rejects(reqB);
  await round2Aborted;

  const checkRest = await fetch(`http://127.0.0.1:${portB}/api/tasks`);
  assert.equal(checkRest.status, 200);
  const saved = await checkRest.json();
  assert.equal(saved.length, 1);
  assert.equal(saved[0].title, "已提交任务");
});
