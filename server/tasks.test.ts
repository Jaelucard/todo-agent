import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { once } from "node:events";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import express from "express";
import { taskRouter } from "./tasks.ts";

test("任务 API 支持创建、查询、更新和删除，并保留未修改的字段", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "todo-agent-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const app = express();
  app.use(express.json());
  app.use("/api/tasks", taskRouter(join(dir, "tasks.json")));
  const server = app.listen(0, "127.0.0.1");
  t.after(async () => {
    await new Promise<void>((resolve, reject) =>
      server.close((err) => (err ? reject(err) : resolve())),
    );
  });
  await once(server, "listening");
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api/tasks`;
  function request(method: string, path = "", body?: object) {
    return fetch(url + path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  assert.deepEqual(await (await request("GET")).json(), []);
  assert.equal((await request("POST", "", { title: " " })).status, 400);
  const created = await request("POST", "", { title: " 买牛奶 " });
  assert.equal(created.status, 201);
  const task = await created.json();
  assert.deepEqual(task, { id: task.id, title: "买牛奶", done: false });
  const path = "/" + task.id;
  assert.equal((await request("PATCH", path, {})).status, 400);
  const updated = await request("PATCH", path, { done: true });
  assert.deepEqual(await updated.json(), { ...task, done: true });
  assert.equal((await request("PATCH", path, { title: "买面包" })).status, 200);
  assert.deepEqual(await (await request("GET")).json(), [{ ...task, title: "买面包", done: true }]);
  assert.equal((await request("DELETE", path)).status, 204);
  assert.equal((await request("DELETE", path)).status, 404);
  assert.equal((await request("PATCH", path, { done: true })).status, 404);
  assert.deepEqual(await (await request("GET")).json(), []);
});
