import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createApp } from "./app.ts";
import { createJsonTaskStore } from "./task-store.ts";
import { createTaskTools } from "./task-tools.ts";

async function temporaryStore(t: TestContext) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-tools-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  return { filename, tasks: createJsonTaskStore(filename) };
}

test("create_task saves a pending task with a trimmed title", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const tools = createTaskTools(tasks);
  assert.deepEqual(tasks.list(), []);

  const created = tools.execute("create_task", '{"title":" 买牛奶 "}');
  assert.ok(created.ok && created.tool === "create_task");
  const task = { id: created.task.id, title: "买牛奶", done: false };

  assert.deepEqual(created.task, task);
  assert.deepEqual(createJsonTaskStore(filename).list(), [task]);
});

test("list, complete, and delete use the current saved tasks", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const milk = tasks.create("买牛奶");
  const parcel = tasks.create("取快递");
  const tools = createTaskTools(tasks);

  assert.deepEqual(tools.execute("list_tasks", "{}"), {
    ok: true,
    tool: "list_tasks",
    tasks: [milk, parcel],
  });
  assert.deepEqual(createJsonTaskStore(filename).list(), [milk, parcel]);

  const completedMilk = { ...milk, done: true };
  assert.deepEqual(tools.execute("complete_task", JSON.stringify({ candidateIds: [milk.id] })), {
    ok: true,
    tool: "complete_task",
    task: completedMilk,
    changed: true,
  });
  assert.deepEqual(createJsonTaskStore(filename).list(), [completedMilk, parcel]);

  assert.deepEqual(tools.execute("list_tasks", "{}"), {
    ok: true,
    tool: "list_tasks",
    tasks: [completedMilk, parcel],
  });
  assert.deepEqual(tools.execute("delete_task", JSON.stringify({ candidateIds: [parcel.id] })), {
    ok: true,
    tool: "delete_task",
    task: parcel,
  });
  assert.deepEqual(createJsonTaskStore(filename).list(), [completedMilk]);
});

test("rejects invalid calls, missing IDs, and ambiguous candidates without changing storage", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const milk = tasks.create("买牛奶");
  const parcel = tasks.create("取快递");
  const tools = createTaskTools(tasks);
  const before = await readFile(filename, "utf8");
  const selectMilk = JSON.stringify({ candidateIds: [milk.id] });

  assert.deepEqual(tools.execute("complete_task", selectMilk), {
    ok: false,
    error: "list_required",
  });
  assert.ok(tools.execute("list_tasks", "{}").ok);
  const rejections = [
    ["constructor", "{}", "unknown_tool"],
    ["create_task", "{", "invalid_arguments"],
    ["create_task", '{"title":"  "}', "invalid_arguments"],
    ["create_task", JSON.stringify({ title: "牛".repeat(201) }), "invalid_arguments"],
    ["create_task", '{"title":"买牛奶","done":true}', "invalid_arguments"],
    ["list_tasks", '{"done":false}', "invalid_arguments"],
    ["complete_task", '{"candidateIds":[7]}', "invalid_arguments"],
    ["delete_task", JSON.stringify({ id: milk.id }), "invalid_arguments"],
    ["delete_task", '{"candidateIds":["missing"],"useBoundReference":true}', "invalid_arguments"],
    ["complete_task", '{"candidateIds":[]}', "task_not_found"],
    [
      "complete_task",
      JSON.stringify({ candidateIds: [milk.id, parcel.id] }),
      "ambiguous_reference",
    ],
    ["complete_task", '{"candidateIds":["missing"]}', "task_not_found"],
    ["delete_task", '{"candidateIds":["missing"]}', "task_not_found"],
  ];
  for (const [name, args, error] of rejections) {
    assert.deepEqual(tools.execute(name, args), { ok: false, error }, `${name} ${args}`);
    assert.equal(await readFile(filename, "utf8"), before);
  }
});

test("a fresh lookup with no matching candidates reports a missing task without writing", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  tasks.create("取快递");
  const before = await readFile(filename, "utf8");
  const tools = createTaskTools(tasks);
  const args = '{"candidateIds":[]}';
  assert.deepEqual(tools.execute("delete_task", args), { ok: false, error: "list_required" });
  assert.ok(tools.execute("list_tasks", "{}").ok);
  assert.deepEqual(tools.execute("delete_task", args), { ok: false, error: "task_not_found" });
  assert.equal(await readFile(filename, "utf8"), before);
});

test("REST sees a tool-created task and a REST update invalidates the previous tool view", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const app = createApp({
    tasks,
    async reply() {
      return "收到";
    },
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;
  const endpoint = `http://127.0.0.1:${port}/api/tasks`;
  const tools = createTaskTools(tasks);

  const created = tools.execute("create_task", '{"title":"买牛奶"}');
  assert.ok(created.ok && created.tool === "create_task");
  const listed = await fetch(endpoint);
  assert.equal(listed.status, 200);
  assert.deepEqual(await listed.json(), [created.task]);
  const toolView = tools.execute("list_tasks", "{}");
  assert.ok(toolView.ok && toolView.tool === "list_tasks");
  assert.deepEqual(toolView.tasks, [created.task]);

  const renamed = await fetch(`${endpoint}/${created.task.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: '{"title":"买面包"}',
  });
  assert.equal(renamed.status, 200);
  assert.deepEqual(await renamed.json(), { ...created.task, title: "买面包" });
  // A caller editing its returned data cannot change what the toolkit originally observed.
  toolView.tasks[0].title = "买面包";
  const before = await readFile(filename, "utf8");
  const selector = JSON.stringify({ candidateIds: [created.task.id] });
  assert.deepEqual(tools.execute("complete_task", selector), { ok: false, error: "stale_view" });
  assert.equal(await readFile(filename, "utf8"), before);

  assert.ok(tools.execute("list_tasks", "{}").ok);
  assert.ok(tools.execute("complete_task", selector).ok);
  assert.deepEqual(tools.execute("delete_task", selector), { ok: false, error: "list_required" });
  assert.deepEqual(await (await fetch(endpoint)).json(), [
    { ...created.task, title: "买面包", done: true },
  ]);
});

test("a repeated normalized create is rejected within one turn and allowed across turns", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const tools = createTaskTools(tasks);
  assert.ok(tools.execute("create_task", '{"title":"买牛奶"}').ok);
  const before = await readFile(filename, "utf8");
  assert.deepEqual(tools.execute("create_task", '{"title":" 买牛奶 "}'), {
    ok: false,
    error: "duplicate_create",
  });
  assert.equal(await readFile(filename, "utf8"), before);
  assert.ok(tools.execute("create_task", '{"title":"取快递"}').ok);
  assert.ok(createTaskTools(tasks).execute("create_task", '{"title":"买牛奶"}').ok);
  assert.deepEqual(
    tasks.list().map((task) => task.title),
    ["买牛奶", "取快递", "买牛奶"],
  );
});

test("a trusted binding can disambiguate duplicate titles and survives unrelated changes", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const unrelated = tasks.create("取快递");
  const duplicate = tasks.create("买牛奶");
  const target = tasks.create("买牛奶");
  const tools = createTaskTools(tasks, { id: target.id, title: target.title });

  assert.ok(tools.execute("list_tasks", "{}").ok);
  assert.deepEqual(tools.execute("delete_task", JSON.stringify({ candidateIds: [target.id] })), {
    ok: false,
    error: "ambiguous_reference",
  });
  assert.equal(tasks.remove(unrelated.id), true);
  const before = await readFile(filename, "utf8");
  const bound = '{"useBoundReference":true}';
  assert.deepEqual(createTaskTools(tasks).execute("delete_task", bound), {
    ok: false,
    error: "reference_required",
  });
  assert.equal(await readFile(filename, "utf8"), before);
  assert.deepEqual(tools.execute("delete_task", bound), {
    ok: true,
    tool: "delete_task",
    task: target,
  });
  assert.deepEqual(tasks.list(), [duplicate]);
  assert.deepEqual(tools.execute("delete_task", bound), { ok: false, error: "task_not_found" });

  const oldBinding = createTaskTools(tasks, duplicate);
  tasks.update(duplicate.id, { title: "买面包" });
  const renamed = await readFile(filename, "utf8");
  assert.deepEqual(oldBinding.execute("complete_task", bound), { ok: false, error: "stale_view" });
  assert.equal(await readFile(filename, "utf8"), renamed);
});

test("tool results omit legacy fields and already completed tasks do not write again", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const task = { id: "legacy-id", title: "买牛奶", done: true };
  const original = JSON.stringify([{ ...task, priority: 2 }]);
  await writeFile(filename, original);
  const tools = createTaskTools(tasks);
  assert.deepEqual(tools.execute("list_tasks", "{}"), {
    ok: true,
    tool: "list_tasks",
    tasks: [task],
  });
  assert.deepEqual(tools.execute("complete_task", '{"candidateIds":["legacy-id"]}'), {
    ok: true,
    tool: "complete_task",
    task,
    changed: false,
  });
  assert.equal(await readFile(filename, "utf8"), original);
});

test("a duplicate stored ID cannot delete more than one saved record", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const task = { id: "legacy-id", title: "买牛奶", done: false };
  const original = JSON.stringify([task, { ...task, title: "取快递" }]);
  await writeFile(filename, original);
  const tools = createTaskTools(tasks, task);
  assert.ok(tools.execute("list_tasks", "{}").ok);
  for (const selector of ['{"candidateIds":["legacy-id"]}', '{"useBoundReference":true}']) {
    assert.deepEqual(tools.execute("delete_task", selector), {
      ok: false,
      error: "ambiguous_reference",
    });
    assert.equal(await readFile(filename, "utf8"), original);
  }
});

test("a failed file write returns a storage error and retains the last saved tasks", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  tasks.create("买牛奶");
  const before = await readFile(filename, "utf8");
  await mkdir(filename + ".tmp");
  const tools = createTaskTools(tasks);
  assert.deepEqual(tools.execute("create_task", '{"title":"取快递"}'), {
    ok: false,
    error: "storage_error",
  });
  assert.equal(await readFile(filename, "utf8"), before);
});
