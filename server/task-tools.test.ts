import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
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

test("metadata creation and edits preserve task identity and other stored fields", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const legacySummary = { id: "legacy", title: "取快递", done: true };
  const legacy = { ...legacySummary, note: "旧记录" };
  await writeFile(filename, JSON.stringify([legacy]));
  const tools = createTaskTools(tasks);
  const created = tools.execute(
    "create_task",
    '{"title":"写报告","priority":"high","workload":"large"}',
  );
  assert.ok(created.ok && created.tool === "create_task");
  const report = {
    id: created.task.id,
    title: "写报告",
    done: false,
    priority: "high",
    workload: "large",
  };
  assert.deepEqual(created.task, report);
  assert.deepEqual(tools.execute("list_tasks", "{}"), {
    ok: true,
    tool: "list_tasks",
    tasks: [legacySummary, report],
  });
  const change = JSON.stringify({ candidateIds: [legacy.id], changes: { priority: "low" } });
  assert.deepEqual(tools.execute("update_task", change), {
    ok: true,
    tool: "update_task",
    changed: true,
    task: { ...legacySummary, priority: "low" },
  });
  assert.deepEqual(createJsonTaskStore(filename).list(), [{ ...legacy, priority: "low" }, report]);

  // CANON-03: verified in-turn writes refresh expectedSnapshot while preserving observedIds,
  // so repeating an identical update on an already-observed ID returns a no-op with zero disk writes.
  const contents = await readFile(filename, "utf8");
  const identity = await stat(filename, { bigint: true });
  assert.deepEqual(tools.execute("update_task", change), {
    ok: true,
    tool: "update_task",
    changed: false,
    task: { ...legacySummary, priority: "low" },
  });
  assert.equal(await readFile(filename, "utf8"), contents);
  const current = await stat(filename, { bigint: true });
  assert.deepEqual([current.ino, current.mtimeNs], [identity.ino, identity.mtimeNs]);
});

test("metadata edits reject invalid changes, ambiguity and a metadata-only stale list", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const first = { id: "first", title: "买牛奶", done: false, priority: "low" };
  const second = { ...first, id: "second" };
  await writeFile(filename, JSON.stringify([first, second]));
  const tools = createTaskTools(tasks);
  assert.ok(tools.execute("list_tasks", "{}").ok);
  const before = await readFile(filename, "utf8");
  // CANON-11 / CANON-13: update_task now accepts valid title/done fields, while empty/invisible/wrong-type patches remain rejected.
  for (const changes of [
    {},
    { title: "" },
    { title: "\u200B\u200C\uFEFF" },
    { done: "yes" },
    { priority: "urgent" },
    { workload: 3 },
  ]) {
    assert.deepEqual(
      tools.execute("update_task", JSON.stringify({ candidateIds: [first.id], changes })),
      {
        ok: false,
        error: "invalid_arguments",
      },
    );
  }
  const edit = JSON.stringify({ candidateIds: [first.id], changes: { workload: "medium" } });
  assert.deepEqual(tools.execute("update_task", edit), { ok: false, error: "ambiguous_reference" });
  assert.equal(await readFile(filename, "utf8"), before);

  const concurrent = JSON.stringify([{ ...first, priority: "high" }, second]);
  await writeFile(filename, concurrent);
  assert.deepEqual(tools.execute("update_task", edit), { ok: false, error: "stale_view" });
  assert.equal(await readFile(filename, "utf8"), concurrent);
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
    // CANON-06: list_tasks now accepts valid optional { done?, priority?, workload? } filters, rejecting unknown/bad filter keys.
    ["list_tasks", '{"done":"no"}', "invalid_arguments"],
    ["list_tasks", '{"priority":"urgent"}', "invalid_arguments"],
    ["list_tasks", '{"unknown":true}', "invalid_arguments"],
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
  assert.deepEqual(await (await fetch(endpoint)).json(), [
    { ...created.task, title: "买面包", done: true },
  ]);
  // CANON-03: an observed ID remains targetable after verified internal mutations,
  // and removing it removes it from observedIds so re-deleting returns task_not_found (not stale_view).
  assert.deepEqual(tools.execute("delete_task", selector), {
    ok: true,
    tool: "delete_task",
    task: { ...created.task, title: "买面包", done: true },
  });
  assert.deepEqual(tools.execute("delete_task", selector), { ok: false, error: "task_not_found" });
  assert.deepEqual(await (await fetch(endpoint)).json(), []);
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

test("in-turn compound mutations preserve observedIds, isolate newly created tasks, and catch external edits", async (t) => {
  const { tasks } = await temporaryStore(t);
  const exam = tasks.create("预约体检");
  const report = tasks.create("写周报");
  const tools = createTaskTools(tasks);

  assert.ok(tools.execute("list_tasks", "{}").ok);
  // Newly created task is NOT automatically added to observedIds.
  const created = tools.execute("create_task", '{"title":"买鸡蛋"}');
  assert.ok(created.ok && created.tool === "create_task");
  assert.deepEqual(
    tools.execute("complete_task", JSON.stringify({ candidateIds: [created.task.id] })),
    { ok: false, error: "list_required" },
  );

  // Already-observed tasks can still be completed and deleted in sequence after internal writes.
  assert.deepEqual(tools.execute("complete_task", JSON.stringify({ candidateIds: [exam.id] })), {
    ok: true,
    tool: "complete_task",
    task: { ...exam, done: true },
    changed: true,
  });
  assert.deepEqual(tools.execute("delete_task", JSON.stringify({ candidateIds: [report.id] })), {
    ok: true,
    tool: "delete_task",
    task: report,
  });
  assert.deepEqual(tools.execute("delete_task", JSON.stringify({ candidateIds: [report.id] })), {
    ok: false,
    error: "task_not_found",
  });

  // External out-of-band edit immediately triggers stale_view.
  tasks.update(exam.id, { priority: "high" });
  assert.deepEqual(tools.execute("delete_task", JSON.stringify({ candidateIds: [exam.id] })), {
    ok: false,
    error: "stale_view",
  });
});

test("duplicate pending+done tasks allow unambiguous complete_task ONLY while blocking delete and update", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const doneMilk = tasks.update(tasks.create("买牛奶").id, { done: true })!;
  const pendingMilk = tasks.create("买牛奶");
  const tools = createTaskTools(tasks);
  assert.ok(tools.execute("list_tasks", "{}").ok);
  const before = await readFile(filename, "utf8");

  // Delete, reopen, rename, and metadata update MUST still require disambiguation.
  assert.deepEqual(
    tools.execute("delete_task", JSON.stringify({ candidateIds: [pendingMilk.id] })),
    { ok: false, error: "ambiguous_reference" },
  );
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: [doneMilk.id], changes: { done: false } }),
    ),
    { ok: false, error: "ambiguous_reference" },
  );
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: [pendingMilk.id], changes: { title: "买鲜牛奶" } }),
    ),
    { ok: false, error: "ambiguous_reference" },
  );
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: [pendingMilk.id], changes: { priority: "high" } }),
    ),
    { ok: false, error: "ambiguous_reference" },
  );
  assert.equal(await readFile(filename, "utf8"), before);

  // complete_task uniquely completes the sole pending copy (even when both duplicate IDs are submitted).
  assert.deepEqual(
    tools.execute("complete_task", JSON.stringify({ candidateIds: [doneMilk.id, pendingMilk.id] })),
    {
      ok: true,
      tool: "complete_task",
      task: { ...pendingMilk, done: true },
      changed: true,
    },
  );
});

test("list_tasks supports done, priority and workload filtering", async (t) => {
  const { tasks } = await temporaryStore(t);
  const a = tasks.create("架构评审", { priority: "high", workload: "large" });
  const b = tasks.update(tasks.create("买牛奶", { priority: "low", workload: "small" }).id, {
    done: true,
  })!;
  const c = tasks.create("取快递", { priority: "high", workload: "small" });
  const tools = createTaskTools(tasks);

  assert.deepEqual(tools.execute("list_tasks", '{"done":false}'), {
    ok: true,
    tool: "list_tasks",
    tasks: [a, c],
  });
  assert.deepEqual(tools.execute("list_tasks", '{"done":true}'), {
    ok: true,
    tool: "list_tasks",
    tasks: [b],
  });
  assert.deepEqual(tools.execute("list_tasks", '{"priority":"high","workload":"large"}'), {
    ok: true,
    tool: "list_tasks",
    tasks: [a],
  });
  assert.deepEqual(tools.execute("list_tasks", '{"workload":"small"}'), {
    ok: true,
    tool: "list_tasks",
    tasks: [b, c],
  });
});

test("update_task supports reopen and rename while enforcing field authorization and Unicode hardening", async (t) => {
  const { tasks, filename } = await temporaryStore(t);
  const legacy = {
    id: "contract",
    title: "寄合同",
    done: true,
    priority: 1,
    extraNote: "保留字段",
  };
  await writeFile(filename, JSON.stringify([legacy]));
  const tools = createTaskTools(tasks);
  assert.ok(tools.execute("list_tasks", "{}").ok);

  // Field-smuggling guard: workload-only grant rejects title or done changes.
  const workloadGrant = {
    allowedPriorities: new Set<never>(),
    allowedWorkloads: new Set(["large"] as const),
    allowTitleChange: false,
    allowDoneValues: new Set<boolean>(),
  };
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: ["contract"], changes: { workload: "large", title: "偷改" } }),
      workloadGrant,
    ),
    { ok: false, error: "metadata_confirmation_required" },
  );
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: ["contract"], changes: { workload: "large", done: false } }),
      workloadGrant,
    ),
    { ok: false, error: "metadata_confirmation_required" },
  );

  // Reopen (done: true -> false) preserves legacy fields and ID.
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({ candidateIds: ["contract"], changes: { done: false } }),
    ),
    {
      ok: true,
      tool: "update_task",
      changed: true,
      task: { id: "contract", title: "寄合同", done: false },
    },
  );
  // Rename strips Bidi controls and preserves legitimate ZWJ emoji + legacy fields.
  assert.deepEqual(
    tools.execute(
      "update_task",
      JSON.stringify({
        candidateIds: ["contract"],
        changes: { title: "👨‍👩‍👧‍👦 \u202E寄正式合同" },
      }),
    ),
    {
      ok: true,
      tool: "update_task",
      changed: true,
      task: { id: "contract", title: "👨‍👩‍👧‍👦 寄正式合同", done: false },
    },
  );
  assert.deepEqual(createJsonTaskStore(filename).list(), [
    {
      id: "contract",
      title: "👨‍👩‍👧‍👦 寄正式合同",
      done: false,
      priority: 1,
      extraNote: "保留字段",
    },
  ]);
  // Invisible-only create_task rejected.
  assert.deepEqual(tools.execute("create_task", JSON.stringify({ title: "\u200B\u200C\uFEFF" })), {
    ok: false,
    error: "invalid_arguments",
  });
});

test("MEDIUM-1: bound-reference writes do not heal a stale expectedSnapshot for subsequent semantic writes", async (t) => {
  for (const boundOp of ["complete_task", "update_task", "delete_task"] as const) {
    const { tasks, filename } = await temporaryStore(t);
    const boundA = tasks.create("买牛奶");
    const otherB = tasks.create("取快递");
    const tools = createTaskTools(tasks, { id: boundA.id, title: boundA.title });

    assert.ok(tools.execute("list_tasks", "{}").ok);
    // Out-of-band external REST edit mutates unrelated task B, making expectedSnapshot stale.
    tasks.update(otherB.id, { priority: "high" });

    // Bound write on boundA still succeeds because boundA itself was not modified externally.
    const boundArgs =
      boundOp === "update_task"
        ? JSON.stringify({ useBoundReference: true, changes: { workload: "small" } })
        : '{"useBoundReference":true}';
    const boundResult = tools.execute(boundOp, boundArgs);
    assert.equal(boundResult.ok, true, boundOp);

    // Crucially, the bound write must NOT heal expectedSnapshot for a subsequent semantic write on B.
    const beforeSemantic = await readFile(filename, "utf8");
    const semanticResult = tools.execute(
      "delete_task",
      JSON.stringify({ candidateIds: [otherB.id] }),
    );
    assert.deepEqual(semanticResult, { ok: false, error: "stale_view" }, boundOp);
    assert.equal(await readFile(filename, "utf8"), beforeSemantic, boundOp);
  }
});
