import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore, type TaskStore } from "./task-store.ts";

type StoredTask = ReturnType<TaskStore["list"]>[number];

async function fixture(t: TestContext, initial: StoredTask[]) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-ordered-completion-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  const store = createJsonTaskStore(filename);
  const writes: StoredTask[][] = [];
  const saved = (): StoredTask[] => JSON.parse(readFileSync(filename, "utf8"));
  const tasks: TaskStore = {
    list: store.list,
    create(title, metadata) {
      const task = store.create(title, metadata);
      writes.push(saved());
      return task;
    },
    update(id, changes) {
      const task = store.update(id, changes);
      if (task) writes.push(saved());
      return task;
    },
    remove(id) {
      const removed = store.remove(id);
      if (removed) writes.push(saved());
      return removed;
    },
  };
  return {
    tasks,
    writes,
    saved,
    async fingerprint() {
      const { ino, mtimeNs } = await stat(filename, { bigint: true });
      return [readFileSync(filename, "utf8"), ino, mtimeNs] as const;
    },
  };
}

function providerWithWrongCreate(title: string): ModelProvider {
  let attempts = 0;
  return async () =>
    attempts++ === 0
      ? {
          finish_reason: "tool_calls",
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "mistaken-create",
                type: "function",
                function: { name: "create_task", arguments: JSON.stringify({ title }) },
              },
              {
                id: "lookup",
                type: "function",
                function: { name: "list_tasks", arguments: "{}" },
              },
            ],
          },
        }
      : { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
}

const distractor = {
  id: "keep",
  title: "保留旅行计划",
  done: false,
  priority: "low",
  workload: "medium",
};

for (const scenario of [
  {
    input: "体检预约好了，然后把写周报删掉",
    complete: "预约体检",
    remove: "写周报",
    mistaken: "体检预约",
  },
  {
    input: "晚饭预订好了，然后把整理发票删掉",
    complete: "预订晚饭",
    remove: "整理发票",
    mistaken: "晚饭预订",
  },
]) {
  test(`an unambiguous ordered completion and deletion follow the request for ${scenario.complete}`, async (t) => {
    const pending = { id: "pending", title: scenario.complete, done: false };
    const removed = { id: "remove", title: scenario.remove, done: false };
    const completed = { ...pending, done: true };
    const { tasks, writes, saved } = await fixture(t, [pending, removed, distractor]);
    const result = await createAgent(tasks, providerWithWrongCreate(scenario.mistaken)).runTurn([
      { role: "user", content: scenario.input },
    ]);

    assert.deepEqual(writes, [
      [completed, removed, distractor],
      [completed, distractor],
    ]);
    assert.deepEqual(saved(), [completed, distractor]);
    assert.deepEqual(result.outcomes, [
      { operation: "complete_task", task: completed },
      { operation: "delete_task", task: removed },
    ]);
    assert.match(
      result.reply,
      new RegExp(`已完成「${pending.title}」。\\n已删除「${removed.title}」。`, "u"),
    );
    assert.doesNotMatch(result.reply, /已添加/u);
  });
}

for (const deleteFirst of [false, true]) {
  const completion = "体检预约好了";
  const deletion = "把「写周报」删掉";
  const input = deleteFirst ? `${deletion}，然后${completion}` : `${completion}，然后${deletion}`;
  test(`a quoted deletion and colloquial completion each authorize their own target (${deleteFirst ? "delete" : "complete"} first)`, async (t) => {
    const exam = { id: "exam", title: "预约体检", done: false, workload: "large" };
    const report = { id: "report", title: "写周报", done: false, priority: "high" };
    const completed = { ...exam, done: true };
    const { tasks, writes, saved } = await fixture(t, [exam, report, distractor]);
    const agent = createAgent(tasks, async () => assert.fail("unexpected model request"));
    const result = await agent.runTurn([{ role: "user", content: input }]);

    assert.deepEqual(writes, [
      deleteFirst ? [exam, distractor] : [completed, report, distractor],
      [completed, distractor],
    ]);
    assert.deepEqual(saved(), [completed, distractor]);
    const outcomes = [
      { operation: "complete_task", task: completed },
      { operation: "delete_task", task: report },
    ];
    const replies = ["已完成「预约体检」。", "已删除「写周报」。"];
    assert.deepEqual(result.outcomes, deleteFirst ? outcomes.reverse() : outcomes);
    assert.equal(result.reply, (deleteFirst ? replies.reverse() : replies).join("\n"));
  });
}

test("an ambiguous later completion blocks an earlier quoted deletion", async (t) => {
  const first = { id: "first", title: "预约体检", done: false, priority: "high" };
  const second = { id: "second", title: "预约体检", done: false, priority: "low" };
  const report = { id: "report", title: "写周报", done: false };
  const initial = [first, second, report, distractor];
  const { tasks, writes, saved, fingerprint } = await fixture(t, initial);
  const before = await fingerprint();
  const result = await createAgent(tasks, providerWithWrongCreate("体检预约")).runTurn([
    { role: "user", content: "把「写周报」删掉，然后体检预约好了" },
  ]);

  assert.deepEqual(writes, []);
  assert.deepEqual(saved(), initial);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /\[#1\].*预约体检[\s\S]*\[#2\].*预约体检/u);
  assert.doesNotMatch(result.reply, /已添加|已完成「|已删除「/u);
});

test("a compound statement clarifies between closely related completion actions before any deletion", async (t) => {
  const report = { id: "report", title: "交报告", done: false };
  const alternate = { id: "alternate", title: "提交报告", done: false };
  const weekly = { id: "weekly", title: "写周报", done: false };
  const initial = [report, alternate, weekly, distractor];
  const { tasks, writes, saved, fingerprint } = await fixture(t, initial);
  const before = await fingerprint();
  const result = await createAgent(tasks, async () =>
    assert.fail("unexpected model request"),
  ).runTurn([{ role: "user", content: "把「写周报」删掉，然后报告交好了。" }]);

  assert.deepEqual(writes, []);
  assert.deepEqual(saved(), initial);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /\[#1\][^\n]*交报告[\s\S]*\[#2\][^\n]*提交报告/u);
  assert.doesNotMatch(result.reply, /已完成「|已删除「/u);
});

test("a reversed noun phrase cannot authorize completion or the subsequent deletion", async (t) => {
  const other = { id: "admin", title: "密码管理员", done: false };
  const report = { id: "report", title: "写周报", done: false };
  const initial = [other, report, distractor];
  const { tasks, writes, saved, fingerprint } = await fixture(t, initial);
  const before = await fingerprint();
  const result = await createAgent(tasks, providerWithWrongCreate("管理员密码")).runTurn([
    { role: "user", content: "管理员密码好了，然后把写周报删掉" },
  ]);

  assert.deepEqual(writes, []);
  assert.deepEqual(saved(), initial);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已添加|已完成「|已删除「/u);
});

for (const deleteFirst of [false, true]) {
  test(`an unresolved completion cannot borrow the quoted deletion's target (${deleteFirst ? "delete" : "complete"} first)`, async (t) => {
    const exam = { id: "exam", title: "预约体检", done: false, priority: "high" };
    const report = { id: "report", title: "写周报", done: false };
    const initial = [exam, report, distractor];
    const { tasks, writes, saved, fingerprint } = await fixture(t, initial);
    const before = await fingerprint();
    const proposedOrder = ["complete_task", "delete_task"];
    if (deleteFirst) proposedOrder.reverse();
    const choices = [
      { id: "list", name: "list_tasks", arguments: "{}" },
      ...proposedOrder.map((name) => ({
        id: name,
        name,
        arguments: JSON.stringify({ candidateIds: [report.id] }),
      })),
    ];
    const provider: ModelProvider = async () => {
      const choice = choices.shift();
      return choice
        ? {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: choice.id,
                  type: "function",
                  function: { name: choice.name, arguments: choice.arguments },
                },
              ],
            },
          }
        : { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
    };
    const completion = "牙医预约好了";
    const deletion = "把「写周报」删掉";
    const input = deleteFirst ? `${deletion}，然后${completion}` : `${completion}，然后${deletion}`;
    const result = await createAgent(tasks, provider).runTurn([{ role: "user", content: input }]);

    assert.deepEqual(writes, []);
    assert.deepEqual(saved(), initial);
    assert.deepEqual(await fingerprint(), before);
    assert.deepEqual(result.outcomes, []);
    assert.doesNotMatch(result.reply, /已完成「|已删除「/u);
  });
}

test("a uniquely named task can be completed and then deleted in order", async (t) => {
  const report = { id: "report", title: "写周报", done: false, workload: "small" };
  const completed = { ...report, done: true };
  const { tasks, writes, saved } = await fixture(t, [report, distractor]);
  const result = await createAgent(tasks, async () =>
    assert.fail("unexpected model request"),
  ).runTurn([{ role: "user", content: "把「写周报」标记为完成，然后把「写周报」删掉" }]);

  assert.deepEqual(writes, [[completed, distractor], [distractor]]);
  assert.deepEqual(saved(), [distractor]);
  assert.deepEqual(result.outcomes, [
    { operation: "complete_task", task: completed },
    { operation: "delete_task", task: completed },
  ]);
  assert.equal(result.reply, "已完成「写周报」。\n已删除「写周报」。");
});

test("a compound plan cannot delete a task before a subsequent requested completion of that ID", async (t) => {
  const report = { id: "report", title: "写周报", done: false };
  const initial = [report, distractor];
  const { tasks, writes, saved, fingerprint } = await fixture(t, initial);
  const before = await fingerprint();
  const result = await createAgent(tasks, async () =>
    assert.fail("unexpected model request"),
  ).runTurn([{ role: "user", content: "把「写周报」删掉，然后把「写周报」标记为完成" }]);

  assert.deepEqual(writes, []);
  assert.deepEqual(saved(), initial);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已完成「|已删除「/u);
});

test("relisting cannot replace a preflight compound target with a different ID", async (t) => {
  const exam = { id: "exam", title: "预约体检", done: false };
  const report = { id: "report", title: "写周报", done: false };
  const { tasks, writes, saved, fingerprint } = await fixture(t, [exam, report, distractor]);
  let replacement: StoredTask | undefined;
  let before: Awaited<ReturnType<typeof fingerprint>> | undefined;
  const scripted: ModelProvider = async () => {
    const name = replacement ? "complete_task" : "list_tasks";
    if (!replacement) {
      assert.ok(tasks.remove(exam.id));
      replacement = tasks.create(exam.title);
      before = await fingerprint();
    }
    return {
      finish_reason: "tool_calls",
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: name,
            type: "function",
            function: {
              name,
              arguments: JSON.stringify(
                name === "list_tasks" ? {} : { candidateIds: [replacement.id] },
              ),
            },
          },
        ],
      },
    };
  };
  const result = await createAgent(tasks, scripted).runTurn([
    { role: "user", content: "体检那件事已经办好了，然后把写周报删掉" },
  ]);

  assert.ok(replacement && before);
  assert.notEqual(replacement.id, exam.id);
  assert.deepEqual(writes, [
    [report, distractor],
    [report, distractor, replacement],
  ]);
  assert.deepEqual(saved(), [report, distractor, replacement]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.equal(result.failure, "stale_view");
});
