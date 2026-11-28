import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

type StoredTask = ReturnType<ReturnType<typeof createJsonTaskStore>["list"]>[number];

async function fixture(t: TestContext, initial: StoredTask[]) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-ambiguity-grant-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  return { tasks: createJsonTaskStore(filename), readTasks: () => readFile(filename, "utf8") };
}

const user = (content: string) => ({ role: "user" as const, content });
const assistant = (content: string) => ({ role: "assistant" as const, content });

function proposal(id: string, name: string, args: object) {
  return {
    finish_reason: "tool_calls",
    message: {
      role: "assistant",
      content: null,
      tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }],
    },
  };
}

function scripted(...steps: unknown[]): ModelProvider {
  let next = 0;
  return async () =>
    steps[next++] ?? { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
}

const first = { id: "first", title: "写摘要", done: false, priority: "medium", workload: "small" };
const duplicate = { id: "duplicate", title: "写摘要", done: true, priority: "low" };
const second = { id: "second", title: "做演示", done: false, priority: "high" };
const orderedRename = user("把「写摘要」改名为交摘要，然后把「做演示」改名为发演示");

for (const proposedTitle of ["发演示", "交摘要"]) {
  test(`numbered clarification preserves the pending authorized rename when offered ${proposedTitle}`, async (t) => {
    const initial = [first, duplicate, second];
    const { tasks, readTasks } = await fixture(t, initial);
    const before = await readTasks();
    const agent = createAgent(
      tasks,
      scripted(
        proposal("lookup", "list_tasks", {}),
        proposal("ambiguous-update", "update_task", {
          candidateIds: [first.id],
          changes: { title: "交摘要" },
        }),
        proposal("selected-update", "update_task", {
          useBoundReference: true,
          changes: { title: proposedTitle },
        }),
      ),
    );
    const clarification = await agent.runTurn([orderedRename]);
    assert.match(clarification.reply, /\[#1\][^\n]*写摘要/u);
    assert.match(clarification.reply, /\[#2\][^\n]*写摘要/u);
    assert.deepEqual(clarification.outcomes, []);
    assert.equal(await readTasks(), before);

    const selection = await agent.runTurn([
      orderedRename,
      assistant(clarification.reply),
      user("#1"),
    ]);
    if (proposedTitle === "交摘要") {
      assert.deepEqual(tasks.list(), [{ ...first, title: "交摘要" }, duplicate, second]);
      assert.match(selection.reply, /已将「写摘要」重命名为「交摘要」/u);
    } else {
      assert.equal(await readTasks(), before);
      assert.deepEqual(selection.outcomes, []);
      assert.doesNotMatch(selection.reply, /已将.*重命名/u);
    }
  });
}

test("a quoted delete does not offer an unrelated duplicate as a numbered choice", async (t) => {
  const initial = [first, duplicate, second];
  const { tasks, readTasks } = await fixture(t, initial);
  const before = await readTasks();
  const deletion = user("把「做演示」删掉");
  const agent = createAgent(
    tasks,
    scripted(
      proposal("lookup", "list_tasks", {}),
      proposal("delete", "delete_task", { candidateIds: [first.id] }),
    ),
  );

  const clarification = await agent.runTurn([deletion]);
  assert.equal(await readTasks(), before);
  assert.deepEqual(clarification.outcomes, []);
  const selection = await agent.runTurn([deletion, assistant(clarification.reply), user("#1")]);
  assert.equal(await readTasks(), before);
  assert.deepEqual(selection.outcomes, []);
  assert.doesNotMatch(clarification.reply, /\[#\d+\][^\n]*写摘要/u);
  assert.doesNotMatch(selection.reply, /已删除/u);
});
