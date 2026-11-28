import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

type StoredTask = ReturnType<ReturnType<typeof createJsonTaskStore>["list"]>[number];

async function fixture(t: TestContext, initial: StoredTask[]) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-ordered-fields-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  return {
    tasks: createJsonTaskStore(filename),
    async fingerprint() {
      const { ino, mtimeNs } = await stat(filename, { bigint: true });
      return [await readFile(filename, "utf8"), ino, mtimeNs] as const;
    },
  };
}

function call(id: string, name: string, args: object = {}) {
  return {
    id,
    type: "function",
    function: { name, arguments: JSON.stringify(args) },
  };
}

function tools(...tool_calls: ReturnType<typeof call>[]) {
  return { finish_reason: "tool_calls", message: { role: "assistant", content: null, tool_calls } };
}

function scripted(...steps: unknown[]): ModelProvider {
  let next = 0;
  return async () =>
    steps[next++] ?? { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
}

const first = { id: "first", title: "写摘要", done: false, priority: "medium", workload: "small" };
const second = { id: "second", title: "做演示", done: true, priority: "medium", workload: "large" };
const initial = [first, second];

for (const scenario of [
  {
    label: "replacement titles",
    prompt: "把「写摘要」改名为交摘要，然后把「做演示」改名为发演示",
    firstPatch: { title: "交摘要" },
    secondPatch: { title: "发演示" },
    firstConfirmation: /已将「写摘要」重命名为「交摘要」/u,
    secondConfirmation: /已将「做演示」重命名为「发演示」/u,
  },
  {
    label: "priority values",
    prompt: "把「写摘要」的优先级设为高，然后把「做演示」的优先级设为低",
    firstPatch: { priority: "high" },
    secondPatch: { priority: "low" },
    firstConfirmation: /已更新「写摘要」/u,
    secondConfirmation: /已更新「做演示」/u,
  },
]) {
  for (const swapped of [true, false]) {
    test(`ordered updates ${swapped ? "reject swapped" : "preserve authorized"} ${scenario.label}`, async (t) => {
      const { tasks, fingerprint } = await fixture(t, initial);
      const before = await fingerprint();
      const result = await createAgent(
        tasks,
        scripted(
          tools(call("lookup", "list_tasks")),
          tools(
            call("first-update", "update_task", {
              candidateIds: [first.id],
              changes: swapped ? scenario.secondPatch : scenario.firstPatch,
            }),
            call("second-update", "update_task", {
              candidateIds: [second.id],
              changes: swapped ? scenario.firstPatch : scenario.secondPatch,
            }),
          ),
        ),
      ).runTurn([{ role: "user", content: scenario.prompt }]);

      if (swapped) {
        assert.deepEqual(tasks.list(), initial);
        assert.deepEqual(await fingerprint(), before);
        assert.deepEqual(result.outcomes, []);
        assert.doesNotMatch(result.reply, /已更新|已将.*重命名/u);
      } else {
        const expected = [
          { ...first, ...scenario.firstPatch },
          { ...second, ...scenario.secondPatch },
        ];
        assert.deepEqual(tasks.list(), expected);
        assert.deepEqual(
          result.outcomes,
          expected.map((task) => ({ operation: "update_task", task })),
        );
        assert.match(result.reply, scenario.firstConfirmation);
        assert.match(result.reply, scenario.secondConfirmation);
      }
    });
  }
}

test("completing an already done task still applies the requested trailing list filter", async (t) => {
  const complete = { id: "done", title: "检查报告", done: true, priority: "low" };
  const high = { id: "high", title: "提交预算", done: false, priority: "high" };
  const { tasks, fingerprint } = await fixture(t, [complete, high]);
  const before = await fingerprint();
  const result = await createAgent(
    tasks,
    scripted(
      tools(call("lookup", "list_tasks")),
      tools(call("complete", "complete_task", { candidateIds: [complete.id] })),
    ),
  ).runTurn([{ role: "user", content: "把「检查报告」标为完成，然后列出高优先级任务" }]);

  assert.match(result.reply, /「检查报告」已经完成/u);
  assert.match(result.reply, /高优先级清单：[\s\S]*提交预算/u);
  assert.doesNotMatch(result.reply, /^- .*检查报告/mu);
  assert.deepEqual(result.displayedList, [high]);
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(await fingerprint(), before);
});

test("a target-first suffix collision names both stored candidates before deletion", async (t) => {
  const plain = { id: "plain", title: "取快递", done: false };
  const suffixed = { id: "literal", title: "取快递这条", done: false };
  const { tasks, fingerprint } = await fixture(t, [plain, suffixed]);
  const before = await fingerprint();
  const result = await createAgent(tasks, scripted()).runTurn([
    { role: "user", content: "取快递这条删掉吧" },
  ]);

  assert.match(result.reply, /取快递(?=\s|（|」)/u);
  assert.match(result.reply, /取快递这条/u);
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(await fingerprint(), before);
  assert.doesNotMatch(result.reply, /已删除/u);
});
