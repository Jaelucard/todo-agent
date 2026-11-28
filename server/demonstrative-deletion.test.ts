import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

const milk = {
  id: "milk",
  title: "买牛奶",
  done: true,
  priority: "high",
  workload: "small",
};
const parcel = { id: "parcel", title: "取快递", done: false };
const stop = { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };

type StoredTask = ReturnType<ReturnType<typeof createJsonTaskStore>["list"]>[number];

async function fixture(t: TestContext, initial: StoredTask[]) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-demonstrative-delete-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  return {
    tasks: createJsonTaskStore(filename),
    async saved() {
      return JSON.parse(await readFile(filename, "utf8"));
    },
    async fingerprint() {
      const { ino, mtimeNs } = await stat(filename, { bigint: true });
      return [await readFile(filename, "utf8"), ino, mtimeNs] as const;
    },
  };
}

test("a target-first demonstrative delete still deletes the exact task when the model would stop", async (t) => {
  const { tasks, saved } = await fixture(t, [milk, parcel]);
  let requests = 0;
  const stopped: ModelProvider = async () => {
    requests++;
    return stop;
  };

  const result = await createAgent(tasks, stopped).runTurn([
    { role: "user", content: "取快递这条删掉吧" },
  ]);
  assert.deepEqual(await saved(), [milk]);
  assert.deepEqual(result.outcomes, [{ operation: "delete_task", task: parcel }]);
  assert.match(result.reply, /已删除「取快递」/u);
  assert.equal(requests, 0);
});

test("the local route accepts another title, demonstrative and deletion verb", async (t) => {
  const receipt = {
    id: "receipt",
    title: "整理收据",
    done: false,
    priority: "medium",
    workload: "large",
  };
  const { tasks, saved } = await fixture(t, [milk, receipt, parcel]);
  const result = await createAgent(tasks, async () =>
    assert.fail("Unexpected model request"),
  ).runTurn([{ role: "user", content: "整理收据这一项待办事项移除。" }]);
  assert.deepEqual(await saved(), [milk, parcel]);
  assert.deepEqual(result.outcomes, [{ operation: "delete_task", task: receipt }]);
  assert.match(result.reply, /已删除「整理收据」/u);
});

const suffixed = { ...parcel, id: "suffixed", title: "取快递这条" };
const duplicate = { ...parcel, id: "duplicate", priority: "low", workload: "large" };

for (const { label, initial } of [
  { label: "the original and suffixed titles both exist", initial: [milk, parcel, suffixed] },
  { label: "only the suffixed title exists", initial: [milk, suffixed] },
  { label: "the exact title belongs to multiple IDs", initial: [milk, parcel, duplicate] },
]) {
  test(`target-first deletion leaves the store unchanged when ${label}`, async (t) => {
    const { tasks, fingerprint } = await fixture(t, initial);
    const before = await fingerprint();
    const result = await createAgent(tasks, async () =>
      assert.fail("Unexpected model request"),
    ).runTurn([{ role: "user", content: "取快递这条删掉吧" }]);
    assert.deepEqual(result.outcomes, []);
    assert.deepEqual(await fingerprint(), before);
    assert.match(result.reply, /本轮未修改任务/u);
    assert.doesNotMatch(result.reply, /已删除/u);
  });
}

for (const { label, prompt } of [
  { label: "a negated target", prompt: "取快递这条先不要删掉" },
  { label: "a target without trusted context", prompt: "刚才那条删掉吧" },
]) {
  test(`target-first deletion cannot remove ${label}`, async (t) => {
    const { tasks, fingerprint } = await fixture(t, [milk, parcel]);
    const before = await fingerprint();
    let firstCall = true;
    const unsafe: ModelProvider = async () => {
      const name = firstCall ? "list_tasks" : "delete_task";
      firstCall = false;
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
                arguments: name === "list_tasks" ? "{}" : '{"candidateIds":["parcel"]}',
              },
            },
          ],
        },
      };
    };
    const result = await createAgent(tasks, unsafe).runTurn([{ role: "user", content: prompt }]);
    assert.deepEqual(result.outcomes, []);
    assert.deepEqual(await fingerprint(), before);
    assert.doesNotMatch(result.reply, /已删除/u);
  });
}
