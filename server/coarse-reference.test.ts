import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createAgent, type ChatMessage } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

function propose(name: string, args: object = {}) {
  return {
    finish_reason: "tool_calls",
    message: {
      role: "assistant",
      content: null,
      tool_calls: [
        { id: name, type: "function", function: { name, arguments: JSON.stringify(args) } },
      ],
    },
  };
}

const cases = [
  {
    message: "买奶那条已经做完了。",
    titles: ["买牛奶", "买燕麦奶"],
    followUp: "我说的是「买燕麦奶」，请把它标为完成。",
  },
  {
    message: "寄合同那一项已经做完了",
    titles: ["寄新版合同", "寄原版合同"],
    followUp: "我说的是「寄原版合同」，请把它标为完成。",
  },
];

for (const { message, titles, followUp } of cases) {
  test(`a broad demonstrative names plausible tasks before completing one: ${message}`, async (t) => {
    const directory = await mkdtemp(join(tmpdir(), "todo-agent-coarse-reference-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const filename = join(directory, "tasks.json");
    const initial = [
      { id: "first", title: titles[0], done: false, priority: "low" },
      { id: "second", title: titles[1], done: false, workload: "small" },
      { id: "unrelated", title: "取快递", done: false },
    ];
    await writeFile(filename, JSON.stringify(initial));
    const bytesBefore = await readFile(filename, "utf8");
    const statBefore = await stat(filename, { bigint: true });
    let candidateId = "first";
    let calls = 0;
    const provider: ModelProvider = async () => {
      if (calls++ === 0) return propose("list_tasks");
      if (calls === 2) return propose("complete_task", { candidateIds: [candidateId] });
      return { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };
    };
    const agent = createAgent(createJsonTaskStore(filename), provider);
    const request: ChatMessage = { role: "user", content: message };
    const result = await agent.runTurn([request]);

    const statAfter = await stat(filename, { bigint: true });
    assert.equal(await readFile(filename, "utf8"), bytesBefore);
    assert.equal(statAfter.ino, statBefore.ino);
    assert.equal(statAfter.mtimeNs, statBefore.mtimeNs);
    assert.deepEqual(result.outcomes, []);
    for (const title of titles) assert.ok(result.reply.includes(title), result.reply);
    assert.doesNotMatch(result.reply, /取快递|已完成「/u);

    candidateId = "second";
    calls = 0;
    const clarified = await agent.runTurn([
      request,
      { role: "assistant", content: result.reply },
      { role: "user", content: followUp },
    ]);

    assert.deepEqual(JSON.parse(await readFile(filename, "utf8")), [
      initial[0],
      { ...initial[1], done: true },
      initial[2],
    ]);
    assert.equal(clarified.outcomes.length, 1);
    assert.match(clarified.reply, /已完成「/u);
    assert.ok(clarified.reply.includes(titles[1]), clarified.reply);
  });
}
