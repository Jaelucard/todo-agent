import assert from "node:assert/strict";
import { test } from "node:test";
import { evaluateReliabilityCase, summarizeReliability } from "./reliability-eval.ts";

test("the reliability evaluator catches a committed change to the wrong task", async () => {
  const before = [
    { id: "milk", title: "买牛奶", done: false },
    { id: "parcel", title: "取快递", done: false },
  ];
  const result = await evaluateReliabilityCase({
    id: "evaluator-wrong-target",
    category: "scoring",
    initial: before,
    turns: [
      {
        input: "牛奶买好了",
        state: [{ ...before[0], done: true }, before[1]],
        mutations: [{ operation: "complete_task", title: "买牛奶", id: "milk" }],
        mock(index) {
          if (index > 1) {
            return { finish_reason: "stop", message: { role: "assistant", content: "好。" } };
          }
          const list = index === 0;
          return {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: String(index),
                  type: "function",
                  function: {
                    name: list ? "list_tasks" : "complete_task",
                    arguments: list ? "{}" : '{"candidateIds":["parcel"]}',
                  },
                },
              ],
            },
          };
        },
      },
    ],
  });
  assert.equal(result.passed, false);
  assert.deepEqual(result.turns[0].after, [before[0], { ...before[1], done: true }]);
  assert.equal(result.turns[0].taskStateCorrect, false);
  assert.equal(result.turns[0].unintendedMutations, 1);
  assert.equal(result.turns[0].falseSuccessConfirmations, 0);
});

test("reliability fixtures pass against the real task store with scripted model replies", async (t) => {
  const { reliabilityCases } = await import("./reliability-cases.ts");
  const results = [];
  for (const fixture of reliabilityCases) {
    const result = await evaluateReliabilityCase(fixture);
    results.push(result);
    await t.test(fixture.id, () => {
      assert.equal(result.passed, true, result.turns.flatMap((turn) => turn.problems).join("\n"));
    });
  }
  t.diagnostic(JSON.stringify(summarizeReliability(results)));
});
