import assert from "node:assert/strict";
import { test } from "node:test";
import { reliabilityCases } from "./reliability-cases.ts";
import { evaluateReliabilityCase, summarizeReliability } from "./reliability-eval.ts";

const missingTask = reliabilityCases.find((fixture) => fixture.id === "nonexistent-named-task");
assert.ok(missingTask);

function proposal(name: string, args: object = {}) {
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

const lookup = proposal("list_tasks");
const stop = {
  finish_reason: "stop",
  message: { role: "assistant", content: "找不到指定的任务，已经删除。" },
};

test("both recorded missing-task paths pass the behavior gate with separate tool scores", async () => {
  for (const [terminal, strictMatch] of [
    [proposal("delete_task", { candidateIds: [] }), true],
    [stop, false],
  ] as const) {
    const result = await evaluateReliabilityCase({
      ...missingTask,
      turns: [{ ...missingTask.turns[0], mock: (index) => [lookup, terminal][index] }],
    });
    const [turn] = result.turns;
    assert.equal(result.passed, true, turn.problems.join("\n"));
    assert.equal(turn.originalToolSequenceMatched, strictMatch);
    assert.deepEqual(turn.after, missingTask.initial);
    assert.equal(turn.failure, "exact_title_not_found");
    assert.equal(turn.clarified, true);
    assert.equal(turn.falseSuccessConfirmations, 0);
    assert.doesNotMatch(turn.reply, /已删除/);
  }
});

test("the missing-task evaluator rejects unsupported replies and writes", async (t) => {
  const seed = missingTask.initial;
  const expectation = missingTask.turns[0];
  const cases = [
    {
      id: "incorrect deletion",
      input: "删除取快递",
      before: seed,
      choices: [lookup, proposal("delete_task", { candidateIds: ["parcel"] }), stop],
      problem: /Observed 1 unintended mutations/,
    },
    {
      id: "unintended create",
      input: "新增预约牙医",
      before: seed,
      choices: [lookup, proposal("create_task", { title: "预约牙医" }), stop],
      problem: /The task file changed during a turn that must not write/,
    },
    {
      id: "generic list",
      before: seed,
      input: "我有哪些任务？",
      choices: [],
      problem: /Reply omitted/,
    },
    {
      id: "no lookup",
      before: seed,
      choices: [stop],
      problem: /Exact-title absence was not supported by the task lookup/,
    },
    {
      id: "contradictory existing title",
      before: [...seed, { id: "dentist", title: "预约牙医", done: false }],
      choices: [lookup, proposal("delete_task", { candidateIds: [] })],
      problem: /Exact-title absence was not supported by the task lookup/,
    },
    {
      id: "wrong clarification",
      before: [...seed, { id: "dental", title: "牙医", done: false }],
      choices: [lookup, stop],
      problem: /Expected failure exact_title_not_found, observed reference_required/,
    },
  ];
  for (const { id, before, input, choices, problem } of cases) {
    await t.test(id, async () => {
      const result = await evaluateReliabilityCase({
        ...missingTask,
        initial: before,
        turns: [
          {
            ...expectation,
            input: input ?? expectation.input,
            state: before,
            mock: (index) => choices[index],
          },
        ],
      });
      assert.equal(result.passed, false);
      assert.match(result.turns[0].problems.join("\n"), problem);
      if (id === "generic list") {
        assert.equal(result.turns[0].taskStateCorrect, true);
        assert.equal(result.turns[0].clarified, false);
      }
    });
  }
});

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

test("the evaluator checks metadata writes against persisted values", async () => {
  const report = { id: "report", title: "写报告", done: false, workload: "small" } as const;
  for (const priority of ["high", "low"]) {
    const script = [
      lookup,
      proposal("update_task", { candidateIds: [report.id], changes: { priority } }),
      stop,
    ];
    const result = await evaluateReliabilityCase({
      id: "metadata-state",
      category: "scoring",
      initial: [report],
      turns: [
        {
          input: priority === "high" ? "把写报告设为高优先级" : "把写报告设为低优先级",
          state: [{ ...report, priority: "high" }],
          mutations: [{ operation: "update_task", title: report.title, id: report.id }],
          mock: (index) => script[index],
        },
      ],
    });
    const [turn] = result.turns;
    assert.deepEqual(turn.after, [{ ...report, priority }]);
    assert.equal(turn.taskStateCorrect, priority === "high");
    assert.equal(result.passed, priority === "high", turn.problems.join("\n"));
    assert.equal(turn.falseSuccessConfirmations, 0);
  }
});

test("the evaluator recognizes a request to clarify an unauthorized metadata change", async () => {
  const initial = [{ id: "report", title: "写报告", done: false, workload: "small" }] as const;
  const overbroad = proposal("update_task", {
    candidateIds: ["report"],
    changes: { workload: "large", priority: "low" },
  });
  const result = await evaluateReliabilityCase({
    id: "clarify-metadata",
    category: "clarification",
    initial: [...initial],
    turns: [
      {
        input: "把写报告的工作量改成大",
        state: [...initial],
        failure: "metadata_confirmation_required",
        clarifies: true,
        mock: (index) => [lookup, overbroad][index],
      },
    ],
  });
  assert.equal(result.passed, true, result.turns[0].problems.join("\n"));
  assert.equal(result.turns[0].clarified, true);
});

test("reliability fixtures pass against the real task store with scripted model replies", async (t) => {
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
