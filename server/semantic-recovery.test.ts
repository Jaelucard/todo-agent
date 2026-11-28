import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent, type ChatMessage } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

const alias = {
  id: "dental-appointment",
  title: "联系口腔诊所约诊",
  done: false,
  priority: "low",
  workload: "small",
};
const unrelated = { id: "access-card", title: "申请补领门禁卡", done: false };
const input: ChatMessage[] = [{ role: "user", content: "把预约牙医删掉。" }];
const stop = { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };

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

async function fixture(t: TestContext, initial = [alias, unrelated]) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-semantic-recovery-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  return {
    tasks: createJsonTaskStore(filename),
    async saved() {
      return JSON.parse(await readFile(filename, "utf8"));
    },
    async fingerprint() {
      const info = await stat(filename, { bigint: true });
      return [await readFile(filename, "utf8"), info.ino, info.mtimeNs] as const;
    },
  };
}

test("a read-only semantic lookup recovers an unquoted deletion after the provider stops", async (t) => {
  const { tasks, saved, fingerprint } = await fixture(t);
  const initial = await fingerprint();
  let requests = 0;
  const scripted: ModelProvider = async (request) => {
    switch (requests++) {
      case 0:
        return proposal("list_tasks");
      case 1:
        return stop;
      case 2:
        assert.deepEqual(await fingerprint(), initial);
        assert.deepEqual(
          request.tools?.map((tool) => tool.type === "function" && tool.function.name),
          ["resolve_task_reference"],
        );
        assert.deepEqual(request.tool_choice, {
          type: "function",
          function: { name: "resolve_task_reference" },
        });
        return proposal("resolve_task_reference", { candidateIds: [alias.id] });
      case 3:
        assert.deepEqual(await fingerprint(), initial);
        assert.deepEqual(
          request.tools?.map((tool) => tool.type === "function" && tool.function.name),
          ["verify_task_reference"],
        );
        assert.deepEqual(request.tool_choice, {
          type: "function",
          function: { name: "verify_task_reference" },
        });
        return proposal("verify_task_reference", { decision: "match", candidateIds: [alias.id] });
      default:
        assert.fail("The successful delete must not ask the provider to act again");
    }
  };

  const result = await createAgent(tasks, scripted).runTurn(input);
  assert.deepEqual(await saved(), [unrelated]);
  assert.deepEqual(result.outcomes, [{ operation: "delete_task", task: alias }]);
  assert.match(result.reply, /已删除「联系口腔诊所约诊」/u);
  assert.equal(requests, 4);
});

const secondAlias = { ...alias, id: "other-dental-appointment", title: "改期口腔检查预约" };

for (const command of ["把预约牙医删掉。", "预约牙医删掉。", "去掉预约牙医。", "清掉预约牙医。"]) {
  test(`a model cannot hide the dental alternative with a filtered view for ${command}`, async (t) => {
    const sibling = { ...secondAlias, title: "打电话预约牙科检查", done: true };
    const { tasks, saved, fingerprint } = await fixture(t, [alias, unrelated, sibling]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks", { done: false }),
      proposal("delete_task", { candidateIds: [alias.id] }),
      proposal("verify_task_reference", { decision: "match", candidateIds: [alias.id] }),
    ];
    const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn([
      { role: "user", content: command },
    ]);

    assert.deepEqual(await saved(), [alias, unrelated, sibling]);
    assert.deepEqual(await fingerprint(), initial);
    assert.deepEqual(result.outcomes, []);
    assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
    assert.match(result.reply, /\[#2\].*打电话预约牙科检查/u);
    assert.doesNotMatch(result.reply, /已删除|申请补领门禁卡/u);
  });
}

test("either dental alias can be omitted without letting the other alias be deleted", async (t) => {
  const sibling = { ...secondAlias, title: "打电话预约牙科检查" };
  const { tasks, saved, fingerprint } = await fixture(t, [alias, unrelated, sibling]);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [sibling.id] }),
    proposal("verify_task_reference", { decision: "match", candidateIds: [sibling.id] }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn(input);

  assert.deepEqual(await saved(), [alias, unrelated, sibling]);
  assert.deepEqual(await fingerprint(), initial);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
  assert.match(result.reply, /\[#2\].*打电话预约牙科检查/u);
});

for (const [wanted, stored] of [
  ["预约牙医", "预约医院体检"],
  ["预约牙医", "医院体检"],
  ["预约牙医", "挂号医院体检"],
  ["预约牙医", "预付物业费"],
  ["预约牙医", "联系宠物诊所约诊"],
  ["预约牙医", "联系骨科诊所约诊"],
  ["预约牙医", "准备预约牙医"],
  ["预约税务咨询", "预约法律咨询"],
  ["预约牙医", "取消预约牙医"],
  ["取消预约牙医", "预约牙医"],
  ["预约牙医", "记得取消预约牙医"],
  ["预约牙医", "打电话取消牙医预约"],
  ["预约牙医", "牙医预约取消"],
  ["预约牙医", "退掉牙医预约"],
]) {
  test(`two model proposals cannot turn ${wanted} into deletion of ${stored}`, async (t) => {
    const other = { ...alias, title: stored };
    const { tasks, saved, fingerprint } = await fixture(t, [other, unrelated]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks"),
      proposal("delete_task", { candidateIds: [other.id] }),
      proposal("verify_task_reference", { decision: "match", candidateIds: [other.id] }),
    ];
    const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn([
      { role: "user", content: `删掉${wanted}。` },
    ]);

    assert.deepEqual(await saved(), [other, unrelated]);
    assert.deepEqual(await fingerprint(), initial);
    assert.deepEqual(result.outcomes, []);
    assert.doesNotMatch(result.reply, /已删除/u);
  });
}

for (const title of ["洗牙", "补牙复诊"]) {
  test(`a hidden ${title} task must also be considered before semantic deletion`, async (t) => {
    const other = { ...secondAlias, title, done: true };
    const { tasks, saved, fingerprint } = await fixture(t, [alias, unrelated, other]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks", { done: false }),
      proposal("delete_task", { candidateIds: [alias.id] }),
      proposal("verify_task_reference", { decision: "match", candidateIds: [alias.id] }),
    ];
    const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn(input);
    assert.deepEqual(await saved(), [alias, unrelated, other]);
    assert.deepEqual(await fingerprint(), initial);
    assert.deepEqual(result.outcomes, []);
    assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
    assert.ok(result.reply.includes(`[#2] ${title}`));
  });
}

test("a matcher that omits a plausible alias cannot turn ambiguity into a deletion", async (t) => {
  const otherAlias = { ...secondAlias, title: "预约牙科医师复诊" };
  const { tasks, saved, fingerprint } = await fixture(t, [alias, unrelated, otherAlias]);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    stop,
    proposal("resolve_task_reference", { candidateIds: [alias.id] }),
  ];
  const agent = createAgent(tasks, async (request) => {
    const next = replies.shift();
    if (next) return next;
    const payload = JSON.parse(String(request.messages.at(-1)?.content));
    if (payload.tasks.some(({ id }: { id: string }) => id === otherAlias.id)) {
      return proposal("verify_task_reference", {
        decision: "match",
        candidateIds: [alias.id, otherAlias.id],
      });
    }
    return proposal("verify_task_reference", { decision: "match" });
  });

  const result = await agent.runTurn(input);

  assert.deepEqual(await fingerprint(), initial);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
  assert.match(result.reply, /\[#2\].*预约牙科医师复诊/u);
  assert.doesNotMatch(result.reply, /申请补领门禁卡|已删除/u);

  const followUp = await agent.runTurn([
    ...input,
    { role: "assistant", content: result.reply },
    { role: "user", content: "#2" },
  ]);
  assert.deepEqual(followUp.outcomes, [{ operation: "delete_task", task: otherAlias }]);
  assert.deepEqual(await saved(), [alias, unrelated]);
});

for (const content of [input[0].content, "删掉预约牙医。"]) {
  test(`a direct delete proposal cannot bypass semantic ambiguity checks for ${content}`, async (t) => {
    const hiddenAlias = { ...secondAlias, done: true };
    const { tasks, fingerprint } = await fixture(t, [alias, unrelated, hiddenAlias]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks", { done: false }),
      proposal("delete_task", { candidateIds: [alias.id] }),
    ];
    const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn([
      { role: "user", content },
    ]);

    assert.deepEqual(await fingerprint(), initial);
    assert.deepEqual(result.outcomes, []);
    assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
    assert.match(result.reply, /\[#2\].*改期口腔检查预约/u);
    assert.doesNotMatch(result.reply, /申请补领门禁卡|已删除/u);
  });
}

for (const selected of [
  {
    label: "an unseen ID",
    choice: proposal("resolve_task_reference", { candidateIds: ["outside-view"] }),
  },
  {
    label: "a repeated ID",
    choice: proposal("resolve_task_reference", { candidateIds: [alias.id, alias.id] }),
  },
  {
    label: "an unsupported mutating tool",
    choice: proposal("delete_task", { candidateIds: [alias.id] }),
  },
  { label: "another prose stop", choice: stop },
]) {
  test(`recovery leaves the store untouched when the model returns ${selected.label}`, async (t) => {
    const { tasks, fingerprint } = await fixture(t, [alias, unrelated, secondAlias]);
    const initial = await fingerprint();
    let requests = 0;
    const scripted: ModelProvider = async () => {
      switch (requests++) {
        case 0:
          return proposal("list_tasks");
        case 1:
          return stop;
        case 2:
          return selected.choice;
        default:
          assert.fail("Recovery may consult the provider only once");
      }
    };
    const result = await createAgent(tasks, scripted).runTurn(input);

    assert.deepEqual(result.outcomes, []);
    assert.doesNotMatch(result.reply, /已删除/u);
    assert.deepEqual(await fingerprint(), initial);
    assert.equal(requests, 3);
  });
}

test("verification confirms ambiguity without deleting either plausible task", async (t) => {
  const { tasks, fingerprint } = await fixture(t, [alias, unrelated, secondAlias]);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    stop,
    proposal("resolve_task_reference", { candidateIds: [alias.id, secondAlias.id] }),
    proposal("verify_task_reference", {
      decision: "match",
      candidateIds: [alias.id, secondAlias.id],
    }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn(input);

  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(await fingerprint(), initial);
  assert.equal(replies.length, 0);
  assert.match(result.reply, /\[#1\].*联系口腔诊所约诊/u);
  assert.match(result.reply, /\[#2\].*改期口腔检查预约/u);
  assert.doesNotMatch(result.reply, /申请补领门禁卡|已删除/u);
});

test("a read-only lookup can establish semantic absence without writing", async (t) => {
  const { tasks, fingerprint } = await fixture(t, [unrelated]);
  const initial = await fingerprint();
  let requests = 0;
  const scripted: ModelProvider = async () => {
    switch (requests++) {
      case 0:
        return proposal("list_tasks");
      case 1:
        return stop;
      case 2:
        return proposal("resolve_task_reference", { candidateIds: [] });
      default:
        assert.fail("A grounded missing result must end recovery");
    }
  };
  const result = await createAgent(tasks, scripted).runTurn(input);

  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /找不到|没有找到|未找到/u);
  assert.doesNotMatch(result.reply, /已删除/u);
  assert.deepEqual(await fingerprint(), initial);
  assert.equal(requests, 3);
});

test("recovery rejects a target from a lookup made stale by an intervening update", async (t) => {
  const { tasks, saved, fingerprint } = await fixture(t);
  const renamed = { ...alias, title: "电话联系新公司" };
  let afterUpdate: Awaited<ReturnType<typeof fingerprint>> | undefined;
  let requests = 0;
  const scripted: ModelProvider = async () => {
    switch (requests++) {
      case 0:
        return proposal("list_tasks");
      case 1:
        return stop;
      case 2:
        tasks.update(alias.id, { title: renamed.title });
        afterUpdate = await fingerprint();
        return proposal("resolve_task_reference", { candidateIds: [alias.id] });
      default:
        assert.fail("A stale decision must end recovery");
    }
  };
  const result = await createAgent(tasks, scripted).runTurn(input);

  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已删除/u);
  assert.deepEqual(await saved(), [renamed, unrelated]);
  assert.deepEqual(await fingerprint(), afterUpdate);
  assert.equal(requests, 3);
});

test("a compound turn that already committed a completion must not attempt recovery on stop", async (t) => {
  const { tasks, saved } = await fixture(t);
  let requests = 0;
  const scripted: ModelProvider = async (request) => {
    assert.ok(
      !request.tools?.some(
        (tool) => tool.type === "function" && tool.function.name === "resolve_task_reference",
      ),
    );
    switch (requests++) {
      case 0:
        return proposal("list_tasks");
      case 1:
        return proposal("complete_task", { candidateIds: [unrelated.id] });
      case 2:
        return stop;
      default:
        assert.fail("A successful completion must not be replayed");
    }
  };
  const result = await createAgent(tasks, scripted).runTurn([
    { role: "user", content: "门禁卡那件事已经办好了，然后把联系口腔诊所约诊删掉。" },
  ]);

  const completed = { ...unrelated, done: true };
  assert.deepEqual(result.outcomes, [{ operation: "complete_task", task: completed }]);
  assert.match(result.reply, /已完成「申请补领门禁卡」/u);
  assert.doesNotMatch(result.reply, /已删除/u);
  assert.deepEqual(await saved(), [alias, completed]);
  assert.equal(requests, 3);
});

const milk = { id: "milk", title: "买牛奶", done: false };
const parcel = { id: "parcel", title: "取快递", done: false };

for (const candidateIds of [[milk.id, parcel.id], [milk.id]]) {
  test(`recovery cannot offer or delete ${candidateIds.length} unrelated current tasks`, async (t) => {
    const { tasks, fingerprint } = await fixture(t, [milk, parcel]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks"),
      stop,
      proposal("resolve_task_reference", { candidateIds }),
    ];
    const agent = createAgent(tasks, async () => replies.shift() ?? stop);
    const result = await agent.runTurn(input);

    assert.deepEqual(result.outcomes, []);
    assert.deepEqual(await fingerprint(), initial);

    const followUp = await agent.runTurn([
      ...input,
      { role: "assistant", content: result.reply },
      { role: "user", content: "#1" },
    ]);
    assert.deepEqual(followUp.outcomes, []);
    assert.deepEqual(await fingerprint(), initial);
    assert.match(result.reply, /找不到|没有找到|未找到|没有完全同名/u);
    assert.doesNotMatch(result.reply, /#\d+|你指的是哪一项|已删除/u);
  });
}

test("recovery cannot turn a mixed candidate set into a single authorized deletion", async (t) => {
  const { tasks, fingerprint } = await fixture(t);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    stop,
    proposal("resolve_task_reference", { candidateIds: [alias.id, unrelated.id] }),
  ];
  const agent = createAgent(tasks, async () => replies.shift() ?? stop);
  const result = await agent.runTurn(input);

  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(await fingerprint(), initial);
  const followUp = await agent.runTurn([
    ...input,
    { role: "assistant", content: result.reply },
    { role: "user", content: "#1" },
  ]);
  assert.deepEqual(followUp.outcomes, []);
  assert.deepEqual(await fingerprint(), initial);
  assert.doesNotMatch(result.reply, /#\d+|已删除/u);
});

const differentAppointment = { id: "physical", title: "预约体检", done: false };

test("agreement on a shared action cannot authorize different objects", async (t) => {
  const { tasks, fingerprint } = await fixture(t, [differentAppointment, unrelated]);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [differentAppointment.id] }),
    proposal("verify_task_reference", {
      decision: "match",
      candidateIds: [differentAppointment.id],
    }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn(input);

  assert.deepEqual(await fingerprint(), initial);
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已删除/u);
});

test("direct alias selection can delete a verified unique match without touching a different appointment", async (t) => {
  const { tasks, saved } = await fixture(t, [alias, differentAppointment, unrelated]);
  const replies = [
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [alias.id] }),
    proposal("verify_task_reference", { decision: "match", candidateIds: [alias.id] }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn(input);

  assert.deepEqual(await saved(), [differentAppointment, unrelated]);
  assert.deepEqual(result.outcomes, [{ operation: "delete_task", task: alias }]);
  assert.match(result.reply, /已删除「联系口腔诊所约诊」/u);
});

test("a mixed request preserves the protected task and can delete a verified alias", async (t) => {
  const { tasks, saved } = await fixture(t, [alias, unrelated]);
  const replies = [
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [alias.id] }),
    proposal("verify_task_reference", { decision: "match", candidateIds: [alias.id] }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn([
    { role: "user", content: "保留申请补领门禁卡，把预约牙医删掉。" },
  ]);
  assert.deepEqual(await saved(), [unrelated]);
  assert.deepEqual(result.outcomes, [{ operation: "delete_task", task: alias }]);
  assert.match(result.reply, /已删除「联系口腔诊所约诊」/u);
});

test("an ungrounded model-resolved alias requires the user's exact task name before deletion", async (t) => {
  const weakAlias = { id: "team-dinner", title: "联系餐厅排定包间", done: false };
  const weakInput: ChatMessage[] = [{ role: "user", content: "把安排团建删掉。" }];
  const { tasks, saved, fingerprint } = await fixture(t, [weakAlias, unrelated]);
  const initial = await fingerprint();
  const replies = [
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [weakAlias.id] }),
    proposal("verify_task_reference", { decision: "match", candidateIds: [weakAlias.id] }),
    proposal("list_tasks"),
    proposal("delete_task", { candidateIds: [weakAlias.id] }),
  ];
  const agent = createAgent(tasks, async () => replies.shift() ?? stop);
  const pending = await agent.runTurn(weakInput);
  assert.deepEqual(await saved(), [weakAlias, unrelated]);
  assert.deepEqual(await fingerprint(), initial);
  assert.deepEqual(pending.outcomes, []);
  assert.match(pending.reply, /联系餐厅排定包间/u);
  assert.match(pending.reply, /请用完整任务名称确认/u);
  assert.doesNotMatch(pending.reply, /已删除/u);

  const confirmed = await agent.runTurn([
    ...weakInput,
    { role: "assistant", content: pending.reply },
    { role: "user", content: "把「联系餐厅排定包间」删掉。" },
  ]);
  assert.deepEqual(await saved(), [unrelated]);
  assert.deepEqual(confirmed.outcomes, [{ operation: "delete_task", task: weakAlias }]);
  assert.match(confirmed.reply, /已删除「联系餐厅排定包间」/u);
});

test("a clarification stop can identify an exact object without deleting it", async (t) => {
  const report = { ...alias, title: "提交年度报告" };
  const { tasks, saved, fingerprint } = await fixture(t, [report, unrelated]);
  const initial = await fingerprint();
  const clarification = {
    finish_reason: "stop",
    message: { role: "assistant", content: "需要澄清！请确认任务。" },
  };
  const replies = [
    clarification,
    proposal("resolve_task_reference", { candidateIds: [report.id] }),
    proposal("verify_task_reference", { decision: "match", candidateIds: [report.id] }),
  ];
  const result = await createAgent(tasks, async () => replies.shift() ?? stop).runTurn([
    { role: "user", content: "把年度报告删掉。" },
  ]);
  assert.deepEqual(await saved(), [report, unrelated]);
  assert.deepEqual(await fingerprint(), initial);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /提交年度报告[\s\S]*请用完整任务名称确认/u);
  assert.doesNotMatch(result.reply, /已删除/u);
});

for (const decision of ["no_match", "uncertain"] as const) {
  test(`semantic recovery cannot delete a different appointment when verification returns ${decision}`, async (t) => {
    const { tasks, fingerprint } = await fixture(t, [differentAppointment, unrelated]);
    const initial = await fingerprint();
    const replies = [
      proposal("list_tasks"),
      stop,
      proposal("resolve_task_reference", { candidateIds: [differentAppointment.id] }),
      proposal("verify_task_reference", { decision, candidateIds: [] }),
    ];
    const agent = createAgent(tasks, async () => replies.shift() ?? stop);
    const result = await agent.runTurn(input);

    assert.deepEqual(await fingerprint(), initial);
    assert.deepEqual(result.outcomes, []);
    assert.equal(replies.length, 0);
    assert.doesNotMatch(result.reply, /#\d+|已删除/u);
  });
}
