import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createAgent, type ChatMessage } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

type Task = ReturnType<ReturnType<typeof createJsonTaskStore>["list"]>[number];

async function fixture(t: TestContext, initial: Task[] = []) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-acceptance-regression-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filename = join(directory, "tasks.json");
  await writeFile(filename, JSON.stringify(initial));
  return {
    tasks: createJsonTaskStore(filename),
    async saved(): Promise<Task[]> {
      return JSON.parse(await readFile(filename, "utf8"));
    },
    async fingerprint() {
      const info = await stat(filename, { bigint: true });
      return [await readFile(filename, "utf8"), info.ino, info.mtimeNs] as const;
    },
  };
}

const user = (content: string): ChatMessage => ({ role: "user", content });
const assistant = (content: string): ChatMessage => ({ role: "assistant", content });
const stop = { finish_reason: "stop", message: { role: "assistant", content: "收到。" } };

function proposal(id: string, name: string, args: object = {}) {
  return {
    finish_reason: "tool_calls",
    message: {
      role: "assistant",
      content: null,
      tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }],
    },
  };
}

function scripted(...choices: unknown[]): ModelProvider {
  let next = 0;
  return async () => choices[next++] ?? stop;
}

test("a known verb-object completion survives an ordinary model stop without rewriting metadata", async (t) => {
  const report = {
    id: "historical-report",
    title: "交报告",
    done: false,
    priority: "low",
    workload: "small",
  };
  const parcel = {
    id: "parcel",
    title: "取快递",
    done: false,
    priority: "medium",
    workload: "small",
  };
  const { tasks, saved, fingerprint } = await fixture(t, [report, parcel]);
  const agent = createAgent(tasks, scripted(stop));
  const result = await agent.runTurn([user("报告交好了。")]);
  const completed = { ...report, done: true };
  assert.deepEqual(await saved(), [completed, parcel]);
  assert.deepEqual(result.outcomes, [{ operation: "complete_task", task: completed }]);
  assert.equal(result.reply, "已完成「交报告」。");

  const before = await fingerprint();
  const repeated = await agent.runTurn([user("报告交好了。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(repeated.outcomes, []);
  assert.match(repeated.reply, /「交报告」已经完成/u);
});

for (const alternate of ["交季度报告", "提交报告"]) {
  test(`an omitted completion clarifies between 交报告 and ${alternate} without writing`, async (t) => {
    const report = { id: "short-report", title: "交报告", done: false };
    const other = { id: "other-report", title: alternate, done: false };
    const parcel = { id: "parcel", title: "取快递", done: false };
    const { tasks, saved, fingerprint } = await fixture(t, [report, other, parcel]);
    const before = await fingerprint();
    const result = await createAgent(tasks, scripted(stop)).runTurn([user("报告交好了。")]);
    assert.deepEqual(await fingerprint(), before);
    assert.deepEqual(await saved(), [report, other, parcel]);
    assert.deepEqual(result.outcomes, []);
    assert.match(
      result.reply,
      new RegExp(`\\[#1\\][^\\n]*交报告[\\s\\S]*\\[#2\\][^\\n]*${alternate}`, "u"),
    );
    assert.doesNotMatch(result.reply, /已完成「|取快递/u);
  });
}

test("an omitted completion cannot infer mutations from a reversed noun or negation", async (t) => {
  const credentials = { id: "admin", title: "密码管理员", done: false };
  const report = { id: "report", title: "交报告", done: false };
  const { tasks, saved, fingerprint } = await fixture(t, [credentials, report]);
  const before = await fingerprint();
  for (const text of ["管理员密码好了。", "别把报告标记完成。", "报告交好了吗？"]) {
    const result = await createAgent(tasks, scripted(stop)).runTurn([user(text)]);
    assert.deepEqual(await saved(), [credentials, report]);
    assert.deepEqual(await fingerprint(), before);
    assert.deepEqual(result.outcomes, []);
    assert.doesNotMatch(result.reply, /已完成「/u);
  }
});

test("ordinary model proposals cannot override a proven compact completion target", async (t) => {
  const report = { id: "report", title: "交报告", done: false, priority: "low" };
  const parcel = { id: "parcel", title: "取快递", done: false };
  const { tasks, saved, fingerprint } = await fixture(t, [report, parcel]);
  const before = await fingerprint();
  const result = await createAgent(
    tasks,
    scripted(
      proposal("list", "list_tasks"),
      proposal("wrong", "complete_task", { candidateIds: [parcel.id] }),
    ),
  ).runTurn([user("报告交好了。")]);

  assert.equal(result.failure, "reference_required");
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(await saved(), [report, parcel]);
  assert.deepEqual(await fingerprint(), before);
  assert.doesNotMatch(result.reply, /已完成「/u);
});

test("an explicit model clarification stop does not trigger compact completion recovery", async (t) => {
  const report = { id: "report", title: "交报告", done: false };
  const { tasks, saved, fingerprint } = await fixture(t, [report]);
  const before = await fingerprint();
  const result = await createAgent(
    tasks,
    scripted({ finish_reason: "stop", message: { role: "assistant", content: "需要澄清" } }),
  ).runTurn([user("报告交好了。")]);

  assert.deepEqual(await saved(), [report]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.equal(result.failure, "ambiguous_reference");
});

test("plain follow-up create persists once even when the provider would stop without an automatic tool", async (t) => {
  const { tasks, saved } = await fixture(t);
  const createReport = user("帮我记一下写项目报告，优先级高，工作量大。");
  const followUp = user("再记一条预约体检。");
  const initial = await createAgent(
    tasks,
    scripted(
      proposal("report", "create_task", {
        title: "写项目报告",
        priority: "high",
        workload: "large",
      }),
    ),
  ).runTurn([createReport]);
  const [report] = await saved();
  assert.deepEqual(report, {
    id: report.id,
    title: "写项目报告",
    done: false,
    priority: "high",
    workload: "large",
  });

  let attempts = 0;
  const result = await createAgent(tasks, async ({ tool_choice, messages }) => {
    if (messages.at(-1)?.role === "tool") return stop;
    if (tool_choice === "auto" && attempts++ === 0) return stop;
    return proposal("appointment", "create_task", { title: "预约体检" });
  }).runTurn([createReport, assistant(initial.reply), followUp]);

  const records = await saved();
  assert.equal(records.length, 2);
  const [, appointment] = records;
  assert.notEqual(appointment.id, report.id);
  assert.deepEqual(records, [report, { id: appointment.id, title: "预约体检", done: false }]);
  assert.match(result.reply, /已(?:经)?添加.{0,8}预约体检/u);
});

for (const edit of [
  {
    name: "rename",
    title: "写项目报告",
    done: false,
    input: "把「写项目报告」改名为提交项目报告。",
    changes: { title: "提交项目报告" },
    reply: /(?:改名|重命名).{0,8}提交项目报告/u,
  },
  {
    name: "reopen",
    title: "提交项目报告",
    done: true,
    input: "把「提交项目报告」改回未完成。",
    changes: { done: false },
    reply: /提交项目报告.{0,8}(?:未完成|待完成)/u,
  },
] as const) {
  test(`an explicit quoted ${edit.name} survives a premature automatic provider stop`, async (t) => {
    const report = {
      id: "report",
      title: edit.title,
      done: edit.done,
      priority: "high",
      workload: "small",
    };
    const distractor = { id: "other", title: "提交周报", done: false, priority: "low" };
    const { tasks, saved } = await fixture(t, [report, distractor]);
    let attempts = 0;
    let issuedLookup = false;
    let issuedUpdate = false;
    const turn = await createAgent(tasks, async ({ tool_choice }) => {
      if (tool_choice === "auto" && attempts++ === 0) return stop;
      if (!issuedLookup) {
        issuedLookup = true;
        return proposal("lookup", "list_tasks");
      }
      if (!issuedUpdate) {
        issuedUpdate = true;
        return proposal("edit", "update_task", {
          candidateIds: [report.id],
          changes: edit.changes,
        });
      }
      return stop;
    }).runTurn([user(edit.input)]);
    assert.deepEqual(await saved(), [{ ...report, ...edit.changes }, distractor]);
    assert.match(turn.reply, edit.reply);
  });
}

for (const input of ["把预约牙医删掉。", "把「预约牙医」删掉。", "把预约牙医这条删掉吧。"]) {
  test(`a grounded lookup acknowledges a missing task for ${input}`, async (t) => {
    const initial = [
      { id: "mail", title: "寄快递", done: false },
      { id: "rent", title: "交房租", done: true, priority: "high" },
    ];
    const { tasks, fingerprint } = await fixture(t, initial);
    const before = await fingerprint();
    const turn = await createAgent(
      tasks,
      scripted(
        proposal("lookup", "list_tasks"),
        stop,
        proposal("empty", "delete_task", { candidateIds: [] }),
      ),
    ).runTurn([user(input)]);
    assert.deepEqual(await fingerprint(), before);
    assert.match(turn.reply, /找不到|没有完全同名|未找到|没有(?:这项|该)任务|不存在/u);
    assert.doesNotMatch(turn.reply, /已删除|你指的是哪一项/u);
  });
}

test("a stored title with a unique semantic match stays actionable", async (t) => {
  const report = { id: "report", title: "整理年度报告", done: false, priority: "high" };
  const other = { id: "other", title: "续办护照", done: false };
  const { tasks, saved } = await fixture(t, [report, other]);
  const turn = await createAgent(
    tasks,
    scripted(
      proposal("lookup", "list_tasks"),
      proposal("delete", "delete_task", { candidateIds: [report.id] }),
      stop,
    ),
  ).runTurn([user("把年度报告删掉。")]);
  assert.deepEqual(await saved(), [other]);
  assert.match(turn.reply, /已(?:经)?删除.{0,8}整理年度报告/u);
});

test("ungrounded ordinals cannot delete similar tasks and the clarification names the real choices", async (t) => {
  const q1 = { id: "quarter-one", title: "Q1季度报告", done: false };
  const q2 = { id: "quarter-two", title: "Q2季度报告", done: true, priority: "high" };
  const unrelated = { id: "retained", title: "寄快递", done: false };
  const { tasks, saved, fingerprint } = await fixture(t, [q1, q2, unrelated]);
  const before = await fingerprint();
  const instruction = user("把季度报告删掉。");
  const clarification = await createAgent(tasks, async ({ tool_choice }) => {
    if (
      typeof tool_choice === "object" &&
      "function" in tool_choice &&
      tool_choice.function.name === "list_tasks"
    ) {
      return proposal("lookup", "list_tasks");
    }
    return proposal("ungrounded", "delete_task", { candidateIds: ["1", "2"] });
  }).runTurn([instruction]);
  assert.deepEqual(await fingerprint(), before);
  assert.match(clarification.reply, /多个|哪一项|哪一条|选择/u);
  assert.match(clarification.reply, /Q1季度报告/u);
  assert.match(clarification.reply, /Q2季度报告/u);
  assert.doesNotMatch(clarification.reply, /\[#\d+\].*寄快递|已删除/u);

  const ordinal = /\[(#\d+)\].*Q2季度报告/u.exec(clarification.reply)?.[1];
  const followUp = await createAgent(
    tasks,
    scripted(
      proposal("lookup", "list_tasks"),
      proposal("selected", "delete_task", { candidateIds: [q2.id] }),
      stop,
    ),
  ).runTurn([
    instruction,
    assistant(clarification.reply),
    user(ordinal ?? "把「Q2季度报告」删掉。"),
  ]);
  assert.deepEqual(await saved(), [q1, unrelated]);
  assert.match(followUp.reply, /已(?:经)?删除.{0,8}Q2季度报告/u);
});

test("a quoted rename still asks before choosing between duplicate IDs", async (t) => {
  const first = { id: "one", title: "撰写报告", done: false, priority: "low" };
  const second = { ...first, id: "two", done: true };
  const { tasks, fingerprint } = await fixture(t, [first, second]);
  const before = await fingerprint();
  const turn = await createAgent(
    tasks,
    scripted(
      proposal("list", "list_tasks"),
      proposal("guess", "update_task", {
        candidateIds: [second.id],
        changes: { title: "提交报告" },
      }),
    ),
  ).runTurn([user("把「撰写报告」重命名为「提交报告」。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.match(turn.reply, /多个|哪一项|哪一条|选择/u);
  assert.doesNotMatch(turn.reply, /已将.*(?:改名|重命名)/u);
});

test("a negated quoted rename leaves the file and both records untouched", async (t) => {
  const report = { id: "one", title: "撰写报告", done: false };
  const distractor = { ...report, id: "two", title: "撰写摘要" };
  const { tasks, fingerprint } = await fixture(t, [report, distractor]);
  const before = await fingerprint();
  const turn = await createAgent(
    tasks,
    scripted(
      proposal("list", "list_tasks"),
      proposal("rename", "update_task", {
        candidateIds: [report.id],
        changes: { title: "提交报告" },
      }),
    ),
  ).runTurn([user("不要把「撰写报告」重命名为「提交报告」。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.doesNotMatch(turn.reply, /已将.*(?:改名|重命名)/u);
});

test("a rename cannot change unrequested status, metadata, or another record", async (t) => {
  const report = {
    id: "one",
    title: "准备简报",
    done: false,
    priority: "high",
    workload: "medium",
  };
  const other = { id: "two", title: "发送演示", done: true, priority: "low" };
  const { tasks, saved } = await fixture(t, [report, other]);
  const turn = await createAgent(
    tasks,
    scripted(
      proposal("lookup", "list_tasks"),
      proposal("broaden", "update_task", {
        candidateIds: [report.id],
        changes: { title: "校对简报", done: true, priority: "low", workload: "small" },
      }),
    ),
  ).runTurn([user("把「准备简报」改名为校对简报。")]);

  const records = await saved();
  const [after] = records;
  assert.ok(after.title === report.title || after.title === "校对简报");
  assert.deepEqual(records, [{ ...report, title: after.title }, other]);
  if (after.title === report.title) assert.doesNotMatch(turn.reply, /已将.*(?:改名|重命名)/u);
});

test("a provider replay in one turn cannot create a second copy", async (t) => {
  const initial = { id: "other", title: "续签租约", done: true };
  const { tasks, saved } = await fixture(t, [initial]);
  const turn = await createAgent(
    tasks,
    scripted(
      proposal("first", "create_task", { title: "领取车票" }),
      proposal("replayed", "create_task", { title: "领取车票" }),
      stop,
    ),
  ).runTurn([user("帮我记一下领取车票。")]);
  const records = await saved();
  assert.equal(records.length, 2);
  assert.notEqual(records[1].id, initial.id);
  assert.deepEqual(records, [initial, { id: records[1].id, title: "领取车票", done: false }]);
  assert.match(turn.reply, /已(?:经)?添加.{0,8}领取车票/u);
});

test("an explicitly sequenced compound request cannot apply the second mutation first", async (t) => {
  const exam = { id: "exam", title: "预约体检", done: false };
  const report = { id: "report", title: "写周报", done: false };
  const distractor = { id: "other", title: "整理照片", done: false };
  const { tasks, fingerprint, saved } = await fixture(t, [exam, report, distractor]);
  const before = await fingerprint();
  const deleteCall = proposal("delete", "delete_task", { candidateIds: [report.id] });
  const completeCall = proposal("complete", "complete_task", { candidateIds: [exam.id] });
  const input = [user("体检那件事已经办好了，然后把写周报删掉")];
  const reversed = await createAgent(
    tasks,
    scripted(proposal("list", "list_tasks"), {
      ...deleteCall,
      message: {
        ...deleteCall.message,
        tool_calls: [...deleteCall.message.tool_calls, ...completeCall.message.tool_calls],
      },
    }),
  ).runTurn(input);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(reversed.outcomes, []);
  assert.doesNotMatch(reversed.reply, /已完成「|已删除「/u);

  const ordered = await createAgent(
    tasks,
    scripted(proposal("list", "list_tasks"), {
      ...completeCall,
      message: {
        ...completeCall.message,
        tool_calls: [...completeCall.message.tool_calls, ...deleteCall.message.tool_calls],
      },
    }),
  ).runTurn(input);
  assert.deepEqual(await saved(), [{ ...exam, done: true }, distractor]);
  assert.match(ordered.reply, /已完成「预约体检」。\n已删除「写周报」。/u);
});

test("an explicit completion and list does not authorize a create proposal", async (t) => {
  const exam = { id: "exam", title: "预约体检", done: false };
  const { tasks, saved } = await fixture(t, [exam]);
  const turn = await createAgent(
    tasks,
    scripted(proposal("extra", "create_task", { title: "重新预约体检" })),
  ).runTurn([user("把预约体检标记为完成，然后给我看看清单。")]);
  assert.deepEqual(await saved(), [{ ...exam, done: true }]);
  assert.match(turn.reply, /已完成「预约体检」。\n当前清单：\n- 预约体检（已完成）/u);
  assert.doesNotMatch(turn.reply, /已添加/u);
});

test("an explicit sequence checks target order even when both operations share a tool", async (t) => {
  const first = { id: "one", title: "准备简报", done: false };
  const second = { id: "two", title: "发送演示", done: false };
  const { tasks, fingerprint, saved } = await fixture(t, [first, second]);
  const input = [user("把「准备简报」标为完成，然后把「发送演示」标为完成。")];
  const firstCall = proposal("first", "complete_task", { candidateIds: [first.id] });
  const secondCall = proposal("second", "complete_task", { candidateIds: [second.id] });
  const before = await fingerprint();
  const reversed = await createAgent(
    tasks,
    scripted(proposal("list", "list_tasks"), {
      ...secondCall,
      message: {
        ...secondCall.message,
        tool_calls: [...secondCall.message.tool_calls, ...firstCall.message.tool_calls],
      },
    }),
  ).runTurn(input);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(reversed.outcomes, []);

  const ordered = await createAgent(
    tasks,
    scripted(proposal("list", "list_tasks"), {
      ...firstCall,
      message: {
        ...firstCall.message,
        tool_calls: [...firstCall.message.tool_calls, ...secondCall.message.tool_calls],
      },
    }),
  ).runTurn(input);
  assert.deepEqual(await saved(), [
    { ...first, done: true },
    { ...second, done: true },
  ]);
  assert.match(ordered.reply, /已完成「准备简报」。\n已完成「发送演示」。/u);
});

test("a plain named deletion cannot use a different valid ID when its exact target exists", async (t) => {
  const intended = { id: "intended", title: "续签租约", done: false };
  const other = { id: "other", title: "预约体检", done: false };
  const { tasks, fingerprint } = await fixture(t, [intended, other]);
  const before = await fingerprint();
  const result = await createAgent(
    tasks,
    scripted(
      proposal("list", "list_tasks"),
      proposal("wrong", "delete_task", { candidateIds: [other.id] }),
    ),
  ).runTurn([user("把续签租约删掉。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已删除/u);
});

test("a later metadata assignment cannot authorize metadata on an earlier create", async (t) => {
  const task = { id: "badge", title: "领取门禁卡", done: false };
  const { tasks, fingerprint } = await fixture(t, [task]);
  const before = await fingerprint();
  const turn = await createAgent(
    tasks,
    scripted(proposal("create", "create_task", { title: "预约摄影", priority: "high" })),
  ).runTurn([user("添加一条预约摄影，然后把领取门禁卡的优先级改成高。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(turn.outcomes, []);
  assert.doesNotMatch(turn.reply, /已添加/u);
});

test("an exact named completion cannot use the valid ID of a distractor", async (t) => {
  const intended = { id: "intended", title: "续签租约", done: false };
  const other = { id: "other", title: "预约体检", done: false };
  const { tasks, fingerprint } = await fixture(t, [intended, other]);
  const before = await fingerprint();
  const result = await createAgent(
    tasks,
    scripted(
      proposal("list", "list_tasks"),
      proposal("wrong", "complete_task", { candidateIds: [other.id] }),
    ),
  ).runTurn([user("把续签租约标记为完成。")]);
  assert.deepEqual(await fingerprint(), before);
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已完成「/u);
});
