import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { APIConnectionTimeoutError } from "openai";
import { createAgent, type ChatMessage } from "./agent.ts";
import { createTaskAgentApp } from "./index.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

async function setup(t: TestContext) {
  const dir = await mkdtemp(join(tmpdir(), "todo-agent-c2-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const filename = join(dir, "tasks.json");
  return { filename, tasks: createJsonTaskStore(filename) };
}

function call(id: string, name: string, args: string) {
  return { id, type: "function", function: { name, arguments: args } };
}

function proposal(...tool_calls: ReturnType<typeof call>[]) {
  return { finish_reason: "tool_calls", message: { role: "assistant", content: null, tool_calls } };
}

const final = (content: string) => ({
  finish_reason: "stop",
  message: { role: "assistant", content },
});
const user = (content: string): ChatMessage => ({ role: "user", content });

// Replay the two observed provider choices. The archived trace omits final prose.
function listThenStop(
  tasks: ReturnType<typeof createJsonTaskStore>,
  onStop?: () => void,
  text = "已执行。",
) {
  const requests: Parameters<ModelProvider>[0][] = [];
  const agent = createAgent(tasks, async (request) => {
    requests.push(request);
    if (requests.length === 1) return proposal(call("lookup", "list_tasks", "{}"));
    assert.equal(requests.length, 2);
    assert.deepEqual(request.messages.at(-1), {
      role: "tool",
      tool_call_id: "lookup",
      content: JSON.stringify({ ok: true, tool: "list_tasks", tasks: tasks.list() }),
    });
    onStop?.();
    return final(text);
  });
  return { agent, requests };
}

for (const [name, input, done, expectedFailure, expectedReply] of [
  [
    "missing named task",
    "请删掉「预约牙医」这项任务。",
    false,
    "exact_title_not_found",
    "当前清单中没有完全同名的任务，请确认名称。",
  ],
  ["already completed task", "再把买牛奶标记完成", true, undefined, "「买牛奶」已经完成。"],
] as const) {
  test(`a lookup followed by stop explains the ${name} without a write`, async (t) => {
    const { filename, tasks } = await setup(t);
    const initial = [
      { id: "milk", title: "买牛奶", done },
      { id: "parcel", title: "取快递", done: false },
    ];
    await writeFile(filename, JSON.stringify(initial));
    const fileBefore = await stat(filename, { bigint: true });
    const { agent, requests } = listThenStop(tasks);
    const turn = await agent.runTurn([user(input)]);
    assert.equal(requests.length, 2);
    assert.deepEqual(tasks.list(), initial);
    assert.equal(await readFile(filename, "utf8"), JSON.stringify(initial));
    const fileAfter = await stat(filename, { bigint: true });
    assert.deepEqual([fileAfter.ino, fileAfter.mtimeNs], [fileBefore.ino, fileBefore.mtimeNs]);
    assert.deepEqual(turn.outcomes, []);
    assert.equal(turn.failure, expectedFailure);
    assert.ok(turn.reply.split("\n").includes(expectedReply), turn.reply);
    assert.doesNotMatch(turn.reply, /已执行/);
  });
}

test("a stopped lookup reports only what a single, identified task establishes", async (t) => {
  const { filename, tasks } = await setup(t);
  const initial = [
    { id: "done", title: "写报告", done: true },
    { id: "pending", title: "预约牙医", done: false },
    { id: "a", title: "登记完成", done: true },
    { id: "b", title: "完成登记", done: true },
  ];
  await writeFile(filename, JSON.stringify(initial));
  for (const [input, expected] of [
    ["请将“写报告”标为已完成。", "「写报告」已经完成。"],
    ["删除「牙医」", "请告诉我你指的是哪一项任务。"],
    ["删掉买菜", "请告诉我你指的是哪一项任务。"],
    ["完成登记完成", "请告诉我你指的是哪一项任务。"],
    ["请把预约牙医标记完成", "尚未执行请求的操作。"],
    ["删除写报告", "尚未执行请求的操作。"],
  ]) {
    const turn = await listThenStop(tasks).agent.runTurn([user(input)]);
    assert.ok(turn.reply.includes(expected), input + " " + turn.reply);
    assert.deepEqual(turn.outcomes, []);
    assert.deepEqual(tasks.list(), initial);
  }
  for (const [input, expected] of [
    ["将写报告标为完成", "「写报告」已经完成。"],
    ["请移除“买面包”", "当前清单中没有完全同名的任务，请确认名称。"],
  ]) {
    const turn = await listThenStop(tasks, undefined, "需要澄清").agent.runTurn([user(input)]);
    assert.ok(turn.reply.includes(expected), turn.reply);
  }
  for (const input of [
    "不要把写报告标记完成",
    "把写报告标记完成？",
    "先列清单，再删除「面包」",
    "删除「面包」或「牙医」",
  ]) {
    const turn = await listThenStop(tasks).agent.runTurn([user(input)]);
    assert.doesNotMatch(turn.reply, /已经完成。|找不到指定的任务|没有完全同名|已删除|已执行/);
    if (input === "不要把写报告标记完成") assert.equal(turn.failure, undefined);
    assert.deepEqual(turn.outcomes, []);
    assert.deepEqual(tasks.list(), initial);
  }
  assert.equal(await readFile(filename, "utf8"), JSON.stringify(initial));
});

test("both no-match paths report exact-title absence when a semantic alias remains possible", async (t) => {
  const { filename, tasks } = await setup(t);
  const alias = tasks.create("联系口腔诊所约诊");
  const before = await readFile(filename, "utf8");
  const input = [user("请删掉「预约牙医」这项任务。")];
  for (const answer of [
    final("没有相关的任务。"),
    proposal(call("no-match", "delete_task", '{"candidateIds":[]}')),
  ]) {
    let requests = 0;
    const agent = createAgent(tasks, async () =>
      requests++ === 0 ? proposal(call("lookup", "list_tasks", "{}")) : answer,
    );
    const turn = await agent.runTurn(input);
    assert.equal(requests, 2);
    assert.equal(turn.failure, "exact_title_not_found");
    assert.match(turn.reply, /^当前清单中没有完全同名的任务，请确认名称。$/m);
    assert.doesNotMatch(turn.reply, /没有相关的任务|找不到指定的任务|已删除/);
    assert.deepEqual(turn.outcomes, []);
    assert.deepEqual(tasks.list(), [alias]);
    assert.equal(await readFile(filename, "utf8"), before);
  }
});

test("empty deletion candidates ask for clarification when the exact named task exists", async (t) => {
  const { filename, tasks } = await setup(t);
  const task = tasks.create("预约牙医");
  const before = await readFile(filename, "utf8");
  let requests = 0;
  const agent = createAgent(tasks, async () =>
    requests++ === 0
      ? proposal(call("lookup", "list_tasks", "{}"))
      : proposal(call("no-match", "delete_task", '{"candidateIds":[]}')),
  );
  const turn = await agent.runTurn([user("请删掉「预约牙医」这项任务。")]);
  assert.equal(requests, 2);
  assert.equal(turn.failure, "reference_required");
  assert.match(turn.reply, /^请告诉我你指的是哪一项任务。$/m);
  assert.doesNotMatch(turn.reply, /没有完全同名|找不到指定的任务|已删除/);
  assert.deepEqual(turn.outcomes, []);
  assert.deepEqual(tasks.list(), [task]);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("lookup ambiguity and a changed view never establish a current task status", async (t) => {
  const { filename, tasks } = await setup(t);
  const original = { id: "one", title: "写报告", done: true };
  const duplicate = { ...original, id: "two" };
  await writeFile(filename, JSON.stringify([original, duplicate]));
  const input = [user("把写报告标为完成")];
  const ambiguous = await listThenStop(tasks).agent.runTurn(input);
  assert.equal(ambiguous.failure, "ambiguous_reference");
  assert.deepEqual(ambiguous.outcomes, []);
  const stale = await listThenStop(tasks, () => {
    tasks.remove(duplicate.id);
  }).agent.runTurn(input);
  assert.equal(stale.failure, "stale_view");
  assert.equal(stale.displayedList, undefined);
  assert.deepEqual(stale.outcomes, []);
  assert.deepEqual(tasks.list(), [original]);
});

test("a stop without a lookup cannot establish that a named task is missing or already done", async (t) => {
  const { tasks } = await setup(t);
  const task = tasks.create("买牛奶");
  tasks.update(task.id, { done: true });
  const agent = createAgent(tasks, async () => final("已经完成。找不到指定的任务。"));
  for (const input of ["请移除「写报告」", "再把买牛奶标记完成"]) {
    const turn = await agent.runTurn([user(input)]);
    assert.deepEqual(turn.outcomes, []);
    assert.doesNotMatch(turn.reply, /已经完成|找不到/);
    assert.match(turn.reply, /本轮未修改任务/);
  }
  assert.deepEqual(tasks.list(), [{ ...task, done: true }]);
});

test("a stop after a saved create and a lookup reports only the committed part of a mixed turn", async (t) => {
  const { tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const script = [
    proposal(call("add", "create_task", '{"title":"取快递"}')),
    proposal(call("list", "list_tasks", "{}")),
    final("两个任务都办好了。"),
  ];
  const turn = await createAgent(tasks, async () => script.shift()).runTurn([
    user("加一条取快递，再把买牛奶标记完成"),
  ]);
  const parcel = tasks.list()[1];
  assert.deepEqual(tasks.list(), [milk, { id: parcel.id, title: "取快递", done: false }]);
  assert.deepEqual(turn.outcomes, [{ operation: "create_task", task: parcel }]);
  assert.match(turn.reply, /已添加待办「取快递」。/);
  assert.match(turn.reply, /本轮只执行了上面确认的更改。/);
  assert.doesNotMatch(turn.reply, /已完成「买牛奶」|两个任务都办好了/);
  assert.equal(script.length, 0);
});

test("terminal inspection reports a failed read and respects cancellation after a lookup", async (t) => {
  const { tasks } = await setup(t);
  const task = tasks.create("写报告");
  tasks.update(task.id, { done: true });
  const input = [user("把写报告标记完成")];
  const controller = new AbortController();
  const cancelled = await listThenStop(tasks, () => controller.abort()).agent.runTurn(
    input,
    controller.signal,
  );
  assert.equal(cancelled.failure, "deadline");
  let readFails = false;
  const store = {
    ...tasks,
    list() {
      if (readFails) throw new Error("synthetic read failure");
      return tasks.list();
    },
  };
  const failed = await listThenStop(store, () => {
    readFails = true;
  }).agent.runTurn(input);
  assert.equal(failed.failure, "storage_error");
  assert.equal(failed.displayedList, undefined);
  assert.deepEqual(failed.outcomes, []);
  assert.deepEqual(tasks.list(), [{ ...task, done: true }]);
});

test("the production app composition shares chat writes with the task API", async (t) => {
  const { tasks } = await setup(t);
  const app = createTaskAgentApp(tasks, async (request) => {
    return request.messages.at(-1)?.role === "tool"
      ? final("已添加。")
      : proposal(call("production_create", "create_task", '{"title":"买牛奶"}'));
  });
  const server = app.listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const url = `http://127.0.0.1:${address.port}`;
  const chat = await fetch(url + "/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [user("帮我记一下买牛奶")] }),
  });
  assert.equal(chat.status, 200);
  assert.deepEqual(await chat.json(), { reply: "已添加待办「买牛奶」。" });
  const rest = await fetch(url + "/api/tasks");
  assert.equal(rest.status, 200);
  const [task] = tasks.list();
  assert.deepEqual(task, { id: task.id, title: "买牛奶", done: false });
  assert.deepEqual(await rest.json(), [task]);
});

test("create sends DeepSeek the saved task under the matching call ID", async (t) => {
  const { tasks } = await setup(t);
  const requests: Parameters<ModelProvider>[0][] = [];
  const toolCall = call("call_milk", "create_task", '{"title":" 买牛奶 "}');
  const agent = createAgent(tasks, async (request) => {
    requests.push(request);
    return requests.length === 1 ? proposal(toolCall) : final("已添加。");
  });
  const input = { role: "user", content: "帮我记一下买牛奶" } as const;
  const result = await agent.runTurn([input]);
  const [task] = tasks.list();
  assert.deepEqual(task, { id: task.id, title: "买牛奶", done: false });
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[0].messages.at(-1), input);
  assert.deepEqual(
    requests[0].tools?.map((tool) => "function" in tool && tool.function.name).sort(),
    ["complete_task", "create_task", "delete_task", "list_tasks", "update_task"],
  );
  assert.deepEqual(requests[1].messages.slice(-2), [
    { role: "assistant", content: null, tool_calls: [toolCall] },
    {
      role: "tool",
      tool_call_id: "call_milk",
      content: JSON.stringify({ ok: true, tool: "create_task", task }),
    },
  ]);
  assert.deepEqual(result, {
    reply: "已添加待办「买牛奶」。",
    outcomes: [{ operation: "create_task", task }],
  });
});

test("the agent reports saved metadata for Chinese create, update and list requests", async (t) => {
  const { tasks, filename } = await setup(t);
  const created = await createAgent(tasks, async (request) => {
    if (request.messages.at(-1)?.role === "tool") return final("好。");
    return proposal(
      call("new", "create_task", '{"title":"写报告","priority":"high","workload":"large"}'),
    );
  }).runTurn([user("帮我记一下写报告，优先级高，工作量大")]);
  const [report] = tasks.list();
  assert.deepEqual(report, {
    id: report.id,
    title: "写报告",
    done: false,
    priority: "high",
    workload: "large",
  });
  assert.deepEqual(created.outcomes, [{ operation: "create_task", task: report }]);
  assert.equal(created.reply.split("\n")[0], "已添加待办「写报告」。");
  assert.match(created.reply, /优先级：高.*工作量：大/);

  const update = async () => {
    const script = [
      proposal(call("list", "list_tasks", "{}")),
      proposal(
        call(
          "edit",
          "update_task",
          JSON.stringify({ candidateIds: [report.id], changes: { workload: "small" } }),
        ),
      ),
      final("好。"),
    ];
    return createAgent(tasks, async () => script.shift()).runTurn([user("把写报告的工作量改成小")]);
  };
  const updated = await update();
  const stored = { ...report, workload: "small" };
  assert.deepEqual(updated.outcomes, [{ operation: "update_task", task: stored }]);
  assert.equal(updated.reply.split("\n")[0], "已更新「写报告」。");
  assert.match(updated.reply, /优先级：高.*工作量：小/);
  assert.deepEqual(createJsonTaskStore(filename).list(), [stored]);

  const bytes = await readFile(filename, "utf8");
  const unchanged = await update();
  assert.deepEqual(unchanged.outcomes, []);
  assert.match(unchanged.reply, /未变化/);
  assert.doesNotMatch(unchanged.reply, /已更新/);
  assert.equal(await readFile(filename, "utf8"), bytes);

  const listed = await createAgent(tasks, async () => {
    throw new Error("direct list must not use the model");
  }).runTurn([user("我有哪些任务？")]);
  assert.deepEqual(listed.displayedList, [stored]);
  assert.match(listed.reply, /写报告（待完成.*优先级：高.*工作量：小/);
});

test("metadata edits require a list delivered to the provider and a named contextual target", async (t) => {
  const { tasks, filename } = await setup(t);
  const milk = tasks.create("买牛奶");
  const parcel = tasks.create("取快递");
  const before = await readFile(filename, "utf8");
  const read = call("list", "list_tasks", "{}");
  const write = call(
    "edit",
    "update_task",
    JSON.stringify({ candidateIds: [milk.id], changes: { priority: "high" } }),
  );
  const combined = await createAgent(tasks, async () => proposal(read, write)).runTurn([
    user("把买牛奶的优先级改成高"),
  ]);
  assert.equal(combined.failure, "list_required");
  assert.deepEqual(combined.outcomes, []);

  const history: ChatMessage[] = [
    user("我有哪些任务？"),
    { role: "assistant", content: "当前清单：\n- 买牛奶（待完成）\n- 取快递（待完成）" },
  ];
  for (const input of [
    "把最后那个改成高优先级",
    "将它的工作量设为大",
    "把第二项的优先级改为高",
    "给第二条设为高优先级",
    "提高刚刚那个的优先级",
  ]) {
    const script = [proposal(read), proposal(write), final("好了。")];
    const result = await createAgent(tasks, async () => script.shift()).runTurn([
      ...history,
      user(input),
    ]);
    assert.equal(result.failure, "reference_required", input);
    assert.deepEqual(result.outcomes, []);
    assert.equal(script.length, 3);
  }
  assert.deepEqual(tasks.list(), [milk, parcel]);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("a list reply reflects stored tasks even when the final model text is wrong", async (t) => {
  const { filename, tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const parcel = { ...tasks.create("取快递"), done: true };
  tasks.update(parcel.id, { done: true });
  const before = await readFile(filename, "utf8");
  const requests: Parameters<ModelProvider>[0][] = [];
  const agent = createAgent(tasks, async (request) => {
    requests.push(request);
    return requests.length === 1
      ? proposal(call("call_list", "list_tasks", "{}"))
      : final("当前电话账单待支付。");
  });

  const turn = await agent.runTurn([user("我手头都记了些什么？")]);
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[1].messages.at(-1), {
    role: "tool",
    tool_call_id: "call_list",
    content: JSON.stringify({ ok: true, tool: "list_tasks", tasks: [milk, parcel] }),
  });
  assert.deepEqual(turn.displayedList, [milk, parcel]);
  assert.deepEqual(turn.outcomes, []);
  assert.match(turn.reply, /买牛奶（待完成）/);
  assert.match(turn.reply, /取快递（已完成）/);
  assert.doesNotMatch(turn.reply, /电话账单/);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("a priority list shows only current matches and distinguishes no matches from an empty store", async (t) => {
  const { filename, tasks } = await setup(t);
  const high = tasks.create("交报告", { priority: "high", workload: "large" });
  tasks.create("取快递", { priority: "medium", workload: "small" });
  tasks.create("未分类");
  const before = await readFile(filename, "utf8");
  const { agent, requests } = listThenStop(tasks);
  const turn = await agent.runTurn([user("有哪些高优先级任务？")]);
  assert.deepEqual(turn.displayedList, [high]);
  assert.match(turn.reply, /交报告（待完成.*优先级：高.*工作量：大/);
  assert.doesNotMatch(turn.reply, /取快递|未分类/);
  const empty = await agent.runTurn([user("请列出低优先级的待办")]);
  assert.deepEqual(empty.displayedList, []);
  assert.match(empty.reply, /没有低优先级/);
  assert.deepEqual([turn.outcomes, empty.outcomes], [[], []]);
  assert.equal(requests.length, 0);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("a list-only follow-up returns the saved list when the model would stop without tools", async (t) => {
  const { filename, tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  tasks.update(milk.id, { done: true });
  const before = await readFile(filename, "utf8");
  const agent = createAgent(tasks, async () => final("收到。"));

  const turn = await agent.runTurn([
    user("我有哪些任务？"),
    { role: "assistant", content: "当前清单：\n- 买牛奶（待完成）\n- 取快递（待完成）" },
    user("再给我看看清单"),
  ]);

  assert.equal(turn.reply, "当前清单：\n- 买牛奶（已完成）");
  assert.deepEqual(turn.displayedList, [{ ...milk, done: true }]);
  assert.deepEqual(turn.outcomes, []);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("explicit list commands and questions show both saved statuses without allowing a write", async (t) => {
  const { filename, tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const parcel = tasks.create("取快递");
  tasks.update(parcel.id, { done: true });
  const before = await readFile(filename, "utf8");
  let requests = 0;
  const agent = createAgent(tasks, async (request) => {
    requests++;
    return request.messages.at(-1)?.role === "tool"
      ? final("收到。")
      : proposal(call("wrong-write", "create_task", '{"title":"意外新增"}'));
  });

  for (const input of [
    "我有哪些任务？",
    "请问我现在都有哪些待办事项？",
    "我的任务有哪些？",
    "请列出我的待办",
    "麻烦再给我查看一下任务列表吧！",
    "帮我看下当前的待办清单",
    "显示所有任务",
  ]) {
    const turn = await agent.runTurn([user(input)]);
    assert.equal(turn.reply, "当前清单：\n- 买牛奶（待完成）\n- 取快递（已完成）", input);
    assert.deepEqual(turn.displayedList, [milk, { ...parcel, done: true }], input);
    assert.deepEqual(turn.outcomes, [], input);
    assert.equal(await readFile(filename, "utf8"), before, input);
  }
  assert.equal(requests, 0);
});

test("a local list distinguishes empty storage, a failed read and a cancelled turn", async (t) => {
  const { filename, tasks } = await setup(t);
  const agent = createAgent(tasks, async () => final("收到。"));
  const input = [user("看看清单")];
  assert.deepEqual(await agent.runTurn(input), {
    reply: "当前没有待办。",
    displayedList: [],
    outcomes: [],
  });
  await assert.rejects(readFile(filename), { code: "ENOENT" });

  await writeFile(filename, "{");
  const failed = await agent.runTurn(input);
  assert.equal(failed.failure, "storage_error");
  assert.match(failed.reply, /任务存储暂时不可用/);
  assert.equal(failed.displayedList, undefined);
  assert.deepEqual(failed.outcomes, []);
  const cancelled = await agent.runTurn(input, AbortSignal.abort());
  assert.equal(cancelled.failure, "deadline");
  assert.equal(cancelled.displayedList, undefined);
  assert.equal(await readFile(filename, "utf8"), "{");
});

test("mixed, negated, scoped and ambiguous requests stay with the model", async (t) => {
  const { filename, tasks } = await setup(t);
  tasks.create("买牛奶");
  const before = await readFile(filename, "utf8");
  const received: Parameters<ModelProvider>[0][] = [];
  const agent = createAgent(tasks, async (request) => {
    received.push(request);
    return final("需要澄清");
  });
  const inputs = [
    "看看清单，再把买牛奶标为完成",
    "别给我看清单",
    "查看清单，不用列出来",
    "请列出未完成的任务",
    "我的清单有哪些问题？",
    "看看那个清单",
    "清单？",
    "删掉清单里的那一条",
  ];
  for (const input of inputs) {
    const turn = await agent.runTurn([user(input)]);
    assert.equal(turn.failure, "ambiguous_reference", input);
    assert.equal(turn.displayedList, undefined, input);
    assert.deepEqual(turn.outcomes, [], input);
  }
  assert.deepEqual(
    received.map((request) => request.messages.at(-1)),
    inputs.map(user),
  );
  assert.equal(await readFile(filename, "utf8"), before);
});

test("mentioning a list command in a task title still creates exactly the requested task", async (t) => {
  const { tasks } = await setup(t);
  const agent = createAgent(tasks, async (request) =>
    request.messages.at(-1)?.role === "tool"
      ? final("收到。")
      : proposal(call("create", "create_task", '{"title":"查看清单"}')),
  );
  const turn = await agent.runTurn([user("帮我记一条查看清单")]);
  const [saved] = tasks.list();
  assert.deepEqual(tasks.list(), [{ id: saved.id, title: "查看清单", done: false }]);
  assert.deepEqual(turn.outcomes, [{ operation: "create_task", task: saved }]);
  assert.equal(turn.reply, "已添加待办「查看清单」。");
});

test("a list command can be a task name when the preceding reply asks which task to delete", async (t) => {
  const { filename, tasks } = await setup(t);
  const target = tasks.create("查看清单");
  const before = await readFile(filename, "utf8");
  const deletion = user("帮我删一条待办");
  const question = await createAgent(tasks, async () => final("需要澄清")).runTurn([deletion]);
  const history: ChatMessage[] = [deletion, { role: "assistant", content: question.reply }];
  const agent = createAgent(tasks, async () =>
    proposal(call("guess", "delete_task", JSON.stringify({ candidateIds: [target.id] }))),
  );

  const browse = await agent.runTurn([...history, user("再给我看看清单")]);
  assert.deepEqual(browse.displayedList, [target]);
  const answer = await agent.runTurn([...history, user("查看清单")]);
  assert.equal(
    answer.failure,
    "list_required",
    "The model must see the list before proposing an ID",
  );
  assert.deepEqual(answer.outcomes, []);
  assert.equal(await readFile(filename, "utf8"), before);

  const resolvingAgent = createAgent(tasks, async (request) => {
    const last = request.messages.at(-1);
    if (last?.role !== "tool") return proposal(call("list", "list_tasks", "{}"));
    if (last.tool_call_id !== "list") return final("收到。");
    assert.ok(typeof last.content === "string");
    const [listed] = JSON.parse(last.content).tasks;
    return proposal(call("delete", "delete_task", JSON.stringify({ candidateIds: [listed.id] })));
  });
  const resolved = await resolvingAgent.runTurn([...history, user("查看清单")]);
  assert.equal(resolved.reply, "已删除「查看清单」。");
  assert.deepEqual(resolved.outcomes, [{ operation: "delete_task", task: target }]);
  assert.deepEqual(tasks.list(), []);
});

test("clarification after observing a list is shown without any task changes", async (t) => {
  const { tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const oatMilk = tasks.create("买燕麦奶");
  let requests = 0;
  const agent = createAgent(tasks, async () =>
    requests++ === 0 ? proposal(call("ambiguous-list", "list_tasks", "{}")) : final("需要澄清"),
  );

  const result = await agent.runTurn([user("奶买好了")]);
  assert.equal(result.failure, "ambiguous_reference");
  assert.deepEqual(result.displayedList, [milk, oatMilk]);
  assert.deepEqual(result.outcomes, []);
  assert.match(result.reply, /请具体说明你指的是哪一项任务/);
  assert.deepEqual(tasks.list(), [milk, oatMilk]);
});

test("unverified model text cannot claim a successful operation", async (t) => {
  const { tasks } = await setup(t);
  const agent = createAgent(tasks, async () => final("已添加待办「买牛奶」。"));
  const result = await agent.runTurn([user("帮我记一下买牛奶")]);
  assert.doesNotMatch(result.reply, /已添加/);
  assert.match(result.reply, /本轮未修改任务/);
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(tasks.list(), []);
});

test("a model closing question does not turn a saved task into a partial failure", async (t) => {
  const { tasks } = await setup(t);
  const agent = createAgent(tasks, async (request) =>
    request.messages.at(-1)?.role === "tool"
      ? final("还需要帮忙吗？")
      : proposal(call("create", "create_task", '{"title":"买牛奶"}')),
  );
  const turn = await agent.runTurn([user("帮我记一下买牛奶")]);
  assert.equal(turn.reply, "已添加待办「买牛奶」。");
  assert.equal(turn.failure, undefined);
  assert.deepEqual(turn.outcomes, [{ operation: "create_task", task: tasks.list()[0] }]);
  assert.equal(tasks.list().length, 1);
});

test("unbound contextual mutations clarify before a model can select a replacement ID", async (t) => {
  const { tasks, filename } = await setup(t);
  const removed = tasks.create("取快递");
  tasks.remove(removed.id);
  const replacement = tasks.create("取快递");
  const before = await readFile(filename, "utf8");
  let requests = 0;
  const agent = createAgent(tasks, async (request) => {
    requests++;
    const last = request.messages.at(-1);
    if (last?.role !== "tool") return proposal(call("list", "list_tasks", "{}"));
    return last.tool_call_id === "list"
      ? proposal(call("delete", "delete_task", JSON.stringify({ candidateIds: [replacement.id] })))
      : final("收到。");
  });
  const prior: ChatMessage[] = [
    user("显示我的待办"),
    { role: "assistant", content: "当前清单：\n- 取快递（待完成）" },
  ];
  for (const input of [
    "把它删掉",
    "删掉第二个",
    "请把这条标记为已完成。",
    "把这项标为完成",
    "请将它标记为完成",
  ]) {
    const turn = await agent.runTurn([...prior, user(input)]);
    assert.equal(turn.failure, "reference_required", input);
    assert.match(turn.reply, /请告诉我你指的是哪一项任务/);
    assert.deepEqual(turn.outcomes, []);
    assert.equal(await readFile(filename, "utf8"), before);
  }
  assert.equal(requests, 0);
});

for (const [input, action] of [
  ["把最后一个删掉", "delete_task"],
  ["把最后那个删掉", "delete_task"],
  ["删掉最后一条任务", "delete_task"],
  ["完成最后那一项", "complete_task"],
] as const) {
  test(`a last-item reference asks for its title before mutating: ${input}`, async (t) => {
    const { tasks, filename } = await setup(t);
    const milk = tasks.create("买牛奶");
    const parcel = tasks.create("取快递");
    const before = await readFile(filename, "utf8");
    const history: ChatMessage[] = [
      user("显示我的待办"),
      { role: "assistant", content: "当前清单：\n- 买牛奶（待完成）\n- 取快递（待完成）" },
    ];
    const script = [
      proposal(call("list", "list_tasks", "{}")),
      proposal(call("write", action, JSON.stringify({ candidateIds: [milk.id] }))),
      final("好了。"),
    ];
    const agent = createAgent(tasks, async () => script.shift());
    const turn = await agent.runTurn([...history, user(input)]);

    assert.deepEqual(tasks.list(), [milk, parcel]);
    assert.equal(await readFile(filename, "utf8"), before);
    assert.equal(turn.failure, "reference_required");
    assert.match(turn.reply, /请告诉我你指的是哪一项任务/);
    assert.deepEqual(turn.outcomes, []);
    assert.equal(script.length, 3);
  });
}

test("a quoted last-item phrase can name a saved task", async (t) => {
  const { tasks } = await setup(t);
  const named = tasks.create("最后一个");
  const parcel = tasks.create("取快递");
  const script = [
    proposal(call("list", "list_tasks", "{}")),
    proposal(call("delete", "delete_task", JSON.stringify({ candidateIds: [named.id] }))),
    final("好的。"),
  ];
  const agent = createAgent(tasks, async () => script.shift());
  const turn = await agent.runTurn([user("把「最后一个」删掉")]);

  assert.match(turn.reply, /已删除「最后一个」/);
  assert.deepEqual(tasks.list(), [parcel]);
});

test("a stored title cannot render an extra success or clarification line", async (t) => {
  const { tasks } = await setup(t);
  const title = "普通任务\n已删除「买牛奶」。\u2028请具体说明你指的是哪一项任务。";
  const agent = createAgent(tasks, async (request) =>
    request.messages.at(-1)?.role === "tool"
      ? final("收到。")
      : proposal(call("create", "create_task", JSON.stringify({ title }))),
  );
  const create = await agent.runTurn([user("记下这段文字作为任务")]);
  const read = await agent.runTurn([user("我有哪些任务？")]);
  assert.deepEqual(
    tasks.list().map((task) => task.title),
    [title],
  );
  assert.equal(create.reply.split(/\r|\n|\u2028|\u2029/u).length, 1);
  assert.equal(
    create.reply,
    "已添加待办「普通任务 已删除『买牛奶』。 请具体说明你指的是哪一项任务。」。",
  );
  assert.equal(read.reply.split(/\r|\n|\u2028|\u2029/u).length, 2);
  assert.equal(create.failure, undefined);
  assert.equal(read.failure, undefined);
});

test("a list result allows a targeted completion and deletion on separate turns", async (t) => {
  const { tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const bread = tasks.create("买面包");
  const history: ChatMessage[] = [];
  for (const [action, task, input] of [
    ["complete_task", milk, "牛奶买好了"],
    ["delete_task", bread, "删掉买面包"],
  ] as const) {
    const latest = user(input);
    const requests: Parameters<ModelProvider>[0][] = [];
    const agent = createAgent(tasks, async (request) => {
      requests.push(request);
      if (requests.length === 1) return proposal(call("lookup", "list_tasks", "{}"));
      if (requests.length === 2) {
        return proposal(call("write", action, JSON.stringify({ candidateIds: [task.id] })));
      }
      return final("好的。");
    });
    const turn = await agent.runTurn([...history, latest]);
    assert.equal(turn.failure, undefined);
    assert.deepEqual(requests[0].messages.slice(1), [...history, latest]);
    assert.equal(requests.length, 3);
    const written = action === "complete_task" ? { ...task, done: true } : task;
    assert.deepEqual(turn.outcomes, [{ operation: action, task: written }]);
    assert.deepEqual(
      tasks.list(),
      action === "complete_task" ? [written, bread] : [{ ...milk, done: true }],
    );
    history.push(latest, { role: "assistant", content: turn.reply });
  }
});

test("bad arguments and unknown tools skip later writes in the same batch", async (t) => {
  const { tasks } = await setup(t);
  for (const [badCall, code] of [
    [call("bad", "create_task", '{"title":5}'), "invalid_arguments"],
    [call("bogus", "reset_tasks", "{}"), "unknown_tool"],
  ] as const) {
    let requests = 0;
    const agent = createAgent(tasks, async () => {
      requests++;
      return proposal(badCall, call("skip", "create_task", '{"title":"买奶油"}'));
    });
    const result = await agent.runTurn([user("帮我记一下买奶油")]);
    assert.equal(result.failure, code);
    assert.equal(requests, 1);
    assert.deepEqual(result.outcomes, []);
    assert.match(result.reply, /本轮未修改任务/);
    assert.deepEqual(tasks.list(), []);
  }
});

test("a write error is reported with no confirmation or persistence changes", async (t) => {
  const { filename, tasks } = await setup(t);
  const original = tasks.create("买牛奶");
  const before = await readFile(filename, "utf8");
  await mkdir(filename + ".tmp");
  const agent = createAgent(tasks, async () =>
    proposal(call("fault", "create_task", '{"title":"买药"}')),
  );
  const result = await agent.runTurn([user("加一个买药")]);
  assert.equal(result.failure, "storage_error");
  assert.deepEqual(result.outcomes, []);
  assert.doesNotMatch(result.reply, /已添加/);
  assert.match(result.reply, /存储.*不可用/);
  assert.deepEqual(tasks.list(), [original]);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("successful calls stay paired and are never replayed when the follow-up model fails", async (t) => {
  const { tasks } = await setup(t);
  const calls = [
    call("milk", "create_task", '{"title":"买牛奶"}'),
    call("bread", "create_task", '{"title":"买面包"}'),
  ];
  const requests: Parameters<ModelProvider>[0][] = [];
  const agent = createAgent(tasks, async (request) => {
    requests.push(request);
    if (requests.length === 1) return proposal(...calls);
    throw new Error("synthetic provider failure");
  });
  const result = await agent.runTurn([user("添加买牛奶和买面包两条待办")]);
  const tasksSaved = tasks.list();
  assert.equal(requests.length, 2);
  assert.deepEqual(
    tasksSaved.map((task) => task.title),
    ["买牛奶", "买面包"],
  );
  assert.deepEqual(requests[1].messages.slice(-3), [
    proposal(...calls).message,
    ...tasksSaved.map((task, i) => ({
      role: "tool",
      tool_call_id: calls[i].id,
      content: JSON.stringify({ ok: true, tool: "create_task", task }),
    })),
  ]);
  assert.equal(result.failure, "model_error");
  assert.deepEqual(
    result.outcomes,
    tasksSaved.map((task) => ({ operation: "create_task", task })),
  );
  assert.match(result.reply, /已添加待办「买牛奶」。[\s\S]*已添加待办「买面包」。/);
  assert.match(result.reply, /变更已保存/);
  assert.doesNotMatch(result.reply, /synthetic provider failure/);
});

test("an SDK timeout after a committed action reports the timeout and keeps the single write", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    if (++requests === 1) return proposal(call("create", "create_task", '{"title":"买牛奶"}'));
    throw new APIConnectionTimeoutError();
  });
  const turn = await agent.runTurn([user("记下买牛奶")]);
  assert.equal(turn.failure, "deadline");
  assert.match(turn.reply, /已添加待办「买牛奶」。\n处理超时。\n以上变更已保存/u);
  const saved = tasks.list();
  assert.equal(saved.length, 1);
  assert.deepEqual(turn.outcomes, [{ operation: "create_task", task: saved[0] }]);
  assert.equal(requests, 2);
});

test("a tool call ID reused with changed arguments cannot create a second task", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () =>
    requests++ === 0
      ? proposal(call("reused_id", "create_task", '{"title":"买牛奶"}'))
      : proposal(call("reused_id", "create_task", '{"title":"买面包"}')),
  );

  const result = await agent.runTurn([user("帮我记一下买牛奶")]);
  const saved = tasks.list();
  assert.equal(result.failure, "invalid_model_response");
  assert.equal(requests, 2);
  assert.deepEqual(
    saved.map((task) => task.title),
    ["买牛奶"],
  );
  assert.deepEqual(result.outcomes, [{ operation: "create_task", task: saved[0] }]);
});

test("an invalid later tool preserves only the earlier confirmed mutation", async (t) => {
  const { tasks } = await setup(t);
  const agent = createAgent(tasks, async () =>
    proposal(
      call("first", "create_task", '{"title":"买牛奶"}'),
      call("bad", "create_task", '{"title":""}'),
      call("last", "create_task", '{"title":"买面包"}'),
    ),
  );
  const result = await agent.runTurn([user("添加三条待办")]);
  const saved = tasks.list();
  assert.equal(result.failure, "invalid_arguments");
  assert.deepEqual(
    saved.map((task) => task.title),
    ["买牛奶"],
  );
  assert.deepEqual(result.outcomes, [{ operation: "create_task", task: saved[0] }]);
  assert.match(result.reply, /变更已保存/);
  assert.doesNotMatch(result.reply, /买面包/);
});

// CANON-03: maxModelRequests raised from 4 to 6 to permit compound multi-mutation turns while preserving the loop cap.
test("tool proposals stop at the sixth model request", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    requests++;
    return requests < 6
      ? proposal(call("list-" + requests, "list_tasks", "{}"))
      : proposal(call("unexecuted", "create_task", '{"title":"买牛奶"}'));
  });
  const result = await agent.runTurn([user("帮我梳理一下手头要做的事")]);
  assert.equal(requests, 6);
  assert.equal(result.failure, "execution_limit");
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(tasks.list(), []);
});

// CANON-03: maxToolCalls raised from 8 to 10; an over-budget batch (5 + 6 = 11) still rejects atomically with 0 prefix writes.
test("a batch exceeding the cumulative tool budget does not execute even its prefix", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    const [start, end] = requests++ === 0 ? [1, 6] : [6, 12];
    const calls = Array.from({ length: end - start }, (_, i) => {
      const step = i + start;
      return call("create-" + step, "create_task", JSON.stringify({ title: "任务 " + step }));
    });
    return proposal(...calls);
  });
  const result = await agent.runTurn([user("记录多件事项")]);
  assert.equal(requests, 2);
  assert.equal(result.failure, "execution_limit");
  assert.deepEqual(
    tasks.list().map((task) => task.title),
    ["任务 1", "任务 2", "任务 3", "任务 4", "任务 5"],
  );
  assert.equal(result.outcomes.length, 5);
});

test("large history drops whole old exchanges and keeps the current input", async (t) => {
  const { tasks } = await setup(t);
  const requests: Parameters<ModelProvider>[0][] = [];
  const agent = createAgent(tasks, async (request) => {
    requests.push(request);
    return final("你好。");
  });
  const old = user("甲".repeat(22_000));
  const oldReply: ChatMessage = { role: "assistant", content: "这是整组移除的回答。" };
  const recent = user("前一个问题");
  const recentReply: ChatMessage = { role: "assistant", content: "前一个回答" };
  const latest = user("打个招呼");
  const greeting = await agent.runTurn([old, oldReply, recent, recentReply, latest]);
  // CANON-06: concise, verified non-mutating Chinese prose is preserved while still enforcing zero mutations and byte limits.
  assert.equal(greeting.reply, "你好。");
  assert.deepEqual(greeting.outcomes, []);
  assert.equal(greeting.displayedList, undefined);
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0].messages.slice(1), [recent, recentReply, latest]);
  assert.ok(Buffer.byteLength(JSON.stringify(requests[0])) <= 64 * 1024);
  const oversizedCurrent = await agent.runTurn([old]);
  assert.equal(oversizedCurrent.failure, "input_limit");
  assert.equal(requests.length, 1);
});

test("oversized list contents never reach the provider or enable a semantic mutation", async (t) => {
  const { filename, tasks } = await setup(t);
  const seed = Array.from({ length: 110 }, (_, i) => ({
    id: "task-" + i,
    title: "甲".repeat(200),
    done: false,
  }));
  const before = JSON.stringify(seed);
  await writeFile(filename, before);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    requests++;
    return proposal(call("look", "list_tasks", "{}"));
  });
  const result = await agent.runTurn([user("完成一个待办")]);
  assert.equal(requests, 1);
  assert.equal(result.failure, "input_limit");
  assert.equal(result.displayedList, undefined);
  assert.deepEqual(result.outcomes, []);
  assert.equal(await readFile(filename, "utf8"), before);
});

test("a mutation cannot use a list that has not been returned to the model yet", async (t) => {
  const { tasks } = await setup(t);
  const milk = tasks.create("买牛奶");
  const agent = createAgent(tasks, async () =>
    proposal(
      call("look", "list_tasks", "{}"),
      call("guess", "delete_task", JSON.stringify({ candidateIds: [milk.id] })),
    ),
  );
  const result = await agent.runTurn([user("把买牛奶删掉")]);
  assert.equal(result.failure, "list_required");
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(tasks.list(), [milk]);
});

test("cancellation stops an active provider request before any tool is dispatched", async (t) => {
  const { tasks } = await setup(t);
  const controller = new AbortController();
  let observedSignal: AbortSignal | undefined;
  const agent = createAgent(tasks, async (_request, signal) => {
    observedSignal = signal;
    controller.abort();
    return new Promise(() => {});
  });
  const result = await agent.runTurn([user("记一下买牛奶")], controller.signal);
  assert.equal(observedSignal?.aborted, true);
  assert.equal(result.failure, "deadline");
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(tasks.list(), []);
});

const metadataSeed = [
  { id: "milk", title: "买牛奶", done: false },
  {
    id: "report",
    title: "写报告",
    done: true,
    priority: "high",
    workload: "small",
    extension: { color: "blue" },
  },
];
const metadataInput = "把「写报告」的工作量改成大";
const metadataCall = (id: string, changes: Record<string, string>) =>
  call("edit", "update_task", JSON.stringify({ candidateIds: [id], changes }));

for (const [label, input, proposals] of [
  [
    "ordinal without provenance",
    "第二条设为高优先级",
    [metadataCall("milk", { priority: "high" })],
  ],
  [
    "combined ordinal",
    "把第二条的优先级改为高，工作量改为大",
    [metadataCall("milk", { priority: "high", workload: "large" })],
  ],
  [
    "combined pronoun",
    "请把它的优先级和工作量分别改为高和大",
    [metadataCall("milk", { priority: "high", workload: "large" })],
  ],
  [
    "quoted title with the wrong valid ID",
    metadataInput,
    [metadataCall("milk", { workload: "large" })],
  ],
  [
    "unrequested priority",
    metadataInput,
    [metadataCall("report", { workload: "large", priority: "low" })],
  ],
  [
    "unrequested workload",
    "把写报告设为低优先级",
    [metadataCall("report", { priority: "low", workload: "large" })],
  ],
  ["wrong enum value", "把写报告设为高优先级", [metadataCall("report", { priority: "low" })]],
  [
    "incomplete combined proposal",
    "把写报告的优先级改为低，工作量改为大",
    [metadataCall("report", { priority: "low" })],
  ],
  [
    "completion substituted for metadata",
    "把买牛奶的工作量改为大",
    [call("finish", "complete_task", '{"candidateIds":["milk"]}')],
  ],
  [
    "creation substituted for metadata",
    metadataInput,
    [call("add", "create_task", '{"title":"其他任务"}')],
  ],
  [
    "deletion bundled with a valid edit",
    metadataInput,
    [
      metadataCall("report", { workload: "large" }),
      call("del", "delete_task", '{"candidateIds":["milk"]}'),
    ],
  ],
  [
    "title in a metadata patch",
    metadataInput,
    [metadataCall("report", { workload: "large", title: "另一个标题" })],
  ],
  ["negated edit", "不要把写报告的工作量改为大", [metadataCall("report", { workload: "large" })]],
  [
    "question about an edit",
    "把写报告的工作量改为大？",
    [metadataCall("report", { workload: "large" })],
  ],
  [
    "unsupported qualified priority edit with substituted deletion",
    "把「写报告」的优先级降到低。",
    [call("del", "delete_task", '{"candidateIds":["milk"]}')],
  ],
  [
    "unsupported workload edit with substituted deletion",
    "把「写报告」的工作量降到小",
    [call("del", "delete_task", '{"candidateIds":["milk"]}')],
  ],
  [
    "verbless workload edit with substituted creation",
    "「写报告」的工作量小一点",
    [call("add", "create_task", '{"title":"其他任务"}')],
  ],
  [
    "read with substituted creation",
    "写报告的工作量是多少？",
    [call("add", "create_task", '{"title":"其他任务"}')],
  ],
  [
    "unresolvable description",
    "把报告的工作量改为大",
    [metadataCall("report", { workload: "large" })],
  ],
  [
    "edit unrelated to the user request",
    "现在几点了？",
    [metadataCall("report", { workload: "large" })],
  ],
] as const) {
  test(`metadata authorization rejects ${label} before writing`, async (t) => {
    const { tasks, filename } = await setup(t);
    const before = JSON.stringify(metadataSeed);
    await writeFile(filename, before);
    const originalFile = await stat(filename, { bigint: true });
    const script = [
      proposal(call("lookup", "list_tasks", "{}")),
      proposal(...proposals),
      final("已修改。"),
    ];
    const turn = await createAgent(tasks, async () => script.shift()).runTurn([
      user("我有哪些任务？"),
      { role: "assistant", content: "当前清单：\n- 买牛奶（待完成）\n- 写报告（已完成）" },
      user(input),
    ]);
    assert.equal(await readFile(filename, "utf8"), before);
    const file = await stat(filename, { bigint: true });
    assert.deepEqual([file.ino, file.mtimeNs], [originalFile.ino, originalFile.mtimeNs]);
    assert.deepEqual(turn.outcomes, []);
    assert.match(turn.reply, /本轮未修改任务/);
    assert.doesNotMatch(turn.reply, /已更新|已删除|已添加|已修改/);
    if (label.includes("ordinal") || label.includes("pronoun")) {
      assert.equal(turn.failure, "reference_required");
      assert.equal(script.length, 3);
    }
  });
}

test("metadata authorization permits both requested edits on one named task", async (t) => {
  const { tasks, filename } = await setup(t);
  for (const input of [
    "把写报告的优先级改为低，工作量改为大。",
    "请把「写报告」的优先级和工作量分别改为低和大",
  ]) {
    await writeFile(filename, JSON.stringify(metadataSeed));
    const edit = { priority: "low", workload: "large" };
    const script = [
      proposal(call("lookup", "list_tasks", "{}")),
      proposal(metadataCall("report", edit)),
      final("好。"),
    ];
    const turn = await createAgent(tasks, async () => script.shift()).runTurn([user(input)]);
    assert.deepEqual(tasks.list(), [metadataSeed[0], { ...metadataSeed[1], ...edit }]);
    assert.equal(turn.outcomes.length, 1);
    assert.match(turn.reply, /已更新「写报告」/);
  }
});

test("metadata authorization preserves duplicate and stale target protections", async (t) => {
  const { tasks, filename } = await setup(t);
  for (const scenario of ["duplicate", "replaced"] as const) {
    await writeFile(
      filename,
      JSON.stringify(
        scenario === "duplicate"
          ? [...metadataSeed, { ...metadataSeed[1], id: "other" }]
          : metadataSeed,
      ),
    );
    let expected = await readFile(filename, "utf8");
    let calls = 0;
    const turn = await createAgent(tasks, async () => {
      if (++calls === 1) return proposal(call("list", "list_tasks", "{}"));
      if (scenario === "replaced") {
        tasks.remove("report");
        tasks.create("写报告");
        expected = await readFile(filename, "utf8");
      }
      return proposal(metadataCall("report", { workload: "large" }));
    }).runTurn([user(metadataInput)]);
    assert.equal(turn.failure, scenario === "duplicate" ? "ambiguous_reference" : "stale_view");
    assert.deepEqual(turn.outcomes, []);
    assert.equal(await readFile(filename, "utf8"), expected);
  }
});

test("metadata authorization keeps create-like and instruction-like task titles intact", async (t) => {
  const { tasks, filename } = await setup(t);
  for (const [title, input] of [
    ["新增用户", "新增用户的优先级改为低"],
    ["工作量改成小", "把「工作量改成小」的优先级改为低。"],
  ]) {
    const seed = [{ ...metadataSeed[0] }, { ...metadataSeed[1], title }];
    for (const write of [
      call("del", "delete_task", '{"candidateIds":["milk"]}'),
      metadataCall("report", { priority: "low" }),
    ]) {
      const before = JSON.stringify(seed);
      await writeFile(filename, before);
      const script = [proposal(call("lookup", "list_tasks", "{}")), proposal(write), final("好。")];
      const turn = await createAgent(tasks, async () => script.shift()).runTurn([user(input)]);
      if (write.function.name === "delete_task") {
        assert.equal(await readFile(filename, "utf8"), before);
        assert.deepEqual(turn.outcomes, []);
      } else {
        assert.deepEqual(tasks.list(), [seed[0], { ...seed[1], priority: "low" }]);
        assert.match(turn.reply, /已更新/);
      }
    }
  }
});

for (const [input, tool] of [
  ["删除「写报告」", "delete_task"],
  ["只删除「写报告」这一条。", "delete_task"],
  ["「写报告」这条删掉吧", "delete_task"],
  ["请把「写报告」标为完成。", "complete_task"],
  ["我说的是“写报告”，请把它标为完成。", "complete_task"],
] as const) {
  test(`quoted task authorization binds ${input} to the stored ID and operation`, async (t) => {
    const { tasks, filename } = await setup(t);
    const seed = [metadataSeed[0], { ...metadataSeed[1], done: false }];
    for (const [name, id] of [
      [tool, "milk"],
      [tool === "delete_task" ? "complete_task" : "delete_task", "report"],
      [tool, "report"],
    ]) {
      const before = JSON.stringify(seed);
      await writeFile(filename, before);
      const script = [
        proposal(call("lookup", "list_tasks", "{}")),
        proposal(call("op", name, JSON.stringify({ candidateIds: [id] }))),
        final("好。"),
      ];
      const turn = await createAgent(tasks, async () => script.shift()).runTurn([user(input)]);
      if (name === tool && id === "report") {
        assert.deepEqual(
          tasks.list(),
          tool === "delete_task" ? [seed[0]] : [seed[0], { ...seed[1], done: true }],
        );
        assert.equal(turn.outcomes.length, 1);
      } else {
        assert.equal(await readFile(filename, "utf8"), before);
        assert.deepEqual(turn.outcomes, []);
        assert.match(turn.reply, /本轮未修改任务/);
      }
    }
  });
}

test("quoted task authorization excludes wrong IDs across unrecognized wording and quote styles", async (t) => {
  const { tasks, filename } = await setup(t);
  const before = JSON.stringify(metadataSeed);
  for (const input of [
    "把「写报告」删了吧。",
    '麻烦清掉 "写报告"。',
    "处理一下 '写报告'。",
    "麻烦清掉『写报告』。",
  ]) {
    await writeFile(filename, before);
    const script = [
      proposal(call("list", "list_tasks", "{}")),
      proposal(call("write", "delete_task", '{"candidateIds":["milk"]}')),
      final("好的。"),
    ];
    const turn = await createAgent(tasks, async () => script.shift()).runTurn([user(input)]);
    assert.equal(await readFile(filename, "utf8"), before, input);
    assert.deepEqual(turn.outcomes, []);
    assert.match(turn.reply, /本轮未修改任务/);
  }
});

test("quoted task authorization permits the literal new title and rejects a substituted one", async (t) => {
  const { tasks, filename } = await setup(t);
  for (const quote of ["「买牛奶」", '"买牛奶"']) {
    for (const title of ["其他事", "买牛奶"]) {
      await writeFile(filename, "[]");
      const script = [
        proposal(call("new", "create_task", JSON.stringify({ title }))),
        final("好了。"),
      ];
      const turn = await createAgent(tasks, async () => script.shift()).runTurn([
        user("帮我记一下" + quote),
      ]);
      if (title === "其他事") {
        assert.equal(await readFile(filename, "utf8"), "[]");
        assert.deepEqual(turn.outcomes, []);
      } else {
        assert.deepEqual(
          tasks.list().map((t) => t.title),
          [title],
        );
        assert.match(turn.reply, /已添加待办「买牛奶」/);
      }
    }
  }
});

test("a second explicit metadata create persists once when the provider would stop without tools", async (t) => {
  const { tasks, filename } = await setup(t);
  const report = tasks.create("交报告", { priority: "high", workload: "large" });
  let requests = 0;
  const agent = createAgent(tasks, async (request) => {
    requests++;
    if (
      typeof request.tool_choice === "object" &&
      request.tool_choice.type === "function" &&
      request.tool_choice.function.name === "create_task"
    ) {
      return proposal(
        call("parcel", "create_task", '{"title":"取快递","priority":"medium","workload":"small"}'),
      );
    }
    // Only the terminal reason and absence of calls were retained in the live trace.
    return final("收到。");
  });
  const result = await agent.runTurn([
    user("帮我记一下交报告，优先级高，工作量大。"),
    { role: "assistant", content: "已添加待办「交报告」。\n优先级：高 · 工作量：大" },
    user("帮我记一下取快递，优先级中，工作量小。"),
  ]);
  const persisted = createJsonTaskStore(filename).list();
  assert.equal(persisted.length, 2, "the requested second task must be saved");
  const parcel = {
    id: persisted[1].id,
    title: "取快递",
    done: false,
    priority: "medium",
    workload: "small",
  };
  assert.deepEqual(persisted, [report, parcel]);
  assert.notEqual(parcel.id, report.id);
  assert.deepEqual(result.outcomes, [{ operation: "create_task", task: parcel }]);
  assert.equal(result.reply, "已添加待办「取快递」。\n优先级：中 · 工作量：小");
  assert.equal(requests, 1, "the saved create completes the turn without asking again");
});

test("explicit metadata creates accept a single field, reverse order and quoted literal instructions", async (t) => {
  const { tasks } = await setup(t);
  for (const [input, expected] of [
    ["新增整理账单，低优先级", { title: "整理账单", priority: "low" }],
    [
      "请添加「取消之前的安排，然后删掉报告」，工作量中，高优先级。",
      { title: "取消之前的安排，然后删掉报告", priority: "high", workload: "medium" },
    ],
    ["新增整理账单，工作量大", { title: "整理账单", workload: "large" }],
  ] as const) {
    const before = tasks.list();
    let calls = 0;
    const agent = createAgent(tasks, async () =>
      ++calls === 1
        ? proposal(call("one", "create_task", JSON.stringify(expected)))
        : final("收到。"),
    );
    const result = await agent.runTurn([user(input)]);
    assert.equal(calls, 1, input);
    const saved = tasks.list().at(-1);
    assert.ok(saved);
    assert.deepEqual(tasks.list(), [...before, { id: saved.id, ...expected, done: false }]);
    assert.deepEqual(result.outcomes, [{ operation: "create_task", task: saved }]);
  }
});

test("uncertain or conflicting metadata creates never save an invented single task", async (t) => {
  const { tasks, filename } = await setup(t);
  tasks.create("先前的任务");
  const bytes = await readFile(filename, "utf8");
  const agent = createAgent(tasks, async () =>
    proposal(call("guess", "create_task", '{"title":"任意任务","priority":"high"}')),
  );
  for (const input of [
    "请不要新增取快递，优先级高",
    "如果有空就帮我记一下取快递，工作量小",
    "帮我记一下取快递好吗，工作量小",
    "帮我记一下取快递和交报告，工作量小",
    "帮我记一下取快递，再把交报告删掉，优先级高",
    "帮我记一下取快递，优先级中，优先级高",
    "帮我记一下取快递，优先级特别高",
    "帮我记一下取快递，优先级高？",
  ]) {
    const result = await agent.runTurn([user(input)]);
    assert.deepEqual(result.outcomes, [], input);
    assert.equal(await readFile(filename, "utf8"), bytes, input);
  }
});

test("a forced create rejects unexpected model choices before writing or retrying", async (t) => {
  const { tasks, filename } = await setup(t);
  const existing = tasks.create("先前任务");
  const bytes = await readFile(filename, "utf8");
  const valid = { title: "交报告", priority: "high" };
  const add = (id: string, args: object) => call(id, "create_task", JSON.stringify(args));
  for (const [choice, failure] of [
    [final("已经添加。"), "action_not_executed"],
    [proposal(add("wrong-task", { ...valid, title: "其他事" })), "reference_required"],
    [proposal(add("wrong-value", { ...valid, priority: "low" })), "metadata_confirmation_required"],
    [proposal(add("extra", { ...valid, workload: "large" })), "metadata_confirmation_required"],
    [proposal(add("omitted", { title: "交报告" })), "metadata_confirmation_required"],
    [proposal(add("first", valid), add("extra", valid)), "invalid_model_response"],
    [
      proposal(add("first", valid), call("extra-list", "list_tasks", "{}")),
      "invalid_model_response",
    ],
    [
      proposal(call("other", "complete_task", JSON.stringify({ candidateIds: [existing.id] }))),
      "invalid_model_response",
    ],
  ] as const) {
    let count = 0;
    const result = await createAgent(tasks, async () => {
      count++;
      return choice;
    }).runTurn([user("帮我记一下交报告，优先级高。")]);
    assert.equal(await readFile(filename, "utf8"), bytes);
    assert.deepEqual(result.outcomes, []);
    assert.equal(result.failure, failure);
    assert.match(result.reply, /本轮未修改任务/);
    assert.equal(count, 1);
  }
});

test("an exact metadata edit returns its saved or unchanged result after the lookup and proposal", async (t) => {
  const { tasks, filename } = await setup(t);
  const report = tasks.create("交报告", { priority: "high", workload: "large" });
  const parcel = tasks.create("取快递", { priority: "medium", workload: "small" });
  const saved = { ...report, priority: "low" };
  for (const changed of [true, false]) {
    const fileBefore = await stat(filename, { bigint: true });
    const requests: Parameters<ModelProvider>[0][] = [];
    const turn = await createAgent(tasks, async (request) => {
      requests.push(request);
      if (requests.length === 1) return proposal(call("view", "list_tasks", "{}"));
      if (requests.length === 2)
        return proposal(
          call(
            "edit",
            "update_task",
            JSON.stringify({ candidateIds: [report.id], changes: { priority: "low" } }),
          ),
        );
      throw new Error("A finished single edit requires no final model request");
    }).runTurn([user("把交报告的优先级改为低。")]);
    assert.deepEqual(createJsonTaskStore(filename).list(), [saved, parcel]);
    assert.deepEqual(turn.outcomes, changed ? [{ operation: "update_task", task: saved }] : []);
    assert.equal(turn.failure, undefined);
    assert.match(turn.reply, changed ? /已更新「交报告」/ : /未变化/);
    assert.match(turn.reply, /优先级：低 · 工作量：大/);
    assert.equal(requests.length, 2);
    const last = requests[1].messages.at(-1);
    assert.ok(last?.role === "tool" && typeof last.content === "string");
    assert.deepEqual(JSON.parse(last.content).tasks, [changed ? report : saved, parcel]);
    if (!changed) {
      const after = await stat(filename, { bigint: true });
      assert.deepEqual([after.ino, after.mtimeNs], [fileBefore.ino, fileBefore.mtimeNs]);
    }
  }
});

test("a named workload edit saves the exact change when automatic selection would stop", async (t) => {
  const { tasks, filename } = await setup(t);
  const report = tasks.create("交报告", { priority: "low", workload: "large" });
  const parcel = tasks.create("取快递", { priority: "medium", workload: "small" });
  const saved = { ...report, workload: "small" };
  const requests: Parameters<ModelProvider>[0][] = [];
  const result = await createAgent(tasks, async (request) => {
    requests.push(request);
    const choice = request.tool_choice;
    if (typeof choice !== "object" || choice?.type !== "function") return final("收到。");
    if (choice.function.name === "list_tasks") return proposal(call("view", "list_tasks", "{}"));
    assert.equal(choice.function.name, "update_task");
    const last = request.messages.at(-1);
    assert.ok(last?.role === "tool" && typeof last.content === "string");
    const lookup = JSON.parse(last.content);
    assert.deepEqual(lookup.tasks, [report, parcel]);
    return proposal(metadataCall(report.id, { workload: "small" }));
  }).runTurn([
    user("帮我记一下交报告，优先级高，工作量大。"),
    { role: "assistant", content: "已添加待办「交报告」。\n优先级：高 · 工作量：大" },
    user("帮我记一下取快递，优先级中，工作量小。"),
    { role: "assistant", content: "已添加待办「取快递」。\n优先级：中 · 工作量：小" },
    user("有哪些高优先级任务？"),
    { role: "assistant", content: "当前清单：\n- 交报告（待完成 · 优先级：高 · 工作量：大）" },
    user("把交报告的优先级改为低。"),
    { role: "assistant", content: "已更新「交报告」。\n优先级：低 · 工作量：大" },
    user("把交报告的工作量改成小。"),
  ]);
  assert.deepEqual(createJsonTaskStore(filename).list(), [saved, parcel]);
  assert.deepEqual(result.outcomes, [{ operation: "update_task", task: saved }]);
  assert.equal(result.reply, "已更新「交报告」。\n优先级：低 · 工作量：小");
  assert.equal(requests.length, 2);
});

test("a forced metadata lookup or update cannot turn a prose response into a write", async (t) => {
  const { tasks, filename } = await setup(t);
  tasks.create("交报告", { priority: "low", workload: "large" });
  const before = await readFile(filename, "utf8");
  const original = await stat(filename, { bigint: true });
  for (const stopRound of [1, 2]) {
    let requests = 0;
    const turn = await createAgent(tasks, async () =>
      ++requests === stopRound ? final("已经修改。") : proposal(call("lookup", "list_tasks", "{}")),
    ).runTurn([user("把交报告的工作量改成小。")]);
    assert.equal(requests, stopRound);
    assert.equal(turn.failure, "action_not_executed");
    assert.deepEqual(turn.outcomes, []);
    assert.match(turn.reply, /本轮未修改任务/);
    assert.doesNotMatch(turn.reply, /已更新|已经修改/);
    assert.equal(await readFile(filename, "utf8"), before);
    const after = await stat(filename, { bigint: true });
    assert.deepEqual([after.ino, after.mtimeNs], [original.ino, original.mtimeNs]);
  }
});

test("a named demonstrative delete removes the intended task when automatic selection would stop", async (t) => {
  const { tasks, filename } = await setup(t);
  const report = tasks.update(tasks.create("交报告", { priority: "low", workload: "small" }).id, {
    done: true,
  })!;
  const parcel = tasks.create("取快递", { priority: "medium", workload: "small" });
  const history: ChatMessage[] = [
    user("帮我记一下交报告，优先级高，工作量大。"),
    { role: "assistant", content: "已添加待办「交报告」。\n优先级：高 · 工作量：大" },
    user("帮我记一下取快递，优先级中，工作量小。"),
    { role: "assistant", content: "已添加待办「取快递」。\n优先级：中 · 工作量：小" },
    user("有哪些高优先级任务？"),
    { role: "assistant", content: "高优先级清单：\n- 交报告（待完成 · 优先级：高 · 工作量：大）" },
    user("把交报告的优先级改为低。"),
    { role: "assistant", content: "已更新「交报告」。\n优先级：低 · 工作量：大" },
    user("把交报告的工作量改成小。"),
    { role: "assistant", content: "已更新「交报告」。\n优先级：低 · 工作量：小" },
    user("报告交好了。"),
    { role: "assistant", content: "已完成「交报告」。" },
    user("再给我看看清单。"),
    {
      role: "assistant",
      content:
        "当前清单：\n- 交报告（已完成 · 优先级：低 · 工作量：小）\n- 取快递（待完成 · 优先级：中 · 工作量：小）",
    },
    user("把取快递这条删掉吧。"),
  ];
  const requests: Parameters<ModelProvider>[0][] = [];
  const turn = await createAgent(tasks, async (request) => {
    requests.push(request);
    const choice = request.tool_choice;
    if (typeof choice !== "object" || choice?.type !== "function") return final("收到。");
    if (choice.function.name === "list_tasks") {
      assert.deepEqual(request.messages.slice(1), history);
      return proposal(call("lookup", "list_tasks", "{}"));
    }
    assert.equal(choice.function.name, "delete_task");
    const lookup = request.messages.at(-1);
    assert.ok(lookup?.role === "tool" && typeof lookup.content === "string");
    assert.equal(lookup.tool_call_id, "lookup");
    assert.deepEqual(JSON.parse(lookup.content).tasks, [report, parcel]);
    return proposal(call("delete", "delete_task", JSON.stringify({ candidateIds: [parcel.id] })));
  }).runTurn(history);

  assert.deepEqual(createJsonTaskStore(filename).list(), [report]);
  assert.deepEqual(turn.outcomes, [{ operation: "delete_task", task: parcel }]);
  assert.equal(turn.reply, "已删除「取快递」。");
  assert.equal(requests.length, 2);
});

test("a demonstrative delete with ambiguous, missing, or invalid authorization never writes", async (t) => {
  const { filename, tasks } = await setup(t);
  const report = tasks.create("交报告");
  const parcel = tasks.create("取快递");
  const alternate = { ...parcel, id: "alternative", title: "取快递这条" };
  const duplicate = { ...parcel, id: "duplicate" };
  const inputs = [
    { name: "suffix collision", tasks: [report, parcel, alternate], requests: 1 },
    { name: "suffixed name alone", tasks: [report, alternate], requests: 1 },
    { name: "duplicate", tasks: [report, parcel, duplicate], requests: 1 },
    { name: "missing", tasks: [report], requests: 1 },
    { name: "wrong candidate", tasks: [report, parcel], candidate: report.id, requests: 2 },
    { name: "provider stop", tasks: [report, parcel], requests: 2 },
  ];
  for (const scenario of inputs) {
    const bytes = JSON.stringify(scenario.tasks);
    await writeFile(filename, bytes);
    let requests = 0;
    const turn = await createAgent(tasks, async () => {
      if (++requests === 1) return proposal(call("lookup", "list_tasks", "{}"));
      if (scenario.candidate) {
        return proposal(
          call("wrong", "delete_task", JSON.stringify({ candidateIds: [scenario.candidate] })),
        );
      }
      return final("已经删除。");
    }).runTurn([user("把取快递这条删掉吧。")]);
    assert.equal(requests, scenario.requests, scenario.name);
    assert.deepEqual(turn.outcomes, [], scenario.name);
    assert.match(turn.reply, /本轮未修改任务/, scenario.name);
    assert.doesNotMatch(turn.reply, /已删除|已经删除/, scenario.name);
    assert.equal(await readFile(filename, "utf8"), bytes, scenario.name);
  }
});

test("contextual labels need a name while the same quoted characters can be a literal new title", async (t) => {
  const { tasks } = await setup(t);
  const existing = tasks.create("交报告");
  for (const title of ["这件事", "刚才说的事"]) {
    for (const quoted of [false, true]) {
      const before = tasks.list();
      const forced: boolean[] = [];
      const turn = await createAgent(tasks, async (request) => {
        forced.push(typeof request.tool_choice === "object");
        return proposal(
          call(
            "write",
            "create_task",
            JSON.stringify({ title, priority: "medium", workload: "small" }),
          ),
        );
      }).runTurn([
        user("我有哪些任务？"),
        { role: "assistant", content: "当前清单：\n- 交报告（待完成）" },
        user("帮我记一下" + (quoted ? "「" + title + "」" : title) + "，优先级中，工作量小。"),
      ]);
      if (quoted) {
        assert.deepEqual(forced, [true]);
        const task = tasks.list().at(-1);
        assert.ok(task && task.id !== existing.id);
        assert.deepEqual(tasks.list(), [
          ...before,
          { id: task.id, title, done: false, priority: "medium", workload: "small" },
        ]);
        assert.equal(turn.outcomes.length, 1);
      } else {
        assert.deepEqual(tasks.list(), before);
        assert.deepEqual(turn.outcomes, []);
        assert.equal(forced.length, 0);
      }
    }
  }
});
