import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
    ["complete_task", "create_task", "delete_task", "list_tasks"],
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

test("tool proposals stop at the fourth model request", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    requests++;
    return requests < 4
      ? proposal(call("list-" + requests, "list_tasks", "{}"))
      : proposal(call("unexecuted", "create_task", '{"title":"买牛奶"}'));
  });
  const result = await agent.runTurn([user("帮我梳理一下手头要做的事")]);
  assert.equal(requests, 4);
  assert.equal(result.failure, "execution_limit");
  assert.deepEqual(result.outcomes, []);
  assert.deepEqual(tasks.list(), []);
});

test("a batch exceeding the cumulative tool budget does not execute even its prefix", async (t) => {
  const { tasks } = await setup(t);
  let requests = 0;
  const agent = createAgent(tasks, async () => {
    const [start, end] = requests++ === 0 ? [1, 6] : [6, 10];
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
  assert.equal(greeting.reply, "请告诉我你想如何管理待办。\n本轮未修改任务。");
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
