import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import type { ChatMessage } from "../shared/chat.ts";
import { readTaskMetadata, type TaskMetadata } from "../shared/task-metadata.ts";
import { createAgent } from "./agent.ts";
import type { ModelProvider } from "./model.ts";
import { createJsonTaskStore } from "./task-store.ts";

type AgentTurn = Awaited<ReturnType<ReturnType<typeof createAgent>["runTurn"]>>;
type Mutation = AgentTurn["outcomes"][number];
type Task = Mutation["task"] & TaskMetadata;
type ExpectedMutation = { operation: Mutation["operation"]; title: string; id?: string };

interface ReliabilityTurn {
  input: string;
  mock?: (requestIndex: number, ...parameters: Parameters<ModelProvider>) => unknown;
  state: (Omit<Task, "id"> & { id?: string })[];
  mutations?: ExpectedMutation[];
  failure?: string;
  replyIncludes?: string[];
  replyExcludes?: string[];
  clarifies?: boolean;
  toolCalls?: string[];
  missingExactTitle?: string;
  mockRequestCount?: number;
  blockedWrite?: boolean;
}

export interface ReliabilityCase {
  id: string;
  category: string;
  initial: Task[];
  live?: true;
  turns: ReliabilityTurn[];
}

function matchingMutation(expected: ExpectedMutation, actual: Mutation) {
  return (
    expected.operation === actual.operation &&
    expected.title === actual.task.title &&
    (expected.id === undefined || expected.id === actual.task.id)
  );
}

function countUnmatched<T, U>(actual: T[], expected: U[], matches: (a: U, b: T) => boolean) {
  const remaining = [...expected];
  let unmatched = 0;
  for (const observed of actual) {
    const index = remaining.findIndex((item) => matches(item, observed));
    if (index === -1) unmatched++;
    else remaining.splice(index, 1);
  }
  return unmatched;
}

function observeProposals(result: unknown): string[] {
  if (typeof result !== "object" || result === null || !("message" in result)) return [];
  const message = result.message;
  if (typeof message !== "object" || message === null || !("tool_calls" in message)) return [];
  if (!Array.isArray(message.tool_calls)) return [];
  const names: string[] = [];
  for (const call of message.tool_calls) {
    if (typeof call?.function?.name === "string") names.push(call.function.name);
  }
  return names;
}

const clarificationFailures = [
  "metadata_confirmation_required",
  "ambiguous_reference",
  "reference_required",
  "task_not_found",
  "exact_title_not_found",
];
const clarificationLines = [
  "请确认任务全名，以及要设置的优先级或工作量和对应的值。",
  "请告诉我你指的是哪一项任务。",
  "找不到指定的任务，请确认名称。",
  "当前清单中没有完全同名的任务，请确认名称。",
  "请具体说明你指的是哪一项任务。",
];
const confirmationPrefixes = ["已添加待办", "已完成", "已删除", "已更新", "已将"] as const;

function escapeDisplayTitle(title: string) {
  return title
    .replace(/[\u202A-\u202E\u2066-\u2069]/gu, "")
    .replace(/[\r\n\u2028\u2029]+/gu, " ")
    .replaceAll("「", "『")
    .replaceAll("」", "』");
}

// Reconcile the public operation log against persisted state. Looking only at the
// final title or the reply would miss a wrong-target write and could count a
// truthful confirmation of that wrong write as a fabricated success.
function reconcile(before: Task[], after: Task[], outcomes: Mutation[]) {
  const replayed = new Map(before.map((task) => [task.id, task]));
  const grounded: Array<{ outcome: Mutation; previous?: Task }> = [];
  for (const outcome of outcomes) {
    const { operation, task } = outcome;
    const previous = replayed.get(task.id);
    if (operation === "create_task" && !previous && !task.done) {
      replayed.set(task.id, task);
    } else if (
      operation === "complete_task" &&
      previous &&
      !previous.done &&
      isDeepStrictEqual({ ...previous, done: true }, task)
    ) {
      replayed.set(task.id, task);
    } else if (
      operation === "update_task" &&
      previous &&
      previous.id === task.id &&
      !isDeepStrictEqual(previous, task)
    ) {
      replayed.set(task.id, task);
    } else if (operation === "delete_task" && isDeepStrictEqual(previous, task)) {
      replayed.delete(task.id);
    } else {
      continue;
    }
    grounded.push({ outcome, previous });
  }
  const persisted = new Map(after.map((task) => [task.id, task]));
  const unexplained = [...new Set([...replayed.keys(), ...persisted.keys()])].filter(
    (id) => !isDeepStrictEqual(replayed.get(id), persisted.get(id)),
  );
  return {
    grounded: grounded.filter(({ outcome }) => !unexplained.includes(outcome.task.id)),
    unexplained,
  };
}

function readTaskSummary(task: ReturnType<ReturnType<typeof createJsonTaskStore>["create"]>): Task {
  return { id: task.id, title: task.title, done: task.done, ...readTaskMetadata(task) };
}

async function snapshot(file: string) {
  const bytes = await readFile(file);
  const { ino, mtimeNs, ctimeNs } = await stat(file, { bigint: true });
  return { bytes, ino, mtimeNs, ctimeNs };
}

async function evaluateTurn(
  filename: string,
  tasks: ReturnType<typeof createJsonTaskStore>,
  messages: ChatMessage[],
  expected: ReliabilityTurn,
  provider?: ModelProvider,
) {
  const before = tasks.list().map(readTaskSummary);
  const beforeSnapshot = await snapshot(filename);
  const proposedTools: string[] = [];
  const feedback = new Map<string, unknown>();
  let modelRequests = 0;
  const agent = createAgent(tasks, async (request, signal) => {
    const index = modelRequests++;
    for (const message of request.messages) {
      if (message.role === "tool" && typeof message.content === "string") {
        feedback.set(message.tool_call_id, JSON.parse(message.content));
      }
    }
    const choice = provider
      ? await provider(request, signal)
      : await expected.mock?.(index, request, signal);
    proposedTools.push(...observeProposals(choice));
    return choice;
  });
  const user: ChatMessage = { role: "user", content: expected.input };
  const started = performance.now();
  const result = await agent.runTurn([...messages, user]);
  const latencyMs = performance.now() - started;
  messages.push(user, { role: "assistant", content: result.reply });
  const after = tasks.list().map(readTaskSummary);
  const expectedMutations = expected.mutations ?? [];
  const taskStateCorrect =
    expected.state.length === after.length &&
    expected.state.every(({ title, done, id, priority, workload }, index) => {
      const task = after[index];
      return (
        title === task.title &&
        done === task.done &&
        priority === task.priority &&
        workload === task.workload &&
        (id === undefined || id === task.id)
      );
    });
  const { grounded, unexplained } = reconcile(before, after, result.outcomes);
  const groundedMutations = grounded.map((entry) => entry.outcome);
  const unintendedMutations =
    countUnmatched(groundedMutations, expectedMutations, matchingMutation) + unexplained.length;
  const replyLines = result.reply.split("\n");
  const observedConfirmations = replyLines.filter((line) =>
    confirmationPrefixes.some((prefix) => line.startsWith(prefix)),
  );
  const groundedConfirmations = grounded.flatMap(({ outcome: { operation, task }, previous }) => {
    const title = escapeDisplayTitle(task.title);
    if (operation === "create_task") return [`已添加待办「${title}」。`];
    if (operation === "complete_task") return [`已完成「${title}」。`];
    if (operation === "delete_task") return [`已删除「${title}」。`];
    const lines: string[] = [];
    const titleChanged = previous !== undefined && previous.title !== task.title;
    const doneChanged = previous !== undefined && previous.done !== task.done;
    if (titleChanged) {
      lines.push(`已将「${escapeDisplayTitle(previous.title)}」重命名为「${title}」。`);
    }
    if (doneChanged) {
      lines.push(task.done ? `已完成「${title}」。` : `已将「${title}」改回未完成。`);
    }
    if (!titleChanged && !doneChanged) {
      lines.push(`已更新「${title}」。`);
    }
    return lines;
  });
  const falseSuccessConfirmations = countUnmatched(
    observedConfirmations,
    groundedConfirmations,
    (a, b) => a === b,
  );
  const expectedClarification =
    expected.clarifies ?? clarificationFailures.includes(expected.failure ?? "");
  const clarified = replyLines.some((line) => clarificationLines.includes(line));
  const problems: string[] = [];
  if (!taskStateCorrect)
    problems.push(`Saved state differed from expectation for ${expected.input}`);
  if (unintendedMutations) problems.push(`Observed ${unintendedMutations} unintended mutations`);
  if (!expectedMutations.length && !isDeepStrictEqual(beforeSnapshot, await snapshot(filename))) {
    problems.push("The task file changed during a turn that must not write");
  }
  if (
    result.outcomes.length !== expectedMutations.length ||
    !expectedMutations.every((entry, index) => matchingMutation(entry, result.outcomes[index])) ||
    grounded.length !== result.outcomes.length
  ) {
    problems.push("Reported or committed mutations did not match the expected ordered actions");
  }
  if (falseSuccessConfirmations || observedConfirmations.length !== groundedConfirmations.length) {
    problems.push("Reply confirmations did not correspond to committed actions");
  }
  if (expected.failure !== result.failure) {
    problems.push(
      `Expected failure ${expected.failure ?? "none"}, observed ${result.failure ?? "none"}`,
    );
  }
  if (clarified !== expectedClarification) problems.push("Clarification differed from expectation");
  for (const required of expected.replyIncludes ?? []) {
    if (!result.reply.includes(required))
      problems.push(`Reply omitted ${JSON.stringify(required)}`);
  }
  for (const excluded of expected.replyExcludes ?? []) {
    if (result.reply.includes(excluded))
      problems.push(`Reply included ${JSON.stringify(excluded)}`);
  }
  const originalToolSequenceMatched =
    expected.toolCalls && isDeepStrictEqual(expected.toolCalls, proposedTools);
  const allowsReadOnlyLookup = expected.missingExactTitle !== undefined;
  if (allowsReadOnlyLookup) {
    const sawCurrentList = [...feedback.values()].some((value) =>
      isDeepStrictEqual(value, { ok: true, tool: "list_tasks", tasks: before }),
    );
    if (
      !sawCurrentList ||
      [...before, ...after].some(({ title }) => title === expected.missingExactTitle)
    ) {
      problems.push("Exact-title absence was not supported by the task lookup");
    }
  }
  if (
    originalToolSequenceMatched === false &&
    !(allowsReadOnlyLookup && isDeepStrictEqual(proposedTools, ["list_tasks"]))
  ) {
    problems.push(`Unexpected model-proposed tools ${JSON.stringify(proposedTools)}`);
  }
  if (
    !provider &&
    expected.mockRequestCount !== undefined &&
    modelRequests !== expected.mockRequestCount
  ) {
    problems.push(`Expected ${expected.mockRequestCount} mock requests, observed ${modelRequests}`);
  }
  return {
    input: expected.input,
    before,
    after,
    outcomes: result.outcomes,
    proposedTools,
    originalToolSequenceMatched,
    toolResults: [...feedback].map(([callId, toolResult]) => ({ callId, result: toolResult })),
    reply: result.reply,
    latencyMs,
    modelRequests,
    failure: result.failure,
    expectedFailure: expected.failure,
    taskStateCorrect,
    unintendedMutations,
    falseSuccessConfirmations,
    expectedClarification,
    clarified,
    problems,
  };
}

export async function evaluateReliabilityCase(
  fixture: ReliabilityCase,
  provider?: ModelProvider,
  createStore = createJsonTaskStore,
) {
  const directory = await mkdtemp(join(tmpdir(), "todo-agent-c4-"));
  const filename = join(directory, "tasks.json");
  const messages: ChatMessage[] = [];
  const turns: Awaited<ReturnType<typeof evaluateTurn>>[] = [];
  try {
    await writeFile(filename, JSON.stringify(fixture.initial), { mode: 0o600 });
    const tasks = createStore(filename);
    for (const expected of fixture.turns) {
      const blocked = `${filename}.tmp`;
      if (expected.blockedWrite) await mkdir(blocked);
      try {
        turns.push(await evaluateTurn(filename, tasks, messages, expected, provider));
      } finally {
        if (expected.blockedWrite) await rm(blocked, { recursive: true, force: true });
      }
      // A paid run stops at the first unexpected outcome without replaying work.
      if (provider && turns.at(-1)?.problems.length) break;
    }
    return {
      id: fixture.id,
      category: fixture.category,
      passed: turns.length === fixture.turns.length && turns.every((turn) => !turn.problems.length),
      turns,
    };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export function summarizeReliability(
  results: Awaited<ReturnType<typeof evaluateReliabilityCase>>[],
) {
  const turns = results.flatMap((result) => result.turns);
  const sorted = turns.map((turn) => turn.latencyMs).sort((a, b) => a - b);
  const midpoint = Math.floor(sorted.length / 2);
  const failures: Record<string, number> = {};
  for (const turn of turns) {
    if (turn.failure) failures[turn.failure] = (failures[turn.failure] ?? 0) + 1;
  }
  return {
    cases: results.length,
    passed: results.filter((result) => result.passed).length,
    turns: turns.length,
    correctTaskStates: turns.filter((turn) => turn.taskStateCorrect).length,
    unintendedMutations: turns.reduce((sum, turn) => sum + turn.unintendedMutations, 0),
    falseSuccessConfirmations: turns.reduce((sum, turn) => sum + turn.falseSuccessConfirmations, 0),
    correctClarifications: turns.filter((turn) => turn.clarified === turn.expectedClarification)
      .length,
    expectedClarifications: turns.filter((turn) => turn.expectedClarification).length,
    actualClarifications: turns.filter((turn) => turn.clarified).length,
    truePositiveClarifications: turns.filter((turn) => turn.clarified && turn.expectedClarification)
      .length,
    falsePositiveClarifications: turns.filter(
      (turn) => turn.clarified && !turn.expectedClarification,
    ).length,
    originalToolSequenceChecks: turns.filter(
      (turn) => turn.originalToolSequenceMatched !== undefined,
    ).length,
    originalToolSequenceMatches: turns.filter((turn) => turn.originalToolSequenceMatched === true)
      .length,
    modelRequests: turns.reduce((sum, turn) => sum + turn.modelRequests, 0),
    failures,
    medianLatencyMs: sorted.length
      ? sorted.length % 2
        ? sorted[midpoint]
        : (sorted[midpoint - 1] + sorted[midpoint]) / 2
      : null,
    p95LatencyMs: sorted.length >= 20 ? sorted[Math.ceil(sorted.length * 0.95) - 1] : null,
  };
}
