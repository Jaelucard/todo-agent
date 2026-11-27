import { parseArgs } from "node:util";
import { getModelClient, model, MODEL_TIMEOUT_MS, type ModelProvider } from "./model.ts";

const scopeModel = "deepseek-flash";
const maxRequestCap = 20;
const requestBodyLimitBytes = 8 * 1024;
const outputTokenLimit = 1024;
const usage = `After approval, use --live --model=${scopeModel} --max-requests=<integer from 1 to ${maxRequestCap}>.`;
const usageFields = [
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "prompt_cache_hit_tokens",
  "prompt_cache_miss_tokens",
] as const;
type TokenUsage = Partial<Record<(typeof usageFields)[number], number>>;
type Outbound = {
  caseId: string;
  number: number;
  bytes: number;
  status?: number;
  elapsedMs?: number;
  usage?: TokenUsage;
};
const knownTools = new Set([
  "create_task",
  "list_tasks",
  "complete_task",
  "delete_task",
  "update_task",
]);

function toolName(name: unknown) {
  return typeof name === "string" && knownTools.has(name) ? name : "[unknown]";
}

function tokenUsage(value: unknown): TokenUsage | undefined {
  if (value === null || typeof value !== "object") return undefined;
  const extracted: TokenUsage = {};
  for (const field of usageFields) {
    const count: unknown = Reflect.get(value, field);
    if (typeof count === "number" && Number.isSafeInteger(count) && count >= 0) {
      extracted[field] = count;
    }
  }
  return Object.keys(extracted).length ? extracted : undefined;
}

async function main() {
  let values;
  try {
    ({ values } = parseArgs({
      options: {
        live: { type: "boolean" },
        model: { type: "string" },
        "max-requests": { type: "string" },
      },
    }));
  } catch {
    console.error(usage);
    return 2;
  }

  const cap = Number(values["max-requests"]);
  if (
    !values.live ||
    values.model !== scopeModel ||
    values.model !== model ||
    !Number.isInteger(cap) ||
    cap < 1 ||
    cap > maxRequestCap
  ) {
    console.error(usage);
    return 2;
  }
  if (!process.env.DEEPSEEK_API_KEY) {
    console.error("Load the server-side DeepSeek configuration before this live run.");
    return 2;
  }

  const [{ reliabilityCases }, { evaluateReliabilityCase, summarizeReliability }] =
    await Promise.all([import("./reliability-cases.ts"), import("./reliability-eval.ts")]);
  const fixtures = reliabilityCases.filter((fixture) => "live" in fixture && fixture.live === true);
  if (!fixtures.length) {
    console.error("No live reliability fixtures are defined.");
    return 2;
  }

  console.log(
    JSON.stringify({
      event: "live_scope",
      caseIds: fixtures.map(({ id }) => id),
      model,
      requestCap: cap,
      requestBodyLimitBytes,
      outputTokenLimit,
      timeoutMs: MODEL_TIMEOUT_MS,
      retries: 0,
      storage: "isolated temporary JSON for each case",
    }),
  );

  const outbound: Outbound[] = [];
  let stopReason: string | undefined;
  const client = getModelClient();
  const liveProvider =
    (caseId: string): ModelProvider =>
    async (body, signal) => {
      let observation: Outbound | undefined;
      let started = 0;
      try {
        const scopedClient = client.withOptions({
          maxRetries: 0,
          fetch: async (input, options) => {
            if (stopReason) throw new Error("Live evaluation stopped");
            if (outbound.length >= cap) {
              stopReason = "request_limit";
              throw new Error("Live request limit reached");
            }
            const request = new Request(input, options);
            const serialized = await request.clone().text();
            const bytes = Buffer.byteLength(serialized);
            const payload = JSON.parse(serialized);
            if (
              request.url !== "https://api.deepseek.com/chat/completions" ||
              request.method !== "POST" ||
              payload.model !== values.model ||
              payload.stream !== false ||
              !Number.isInteger(payload.max_tokens) ||
              payload.max_tokens < 1 ||
              payload.max_tokens > outputTokenLimit ||
              bytes > requestBodyLimitBytes
            ) {
              stopReason = "request_outside_scope";
              throw new Error("Request is outside the live scope");
            }
            request.signal.throwIfAborted();
            // Recheck immediately before sending, after reading the body.
            if (stopReason || outbound.length >= cap) {
              stopReason ??= "request_limit";
              throw new Error("Live evaluation stopped");
            }
            observation = { caseId, number: outbound.length + 1, bytes };
            outbound.push(observation);
            started = performance.now();
            const response = await fetch(request, { redirect: "error" });
            observation.status = response.status;
            if (!response.ok) stopReason ??= "provider_http_error";
            return response;
          },
        });
        const response = await scopedClient.chat.completions.create(body, { signal });
        if (observation) observation.usage = tokenUsage(response.usage);
        return response.choices?.[0];
      } catch (error) {
        stopReason ??= signal.aborted ? "deadline" : "provider_error";
        throw error;
      } finally {
        if (observation) observation.elapsedMs = Math.round(performance.now() - started);
      }
    };

  const results = [];
  try {
    for (const fixture of fixtures) {
      const result = await evaluateReliabilityCase(fixture, liveProvider(fixture.id));
      results.push(result);
      if (!result.passed) stopReason ??= "case_failed";
      if (stopReason) break;
    }
  } catch {
    stopReason ??= "runner_error";
  }
  const totalUsage: TokenUsage = {};
  for (const request of outbound) {
    for (const field of usageFields) {
      const tokens = request.usage?.[field];
      if (tokens !== undefined) totalUsage[field] = (totalUsage[field] ?? 0) + tokens;
    }
  }
  console.log(
    JSON.stringify(
      {
        event: "live_result",
        cases: results.map(({ id, category, passed, turns }) => ({
          id,
          category,
          passed,
          turns: turns.map((turn) => ({
            input: turn.input,
            before: turn.before.map(({ id, done }) => ({ id, done })),
            after: turn.after.map(({ id, done }) => ({ id, done })),
            outcomes: turn.outcomes.map(({ operation, task }) => ({
              operation,
              taskId: task.id,
              done: task.done,
            })),
            proposedTools: turn.proposedTools.map(toolName),
            toolResults: turn.toolResults.map(({ result }) =>
              typeof result === "object" && result !== null && "tool" in result
                ? toolName(result.tool)
                : "[unknown]",
            ),
            taskStateCorrect: turn.taskStateCorrect,
            expectedClarification: turn.expectedClarification,
            clarified: turn.clarified,
            falseSuccessConfirmations: turn.falseSuccessConfirmations,
            unintendedMutations: turn.unintendedMutations,
            failure: turn.failure,
            expectedFailure: turn.expectedFailure,
            modelRequests: turn.modelRequests,
            latencyMs: turn.latencyMs,
          })),
        })),
        skippedCaseIds: fixtures.slice(results.length).map(({ id }) => id),
        metrics: summarizeReliability(results),
        outbound,
        requestCount: outbound.length,
        requestsWithUsage: outbound.filter((request) => request.usage).length,
        usage: totalUsage,
        billingCost: null,
        stopReason,
      },
      null,
      2,
    ),
  );
  return stopReason ? 1 : 0;
}

if (import.meta.main) {
  main().then(
    (code) => (process.exitCode = code),
    () => {
      console.error("Live reliability evaluation could not complete.");
      process.exitCode = 1;
    },
  );
}
