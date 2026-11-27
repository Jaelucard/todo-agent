import OpenAI, { APIConnectionTimeoutError } from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";

export type { ChatCompletionFunctionTool, ChatCompletionMessageParam };

export type ModelProvider = (
  request: ChatCompletionCreateParamsNonStreaming,
  signal: AbortSignal,
) => Promise<unknown>;

export const model = process.env.DEEPSEEK_MODEL || "deepseek-flash";
export const MODEL_TIMEOUT_MS = 20_000;
const DEEPSEEK_BASE_URL = "https://api.deepseek.com";

export class ModelConfigError extends Error {
  override name = "ModelConfigError";

  constructor(message = "请在 .env 中设置 DEEPSEEK_API_KEY 后重启服务。") {
    super(message);
  }
}

let cachedClient: { apiKey: string; baseURL: string; client: OpenAI } | undefined;

export function isModelTimeout(error: unknown): boolean {
  return error instanceof APIConnectionTimeoutError;
}

export function getModelClient(): OpenAI {
  const apiKey = process.env.DEEPSEEK_API_KEY?.trim();
  if (!apiKey) throw new ModelConfigError();
  const baseURL = DEEPSEEK_BASE_URL;
  if (cachedClient && cachedClient.apiKey === apiKey && cachedClient.baseURL === baseURL) {
    return cachedClient.client;
  }
  const client = new OpenAI({
    apiKey,
    baseURL,
    maxRetries: 0,
    timeout: MODEL_TIMEOUT_MS,
    logLevel: "off",
  });
  cachedClient = { apiKey, baseURL, client };
  return client;
}

export function buildModelRequest(
  messages: ChatCompletionMessageParam[],
  tools: ChatCompletionFunctionTool[],
  toolChoice: ChatCompletionCreateParamsNonStreaming["tool_choice"] = "auto",
): ChatCompletionCreateParamsNonStreaming {
  return {
    model,
    messages,
    tools,
    tool_choice: toolChoice,
    reasoning_effort: "none",
    max_tokens: 1024,
    stream: false,
  };
}

export const requestModel: ModelProvider = async (request, signal) => {
  const response = await getModelClient().chat.completions.create(request, { signal });
  return response.choices?.[0];
};
