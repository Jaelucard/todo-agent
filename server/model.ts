import OpenAI from "openai";
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

export function getModelClient(): OpenAI {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("请在 .env 中设置 DEEPSEEK_API_KEY 后重启服务。");
  return new OpenAI({
    apiKey,
    baseURL: "https://api.deepseek.com",
    maxRetries: 0,
    timeout: MODEL_TIMEOUT_MS,
    logLevel: "off",
  });
}

export function buildModelRequest(
  messages: ChatCompletionMessageParam[],
  tools: ChatCompletionFunctionTool[],
): ChatCompletionCreateParamsNonStreaming {
  return {
    model,
    messages,
    tools,
    tool_choice: "auto",
    reasoning_effort: "none",
    max_tokens: 1024,
    stream: false,
  };
}

export const requestModel: ModelProvider = async (request, signal) => {
  const response = await getModelClient().chat.completions.create(request, { signal });
  return response.choices?.[0];
};
