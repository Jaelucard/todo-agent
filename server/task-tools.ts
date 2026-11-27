import { z } from "zod";
import { createTaskSchema } from "../shared/task-input.ts";
import type { TaskStore } from "./task-store.ts";

type Task = Pick<ReturnType<TaskStore["create"]>, "id" | "title" | "done">;
type BoundReference = Pick<Task, "id" | "title">;
type ResolvedTarget = { ok: true; task: Task } | Extract<ToolResult, { ok: false }>;
type ToolError =
  | "unknown_tool"
  | "invalid_arguments"
  | "duplicate_create"
  | "list_required"
  | "reference_required"
  | "stale_view"
  | "task_not_found"
  | "ambiguous_reference"
  | "storage_error";
type ToolResult =
  | { ok: true; tool: "list_tasks"; tasks: Task[] }
  | { ok: true; tool: "create_task"; task: Task }
  | { ok: true; tool: "complete_task"; task: Task; changed: boolean }
  | { ok: true; tool: "delete_task"; task: Task }
  | { ok: false; error: ToolError };

const listTaskSchema = z.strictObject({});
const selectorSchema = z.union([
  z.strictObject({ candidateIds: z.array(z.string()) }),
  z.strictObject({ useBoundReference: z.literal(true) }),
]);

function taskSummary({ id, title, done }: Task): Task {
  return { id, title, done };
}

// Create one toolkit per turn. A bound reference comes from trusted server context.
export function createTaskTools(tasks: TaskStore, boundReference?: BoundReference) {
  const binding = boundReference && { ...boundReference };
  const createdTitles = new Set<string>();
  let lastView: string | undefined;

  function resolveTarget(selector: z.infer<typeof selectorSchema>): ResolvedTarget {
    const semantic = "candidateIds" in selector;
    if (semantic && lastView === undefined) {
      return {
        ok: false,
        error: selector.candidateIds.length > 1 ? "ambiguous_reference" : "list_required",
      };
    }
    const id = semantic ? selector.candidateIds[0] : binding?.id;
    if (!semantic && id === undefined) return { ok: false, error: "reference_required" };

    // Revalidate the view before either describing a target or modifying it.
    const current = tasks.list().map(taskSummary);
    if (semantic && JSON.stringify(current) !== lastView) {
      lastView = undefined;
      return { ok: false, error: "stale_view" };
    }
    if (semantic && selector.candidateIds.length > 1)
      return { ok: false, error: "ambiguous_reference" };
    if (id === undefined) return { ok: false, error: "task_not_found" };
    const [target, duplicateId] = current.filter((task) => task.id === id);
    if (!target) return { ok: false, error: "task_not_found" };
    if (
      duplicateId ||
      (semantic && current.some((task) => task.id !== id && task.title === target.title))
    ) {
      return { ok: false, error: "ambiguous_reference" };
    }
    if (!semantic && target.title !== binding?.title) return { ok: false, error: "stale_view" };
    return { ok: true, task: target };
  }

  function mutateTask(tool: "complete_task" | "delete_task", args: unknown): ToolResult {
    const input = selectorSchema.safeParse(args);
    if (!input.success) return { ok: false, error: "invalid_arguments" };
    const resolved = resolveTarget(input.data);
    if (!resolved.ok) return resolved;
    const target = resolved.task;
    if (tool === "complete_task") {
      if (target.done) return { ok: true, tool, task: target, changed: false };
      const updated = tasks.update(target.id, { done: true });
      if (!updated) return { ok: false, error: "task_not_found" };
      lastView = undefined;
      return { ok: true, tool, task: taskSummary(updated), changed: true };
    }

    if (!tasks.remove(target.id)) return { ok: false, error: "task_not_found" };
    lastView = undefined;
    return { ok: true, tool, task: target };
  }

  return {
    // Server-only observation. Never dispatch a mutation to find out whether it is a no-op.
    inspectCandidates(candidateIds: string[]): ResolvedTarget {
      try {
        return resolveTarget({ candidateIds });
      } catch {
        return { ok: false, error: "storage_error" };
      }
    },
    execute(tool: string, argumentsJson: string): ToolResult {
      if (
        tool !== "list_tasks" &&
        tool !== "create_task" &&
        tool !== "complete_task" &&
        tool !== "delete_task"
      ) {
        return { ok: false, error: "unknown_tool" };
      }

      let args: unknown;
      try {
        args = JSON.parse(argumentsJson);
      } catch {
        return { ok: false, error: "invalid_arguments" };
      }

      try {
        switch (tool) {
          case "list_tasks": {
            if (!listTaskSchema.safeParse(args).success)
              return { ok: false, error: "invalid_arguments" };
            const view = tasks.list().map(taskSummary);
            lastView = JSON.stringify(view);
            return { ok: true, tool, tasks: view };
          }
          case "create_task": {
            const input = createTaskSchema.safeParse(args);
            if (!input.success) return { ok: false, error: "invalid_arguments" };
            const { title } = input.data;
            if (createdTitles.has(title)) return { ok: false, error: "duplicate_create" };
            const task = tasks.create(title);
            createdTitles.add(title);
            lastView = undefined;
            return { ok: true, tool, task: taskSummary(task) };
          }
          case "complete_task":
          case "delete_task":
            return mutateTask(tool, args);
        }
      } catch {
        return { ok: false, error: "storage_error" };
      }
    },
  };
}
