import { z } from "zod";
import { createTaskSchema, taskUpdatePatchSchema } from "../shared/task-input.ts";
import {
  priorities,
  readTaskMetadata,
  workloads,
  type TaskMetadata,
} from "../shared/task-metadata.ts";
import type { TaskStore } from "./task-store.ts";

export type Task = Pick<ReturnType<TaskStore["create"]>, "id" | "title" | "done"> & TaskMetadata;
export type BoundReference = Pick<Task, "id" | "title">;
export type TaskUpdateChanges = z.infer<typeof taskUpdatePatchSchema>;
export type ListTasksFilter = z.infer<typeof listTaskFilterSchema>;

export interface RequestedTarget {
  title?: string;
  changes?: TaskUpdateChanges;
  requireExactMetadata?: boolean;
  allowedPriorities?: ReadonlySet<NonNullable<TaskMetadata["priority"]>>;
  allowedWorkloads?: ReadonlySet<NonNullable<TaskMetadata["workload"]>>;
  allowTitleChange?: boolean;
  allowDoneValues?: ReadonlySet<boolean>;
  allowedTitles?: ReadonlySet<string>;
  forbiddenTitles?: ReadonlySet<string>;
  requireBoundReference?: boolean;
}

type ResolvedTarget = { ok: true; task: Task } | Extract<ToolResult, { ok: false }>;
export type ToolError =
  | "metadata_confirmation_required"
  | "unknown_tool"
  | "invalid_arguments"
  | "duplicate_create"
  | "list_required"
  | "reference_required"
  | "stale_view"
  | "task_not_found"
  | "ambiguous_reference"
  | "storage_error";

export type ToolResult =
  | { ok: true; tool: "list_tasks"; tasks: Task[] }
  | { ok: true; tool: "create_task"; task: Task }
  | { ok: true; tool: "complete_task"; task: Task; changed: boolean }
  | { ok: true; tool: "update_task"; task: Task; changed: boolean }
  | { ok: true; tool: "delete_task"; task: Task }
  | { ok: false; error: ToolError };

export const listTaskFilterSchema = z.strictObject({
  done: z.boolean().optional(),
  priority: z.enum(priorities).optional(),
  workload: z.enum(workloads).optional(),
});

const semanticSelectorSchema = z.strictObject({ candidateIds: z.array(z.string()) });
const boundSelectorSchema = z.strictObject({ useBoundReference: z.literal(true) });
const selectorSchema = z.union([semanticSelectorSchema, boundSelectorSchema]);
const updateTaskSchema = z.union([
  semanticSelectorSchema.extend({ changes: taskUpdatePatchSchema }),
  boundSelectorSchema.extend({ changes: taskUpdatePatchSchema }),
]);

export function taskSummary(task: ReturnType<TaskStore["create"]>): Task {
  const { id, title, done } = task;
  return { id, title, done, ...readTaskMetadata(task) };
}

function matchesFilter(task: Task, filter: ListTasksFilter): boolean {
  if (filter.done !== undefined && task.done !== filter.done) return false;
  if (filter.priority !== undefined && task.priority !== filter.priority) return false;
  if (filter.workload !== undefined && task.workload !== filter.workload) return false;
  return true;
}

// Create one toolkit per turn. A bound reference comes from trusted server context.
export function createTaskTools(tasks: TaskStore, boundReference?: BoundReference) {
  const binding = boundReference && { ...boundReference };
  const createdTitles = new Set<string>();
  let expectedSnapshot: string | undefined;
  const observedIds = new Set<string>();
  let lastBeforeMutation: Task | undefined;
  let lastAmbiguousDuplicateTitle: string | undefined;

  function resolveTarget(
    selector: z.infer<typeof selectorSchema>,
    tool: "complete_task" | "delete_task" | "update_task" | "inspect",
    grant?: RequestedTarget,
  ): ResolvedTarget {
    lastAmbiguousDuplicateTitle = undefined;
    const semantic = "candidateIds" in selector;
    if (grant?.requireBoundReference && semantic) {
      return { ok: false, error: "reference_required" };
    }
    if (semantic && expectedSnapshot === undefined) {
      return {
        ok: false,
        error: selector.candidateIds.length > 1 ? "ambiguous_reference" : "list_required",
      };
    }
    if (!semantic && binding === undefined) {
      return { ok: false, error: "reference_required" };
    }

    // Revalidate the view before either describing a target or modifying it.
    const current = tasks.list().map(taskSummary);
    if (semantic && JSON.stringify(current) !== expectedSnapshot) {
      expectedSnapshot = undefined;
      observedIds.clear();
      return { ok: false, error: "stale_view" };
    }

    let target: Task | undefined;

    if (semantic) {
      if (selector.candidateIds.length === 0) {
        return { ok: false, error: "task_not_found" };
      }
      if (selector.candidateIds.length > 1) {
        const matched: Task[] = [];
        let allObserved = true;
        let hasDuplicateStoredId = false;
        for (const candidateId of selector.candidateIds) {
          const hits = current.filter((item) => item.id === candidateId);
          if (hits.length > 1) hasDuplicateStoredId = true;
          if (hits.length === 1) {
            if (!observedIds.has(candidateId)) allObserved = false;
            matched.push(hits[0]);
          }
        }
        if (
          !hasDuplicateStoredId &&
          allObserved &&
          matched.length === selector.candidateIds.length &&
          new Set(matched.map((item) => item.id)).size === matched.length
        ) {
          const titles = new Set(matched.map((item) => item.title));
          if (titles.size === 1) {
            const sharedTitle = matched[0].title;
            const allWithTitle = current.filter((item) => item.title === sharedTitle);
            const pendingWithTitle = allWithTitle.filter((item) => !item.done);
            if (
              tool === "complete_task" &&
              pendingWithTitle.length === 1 &&
              matched.some((item) => item.id === pendingWithTitle[0].id)
            ) {
              target = pendingWithTitle[0];
            } else if (allWithTitle.length >= 2) {
              lastAmbiguousDuplicateTitle = sharedTitle;
              return { ok: false, error: "ambiguous_reference" };
            }
          }
        }
        if (!target) {
          return { ok: false, error: "ambiguous_reference" };
        }
      } else {
        const id = selector.candidateIds[0];
        const [found, duplicateId] = current.filter((task) => task.id === id);
        if (!found) return { ok: false, error: "task_not_found" };
        if (!observedIds.has(id)) return { ok: false, error: "list_required" };
        if (duplicateId) return { ok: false, error: "ambiguous_reference" };
        const sameTitleTasks = current.filter((task) => task.title === found.title);
        if (sameTitleTasks.length > 1) {
          // For complete_task ONLY: allow unambiguous completion when uniquely one copy is pending.
          const pendingSameTitle = sameTitleTasks.filter((task) => !task.done);
          if (tool === "complete_task" && pendingSameTitle.length === 1 && !found.done) {
            target = found;
          } else {
            lastAmbiguousDuplicateTitle = found.title;
            return { ok: false, error: "ambiguous_reference" };
          }
        } else {
          target = found;
        }
      }
    } else {
      const id = binding!.id;
      const [found, duplicateId] = current.filter((task) => task.id === id);
      if (!found) return { ok: false, error: "task_not_found" };
      if (duplicateId) return { ok: false, error: "ambiguous_reference" };
      if (found.title !== binding!.title) return { ok: false, error: "stale_view" };
      target = found;
    }

    if (grant?.forbiddenTitles?.has(target.title)) {
      return { ok: false, error: "reference_required" };
    }
    if (grant?.title !== undefined && target.title !== grant.title) {
      return { ok: false, error: "reference_required" };
    }
    if (grant?.allowedTitles && !grant.allowedTitles.has(target.title)) {
      return { ok: false, error: "reference_required" };
    }
    return { ok: true, task: target };
  }

  function mutateTask(
    tool: "complete_task" | "delete_task",
    args: unknown,
    grant?: RequestedTarget,
  ): ToolResult {
    const input = selectorSchema.safeParse(args);
    if (!input.success) return { ok: false, error: "invalid_arguments" };
    const resolved = resolveTarget(input.data, tool, grant);
    if (!resolved.ok) return resolved;
    const target = resolved.task;
    lastBeforeMutation = target;
    if (tool === "complete_task") {
      if (target.done) return { ok: true, tool, task: target, changed: false };
      const updated = tasks.update(target.id, { done: true });
      if (!updated) return { ok: false, error: "task_not_found" };
      if (expectedSnapshot !== undefined) {
        expectedSnapshot = JSON.stringify(tasks.list().map(taskSummary));
      }
      return { ok: true, tool, task: taskSummary(updated), changed: true };
    }

    if (!tasks.remove(target.id)) return { ok: false, error: "task_not_found" };
    observedIds.delete(target.id);
    if (expectedSnapshot !== undefined) {
      expectedSnapshot = JSON.stringify(tasks.list().map(taskSummary));
    }
    return { ok: true, tool, task: target };
  }

  function updateTask(args: unknown, grant?: RequestedTarget): ToolResult {
    const input = updateTaskSchema.safeParse(args);
    if (!input.success) return { ok: false, error: "invalid_arguments" };
    const resolved = resolveTarget(input.data, "update_task", grant);
    if (!resolved.ok) return resolved;
    const target = resolved.task;
    const changes = input.data.changes;

    if (grant) {
      if (grant.requireExactMetadata) {
        if (
          changes.title !== undefined ||
          changes.done !== undefined ||
          changes.priority !== grant.changes?.priority ||
          changes.workload !== grant.changes?.workload
        ) {
          return { ok: false, error: "metadata_confirmation_required" };
        }
      } else {
        if (changes.priority !== undefined && !grant.allowedPriorities?.has(changes.priority)) {
          return { ok: false, error: "metadata_confirmation_required" };
        }
        if (changes.workload !== undefined && !grant.allowedWorkloads?.has(changes.workload)) {
          return { ok: false, error: "metadata_confirmation_required" };
        }
        if (changes.title !== undefined && !grant.allowTitleChange) {
          return { ok: false, error: "metadata_confirmation_required" };
        }
        if (changes.done !== undefined && !grant.allowDoneValues?.has(changes.done)) {
          return { ok: false, error: "metadata_confirmation_required" };
        }
      }
    }

    lastBeforeMutation = target;
    const titleChanged = changes.title !== undefined && changes.title !== target.title;
    const doneChanged = changes.done !== undefined && changes.done !== target.done;
    const priorityChanged = changes.priority !== undefined && changes.priority !== target.priority;
    const workloadChanged = changes.workload !== undefined && changes.workload !== target.workload;
    if (!titleChanged && !doneChanged && !priorityChanged && !workloadChanged) {
      return { ok: true, tool: "update_task", task: target, changed: false };
    }
    const patch: Partial<Pick<Task, "title" | "done">> & TaskMetadata = {
      ...(changes.title !== undefined && { title: changes.title }),
      ...(changes.done !== undefined && { done: changes.done }),
      ...(changes.priority !== undefined && { priority: changes.priority }),
      ...(changes.workload !== undefined && { workload: changes.workload }),
    };
    const updated = tasks.update(target.id, patch);
    if (!updated) return { ok: false, error: "task_not_found" };
    const summary = taskSummary(updated);
    if (!("candidateIds" in input.data) && binding && changes.title !== undefined) {
      binding.title = summary.title;
    }
    if (expectedSnapshot !== undefined) {
      expectedSnapshot = JSON.stringify(tasks.list().map(taskSummary));
    }
    return { ok: true, tool: "update_task", task: summary, changed: true };
  }

  return {
    // Server-only observation. Never dispatch a mutation to find out whether it is a no-op.
    inspectCandidates(candidateIds: string[]): ResolvedTarget {
      try {
        return resolveTarget({ candidateIds }, "inspect");
      } catch {
        return { ok: false, error: "storage_error" };
      }
    },
    lastBeforeUpdate(): Task | undefined {
      return lastBeforeMutation;
    },
    lastAmbiguousTitle(): string | undefined {
      return lastAmbiguousDuplicateTitle;
    },
    execute(tool: string, argumentsJson: string, grant?: RequestedTarget): ToolResult {
      lastBeforeMutation = undefined;
      lastAmbiguousDuplicateTitle = undefined;
      if (
        tool !== "list_tasks" &&
        tool !== "create_task" &&
        tool !== "complete_task" &&
        tool !== "delete_task" &&
        tool !== "update_task"
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
            const filter = listTaskFilterSchema.safeParse(args);
            if (!filter.success) return { ok: false, error: "invalid_arguments" };
            const all = tasks.list().map(taskSummary);
            expectedSnapshot = JSON.stringify(all);
            const allIds = new Set(all.map((task) => task.id));
            for (const existing of observedIds) {
              if (!allIds.has(existing)) observedIds.delete(existing);
            }
            const view = all.filter((task) => matchesFilter(task, filter.data));
            for (const task of view) {
              observedIds.add(task.id);
            }
            return { ok: true, tool, tasks: view };
          }
          case "create_task": {
            const input = createTaskSchema.safeParse(args);
            if (!input.success) return { ok: false, error: "invalid_arguments" };
            const { title, ...metadata } = input.data;
            if (grant?.forbiddenTitles?.has(title)) {
              return { ok: false, error: "reference_required" };
            }
            if (grant?.title !== undefined && title !== grant.title) {
              return { ok: false, error: "reference_required" };
            }
            if (grant?.requireExactMetadata) {
              if (
                metadata.priority !== grant.changes?.priority ||
                metadata.workload !== grant.changes?.workload
              ) {
                return { ok: false, error: "metadata_confirmation_required" };
              }
            } else if (grant) {
              if (
                metadata.priority !== undefined &&
                !grant.allowedPriorities?.has(metadata.priority)
              ) {
                return { ok: false, error: "metadata_confirmation_required" };
              }
              if (
                metadata.workload !== undefined &&
                !grant.allowedWorkloads?.has(metadata.workload)
              ) {
                return { ok: false, error: "metadata_confirmation_required" };
              }
              if (
                grant.changes?.priority !== undefined &&
                metadata.priority !== grant.changes.priority
              ) {
                return { ok: false, error: "metadata_confirmation_required" };
              }
              if (
                grant.changes?.workload !== undefined &&
                metadata.workload !== grant.changes.workload
              ) {
                return { ok: false, error: "metadata_confirmation_required" };
              }
            }
            if (createdTitles.has(title)) return { ok: false, error: "duplicate_create" };
            const snapshotStillMatches =
              expectedSnapshot !== undefined &&
              JSON.stringify(tasks.list().map(taskSummary)) === expectedSnapshot;
            const task = tasks.create(title, metadata);
            createdTitles.add(title);
            if (snapshotStillMatches) {
              expectedSnapshot = JSON.stringify(tasks.list().map(taskSummary));
            }
            return { ok: true, tool, task: taskSummary(task) };
          }
          case "complete_task":
          case "delete_task":
            return mutateTask(tool, args, grant);
          case "update_task":
            return updateTask(args, grant);
        }
      } catch {
        return { ok: false, error: "storage_error" };
      }
    },
  };
}
