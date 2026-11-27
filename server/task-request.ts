import {
  priorities,
  priorityLabels,
  workloads,
  workloadLabels,
  type TaskMetadata,
} from "../shared/task-metadata.ts";

const quotedName = String.raw`(?:「[^「」]+」|“[^“”]+”|"[^"]+"|'[^']+'|‘[^‘’]+’|『[^『』]+』)`;
const namedTask = new RegExp(`^(${quotedName})(?:这(?:一)?[项条](?:任务|待办)?)?$`, "u");
const literalCreate = new RegExp(
  String.raw`^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?(?:新增|新建|创建|添加|再?记(?:一下|一条|下))(?:一(?:项|条|个)(?:任务|待办)?)?\s*(${quotedName})([，,、].*)?[。！.!]?$`,
  "u",
);
const createWithMetadata = new RegExp(
  String.raw`^(?:新增|新建|创建|添加|再?记(?:一下|一条|下))\s*(${quotedName}|[\p{L}\p{N} ]+?)[，,、](.+)$`,
  "u",
);

export const clarifiedTitle = new RegExp(
  String.raw`^我说的是(${quotedName})[，,]\s*请?(?:把|将)它`,
  "u",
);
const targetAndAssignment = new RegExp(
  String.raw`^(${quotedName}|[^「」“”"'‘’『』]+?)(?:的\s*)?((?:优先级|工作量)\s*(?:改|设|调|和).+|(?:改|设(?:置)?|调整?)(?:为|成|到).+)$`,
  "u",
);

export function maskQuotedTitles(input: string): string {
  return input.replace(new RegExp(quotedName, "gu"), "任务名称");
}

export function quotedTaskTitle(target: string) {
  return namedTask.exec(target)?.[1].slice(1, -1);
}

export function quotedCreateTitle(input: string, metadataCreate: boolean) {
  const match = literalCreate.exec(input);
  if (match && (!match[2] || metadataCreate)) return match[1].slice(1, -1);
}

// Only a complete assignment in the user's current message can authorize a value.
function assignments(text: string): TaskMetadata | undefined {
  const clauses = text.replace(/\s/gu, "");
  const paired =
    /^(优先级|工作量)和(优先级|工作量)分别(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])和([低中高小大])$/u.exec(
      clauses,
    );
  const parts = paired
    ? [`${paired[1]}改为${paired[3]}`, `${paired[2]}改为${paired[4]}`]
    : clauses.split(/[，,、]|以及|和/u);
  return fieldValues(
    parts,
    /^(优先级|工作量)(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])$/u,
    /^(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])(优先级|工作量)$/u,
  );
}

function fieldValues(
  parts: string[],
  fieldFirstPattern: RegExp,
  valueFirstPattern: RegExp,
): TaskMetadata | undefined {
  const changes: TaskMetadata = {};
  for (const part of parts) {
    const fieldFirst = fieldFirstPattern.exec(part);
    const valueFirst = valueFirstPattern.exec(part);
    const field = fieldFirst?.[1] ?? valueFirst?.[2];
    const value = fieldFirst?.[2] ?? valueFirst?.[1];
    if (field === "优先级" && !changes.priority) {
      const priority = priorities.find((item) => priorityLabels[item] === value);
      if (!priority) return undefined;
      changes.priority = priority;
    } else if (field === "工作量" && !changes.workload) {
      const workload = workloads.find((item) => workloadLabels[item] === value);
      if (!workload) return undefined;
      changes.workload = workload;
    } else {
      return undefined;
    }
  }
  return changes;
}

type MetadataRequest =
  | { operation: "update_task"; target?: string; changes?: TaskMetadata }
  | { operation: "create_task"; target: string; changes: TaskMetadata | undefined }
  | { operation: "list_tasks" };

export function metadataRequest(content: string): MetadataRequest | undefined {
  const text = content
    .trim()
    .replace(/^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)/u, "")
    .replace(/吧?[。！.!]?$/u, "");
  const instruction = maskQuotedTitles(text);
  if (!/(?:优先级|工作量)/u.test(instruction)) return undefined;

  const body = text.replace(/^(?:把|将|给)\s*/u, "");
  const assignment = targetAndAssignment.exec(body);
  if (assignment) {
    return {
      operation: "update_task",
      target: assignment[1].trim(),
      changes: assignments(assignment[2]),
    };
  }
  const relative = /^(?:提高|降低)(.+?)的?优先级$/u.exec(body);
  if (relative) return { operation: "update_task", target: relative[1].trim() };

  const create = createWithMetadata.exec(text);
  if (create) {
    const changes = fieldValues(
      create[2].split(/[，,、]/u).map((part) => part.trim()),
      /^(优先级|工作量)([低中高小大])$/u,
      /^([低中高小大])(优先级|工作量)$/u,
    );
    return { operation: "create_task", target: create[1].trim(), changes };
  }
  if (/[?？]$/u.test(instruction)) return { operation: "list_tasks" };
  return { operation: "update_task" };
}
