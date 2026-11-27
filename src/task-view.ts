import type { TaskStatus } from "./chat-client.ts";

export const taskStatusFilters = [
  { key: "all", label: "全部" },
  { key: "pending", label: "待完成" },
  { key: "completed", label: "已完成" },
] as const;

export type TaskStatusFilter = (typeof taskStatusFilters)[number]["key"];

export interface TaskFilterState {
  status: TaskStatusFilter;
  priority: string;
  workload: string;
}

export const defaultTaskFilterState: TaskFilterState = {
  status: "all",
  priority: "*",
  workload: "*",
};

export interface TaskGroup {
  key: "pending" | "completed";
  title: string;
  items: TaskStatus[];
}

export interface TaskViewSummary {
  total: number;
  pendingCount: number;
  completedCount: number;
  percent: number;
  filteredCount: number;
  hasActiveFilters: boolean;
  groups: TaskGroup[];
}

export interface SendShortcutEvent {
  key: string;
  shiftKey: boolean;
  isComposing?: boolean;
  keyCode?: number;
  nativeEvent?: {
    isComposing?: boolean;
    keyCode?: number;
  };
}

function matchesMetadataFilter(value: string | undefined, selection: string): boolean {
  return selection === "*" || (value ?? "") === selection;
}

export function hasActiveTaskFilters(filters: TaskFilterState): boolean {
  return filters.status !== "all" || filters.priority !== "*" || filters.workload !== "*";
}

export function filterAndGroupTasks(
  tasks: readonly TaskStatus[],
  filters: TaskFilterState = defaultTaskFilterState,
): TaskViewSummary {
  const pending = tasks.filter((task) => !task.done);
  const completed = tasks.filter((task) => task.done);
  const total = tasks.length;
  const percent = total ? Math.round((completed.length / total) * 100) : 0;

  const candidateGroups: TaskGroup[] = [
    { key: "pending", title: "待完成", items: pending },
    { key: "completed", title: "已完成", items: completed },
  ];

  const groups = candidateGroups
    .filter(({ key }) => filters.status === "all" || filters.status === key)
    .map((group) => ({
      ...group,
      items: group.items.filter(
        (task) =>
          matchesMetadataFilter(task.priority, filters.priority) &&
          matchesMetadataFilter(task.workload, filters.workload),
      ),
    }))
    .filter(({ items }) => items.length > 0);

  const filteredCount = groups.reduce((sum, group) => sum + group.items.length, 0);

  return {
    total,
    pendingCount: pending.length,
    completedCount: completed.length,
    percent,
    filteredCount,
    hasActiveFilters: hasActiveTaskFilters(filters),
    groups,
  };
}

export function isSendShortcut(event: SendShortcutEvent): boolean {
  // 中文输入法会用 Enter 确认选词，部分浏览器此时只报告键码 229。
  const composing =
    Boolean(event.nativeEvent?.isComposing) ||
    event.nativeEvent?.keyCode === 229 ||
    Boolean(event.isComposing) ||
    event.keyCode === 229;
  return event.key === "Enter" && !event.shiftKey && !composing;
}
