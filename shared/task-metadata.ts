export const priorities = ["low", "medium", "high"] as const;
export const workloads = ["small", "medium", "large"] as const;

export interface TaskMetadata {
  priority?: (typeof priorities)[number];
  workload?: (typeof workloads)[number];
}

export const priorityLabels = { low: "低", medium: "中", high: "高" } as const;
export const workloadLabels = { small: "小", medium: "中", large: "大" } as const;

export function readTaskMetadata(task: object): TaskMetadata {
  const priority = priorities.find((value) => "priority" in task && task.priority === value);
  const workload = workloads.find((value) => "workload" in task && task.workload === value);
  return { ...(priority && { priority }), ...(workload && { workload }) };
}
