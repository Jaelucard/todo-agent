import { useState } from "react";
import { priorities, priorityLabels, workloads, workloadLabels } from "../shared/task-metadata.ts";
import type { TaskStatus } from "./chat-client.ts";

const filters = [
  { key: "all", label: "全部" },
  { key: "pending", label: "待完成" },
  { key: "completed", label: "已完成" },
] as const;

export interface DashboardState {
  tasks: TaskStatus[] | null;
  status: "loading" | "ready" | "error";
}

function matchesFilter(value: string | undefined, selection: string) {
  return selection === "*" || (value ?? "") === selection;
}

export function TaskDashboard({
  state,
  onRefresh,
}: {
  state: DashboardState;
  onRefresh: () => void;
}) {
  const [filter, setFilter] = useState<(typeof filters)[number]["key"]>("all");
  const [priorityFilter, setPriorityFilter] = useState("*");
  const [workloadFilter, setWorkloadFilter] = useState("*");
  const { tasks, status } = state;
  const pending = tasks?.filter((task) => !task.done) ?? [];
  const completed = tasks?.filter((task) => task.done) ?? [];
  const total = tasks?.length ?? 0;
  const percent = total ? Math.round((completed.length / total) * 100) : 0;
  const groups = [
    { key: "pending", title: "待完成", items: pending },
    { key: "completed", title: "已完成", items: completed },
  ]
    .filter(({ key }) => filter === "all" || filter === key)
    .map((group) => ({
      ...group,
      items: group.items.filter(
        (task) =>
          matchesFilter(task.priority, priorityFilter) &&
          matchesFilter(task.workload, workloadFilter),
      ),
    }))
    .filter(({ items }) => items.length > 0);

  return (
    <section className="tasks pane" aria-labelledby="tasks-heading" id="workspace-tasks">
      <div className="pane-heading">
        <h2 id="tasks-heading" tabIndex={-1}>
          任务清单
        </h2>
        <button type="button" className="secondary-button" onClick={onRefresh}>
          刷新任务
        </button>
      </div>

      {tasks !== null && (
        <>
          <div className="task-overview">
            <dl className="task-counts">
              <div>
                <dt>全部任务</dt>
                <dd>{total}</dd>
              </div>
              <div>
                <dt>待完成</dt>
                <dd>{pending.length}</dd>
              </div>
              <div>
                <dt>已完成</dt>
                <dd>{completed.length}</dd>
              </div>
            </dl>
            <div className="progress-label">
              <label htmlFor="completion-progress">完成进度</label>
              <span>{percent}%</span>
            </div>
            <progress id="completion-progress" value={completed.length} max={total || 1} />
          </div>
          <div className="metadata-filters">
            <div>
              <label htmlFor="filter-priority">优先级</label>
              <select
                id="filter-priority"
                value={priorityFilter}
                onChange={(event) => setPriorityFilter(event.target.value)}
              >
                <option value="*">全部优先级</option>
                {priorities.map((priority) => (
                  <option key={priority} value={priority}>
                    {priorityLabels[priority]}
                  </option>
                ))}
                <option value="">未设置</option>
              </select>
            </div>
            <div>
              <label htmlFor="filter-workload">工作量</label>
              <select
                id="filter-workload"
                value={workloadFilter}
                onChange={(event) => setWorkloadFilter(event.target.value)}
              >
                <option value="*">全部工作量</option>
                {workloads.map((workload) => (
                  <option key={workload} value={workload}>
                    {workloadLabels[workload]}
                  </option>
                ))}
                <option value="">未设置</option>
              </select>
            </div>
          </div>
          <div className="task-filters" role="group" aria-label="任务状态筛选">
            {filters.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                aria-pressed={filter === key}
                onClick={() => setFilter(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </>
      )}
      <div className="task-content" aria-busy={status === "loading"}>
        {status === "loading" && <p role="status">正在读取任务…</p>}
        {status === "error" && (
          <p className="error" role="alert">
            {tasks === null ? "读取任务失败，请刷新重试。" : "刷新任务失败，当前清单可能已过时。"}
          </p>
        )}
        {tasks !== null && (
          <div aria-label="当前任务">
            {!groups.length && (
              <p className="task-empty">{total ? "没有符合筛选条件的任务" : "暂无任务"}</p>
            )}
            {groups.map(({ key, title, items }) => (
              <div className="task-group" key={key}>
                <h3>{title}</h3>
                <ul>
                  {items.map((task) => (
                    <li key={task.id} className="task-row">
                      <div className="task-details">
                        <span className="task-title">{task.title}</span>
                        {(task.priority || task.workload) && (
                          <div className="task-metadata">
                            {task.priority && <span>优先级：{priorityLabels[task.priority]}</span>}
                            {task.workload && <span>工作量：{workloadLabels[task.workload]}</span>}
                          </div>
                        )}
                      </div>
                      <span className={task.done ? "task-status completed" : "task-status"}>
                        {task.done ? "已完成" : "待完成"}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
