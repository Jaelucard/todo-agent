import { useState } from "react";
import { priorities, priorityLabels, workloads, workloadLabels } from "../shared/task-metadata.ts";
import type { TaskStatus } from "./chat-client.ts";
import {
  defaultTaskFilterState,
  filterAndGroupTasks,
  taskStatusFilters,
  type TaskStatusFilter,
} from "./task-view.ts";

export interface DashboardState {
  tasks: TaskStatus[] | null;
  status: "loading" | "ready" | "error";
}

export function TaskDashboard({
  state,
  onRefresh,
}: {
  state: DashboardState;
  onRefresh: () => void;
}) {
  const [filter, setFilter] = useState<TaskStatusFilter>(defaultTaskFilterState.status);
  const [priorityFilter, setPriorityFilter] = useState(defaultTaskFilterState.priority);
  const [workloadFilter, setWorkloadFilter] = useState(defaultTaskFilterState.workload);
  const { tasks, status } = state;

  const { total, pendingCount, completedCount, percent, filteredCount, hasActiveFilters, groups } =
    filterAndGroupTasks(tasks ?? [], {
      status: filter,
      priority: priorityFilter,
      workload: workloadFilter,
    });

  function resetFilters() {
    setFilter(defaultTaskFilterState.status);
    setPriorityFilter(defaultTaskFilterState.priority);
    setWorkloadFilter(defaultTaskFilterState.workload);
  }

  return (
    <section className="tasks pane" aria-labelledby="tasks-heading" id="workspace-tasks">
      <div className="pane-heading">
        <h2 id="tasks-heading" tabIndex={-1}>
          任务清单
        </h2>
        <button
          type="button"
          className="secondary-button"
          disabled={status === "loading"}
          onClick={onRefresh}
        >
          {status === "loading" ? "刷新中…" : "刷新任务"}
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
                <dd>{pendingCount}</dd>
              </div>
              <div>
                <dt>已完成</dt>
                <dd>{completedCount}</dd>
              </div>
            </dl>
            <div className="progress-label">
              <label htmlFor="completion-progress">完成进度</label>
              <span>{percent}%</span>
            </div>
            <progress id="completion-progress" value={completedCount} max={total || 1} />
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
            {taskStatusFilters.map(({ key, label }) => (
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
          {hasActiveFilters && (
            <div className="active-filter-bar">
              <span>{`已筛选：显示 ${filteredCount} / ${total} 项`}</span>
              {groups.length > 0 && (
                <button type="button" className="secondary-button" onClick={resetFilters}>
                  重置筛选
                </button>
              )}
            </div>
          )}
        </>
      )}
      <div className="task-content" aria-busy={status === "loading"}>
        {status === "loading" && tasks === null && <p role="status">正在读取任务…</p>}
        {status === "error" && (
          <p className="error" role="alert">
            {tasks === null ? "读取任务失败，请刷新重试。" : "刷新任务失败，当前清单可能已过时。"}
          </p>
        )}
        {tasks !== null && (
          <div>
            {!groups.length && (
              <div className="task-empty">
                <p>{total ? "没有符合筛选条件的任务" : "暂无任务"}</p>
                {hasActiveFilters && (
                  <button type="button" className="secondary-button" onClick={resetFilters}>
                    重置筛选
                  </button>
                )}
              </div>
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
