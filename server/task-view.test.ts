import assert from "node:assert/strict";
import { test } from "node:test";
import type { TaskStatus } from "../src/chat-client.ts";
import {
  defaultTaskFilterState,
  filterAndGroupTasks,
  hasActiveTaskFilters,
  isSendShortcut,
  taskStatusFilters,
} from "../src/task-view.ts";

const sampleTasks: TaskStatus[] = [
  { id: "1", title: "架构评审", done: false, priority: "high", workload: "large" },
  { id: "2", title: "买牛奶", done: true, priority: "low", workload: "small" },
  { id: "3", title: "取快递", done: false },
  { id: "4", title: "写周报", done: true, priority: "high", workload: "medium" },
];

test("filterAndGroupTasks groups pending before completed and computes progress accurately", () => {
  const view = filterAndGroupTasks(sampleTasks);
  assert.equal(view.total, 4);
  assert.equal(view.pendingCount, 2);
  assert.equal(view.completedCount, 2);
  assert.equal(view.percent, 50);
  assert.equal(view.filteredCount, 4);
  assert.equal(view.hasActiveFilters, false);
  assert.deepEqual(
    view.groups.map((group) => [group.key, group.items.map((item) => item.id)]),
    [
      ["pending", ["1", "3"]],
      ["completed", ["2", "4"]],
    ],
  );

  const empty = filterAndGroupTasks([]);
  assert.equal(empty.total, 0);
  assert.equal(empty.percent, 0);
  assert.deepEqual(empty.groups, []);
  assert.equal(taskStatusFilters.length, 3);
});

test("filterAndGroupTasks applies status, priority, workload, and unset-metadata filters", () => {
  const highPending = filterAndGroupTasks(sampleTasks, {
    status: "pending",
    priority: "high",
    workload: "*",
  });
  assert.equal(highPending.hasActiveFilters, true);
  assert.equal(highPending.filteredCount, 1);
  assert.equal(highPending.total, 4);
  assert.deepEqual(
    highPending.groups.map((g) => g.items.map((i) => i.id)),
    [["1"]],
  );

  // Unset priority/workload ("") matches tasks without metadata.
  const unset = filterAndGroupTasks(sampleTasks, {
    status: "all",
    priority: "",
    workload: "",
  });
  assert.equal(unset.filteredCount, 1);
  assert.equal(unset.groups[0].items[0].id, "3");

  // Filtered-empty preserves total counts while reporting filteredCount: 0.
  const noMatch = filterAndGroupTasks(sampleTasks, {
    status: "completed",
    priority: "high",
    workload: "large",
  });
  assert.equal(noMatch.total, 4);
  assert.equal(noMatch.filteredCount, 0);
  assert.equal(noMatch.hasActiveFilters, true);
  assert.deepEqual(noMatch.groups, []);
});

test("hasActiveTaskFilters detects default vs active filters", () => {
  assert.equal(hasActiveTaskFilters(defaultTaskFilterState), false);
  assert.equal(hasActiveTaskFilters({ ...defaultTaskFilterState, status: "pending" }), true);
  assert.equal(hasActiveTaskFilters({ ...defaultTaskFilterState, priority: "high" }), true);
  assert.equal(hasActiveTaskFilters({ ...defaultTaskFilterState, workload: "" }), true);
});

test("isSendShortcut allows plain Enter and blocks Shift+Enter and IME composition", () => {
  assert.equal(isSendShortcut({ key: "Enter", shiftKey: false }), true);
  assert.equal(isSendShortcut({ key: "Enter", shiftKey: true }), false);
  assert.equal(isSendShortcut({ key: "a", shiftKey: false }), false);
  assert.equal(isSendShortcut({ key: "Enter", shiftKey: false, isComposing: true }), false);
  assert.equal(isSendShortcut({ key: "Enter", shiftKey: false, keyCode: 229 }), false);
  assert.equal(
    isSendShortcut({ key: "Enter", shiftKey: false, nativeEvent: { isComposing: true } }),
    false,
  );
  assert.equal(
    isSendShortcut({ key: "Enter", shiftKey: false, nativeEvent: { keyCode: 229 } }),
    false,
  );
});
