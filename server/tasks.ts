import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { Router } from "express";
import { z } from "zod";

const createTask = z.strictObject({ title: z.string().trim().min(1).max(200) });
const updateTask = createTask
  .partial()
  .extend({ done: z.boolean().optional() })
  .refine((value) => Object.keys(value).length > 0);

export function taskRouter(filename = join("data", "tasks.json")) {
  const router = Router();
  type Task = { id: string; title: string; done: boolean };

  function readTasks(): Task[] {
    return existsSync(filename) ? JSON.parse(readFileSync(filename, "utf8")) : [];
  }

  function saveTasks(tasks: Task[]) {
    mkdirSync(dirname(filename), { recursive: true });
    const temporary = filename + ".tmp";
    writeFileSync(temporary, JSON.stringify(tasks, null, 2) + "\n");
    renameSync(temporary, filename);
  }

  router.get("/", (_req, res) => {
    res.json(readTasks());
  });

  router.post("/", (req, res) => {
    const input = createTask.safeParse(req.body);
    if (!input.success)
      return res.status(400).json({ error: "请提供 1 至 200 个字符的任务标题。" });
    const task = { id: randomUUID(), title: input.data.title, done: false };
    saveTasks([...readTasks(), task]);
    res.status(201).json(task);
  });

  router.patch("/:id", (req, res) => {
    const input = updateTask.safeParse(req.body);
    if (!input.success) return res.status(400).json({ error: "请提供有效的 title 或 done 字段。" });
    const tasks = readTasks();
    const task = tasks.find((item) => item.id === req.params.id);
    if (!task) return res.status(404).json({ error: "找不到这条任务。" });
    Object.assign(task, input.data);
    saveTasks(tasks);
    res.json(task);
  });

  router.delete("/:id", (req, res) => {
    const tasks = readTasks();
    const remaining = tasks.filter((item) => item.id !== req.params.id);
    if (remaining.length === tasks.length)
      return res.status(404).json({ error: "找不到这条任务。" });
    saveTasks(remaining);
    res.sendStatus(204);
  });

  return router;
}
