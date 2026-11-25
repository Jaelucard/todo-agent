import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { Router } from "express";
import { z } from "zod";

const createTask = z.strictObject({ title: z.string().trim().min(1).max(200) });
const updateTask = createTask
  .partial()
  .extend({ done: z.boolean().optional() })
  .refine((value) => Object.keys(value).length > 0);

export function taskRouter(filename = join("data", "tasks.sqlite")) {
  const router = Router();
  let connection: DatabaseSync | undefined;

  function db() {
    if (!connection) {
      mkdirSync(dirname(filename), { recursive: true });
      const database = new DatabaseSync(filename);
      database.exec(
        "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0)",
      );
      connection = database;
    }
    return connection;
  }

  function task(row: Record<string, unknown>) {
    return { id: row.id, title: row.title, done: Boolean(row.done) };
  }
  router.get("/", (_req, res) => {
    res.json(db().prepare("SELECT * FROM tasks ORDER BY rowid").all().map(task));
  });

  router.post("/", (req, res) => {
    const input = createTask.safeParse(req.body);
    if (!input.success)
      return res.status(400).json({ error: "请提供 1 至 200 个字符的任务标题。" });
    const id = randomUUID();
    db().prepare("INSERT INTO tasks (id, title) VALUES (?, ?)").run(id, input.data.title);
    res.status(201).json({ id, title: input.data.title, done: false });
  });

  router.patch("/:id", (req, res) => {
    const input = updateTask.safeParse(req.body);
    if (!input.success) return res.status(400).json({ error: "请提供有效的 title 或 done 字段。" });
    const { title, done } = input.data;
    const row = db()
      .prepare(
        "UPDATE tasks SET title = COALESCE(?, title), done = COALESCE(?, done) WHERE id = ? RETURNING *",
      )
      .get(title ?? null, done === undefined ? null : Number(done), req.params.id);
    if (!row) return res.status(404).json({ error: "找不到这条任务。" });
    res.json(task(row));
  });

  router.delete("/:id", (req, res) => {
    const result = db().prepare("DELETE FROM tasks WHERE id = ?").run(req.params.id);
    if (!result.changes) return res.status(404).json({ error: "找不到这条任务。" });
    res.sendStatus(204);
  });

  return router;
}
