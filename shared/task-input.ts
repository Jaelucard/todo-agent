import { z } from "zod";

export const createTaskSchema = z.strictObject({ title: z.string().trim().min(1).max(200) });
