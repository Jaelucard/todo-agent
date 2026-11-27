import { z } from "zod";
import { priorities, workloads } from "./task-metadata.ts";

export const taskMetadataShape = {
  priority: z.enum(priorities).optional(),
  workload: z.enum(workloads).optional(),
};

export const taskMetadataPatchSchema = z
  .strictObject(taskMetadataShape)
  .refine(({ priority, workload }) => priority !== undefined || workload !== undefined);

export const createTaskSchema = z.strictObject({
  title: z.string().trim().min(1).max(200),
  ...taskMetadataShape,
});
