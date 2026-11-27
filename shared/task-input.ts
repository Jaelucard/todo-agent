import { z } from "zod";
import { priorities, workloads } from "./task-metadata.ts";

export const bidiControlPattern = /[\u202A-\u202E\u2066-\u2069]/gu;
const invisibleOnlyPattern = /[\s\p{Cf}]/gu;

export function stripBidiControls(text: string): string {
  return text.replace(bidiControlPattern, "");
}

export function hasVisibleText(text: string): boolean {
  return text.replace(invisibleOnlyPattern, "").length > 0;
}

export const taskTitleSchema = z
  .string()
  .transform((value) => stripBidiControls(value).trim())
  .refine((value) => value.length >= 1 && value.length <= 200, {
    message: "title_length",
  })
  .refine((value) => hasVisibleText(value), {
    message: "title_invisible",
  });

export const taskMetadataShape = {
  priority: z.enum(priorities).optional(),
  workload: z.enum(workloads).optional(),
};

export const taskMetadataPatchSchema = z
  .strictObject(taskMetadataShape)
  .refine(({ priority, workload }) => priority !== undefined || workload !== undefined);

export const taskUpdatePatchSchema = z
  .strictObject({
    title: taskTitleSchema.optional(),
    done: z.boolean().optional(),
    ...taskMetadataShape,
  })
  .refine(
    ({ title, done, priority, workload }) =>
      title !== undefined || done !== undefined || priority !== undefined || workload !== undefined,
  );

export const createTaskSchema = z.strictObject({
  title: taskTitleSchema,
  ...taskMetadataShape,
});
