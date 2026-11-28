import { createTaskSchema, taskTitleSchema } from "../shared/task-input.ts";
import {
  priorities,
  priorityLabels,
  workloads,
  workloadLabels,
  type TaskMetadata,
} from "../shared/task-metadata.ts";
import type { ListTasksFilter } from "./task-tools.ts";

type Priority = NonNullable<TaskMetadata["priority"]>;
type Workload = NonNullable<TaskMetadata["workload"]>;

const quotedName = String.raw`(?:「[^「」]+」|“[^“”]+”|"[^"]+"|'[^']+'|‘[^‘’]+’|『[^『』]+』)`;
const quotedGlobal = new RegExp(quotedName, "gu");
const namedTask = new RegExp(`^(${quotedName})(?:这(?:一)?[项条个](?:任务|待办)?)?$`, "u");
const literalCreate = new RegExp(
  String.raw`^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?(?:新增|新建|创建|添加|再?记(?:一下|一条|下))(?:一(?:项|条|个)(?:任务|待办)?)?\s*(${quotedName})([，,、].*)?[。！.!]?$`,
  "u",
);
const createWithMetadata = new RegExp(
  String.raw`^(?:新增|新建|创建|添加|再?记(?:一下|一条|下))(?:一(?:项|条|个)(?:任务|待办)?)?\s*(${quotedName}|[^「」“”"'‘’『』，,、]+?)[，,、](.+)$`,
  "u",
);
const adjectiveCreateWithMetadata = new RegExp(
  String.raw`^(?:加|新增|新建|创建|添加|再?记)(?:一?[个条项])?(?:的)?(.+?)(?:的)?(?:任务|待办(?:事项)?)[：:]\s*(.+)$`,
  "u",
);
const standaloneCreate =
  /^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?\s*(?:再?记(?:(一[条项个])|一下|下)|再?(?:新增|新建|创建|添加)(一[条项个])?)\s*(?:(?:任务|待办)(?=[\s：:])\s*)?[：:]?\s*(.+)$/u;
const standaloneNamedEdit = new RegExp(
  String.raw`^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?\s*再?(?:把|将)\s*(${quotedName})\s*(.+)$`,
  "u",
);
const standaloneQuotedTitle = new RegExp(`^(${quotedName})$`, "u");
const renameAssignment =
  /^(?:(?:重命名|改名|更名)(?:为|成|叫)|(?:的)?(?:标题|名称|名字)(?:改|修改|换|设)(?:为|成))\s*[：:]?\s*(.+)$/u;
const reopenAssignment =
  /^(?:(?:改(?:回|为|成)|恢复(?:为|成)?|重新标记为|设(?:置)?为|标记为)(?:未|待)完成|取消(?:已)?完成)(?:状态)?$/u;
const plainTitle = /^[\p{L}\p{N}][\p{L}\p{N} \t]*$/u;
const ambiguousTitle =
  /(?:和|与|或|并|以及|然后|再|新增|新建|创建|添加|加上|接着|顺便|同时|另外|还有|还要|算了|还是|如果|假如|若是|先不|不用|不必|同名|一样的|相同的|随便|任选|到期|截止|逾期|提醒|[吗呢嘛吧啊呀呗]$|^(?:上次|前面|之前|刚才|刚刚|上述|上面)(?:的|这个|那个))/u;

export const clarifiedTitle = new RegExp(
  String.raw`^我说的是(${quotedName})[，,]\s*请?(?:把|将)它`,
  "u",
);

const targetAndAssignment = new RegExp(
  String.raw`^(${quotedName}|[^「」“”"'‘’『』]+?)(?:的\s*)?((?:优先级|工作量)\s*(?:改|设|调|降|升|和).+|(?:改|设(?:置)?|调整?)(?:为|成|到).+)$`,
  "u",
);

const anaphoricPattern =
  /^(?:它|(?:这一?|那一?|上一|刚(?:才|刚)那|上面那)(?:条|项|个|件)?|这件事|那件事|刚才说的事|刚刚说的事|(?:刚(?:才|刚)?|前面|之前|上面)(?:说|提|加|添加|新增|新建|创建|记(?:录|下)?|改|修改|更新|完成)(?:到|过|了)?的(?:[这那]?一?[条项个件])?(?:事(?:情)?|任务|待办(?:事项)?)?)(?:任务|待办(?:事项)?|事(?:情)?)?$/u;

const positionalPattern =
  /^(?:(?:\[#\d+\]|#\d+)|最后(?:一|那一?)(?:条|项|个|件)|(?:倒数)?第[\d一二三四五六七八九十两百]+(?:条|项|个|件))(?:任务|待办(?:事项)?|事(?:情)?)?$/u;

export function maskQuotedTitles(input: string): string {
  return input.replace(quotedGlobal, "任务名称");
}

export function extractQuotedTitles(input: string): string[] {
  const titles: string[] = [];
  for (const match of input.matchAll(quotedGlobal)) {
    titles.push(match[0].slice(1, -1));
  }
  return titles;
}

export function quotedTaskTitle(target: string): string | undefined {
  return namedTask.exec(target.trim())?.[1].slice(1, -1);
}

export function quotedCreateTitle(input: string, metadataCreate: boolean): string | undefined {
  const match = literalCreate.exec(input);
  if (match && (!match[2] || metadataCreate)) return match[1].slice(1, -1);
  return undefined;
}

export function isAnaphoricTarget(target: string): boolean {
  const text = target.replace(/\s/gu, "");
  if (!text) return false;
  return anaphoricPattern.test(text);
}

export function isPositionalTarget(target: string): boolean {
  const text = target.replace(/\s/gu, "");
  if (!text) return false;
  return positionalPattern.test(text);
}

export function isUnboundTarget(target: string): boolean {
  return isAnaphoricTarget(target) || isPositionalTarget(target);
}

// A literal quoted name can establish absence. A clean unquoted name can establish an exact match.
export function namedTarget(target: string): { title: string; literal: boolean } | undefined {
  const trimmed = target.trim();
  const quoted = quotedTaskTitle(trimmed);
  if (quoted !== undefined) return { title: quoted, literal: true };
  if (
    !trimmed ||
    /[「」“”"'‘’『』\r\n]/u.test(trimmed) ||
    /(?:不|别|勿|如果|是否|然后|或|并|和|以及|再把|[个条项]$)/u.test(trimmed)
  ) {
    return undefined;
  }
  return { title: trimmed, literal: false };
}

function splitClausesOutsideQuotes(input: string): string[] {
  const clauses: string[] = [];
  const openToClose = new Map<string, string>([
    ["「", "」"],
    ["“", "”"],
    ["『", "』"],
    ["‘", "’"],
    ['"', '"'],
    ["'", "'"],
  ]);
  let current = "";
  let closing: string | undefined;
  for (const char of input) {
    if (closing) {
      current += char;
      if (char === closing) closing = undefined;
      continue;
    }
    const maybeClose = openToClose.get(char);
    if (maybeClose) {
      closing = maybeClose;
      current += char;
      continue;
    }
    if (/[，,；;。！!]/u.test(char)) {
      if (current.trim()) clauses.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  if (current.trim()) clauses.push(current.trim());
  return clauses;
}

function parseChineseOrArabicNumber(token: string): number | undefined {
  if (/^\d+$/u.test(token)) {
    const value = Number(token);
    return value >= 1 && value <= 20 ? value : undefined;
  }
  const digits: Record<string, number> = {
    一: 1,
    二: 2,
    两: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
  };
  if (token === "十") return 10;
  if (token in digits) return digits[token];
  if (token.startsWith("十") && token.slice(1) in digits) {
    return 10 + digits[token.slice(1)];
  }
  if (token === "二十") return 20;
  return undefined;
}

export function parseDisambiguationIndex(
  input: string,
): { index: number; bare: boolean } | undefined {
  const trimmed = input
    .trim()
    .replace(/[。！!？?吧啊呀呢]+$/u, "")
    .trim();
  const bareHash = /^(?:(?:选|要|是|就|用)?\s*)?(?:\[#(\d+)\]|#(\d+)|(\d+))$/u.exec(trimmed);
  if (bareHash) {
    const value = parseChineseOrArabicNumber(bareHash[1] ?? bareHash[2] ?? bareHash[3]);
    if (value !== undefined) return { index: value, bare: true };
  }
  const bareOrdinal =
    /^(?:(?:选|要|是|就|用)?\s*)?第([\d一二三四五六七八九十两]+)(?:条|个|项|件)(?:任务|待办)?$/u.exec(
      trimmed,
    );
  if (bareOrdinal) {
    const value = parseChineseOrArabicNumber(bareOrdinal[1]);
    if (value !== undefined) return { index: value, bare: true };
  }
  const masked = maskQuotedTitles(trimmed);
  const embeddedHash = /(?:\[#(\d+)\]|#(\d+))/u.exec(masked);
  if (embeddedHash) {
    const value = parseChineseOrArabicNumber(embeddedHash[1] ?? embeddedHash[2]);
    if (value !== undefined) return { index: value, bare: false };
  }
  const embeddedOrdinal = /(?<!倒数)第([\d一二三四五六七八九十两]+)(?:条|个|项|件)/u.exec(masked);
  if (embeddedOrdinal) {
    const value = parseChineseOrArabicNumber(embeddedOrdinal[1]);
    if (value !== undefined) return { index: value, bare: false };
  }
  return undefined;
}

export function isUnsupportedDueDateQuery(input: string): boolean {
  const masked = maskQuotedTitles(input.trim());
  if (!/(?:到期|截止|ddl|deadline|过期|逾期|提醒时间|定时提醒)/iu.test(masked)) return false;
  if (
    /^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?(?:新增|新建|创建|添加|加一|加个|加条|再?记)/u.test(masked)
  ) {
    return false;
  }
  return true;
}

const trailingListClausePattern =
  /^(?:再|然后|最后|顺便|并|并且|同时|接着|之后|另外)?(?:请)?(?:给我|帮我)?(?:看看|看一看|看下|看一下|查看(?:一下|下)?|显示(?:一下|下)?|展示(?:一下|下)?|列出|告诉我)(?:一下|下)?.*(?:清单|列表|待办|任务|剩|有什么|有哪些)/u;

const clauseNegationPattern =
  /(?:千万)?(?:不要|(?<!分|特|级|类|区|告|差|识|判|个|性)别(?!人|的|处|扭)|勿|不用|不必|暂不|不想|不打算|不要动|别动)/u;

export function extractTrailingListRequest(input: string): {
  wantsList: boolean;
  commandText: string;
  filter: ListTasksFilter;
} {
  const clauses = splitClausesOutsideQuotes(input);
  if (clauses.length < 2) {
    return { wantsList: false, commandText: input.trim(), filter: {} };
  }
  const lastClause = clauses.at(-1)!;
  const maskedLast = maskQuotedTitles(lastClause);
  if (clauseNegationPattern.test(maskedLast) || !trailingListClausePattern.test(maskedLast)) {
    return { wantsList: false, commandText: input.trim(), filter: {} };
  }
  const filter: ListTasksFilter = {};
  if (/(?:未完成|待完成|还没完成|没做完)/u.test(maskedLast)) filter.done = false;
  else if (/(?:已完成|做完了)/u.test(maskedLast)) filter.done = true;
  const prio = /([低中高])优先级/u.exec(maskedLast);
  if (prio) filter.priority = priorities.find((item) => priorityLabels[item] === prio[1]);
  const work = /([小中大])工作量/u.exec(maskedLast);
  if (work) filter.workload = workloads.find((item) => workloadLabels[item] === work[1]);

  const commandText = clauses.slice(0, -1).join("，");
  return { wantsList: true, commandText, filter };
}

function fieldValues(
  parts: string[],
  fieldFirstPattern: RegExp,
  valueFirstPattern: RegExp,
): TaskMetadata | undefined {
  const changes: TaskMetadata = {};
  for (const rawPart of parts) {
    const part = rawPart.trim();
    if (!part) return undefined;
    const fieldFirst = fieldFirstPattern.exec(part);
    const valueFirst = valueFirstPattern.exec(part);
    const field = fieldFirst?.[1] ?? valueFirst?.[2];
    const value = fieldFirst?.[2] ?? valueFirst?.[1];
    if (field === "优先级" && !changes.priority) {
      const priority = priorities.find((item) => priorityLabels[item] === value);
      if (!priority) return undefined;
      changes.priority = priority;
    } else if (field === "工作量" && !changes.workload) {
      const workload = workloads.find((item) => workloadLabels[item] === value);
      if (!workload) return undefined;
      changes.workload = workload;
    } else {
      return undefined;
    }
  }
  return changes.priority || changes.workload ? changes : undefined;
}

function assignments(text: string): TaskMetadata | undefined {
  const clauses = text.replace(/\s/gu, "");
  const paired =
    /^(优先级|工作量)和(优先级|工作量)分别(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])和([低中高小大])$/u.exec(
      clauses,
    );
  const parts = paired
    ? [`${paired[1]}改为${paired[3]}`, `${paired[2]}改为${paired[4]}`]
    : clauses.split(/[，,、]|以及|和/u);
  return fieldValues(
    parts,
    /^(优先级|工作量)(?:改|设(?:置)?|调整?|降|升)(?:为|成|到|至)([低中高小大])$/u,
    /^(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])(优先级|工作量)$/u,
  );
}

function parseCreateMetadataClauses(text: string): TaskMetadata | undefined {
  const parts = text
    .split(/[，,、]|以及|和/u)
    .map((part) => part.replace(/\s/gu, ""))
    .filter(Boolean);
  if (!parts.length) return undefined;
  return fieldValues(
    parts,
    /^(优先级|工作量)(?:改为|改成|设为|设成|为|是|：|:)?([低中高小大])$/u,
    /^([低中高小大])(优先级|工作量)$/u,
  );
}

export function singleCreateGrant(target: string, changes: TaskMetadata | undefined) {
  const name = namedTarget(target);
  if (!name || !changes) return undefined;
  if (
    !name.literal &&
    (isUnboundTarget(target) ||
      /(?:吗|呢|么|嘛|还是|再|顺便|同时|接着|还要|若|否则|取消|删掉|删除|移除|清空|^一[条项个]任务)/u.test(
        target,
      ))
  ) {
    return undefined;
  }
  const input = createTaskSchema.safeParse({ title: name.title, ...changes });
  if (input.success) return { title: input.data.title, changes };
  return undefined;
}

export type SingleMetadataRequest =
  | { operation: "update_task"; target?: string; changes?: TaskMetadata }
  | { operation: "create_task"; target: string; changes: TaskMetadata | undefined }
  | { operation: "list_tasks" }
  | { operation: "invalid_metadata" };

// Attempts to recognize clean single-task metadata creates/updates or clearly invalid metadata requests.
// Returns undefined for general/compound requests so the LLM can handle natural phrasing under TurnPolicy.
export function singleMetadataRequest(content: string): SingleMetadataRequest | undefined {
  const text = content
    .trim()
    .replace(/^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)/u, "")
    .replace(/吧?[。！.!]?$/u, "");
  const instruction = maskQuotedTitles(text);
  if (!/(?:优先级|工作量)/u.test(instruction)) return undefined;

  if (/[?？]$/u.test(instruction)) return { operation: "list_tasks" };

  // Reject clearly conflicting/vague/negated/hypothetical metadata expressions upfront.
  if (
    /^(?:不要|别|勿|不用|不必|先不要|先别|如果|假如|要是|若是)/u.test(instruction) ||
    /(?:好吗|行吗|可以吗|吗|是否)/u.test(instruction) ||
    /(?:优先级|工作量)(?:特别|非常|超|很|挺|稍微|略|更|较)[低中高小大]/u.test(instruction) ||
    /(?:优先级|工作量)[低中高小大]一点/u.test(instruction)
  ) {
    return { operation: "invalid_metadata" };
  }

  // Reject dangling metadata after a deletion clause (e.g. "帮我记一下取快递，再把交报告删掉，优先级高").
  if (
    /(?:删掉|删除|移除)[，,、]\s*(?:优先级|工作量|[低中高]优先级|[小中大]工作量)/u.test(instruction)
  ) {
    return { operation: "invalid_metadata" };
  }

  const body = text.replace(/^(?:把|将|给)\s*/u, "");
  const assignment = targetAndAssignment.exec(body);
  if (assignment) {
    const parsedTarget = assignment[1].trim();
    const parsedChanges = assignments(assignment[2]);
    // Only treat as single-task edit when target does not look like a compound multi-clause sentence.
    if (!/[，,；;。！!]/u.test(maskQuotedTitles(parsedTarget))) {
      if (
        !parsedChanges &&
        /(?:另外|再把|然后把|并且把|同时把|删掉|删除|移除|新增|添加|完成|重命名|改名|改回)/u.test(
          maskQuotedTitles(assignment[2]),
        )
      ) {
        return undefined;
      }
      return {
        operation: "update_task",
        target: parsedTarget,
        changes: parsedChanges,
      };
    }
  }

  const relative = /^(?:提高|降低|调高|调低)(.+?)的?优先级$/u.exec(body);
  if (relative) return { operation: "update_task", target: relative[1].trim() };

  const adjCreate = adjectiveCreateWithMetadata.exec(text);
  if (adjCreate) {
    const changes = parseCreateMetadataClauses(adjCreate[1]);
    if (changes) {
      return { operation: "create_task", target: adjCreate[2].trim(), changes };
    }
  }

  const create = createWithMetadata.exec(text);
  if (create) {
    const changes = parseCreateMetadataClauses(create[2]);
    if (changes) {
      return { operation: "create_task", target: create[1].trim(), changes };
    }
    // Check for duplicate conflicting fields (e.g. "优先级中，优先级高") or ambiguous unquoted multi-target create.
    if (
      /(?:优先级.*优先级|工作量.*工作量)/u.test(maskQuotedTitles(create[2])) ||
      /(?:和|以及)/u.test(maskQuotedTitles(create[1]))
    ) {
      return { operation: "invalid_metadata" };
    }
  }

  return undefined;
}

export interface ClauseAnalysis {
  mutationsAllowed: boolean;
  hasProhibition: boolean;
  forbiddenTitles: Set<string>;
  clauseCount: number;
  positiveClauses: string[];
  positiveQuotedTitles: string[];
  hasUnquotedPositiveActionClause: boolean;
  allowedPriorities: Set<Priority>;
  allowedWorkloads: Set<Workload>;
  hasMetadataMention: boolean;
  hasInvalidMetadataValue: boolean;
  allowCreate: boolean;
  allowComplete: boolean;
  allowDelete: boolean;
  allowReopen: boolean;
  allowRename: boolean;
  isPureCreate: boolean;
  isPureComplete: boolean;
  isPureDelete: boolean;
  isPureReopen: boolean;
  isPureRename: boolean;
  hasCompleteVerb: boolean;
  hasDeleteVerb: boolean;
  hasPositiveCreateCommand: boolean;
}

const retractionClausePattern = /^(?:(?:还是)?算了(?:吧)?|罢了|没事了|不麻烦了)$/u;
const wholeTurnRetractionClausePattern =
  /^(?:(?:还是)?算了(?:吧)?|罢了|没事了|不麻烦了|(?:(?:还是)?算了(?:吧)?\s*)?(?:还是|先)?(?:不用|不要|(?<!分|特|级|类|区|告|差|识|判|个|性)别(?!人|的|处|扭)|不必)(?:记|加|新增|创建|添加|删|改|做|办|完成|动|管)了)$/u;
const keepNoOpClausePattern =
  /^(?:(?:先|还是|就)?(?:留着|保留|保持原样|不动|别动|放着|搁着)(?:吧|好了|就行)?)$/u;
const tentativeIntentPattern = /^(?:我)?(?:本来|原本|刚才|刚刚)想/u;
const retractionCuePattern =
  /(?:算了|(?:不用|不要|(?<!分|特|级|类|区|告|差|识|判|个|性)别(?!人|的|处|扭)|不必)(?:记|加|新增|创建|添加|删|改|做|办|完成|动|管)了?)/u;

const deletionVerb = "(?:删掉|删除|删|移除|清掉|去掉|清除)";
const deletionWithoutObject = new RegExp(`^${deletionVerb}(?:了|吧)?$`, "u");
const deletionWithObjectFirst = new RegExp(
  String.raw`^(?:把|将)\s*(.+?)\s*${deletionVerb}(?:了|吧)?$`,
  "u",
);
const deletionWithVerbFirst = new RegExp(String.raw`^${deletionVerb}\s*(.+)$`, "u");

function unquotedProhibitedDeletionTargets(clause: string): string[] {
  const negation = clauseNegationPattern.exec(clause);
  if (!negation) return [];
  const before = clause.slice(0, negation.index).trim();
  const after = clause
    .slice(negation.index + negation[0].length)
    .trimStart()
    .replace(/^(?:先|再)\s*/u, "");
  const objectBeforeNegation = deletionWithoutObject.test(after);
  const target = objectBeforeNegation
    ? before
    : (deletionWithObjectFirst.exec(after)?.[1] ?? deletionWithVerbFirst.exec(after)?.[1]);
  if (!target) return [];

  const undecorated = (
    objectBeforeNegation
      ? target.replace(/^(?:但是|不过|并且|而且|另外|但)\s*/u, "").replace(/先$/u, "")
      : target.replace(/吧$/u, "")
  ).replace(/(?:这|那)(?:一)?[条项个](?:任务|待办(?:事项)?)?$/u, "");
  return [...new Set([target, undecorated])].flatMap((title) => {
    const parsed = taskTitleSchema.safeParse(title);
    return parsed.success ? [parsed.data] : [];
  });
}

export function analyzeClauses(input: string): ClauseAnalysis {
  const trimmed = input.trim();
  const maskedWhole = maskQuotedTitles(trimmed);
  const clauses = splitClausesOutsideQuotes(trimmed);

  const forbiddenTitles = new Set<string>();
  const positiveEntries: Array<{
    text: string;
    masked: string;
    quoted: string[];
    tentative: boolean;
  }> = [];
  let previousWasProhibition = false;
  let sawAnyProhibition = false;

  const isQuestion =
    /[?？]$/u.test(maskedWhole) ||
    /(?:好吗|行吗|可以吗|吗|是否|要不要|需不需要)/u.test(maskedWhole);
  const isHypothetical =
    /^(?:请)?(?:如果|假如|要是|若是|万一)/u.test(maskedWhole) ||
    /(?:如果|假如|要是|若是|万一).*(?:就|再|才)/u.test(maskedWhole);

  for (const clause of clauses) {
    const maskedClause = maskQuotedTitles(clause);
    const clauseQuoted = extractQuotedTitles(clause);
    const isSelfCancelledTentative =
      tentativeIntentPattern.test(maskedClause) && retractionCuePattern.test(maskedClause);
    const isWholeRetraction = wholeTurnRetractionClausePattern.test(maskedClause);
    const isProhibition =
      isSelfCancelledTentative ||
      isWholeRetraction ||
      clauseNegationPattern.test(maskedClause) ||
      retractionClausePattern.test(maskedClause) ||
      keepNoOpClausePattern.test(maskedClause) ||
      (previousWasProhibition && /^(?:尤其是|特别是|包括)/u.test(maskedClause));

    if (isProhibition) {
      sawAnyProhibition = true;
      previousWasProhibition = true;
      const prohibitedTitles = clauseQuoted.length
        ? clauseQuoted
        : unquotedProhibitedDeletionTargets(clause);
      for (const title of prohibitedTitles) {
        forbiddenTitles.add(title);
      }
      if (isWholeRetraction) {
        for (const entry of positiveEntries) {
          for (const title of entry.quoted) {
            forbiddenTitles.add(title);
          }
        }
        positiveEntries.length = 0;
      } else if (retractionCuePattern.test(maskedClause)) {
        for (let i = positiveEntries.length - 1; i >= 0; i--) {
          if (positiveEntries[i].tentative) {
            for (const title of positiveEntries[i].quoted) {
              forbiddenTitles.add(title);
            }
            positiveEntries.splice(i, 1);
          }
        }
      }
      continue;
    }

    previousWasProhibition = false;
    positiveEntries.push({
      text: clause,
      masked: maskedClause,
      quoted: clauseQuoted,
      tentative: tentativeIntentPattern.test(maskedClause),
    });
  }

  // A bare index token (e.g. "#2" in "#2，不要改了") never independently enables mutations when another clause negates.
  if (sawAnyProhibition) {
    for (let i = positiveEntries.length - 1; i >= 0; i--) {
      if (parseDisambiguationIndex(positiveEntries[i].masked)?.bare) {
        positiveEntries.splice(i, 1);
      }
    }
  }

  const positiveMaskedClauses: string[] = [];
  const positiveQuotedTitles: string[] = [];
  let hasUnquotedPositiveActionClause = false;
  for (const entry of positiveEntries) {
    positiveMaskedClauses.push(entry.masked);
    for (const title of entry.quoted) {
      positiveQuotedTitles.push(title);
    }
    // Check if this positive clause performs an existing-task mutation without quoting its target.
    if (
      entry.quoted.length === 0 &&
      /(?:完成|做好|买好|办好|交好|搞定|删掉|删除|移除|清空|改回|重命名|改名|改|设|调|优先级|工作量)/u.test(
        entry.masked,
      )
    ) {
      hasUnquotedPositiveActionClause = true;
    }
  }

  // Remove any title that appeared in a positive clause from forbiddenTitles.
  for (const title of positiveQuotedTitles) {
    forbiddenTitles.delete(title);
  }

  const positiveText = positiveMaskedClauses.join("，");
  const hasAnyPositiveClause = positiveMaskedClauses.length > 0;
  const mutationsAllowed = hasAnyPositiveClause && !isQuestion && !isHypothetical;

  // Extract explicit priority and workload values from positive clauses.
  const allowedPriorities = new Set<Priority>();
  const allowedWorkloads = new Set<Workload>();
  const hasMetadataMention = /(?:优先级|工作量)/u.test(positiveText);
  let hasInvalidMetadataValue = false;

  if (
    /(?:优先级|工作量)(?:特别|非常|超|很|挺|稍微|略|更|较)[低中高小大]/u.test(positiveText) ||
    /(?:优先级|工作量)[低中高小大]一点/u.test(positiveText) ||
    /(?:提高|降低|调高|调低)(?:[^，,。]*?)优先级(?!\s*(?:到|至|为|成)\s*[低中高])/u.test(
      positiveText,
    )
  ) {
    hasInvalidMetadataValue = true;
  }

  const pairedMatches = positiveText.matchAll(
    /(优先级|工作量)和(优先级|工作量)分别(?:改|设(?:置)?|调整?)(?:为|成|到)([低中高小大])和([低中高小大])/gu,
  );
  for (const m of pairedMatches) {
    const pairs: Array<[string, string]> = [
      [m[1], m[3]],
      [m[2], m[4]],
    ];
    for (const [field, val] of pairs) {
      if (field === "优先级") {
        const p = priorities.find((item) => priorityLabels[item] === val);
        if (p) allowedPriorities.add(p);
      } else {
        const w = workloads.find((item) => workloadLabels[item] === val);
        if (w) allowedWorkloads.add(w);
      }
    }
  }

  for (const m of positiveText.matchAll(
    /优先级\s*(?:改为|改成|改到|设为|设成|设置为|设置成|调整为|调整成|调为|调成|调到|降到|降至|降为|升到|升至|升为|为|是|：|:)?\s*([低中高])/gu,
  )) {
    const p = priorities.find((item) => priorityLabels[item] === m[1]);
    if (p) allowedPriorities.add(p);
  }
  for (const m of positiveText.matchAll(/([低中高])\s*优先级/gu)) {
    const p = priorities.find((item) => priorityLabels[item] === m[1]);
    if (p) allowedPriorities.add(p);
  }
  for (const m of positiveText.matchAll(
    /工作量\s*(?:改为|改成|改到|设为|设成|设置为|设置成|调整为|调整成|调为|调成|调到|降到|降至|降为|升到|升至|升为|为|是|：|:)?\s*([小中大])/gu,
  )) {
    const w = workloads.find((item) => workloadLabels[item] === m[1]);
    if (w) allowedWorkloads.add(w);
  }
  for (const m of positiveText.matchAll(/([小中大])\s*工作量/gu)) {
    const w = workloads.find((item) => workloadLabels[item] === m[1]);
    if (w) allowedWorkloads.add(w);
  }

  // Detect action intents in positive clauses.
  const hasCreateVerb =
    /(?:新增|新建|创建|添加|加入|加上|加一|加个|加条|(?<!标)(?:再)?记(?:一下|一条|下|个|条|住|录)?|备忘|提醒我)/u.test(
      positiveText,
    );
  const textWithoutReopenPhrases = positiveText
    .replace(/取消[^，,。]*?(?:已)?完成(?:状态)?/gu, "")
    .replace(/(?:未|待|没|不)完成/gu, "");
  const hasCompleteVerb =
    /(?:标(?:记)?(?:为|成)?(?:已)?完成|完成(?:了)?|[\p{Script=Han}](?:好了|完了)|买回来了|搞定了|勾掉|打勾)/u.test(
      textWithoutReopenPhrases,
    );
  const hasDeleteVerb = /(?:删掉|删除|移除|清掉|去掉|清除|删了|删一)/u.test(positiveText);
  const allowReopen =
    /(?:改回|恢复|退回|变回|重置(?:为|成)?|重新(?:打开|开启|标记|设为|变为)|取消(?:已)?完成|标(?:记)?(?:为|成)?(?:未|待)完成|设(?:置)?(?:为|成)?(?:未|待)完成|还没(?:做|买|寄|写|交|办|完成|搞定)|其实没|其实还没|未完成|待完成)/u.test(
      positiveText,
    );

  // Rename intent must NOT be triggered by metadata or status changes like "工作量改成大" or "改回未完成".
  const textWithoutMetadataOrStatus = positiveText
    .replace(
      /(?:优先级|工作量)\s*(?:和\s*(?:优先级|工作量)\s*)?(?:分别)?(?:改|设(?:置)?|调整?|降|升)(?:为|成|到|至)?\s*[低中高小大]+(?:和[低中高小大]+)?/gu,
      "",
    )
    .replace(/(?:改|设(?:置)?|调整?)(?:为|成|到)\s*[低中高小大](?:优先级|工作量)/gu, "")
    .replace(/(?:改回|改为|改成|设为|设成|标(?:记)?(?:为|成)?)\s*(?:已|未|待)完成/gu, "");
  const allowRename =
    /(?:重命名|改名(?:为|成|叫)?|更名(?:为|成|叫)?|(?:改|修改|换)(?:一下|下)?标题|标题(?:改|换|设)(?:为|成)?|名称(?:改|换|设)(?:为|成)?|名字(?:改|换|设)(?:为|成)?|(?:把|将)[^，,。]+?(?:改|换)(?:为|成))/u.test(
      textWithoutMetadataOrStatus,
    );

  // Restrict unrelated verbs when the request clearly expresses a single pure intent.
  const isSinglePositiveClause = positiveMaskedClauses.length === 1;
  const isPureMetadataRequest =
    hasMetadataMention &&
    !hasCreateVerb &&
    !hasCompleteVerb &&
    !hasDeleteVerb &&
    !allowReopen &&
    !allowRename;
  const isPureCreate =
    isSinglePositiveClause &&
    hasCreateVerb &&
    !hasCompleteVerb &&
    !hasDeleteVerb &&
    !allowReopen &&
    !allowRename &&
    !/(?:改|设|调|降|升)(?:为|成|到|至)/u.test(positiveText);
  const isPureComplete =
    isSinglePositiveClause &&
    hasCompleteVerb &&
    !hasDeleteVerb &&
    !hasCreateVerb &&
    !allowReopen &&
    !allowRename &&
    !hasMetadataMention;
  const isPureDelete =
    isSinglePositiveClause &&
    hasDeleteVerb &&
    !hasCompleteVerb &&
    !hasCreateVerb &&
    !allowReopen &&
    !allowRename &&
    !hasMetadataMention;
  const isPureReopen =
    allowReopen &&
    !hasCompleteVerb &&
    !hasDeleteVerb &&
    !hasCreateVerb &&
    !allowRename &&
    !hasMetadataMention;
  const isPureRename =
    isSinglePositiveClause &&
    allowRename &&
    !hasCompleteVerb &&
    !hasDeleteVerb &&
    !hasCreateVerb &&
    !allowReopen &&
    !hasMetadataMention;

  const allowCreate =
    mutationsAllowed &&
    !isPureMetadataRequest &&
    !isPureComplete &&
    !isPureDelete &&
    !isPureReopen &&
    !isPureRename &&
    (hasCreateVerb || (!hasCompleteVerb && !hasDeleteVerb && !allowReopen && !allowRename));
  const allowComplete =
    mutationsAllowed &&
    !isPureMetadataRequest &&
    !isPureCreate &&
    !isPureDelete &&
    !isPureReopen &&
    !isPureRename;
  const allowDelete =
    mutationsAllowed &&
    !isPureMetadataRequest &&
    !isPureCreate &&
    !isPureComplete &&
    !isPureReopen &&
    !isPureRename;

  const hasPositiveCreateCommand =
    mutationsAllowed &&
    hasCreateVerb &&
    !/^(?:看看|列出|查看|显示|展示|我有哪些)/u.test(positiveText);

  return {
    mutationsAllowed,
    hasProhibition: sawAnyProhibition,
    forbiddenTitles,
    clauseCount: clauses.length,
    positiveClauses: positiveEntries.map(({ text }) => text),
    positiveQuotedTitles,
    hasUnquotedPositiveActionClause,
    allowedPriorities,
    allowedWorkloads,
    hasMetadataMention,
    hasInvalidMetadataValue,
    allowCreate,
    allowComplete,
    allowDelete,
    allowReopen: mutationsAllowed && allowReopen,
    allowRename: mutationsAllowed && allowRename,
    isPureCreate: mutationsAllowed && isPureCreate,
    isPureComplete: mutationsAllowed && isPureComplete,
    isPureDelete: mutationsAllowed && isPureDelete,
    isPureReopen: mutationsAllowed && isPureReopen,
    isPureRename: mutationsAllowed && isPureRename,
    hasCompleteVerb: mutationsAllowed && hasCompleteVerb,
    hasDeleteVerb: mutationsAllowed && hasDeleteVerb,
    hasPositiveCreateCommand,
  };
}

export type ExplicitMutation =
  | { operation: "create_task"; title: string; explicitQuantity: boolean }
  | { operation: "update_task"; title: string; changes: { title: string } | { done: false } };

function parsePlainTitle(value: string): string | undefined {
  const title = value.trim();
  if (
    !plainTitle.test(title) ||
    /^(?:一[条项个])?(?:任务|待办(?:事项)?)$|^一[条项个]$/u.test(title) ||
    ambiguousTitle.test(title) ||
    isUnboundTarget(title)
  ) {
    return;
  }
  const parsed = taskTitleSchema.safeParse(title);
  return parsed.success ? parsed.data : undefined;
}

export function explicitMutation(
  input: string,
  clauses = analyzeClauses(input),
): ExplicitMutation | undefined {
  if (!clauses.mutationsAllowed || clauses.hasMetadataMention || /[\r\n]/u.test(input)) return;
  const text = input
    .trim()
    .replace(/[。！.!]+$/u, "")
    .trimEnd();
  const create = clauses.isPureCreate && standaloneCreate.exec(text);
  if (create) {
    const title = parsePlainTitle(create[3]);
    if (title) {
      return {
        operation: "create_task",
        title,
        explicitQuantity: create[1] !== undefined || create[2] !== undefined,
      };
    }
  }

  const edit = standaloneNamedEdit.exec(text);
  if (!edit) return;
  const title = edit[1].slice(1, -1);
  const parsedTarget = taskTitleSchema.safeParse(title);
  if (!parsedTarget.success || parsedTarget.data !== title) return;
  if (clauses.allowReopen && reopenAssignment.test(edit[2])) {
    return { operation: "update_task", title, changes: { done: false } };
  }

  const rename = clauses.allowRename && renameAssignment.exec(edit[2]);
  if (!rename || clauses.hasCompleteVerb || clauses.hasDeleteVerb || clauses.allowReopen) return;
  const literal = standaloneQuotedTitle.exec(rename[1]);
  const parsedTitle = literal
    ? taskTitleSchema.safeParse(literal[1].slice(1, -1)).data
    : parsePlainTitle(rename[1]);
  if (parsedTitle) return { operation: "update_task", title, changes: { title: parsedTitle } };
}

export function orderedMutationOperations(input: string) {
  const parts = splitClausesOutsideQuotes(input);
  if (
    parts.length < 2 ||
    parts.length > 6 ||
    parts.slice(1).some((part) => !/^(?:然后|接着|随后)/u.test(part))
  ) {
    return undefined;
  }
  const operations = parts.map((part) => {
    const text = part.replace(/^(?:然后|接着|随后)\s*/u, "");
    const intent = analyzeClauses(text);
    const candidates = [
      ...(intent.hasPositiveCreateCommand ? ["create_task" as const] : []),
      ...(intent.hasCompleteVerb ? ["complete_task" as const] : []),
      ...(intent.hasDeleteVerb ? ["delete_task" as const] : []),
      ...(intent.allowRename ||
      intent.allowReopen ||
      (intent.hasMetadataMention && !intent.hasPositiveCreateCommand)
        ? ["update_task" as const]
        : []),
    ];
    return intent.mutationsAllowed && candidates.length === 1
      ? { operation: candidates[0], text }
      : undefined;
  });
  return operations.every((operation) => operation !== undefined) ? operations : undefined;
}

export function explicitCreationList(input: string, clauses: ClauseAnalysis): string[] | undefined {
  if (
    !clauses.hasPositiveCreateCommand ||
    clauses.hasCompleteVerb ||
    clauses.hasDeleteVerb ||
    clauses.allowRename ||
    clauses.allowReopen ||
    clauses.hasMetadataMention
  )
    return undefined;
  const command =
    /^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?\s*(?:记|添加|新增)([二两三四五六2-6])(?:条|项|个)(?:任务|待办)?[，,：:]\s*(.+)$/u.exec(
      input.trim().replace(/[。！!]$/u, ""),
    );
  if (!command) return undefined;
  const count = parseChineseOrArabicNumber(command[1]);
  const titles = command[2].split(/、|和|以及|[，,]/u).map(parsePlainTitle);
  return titles.length === count &&
    new Set(titles).size === titles.length &&
    titles.every((title) => title !== undefined)
    ? titles
    : undefined;
}

export function detectUnboundCommand(input: string): "anaphoric" | "positional" | undefined {
  const trimmed = input
    .trim()
    .replace(/^(?:(?:还是)?算了(?:吧)?|好吧|好的|那|嗯|哎呀|对了)[，,；;\s]*/u, "")
    .trim();
  const masked = maskQuotedTitles(trimmed);
  if (clarifiedTitle.test(trimmed)) return undefined;
  if (parseDisambiguationIndex(trimmed)?.bare) return "positional";
  const clauses = splitClausesOutsideQuotes(trimmed);
  // If multiple clauses exist and an earlier clause already names a task (e.g. "牛奶已经买回来了，帮我把那项标成完成"),
  // do not treat the later pronoun as an unbound cross-turn reference.
  if (clauses.length > 1) {
    const firstMasked = maskQuotedTitles(clauses[0]);
    const restMasked = clauses.slice(1).map(maskQuotedTitles).join("，");
    const firstHasSubject =
      !isUnboundTarget(
        firstMasked.replace(/^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?(?:把|将|给)?/u, ""),
      ) &&
      !/^(?:把|将|给)?(?:它|这|那|刚|上|最后|第|#|\[#)/u.test(
        firstMasked.replace(/^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?/u, ""),
      );
    // Still catch pure compound metadata updates on one unbound target ("把第二条的优先级改为高，工作量改为大").
    const restIsOnlyMetadata = /^(?:优先级|工作量|再?(?:把|将)?(?:优先级|工作量))/u.test(
      restMasked,
    );
    if (firstHasSubject && !restIsOnlyMetadata) return undefined;
  }

  const normalized = masked
    .replace(/^(?:请(?:帮我)?|帮我|麻烦(?:帮我)?)?/u, "")
    .replace(/^(?:再|只)(?=把|将|给|删除|删掉|移除|完成|标记|改|设|调|提高|降低|重命名|改名)/u, "")
    .replace(/吧?[。！？.!?]?$/u, "")
    .trim();

  if (isPositionalTarget(normalized)) return "positional";
  if (isAnaphoricTarget(normalized)) return "anaphoric";

  // Extract candidate target from common single-target command structures.
  const candidates: string[] = [];
  const baPattern =
    /^(?:把|将|给)\s*(.+?)(?:的\s*(?:优先级|工作量|标题|名称|名字|完成状态)|(?:标(?:记)?(?:为|成)?(?:已|未|待)?完成|完成(?:了)?|做完了|删掉|删除|移除|改回|重命名|改名|更名|改(?:为|成|到)|设(?:置)?(?:为|成|到)|调(?:整)?(?:为|成|到)))/u.exec(
      normalized,
    );
  if (baPattern) candidates.push(baPattern[1].trim());

  const actionFirst =
    /^(?:删掉|删除|移除|完成|标(?:记)?(?:为|成)?(?:已|未|待)?完成|提高|降低|调高|调低|重新打开|重命名|改名)\s*(.+?)(?:的\s*(?:优先级|工作量|标题|名称|名字)|(?:吧)?)$/u.exec(
      normalized,
    );
  if (actionFirst) candidates.push(actionFirst[1].trim());

  const targetFirstMetadata =
    /^(.+?)(?:的\s*)?(?:(?:优先级|工作量)(?:和(?:优先级|工作量))?(?:分别)?(?:改|设|调|降|升)|(?:改|设(?:置)?|调整?)(?:为|成|到)[低中高小大](?:优先级|工作量))/u.exec(
      normalized,
    );
  if (targetFirstMetadata) candidates.push(targetFirstMetadata[1].trim());

  const subjectFirst =
    /^(.+?)(?:已经|都|也|快)?(?:请)?(?:帮我|给我)?(?:标(?:记)?(?:为|成)?(?:已|未|待)?完成|完成(?:了)?|做完了|做好了|买好了|搞定了|删掉|删除|移除|改回|重命名|改名|更名|改(?:为|成|到)|设(?:置)?(?:为|成|到)|调(?:整)?(?:为|成|到))/u.exec(
      normalized,
    );
  if (subjectFirst) candidates.push(subjectFirst[1].trim());

  for (const candidate of candidates) {
    if (isPositionalTarget(candidate)) return "positional";
    if (isAnaphoricTarget(candidate)) return "anaphoric";
  }
  return undefined;
}
