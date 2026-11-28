import assert from "node:assert/strict";
import { test } from "node:test";
import { explicitMutation } from "./task-request.ts";

test("a standalone plain create extracts the exact title and explicit quantity", () => {
  for (const [input, title, explicitQuantity] of [
    ["再记一条预约体检。", "预约体检", true],
    ["再记一个预约体检。", "预约体检", true],
    ["帮我记一下买牛奶", "买牛奶", false],
    ["麻烦帮我添加一项任务：整理票据！", "整理票据", true],
    ["请新增一个待办：清点物料", "清点物料", true],
    ["再记一条任务管理课", "任务管理课", true],
  ] as const) {
    assert.deepEqual(explicitMutation(input), {
      operation: "create_task",
      title,
      explicitQuantity,
    });
  }
});

test("standalone quoted rename and reopen commands authorize only the specified patch", () => {
  assert.deepEqual(explicitMutation("把「写项目报告」改名为提交项目报告。"), {
    operation: "update_task",
    title: "写项目报告",
    changes: { title: "提交项目报告" },
  });
  assert.deepEqual(explicitMutation("请将“旧名称”的标题修改为「费用、会议和采购」！"), {
    operation: "update_task",
    title: "旧名称",
    changes: { title: "费用、会议和采购" },
  });
  assert.deepEqual(explicitMutation("把「旧名称」更名为「不要存档，然后删掉记录」。"), {
    operation: "update_task",
    title: "旧名称",
    changes: { title: "不要存档，然后删掉记录" },
  });
  for (const input of [
    "把「提交项目报告」改回未完成。",
    "请将「提交项目报告」设为待完成。",
    "帮我把「提交项目报告」取消已完成状态。",
  ]) {
    assert.deepEqual(explicitMutation(input), {
      operation: "update_task",
      title: "提交项目报告",
      changes: { done: false },
    });
  }
});

test("uncertain, contextual, compound and metadata expressions do not authorize the standalone route", () => {
  for (const input of [
    "不要再记一条预约体检",
    "再记一条预约体检，还是算了",
    "如果有空就帮我记一下预约体检",
    "帮我记一下买牛奶吗？",
    "再记一条刚才那个",
    "再记一条之前的任务",
    "再记一条一样的",
    "再记一条随便什么",
    "再记一条第二条",
    "再记一条买牛奶和取快递",
    "再记一条买牛奶或取快递",
    "帮我记两条，买牛奶和取快递",
    "再记一条买牛奶然后取快递",
    "再记一条买牛奶再创建取快递",
    "帮我记一下买牛奶，优先级高",
    "再记一条明天到期交报告",
    "帮我记一下「买牛奶」",
    "新增一条",
    "新增一条任务",
    "新增待办",
    "把「写项目报告」改名为",
    "把写项目报告改名为提交项目报告",
    "把刚刚那个改名为提交项目报告",
    "不要把「写项目报告」改名为提交项目报告",
    "把「写项目报告」改名为提交项目报告还是写季度报告",
    "把「写项目报告」改名为提交项目报告并标为已完成",
    "把「写项目报告」改回未完成，顺便设为高优先级",
    "如果还没交，把「写项目报告」改回未完成",
    "把「 写项目报告 」改回未完成",
    "把「写项目报告」改名为「\u202e」",
    "再记一条" + "字".repeat(201),
  ]) {
    assert.equal(explicitMutation(input), undefined, input);
  }
});
