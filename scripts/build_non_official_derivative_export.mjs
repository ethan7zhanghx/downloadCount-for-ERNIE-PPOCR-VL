import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const dbPath = "/Users/zhanghaoxin/Desktop/Baidu/DownloadData/data/ernie_downloads.db";
const outputDir = "/Users/zhanghaoxin/Desktop/Baidu/DownloadData/outputs/non_official_derivative_models_20260514";
const outputPath = `${outputDir}/non_official_derivative_models.xlsx`;

const conditionSql = `
  CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) > 0
  AND LOWER(COALESCE(publisher, '')) NOT LIKE '%baidu%'
  AND COALESCE(publisher, '') NOT LIKE '%百度%'
  AND LOWER(COALESCE(publisher, '')) NOT LIKE '%paddle%'
  AND LOWER(COALESCE(publisher, '')) NOT LIKE '%yiyan%'
  AND COALESCE(publisher, '') NOT LIKE '%一言%'
  AND (
    (
      LOWER(TRIM(COALESCE(model_type, ''))) IN ('finetune', 'lora', 'adapter', 'adaptor', '适配', '微调')
      AND CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) > 100
    )
    OR (
      (LOWER(TRIM(COALESCE(model_type, ''))) = 'quantized' OR TRIM(COALESCE(model_type, '')) = '')
      AND CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) > 500
    )
  )
`;

const rowsSql = `
WITH filtered AS (
  SELECT
    date,
    repo,
    model_name,
    publisher,
    CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) AS download_count,
    COALESCE(NULLIF(TRIM(model_type), ''), '未标注') AS model_type,
    COALESCE(model_category, '') AS model_category,
    COALESCE(tags, '') AS tags,
    COALESCE(base_model, '') AS base_model,
    COALESCE(base_model_from_api, '') AS base_model_from_api,
    COALESCE(data_source, '') AS data_source,
    COALESCE(likes, '') AS likes,
    COALESCE(library_name, '') AS library_name,
    COALESCE(pipeline_tag, '') AS pipeline_tag,
    COALESCE(search_keyword, '') AS search_keyword,
    COALESCE(url, '') AS url,
    CASE
      WHEN LOWER(TRIM(COALESCE(model_type, ''))) IN ('finetune', 'lora', 'adapter', 'adaptor', '适配', '微调')
        THEN 'finetune/lora/adapter 且下载量>100'
      WHEN LOWER(TRIM(COALESCE(model_type, ''))) = 'quantized'
        THEN 'quantized 且下载量>500'
      ELSE '未标注 且下载量>500'
    END AS matched_rule
  FROM model_downloads
  WHERE ${conditionSql}
),
first_seen AS (
  SELECT
    repo,
    publisher,
    model_name,
    MIN(date) AS first_seen_date
  FROM model_downloads
  GROUP BY repo, publisher, model_name
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY repo, publisher, model_name
      ORDER BY download_count DESC, date DESC
    ) AS rn
  FROM filtered
)
SELECT
  ranked.date AS "统计日期",
  first_seen.first_seen_date AS "最早记录日期",
  ranked.repo AS "平台",
  ranked.publisher AS "发布者",
  ranked.model_name AS "模型名称",
  CASE WHEN ranked.model_type = '未标注' THEN '' ELSE ranked.model_type END AS "模型类型",
  ranked.download_count AS "下载量",
  ranked.model_category AS "模型分类",
  ranked.matched_rule AS "命中规则",
  ranked.url AS "链接",
  ranked.base_model AS "Base Model",
  ranked.base_model_from_api AS "Base Model API",
  ranked.data_source AS "数据来源",
  ranked.likes AS "点赞数",
  ranked.library_name AS "Library",
  ranked.pipeline_tag AS "Pipeline",
  ranked.search_keyword AS "搜索关键词",
  ranked.tags AS "Tags"
FROM ranked
LEFT JOIN first_seen
  ON ranked.repo = first_seen.repo
  AND ranked.publisher = first_seen.publisher
  AND ranked.model_name = first_seen.model_name
WHERE rn = 1
ORDER BY download_count DESC, "平台", "发布者", "模型名称";
`;

const summarySql = `
WITH result AS (
  SELECT
    repo,
    COALESCE(NULLIF(TRIM(model_type), ''), '未标注') AS model_type,
    CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) AS download_count,
    ROW_NUMBER() OVER (
      PARTITION BY repo, publisher, model_name
      ORDER BY CAST(REPLACE(COALESCE(download_count, '0'), ',', '') AS INTEGER) DESC, date DESC
    ) AS rn
  FROM model_downloads
  WHERE ${conditionSql}
)
SELECT
  repo AS "平台",
  model_type AS "模型类型",
  COUNT(*) AS "模型数",
  SUM(download_count) AS "下载量合计",
  MAX(download_count) AS "最高下载量"
FROM result
WHERE rn = 1
GROUP BY repo, model_type
ORDER BY "模型数" DESC, "下载量合计" DESC;
`;

function queryJson(sql) {
  const out = execFileSync("sqlite3", ["-json", dbPath, sql], { encoding: "utf8", maxBuffer: 1024 * 1024 * 64 });
  return JSON.parse(out || "[]");
}

function colLetter(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

async function main() {
  const rows = queryJson(rowsSql);
  const summary = queryJson(summarySql);
  const workbook = Workbook.create();

  const resultSheet = workbook.worksheets.add("筛选结果");
  const headers = Object.keys(rows[0] || {
    "统计日期": "",
    "最早记录日期": "",
    "平台": "",
    "发布者": "",
    "模型名称": "",
    "模型类型": "",
    "下载量": "",
    "模型分类": "",
    "命中规则": "",
    "链接": "",
    "Base Model": "",
    "Base Model API": "",
    "数据来源": "",
    "点赞数": "",
    "Library": "",
    "Pipeline": "",
    "搜索关键词": "",
    "Tags": "",
  });
  const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];
  const resultRange = `A1:${colLetter(headers.length)}${Math.max(matrix.length, 1)}`;
  resultSheet.getRange(resultRange).values = matrix;
  resultSheet.getRange(`A1:${colLetter(headers.length)}1`).format = {
    font: { bold: true, color: "#FFFFFF" },
    fill: { color: "#1F4E78" },
  };
  resultSheet.getRange(`A2:${colLetter(headers.length)}${Math.max(matrix.length, 2)}`).format = {
    font: { color: "#222222" },
  };
  resultSheet.getRange("A:A").numberFormat = "yyyy-mm-dd";
  resultSheet.getRange("B:B").numberFormat = "yyyy-mm-dd";
  resultSheet.getRange("G:G").numberFormat = "#,##0";
  resultSheet.getRange("A:A").columnWidthPx = 96;
  resultSheet.getRange("B:B").columnWidthPx = 110;
  resultSheet.getRange("C:C").columnWidthPx = 120;
  resultSheet.getRange("D:D").columnWidthPx = 170;
  resultSheet.getRange("E:E").columnWidthPx = 340;
  resultSheet.getRange("F:F").columnWidthPx = 110;
  resultSheet.getRange("G:G").columnWidthPx = 95;
  resultSheet.getRange("H:I").columnWidthPx = 170;
  resultSheet.getRange("J:J").columnWidthPx = 330;
  resultSheet.getRange("K:R").columnWidthPx = 160;

  const summarySheet = workbook.worksheets.add("筛选说明");
  const summaryHeaders = Object.keys(summary[0] || { "平台": "", "模型类型": "", "模型数": "", "下载量合计": "", "最高下载量": "" });
  const notes = [
    ["导出说明", ""],
    ["数据源", dbPath],
    ["数据库日期范围", "2025-06-30 至 2026-05-08"],
    ["导出口径", "按 repo + publisher + model_name 去重，保留满足条件记录中的最高下载量；若下载量相同，取日期较新的记录。"],
    ["非官方规则", "publisher 不包含 百度、baidu、Paddle、yiyan、一言。"],
    ["条件1", "model_type 为 finetune、lora、adapter/adaptor（含“适配”“微调”）且下载量 > 100。"],
    ["条件2", "model_type 为 quantized 或空值，且下载量 > 500；结果表中空类型保持空白。"],
    ["最早记录日期", "同一 repo + publisher + model_name 在全库 model_downloads 中的最早 date。"],
    ["结果模型数", rows.length],
    ["", ""],
    ["汇总", ""],
    summaryHeaders,
    ...summary.map((row) => summaryHeaders.map((header) => row[header] ?? "")),
  ];
  summarySheet.getRange(`A1:${colLetter(Math.max(...notes.map((r) => r.length)))}${notes.length}`).values = notes;
  summarySheet.getRange("A1:B1").format = { font: { bold: true, color: "#FFFFFF" }, fill: { color: "#1F4E78" } };
  summarySheet.getRange("A10:E10").format = { font: { bold: true, color: "#FFFFFF" }, fill: { color: "#1F4E78" } };
  summarySheet.getRange("A:A").columnWidthPx = 135;
  summarySheet.getRange("B:B").columnWidthPx = 620;
  summarySheet.getRange("C:E").columnWidthPx = 115;
  summarySheet.getRange("C:E").numberFormat = "#,##0";

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "final formula error scan",
  });
  if (errors.ndjson && errors.ndjson.trim()) {
    console.log(errors.ndjson);
  }

  await workbook.render({ sheetName: "筛选结果", range: `A1:I${Math.min(rows.length + 1, 40)}`, scale: 1 });
  await workbook.render({ sheetName: "筛选说明", range: "A1:E30", scale: 1 });

  await fs.mkdir(outputDir, { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(JSON.stringify({ outputPath, rows: rows.length, summaryRows: summary.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
