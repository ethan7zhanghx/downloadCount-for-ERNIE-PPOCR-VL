# 衍生模型数据回填说明

## 概述

为了避免在统计当周新增模型时，将历史新增的衍生模型误认为是本周新增，我们提供了数据回填功能。

## 背景

当从 AI Studio 和 ModelScope 获取数据时，可能会发现一些之前从未获取过的衍生模型。如果这些模型实际上是在更早的时间创建的，那么在统计"本周新增"时，它们会被错误地计入本周新增，导致统计数据失真。

## 解决方案

通过回填功能，在模型的创建日期（`created_at`）或最后修改日期（`last_modified`）插入一条下载量为0的记录，这样在进行周报统计时，这些模型就不会被误认为是本周新增的。

## 使用方法

### 1. 回填新增衍生模型数据

```bash
python3 scripts/backfill_derivative_models.py
```

**功能说明：**
- 识别 2026-01-16 获取的 AI Studio 和 ModelScope 数据中的新增衍生模型
- 只处理 `model_category` 为 `ernie-4.5` 或 `paddleocr-vl` 的模型
- 对于新增模型，在其创建日期（优先 `created_at`，备选 `last_modified`）插入一条下载量为0的记录
- 自动运行 `backfill_model_category.py` 确保 `model_category` 字段已填充

**处理逻辑：**
1. 填充 `model_category` 字段（如果尚未填充）
2. 识别 2026-01-16 之前的历史模型
3. 获取 2026-01-16 的所有相关模型记录
4. 识别新增的衍生模型（在历史记录中不存在的）
5. 为每个新增衍生模型在其创建日期插入0下载量记录

**注意事项：**
- 脚本会跳过没有创建时间的模型
- 脚本会跳过创建时间不在目标日期之前的模型
- 脚本会跳过该日期已有记录的模型
- 使用 `GROUP BY repo, model_name` 确保每个模型只处理一次

### 2. 清理重复的回填记录

```bash
python3 scripts/cleanup_duplicate_backfill.py
```

**功能说明：**
- 删除所有平台重复的回填记录（同一平台、同一模型、同一日期的多条记录）
- 保留每组重复记录中最早插入的（即 `rowid` 最小的）
- 确保每个模型的每个日期只有一条回填记录

## 回填效果示例

以 ModelScope 的 `PaddleOCR-VL-GGUF` 模型为例：

| date | repo | model_name | download_count |
|------|------|------------|----------------|
| 2026-01-14 | ModelScope | PaddleOCR-VL-GGUF | 0 (回填记录) |
| 2026-01-16 | ModelScope | PaddleOCR-VL-GGUF | 23 (实际下载量) |

这样，在统计 2026-01-09 到 2026-01-16 的周报时，`PaddleOCR-VL-GGUF` 模型不会被计入本周新增，因为在 2026-01-14 就已经有了记录。

## 数据统计

回填完成后，各平台的回填记录数量如下（截至清理后）：

| 平台 | 回填记录数 |
|------|-----------|
| AI Studio | 52 |
| ModelScope | 28 |

## 技术细节

### 数据库表结构

```sql
CREATE TABLE model_downloads (
    date TEXT,
    repo TEXT,
    model_name TEXT,
    download_count TEXT,
    model_category TEXT,
    ...
    created_at TEXT,
    last_modified TEXT,
    ...
)
```

### 回填记录的特征

- `download_count = '0'`
- `date = created_at` 或 `date = last_modified`
- `date < '2026-01-16'`（目标抓取日期）
- `model_category IN ('ernie-4.5', 'paddleocr-vl')`

### 查询回填记录

```sql
-- 查看所有回填记录
SELECT repo, model_name, date, created_at
FROM model_downloads
WHERE download_count = '0'
AND model_category IN ('ernie-4.5', 'paddleocr-vl')
ORDER BY repo, date;

-- 统计各平台的回填记录
SELECT repo, COUNT(*) as count
FROM model_downloads
WHERE download_count = '0'
AND model_category IN ('ernie-4.5', 'paddleocr-vl')
GROUP BY repo;
```

## 相关脚本

- `scripts/backfill_model_category.py`: 填充 model_category 字段
- `scripts/backfill_derivative_models.py`: 回填衍生模型数据
- `scripts/cleanup_duplicate_backfill.py`: 清理重复的回填记录

## 注意事项

1. **备份**: 在运行任何脚本之前，建议先备份数据库
   ```bash
   cp data/ernie_downloads.db data/ernie_downloads.db.backup
   ```

2. **重复运行**: `backfill_derivative_models.py` 可以安全地重复运行，它会自动跳过已回填的模型

3. **清理时机**: 如果多次运行了回填脚本，建议运行 `cleanup_duplicate_backfill.py` 清理重复记录

4. **性能**: 对于大量数据，回填可能需要一些时间，请耐心等待

## 常见问题

**Q: 为什么要插入0下载量的记录，而不是修改1月16日的记录日期？**
A: 保持数据完整性。1月16日的记录反映了当天的实际下载量，而回填记录标记了模型的创建时间，两者都有意义。

**Q: 回填会影响下载量统计吗？**
A: 不会。回填记录的下载量为0，不会影响总下载量的统计。它只是用于判断模型的新增时间。

**Q: 如何验证回填是否成功？**
A: 运行脚本后会显示详细的统计信息。也可以手动查询数据库验证：
```bash
sqlite3 data/ernie_downloads.db "
SELECT repo, model_name, date, download_count, created_at
FROM model_downloads
WHERE repo IN ('AI Studio', 'ModelScope')
AND download_count = '0'
AND model_category IN ('ernie-4.5', 'paddleocr-vl')
ORDER BY repo, date DESC
LIMIT 20"
```

**Q: 如何撤销回填？**
A: 如果需要撤销，可以删除所有下载量为0的记录：
```sql
DELETE FROM model_downloads WHERE download_count = '0';
```

但请注意，这会删除所有平台的回填记录，不仅仅是 AI Studio 和 ModelScope。
