# 数据标准化一致性修复总结

**修复日期**: 2026-01-04
**问题**: 累计模型数、当前模型数、已删除模型数三者之间数学关系不一致

## 问题描述

### 原始数据（修复前）

**ERNIE-4.5**:
- 累计衍生模型（周报显示）: 204 个
- 当前日期衍生模型: 187 个
- 已删除模型: 19 个
- **问题**: 187 + 19 = 206 ≠ 204 ❌

**PaddleOCR-VL**:
- 类似问题存在

### 根本原因

1. **周报统计** (`app.py`) 使用 `enforce_deduplication_and_standardization()` 函数对数据进行标准化和去重
2. **已删除模型检测** (`get_deleted_or_hidden_models()`) 使用原始回填数据，未应用标准化
3. **结果**: 两者使用不同的数据集进行计算，导致数学关系不一致

#### 标准化逻辑

`enforce_deduplication_and_standardization()` 执行三个步骤:
1. **标准化 publisher**: 转换为 Title Case (`wasirmen123` → `Wasirmen123`)
2. **标准化 model_name**: 移除 publisher 前缀
3. **去重**: 按 `(date, repo, publisher, model_name)` 去重，保留下载量最高的记录

#### 数据差异

- **回填模式原始数据**: 206 个模型（ERNIE-4.5）
- **回填模式标准化后**: 204 个模型（去除 2 个重复）
- **重复模型**:
  - `Cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit` (data_source='both', 出现2次)
  - `Cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-8bit` (data_source='both', 出现2次)

## 修复方案

### 修改文件

**`ernie_tracker/analysis.py`**: `get_deleted_or_hidden_models()` 函数

### 修复内容

#### 1. 应用相同的标准化逻辑（第 1072-1105 行）

```python
# 3.5. 应用与周报相同的标准化逻辑
# 标准化 publisher 名称
historical_derivatives['publisher'] = historical_derivatives['publisher'].astype(str).apply(
    lambda x: x.title() if x.lower() != 'nan' else x
)
if not current_derivatives.empty:
    current_derivatives['publisher'] = current_derivatives['publisher'].astype(str).apply(
        lambda x: x.title() if x.lower() != 'nan' else x
    )

# 标准化模型名称
historical_derivatives = normalize_model_names(historical_derivatives)
if not current_derivatives.empty:
    current_derivatives = normalize_model_names(current_derivatives)

# 去重（按下载量降序，保留最高的）
historical_derivatives['download_count'] = pd.to_numeric(
    historical_derivatives['download_count'], errors='coerce'
).fillna(0)
historical_derivatives = historical_derivatives.sort_values(
    by='download_count', ascending=False
).drop_duplicates(
    subset=['date', 'repo', 'publisher', 'model_name'], keep='first'
)

if not current_derivatives.empty:
    current_derivatives['download_count'] = pd.to_numeric(
        current_derivatives['download_count'], errors='coerce'
    ).fillna(0)
    current_derivatives = current_derivatives.sort_values(
        by='download_count', ascending=False
    ).drop_duplicates(
        subset=['date', 'repo', 'publisher', 'model_name'], keep='first'
    )
```

#### 2. 修复 last_seen_date 查询（第 1146-1156 行）

**问题**: 标准化后，`publisher` 从 `wasirmen123` 变为 `Wasirmen123`，但数据库中仍是原始值，导致 SQL 查询失败

**解决方案**: 使用 `LOWER()` 函数进行不区分大小写的匹配

```python
# 查询该模型在数据库中最后出现的日期
# 使用 LOWER() 进行不区分大小写的匹配，因为标准化后的 publisher 可能与数据库中的原始值大小写不同
conn = sqlite3.connect(DB_PATH)
query = """
    SELECT date, download_count
    FROM model_downloads
    WHERE repo = ? AND LOWER(publisher) = LOWER(?) AND model_name = ?
    ORDER BY date DESC
    LIMIT 1
"""
result = pd.read_sql_query(query, conn, params=(repo, publisher, model_name))
```

**修复前的错误行为**:
- 查询 `publisher='Wasirmen123'` → 无结果
- 回退到使用 DataFrame 的 date（回填日期 2026-01-02）
- 显示错误的 `last_seen_date`

**修复后的正确行为**:
- 查询 `LOWER(publisher)=LOWER('Wasirmen123')` → 匹配 `wasirmen123`
- 返回正确的最后出现日期 (2025-12-17)

## 修复效果验证

### ERNIE-4.5

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 累计衍生模型（标准化） | 204 | 204 |
| 当前衍生模型 | 187 | 187 |
| 已删除模型 | 19 ❌ | 17 ✅ |
| 数学验证 | 187 + 19 = 206 ≠ 204 ❌ | 187 + 17 = 204 ✅ |

### PaddleOCR-VL

| 指标 | 修复后 |
|------|--------|
| 累计衍生模型（标准化） | 45 |
| 当前衍生模型 | 35 |
| 已删除模型 | 10 |
| 数学验证 | 35 + 10 = 45 ✅ |

### 已删除模型详情验证

**修复前** (错误的 last_seen_date):
- `Wasirmen123/ERNIE-4.5-21B-A3B-PT-Q3_K_L-GGUF`: 2026-01-02 ❌
- `Fixart/ERNIE-4.5-21B-A3B-Thinking-mlx-3Bit`: 2026-01-02 ❌

**修复后** (正确的 last_seen_date):
- `Wasirmen123/ERNIE-4.5-21B-A3B-PT-Q3_K_L-GGUF`: 2025-12-17 ✅
- `Fixart/ERNIE-4.5-21B-A3B-Thinking-mlx-3Bit`: 2025-11-28 ✅

### 重复模型处理

修复后不再重复计数的模型（已从删除列表中移除）:
1. `Cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit` (重复记录之一)
2. `Cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-8bit` (重复记录之一)

这两个模型在回填数据中有重复记录（`data_source='both'`），标准化后只保留一条，因此从删除列表中正确移除。

## 技术细节

### 标准化的重要性

1. **一致性**: 确保所有统计使用相同的数据处理逻辑
2. **去重**: 移除 `data_source='both'` 等原因造成的重复记录
3. **准确性**: publisher 名称标准化避免大小写不同导致的重复计数

### SQL 查询优化

使用 `LOWER()` 函数的优缺点:
- ✅ 优点: 不修改数据库，保持原始数据完整性
- ✅ 优点: 兼容标准化前后的 publisher 值
- ⚠️ 注意: 可能略微影响查询性能（但数据量小，影响可忽略）

### 回填模式 vs 正常模式

**回填模式** (`last_value_per_model=True`):
- 获取每个模型截止到当前日期的最后一条记录
- 用于"累计统计"
- 可能包含重复记录（需要标准化去重）

**正常模式** (`last_value_per_model=False`):
- 只获取当前日期的实际数据
- 用于"当天快照"
- 通常无重复（因为是单日数据）

## 影响范围

### 前端显示

**ERNIE-4.5 Tab** (`app.py:686-712`):
- 已删除模型列表现在显示 17 个（之前 19 个）
- `last_seen_date` 现在显示正确的日期

**PaddleOCR-VL Tab** (`app.py:948-974`):
- 已删除模型列表显示 10 个
- `last_seen_date` 显示正确的日期

### 数学一致性

所有统计现在满足:
```
当前模型数 + 已删除模型数 = 累计模型数（标准化后）
```

## 测试验证

```python
from ernie_tracker.analysis import get_deleted_or_hidden_models
from ernie_tracker.db import load_data_from_db

# ERNIE-4.5
deleted_ernie = get_deleted_or_hidden_models('2026-01-02', model_series='ERNIE-4.5')
assert len(deleted_ernie) == 17  # ✅

# PaddleOCR-VL
deleted_paddle = get_deleted_or_hidden_models('2026-01-02', model_series='PaddleOCR-VL')
assert len(deleted_paddle) == 10  # ✅
```

## 相关文件

- 核心修复: `ernie_tracker/analysis.py` (lines 1072-1105, 1146-1156)
- 周报逻辑: `app.py` (使用 `enforce_deduplication_and_standardization()`)
- 标准化函数: `ernie_tracker/analysis.py::normalize_model_names()`

## 结论

✅ 数据标准化逻辑现在在所有统计中保持一致
✅ 累计、当前、已删除模型数的数学关系正确
✅ `last_seen_date` 显示正确的日期（不再显示回填日期）
✅ 所有测试验证通过

## 未来建议

1. **考虑数据库标准化**: 定期运行脚本标准化数据库中的 publisher 名称，避免大小写不一致
2. **添加单元测试**: 为 `get_deleted_or_hidden_models()` 添加测试，确保标准化逻辑正确
3. **监控重复记录**: 定期检查 `data_source='both'` 的重复记录，理解其产生原因
