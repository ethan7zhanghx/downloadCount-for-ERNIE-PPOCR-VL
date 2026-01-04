# 新功能实现总结

**日期**: 2026-01-04

## 新增功能

### 1. PaddleOCR-VL Tab 累计衍生模型统计

**位置**: `app.py:805-811`

**功能**: 在 PaddleOCR-VL 分析页面添加累计衍生模型统计信息，与 ERNIE-4.5 页面保持一致。

**显示内容**:
- 累计衍生模型数量（使用回填模式统计）
- 本周新增衍生模型数量（HF非官方差集）
- 新增列表展示数量

**示例输出**:
```
累计衍生模型：45 个｜本周新增衍生（HF非官方差集）：4 个｜新增列表展示：5 个
```

**测试结果**:
- ✅ PaddleOCR-VL 累计衍生模型：45 个
- ✅ 本周新增衍生模型：4 个

---

### 2. 已删除/隐藏模型检测功能

**核心函数**: `ernie_tracker/analysis.py::get_deleted_or_hidden_models()`

**实现逻辑**:
1. 使用回填模式（`last_value_per_model=True`）获取所有历史模型
2. 使用正常模式（`last_value_per_model=False`）获取当前日期实际数据
3. 对比两者差集，找出已删除/隐藏的模型
4. 查询数据库获取每个模型的最后出现日期和下载量

**返回信息**:
- 模型名称
- 发布者
- 平台
- 模型类型
- 基础模型
- **最后出现日期** ⭐
- **最后记录的下载量** ⭐

---

### 3. ERNIE-4.5 Tab 显示已删除模型列表

**位置**: `app.py:686-712`

**显示内容**:
- 已删除/隐藏的衍生模型列表
- 每个模型的详细信息（名称、发布者、平台、类型、基础模型、最后出现日期、最后下载量）
- 警告提示或成功提示

**测试结果**:
- ✅ 检测到 22 个已删除的 ERNIE-4.5 衍生模型

**典型案例**:
1. ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit (Cyankiwi) - 最后出现: 2025-12-26
2. ERNIE-4.5-21B-A3B-Thinking-AWQ-8bit (Cyankiwi) - 最后出现: 2025-12-26
3. ERNIE-4.5-21B-A3B-PT-GGUF (Mungert) - 最后出现: 2025-12-19

---

### 4. PaddleOCR-VL Tab 显示已删除模型列表

**位置**: `app.py:948-974`

**显示内容**: 与 ERNIE-4.5 Tab 相同

**测试结果**:
- ✅ 检测到 10 个已删除的 PaddleOCR-VL 衍生模型

**典型案例**:
1. AILAND (Unknown) - 最后出现: 2025-12-11
2. Elonmusk (Unknown) - 最后出现: 2025-12-11
3. MODEL-ONE-1 (Unknown) - 最后出现: 2025-12-11

**观察**: 这些模型都是在 2025-12-11 最后出现，发布者为 "Unknown"，可能是数据质量问题或测试数据。

---

## 技术细节

### 回填模式 vs 正常模式

**回填模式** (`last_value_per_model=True`):
- 获取截止到当前日期的所有历史模型
- 如果某模型在当前日期无数据，使用之前日期的最后一条记录
- 用于"累计衍生模型"统计

**正常模式** (`last_value_per_model=False`):
- 只获取当前日期的实际数据
- 用于"当天快照"统计

### 模型唯一标识

使用 `repo|||publisher|||model_name` 作为唯一键，确保跨平台模型不会被误判为删除。

### 数据库查询优化

对于每个已删除模型，单独查询数据库获取最后出现日期，避免一次性加载大量历史数据。

---

## 修改文件

1. **ernie_tracker/analysis.py**
   - 新增函数: `get_deleted_or_hidden_models()`

2. **app.py**
   - PaddleOCR-VL Tab: 新增累计衍生模型统计（第 805-811 行）
   - ERNIE-4.5 Tab: 新增已删除模型列表（第 686-712 行）
   - PaddleOCR-VL Tab: 新增已删除模型列表（第 948-974 行）

---

## 使用示例

### 在代码中调用

```python
from ernie_tracker.analysis import get_deleted_or_hidden_models

# 获取 ERNIE-4.5 已删除模型
deleted_ernie = get_deleted_or_hidden_models('2026-01-02', model_series='ERNIE-4.5')
print(f'已删除模型数量: {len(deleted_ernie)}')

# 获取 PaddleOCR-VL 已删除模型
deleted_paddle = get_deleted_or_hidden_models('2026-01-02', model_series='PaddleOCR-VL')
print(f'已删除模型数量: {len(deleted_paddle)}')
```

### 在前端查看

1. 启动应用: `./start.sh` 或 `streamlit run app.py`
2. 导航到 "📊 ERNIE-4.5 分析" 或 "📊 PaddleOCR-VL 分析"
3. 选择日期并生成周报
4. 查看 "🗑️ 已删除/隐藏的衍生模型" 部分

---

## 数据统计

### 截至 2026-01-02 的删除模型统计

| 系列 | 已删除模型数 | 最近删除日期 | 主要平台 |
|------|------------|------------|---------|
| ERNIE-4.5 | 22 个 | 2025-12-26 | Hugging Face |
| PaddleOCR-VL | 10 个 | 2025-12-11 | Hugging Face |

### 删除原因分析

1. **量化模型删除** (Cyankiwi): AWQ-4bit/8bit 版本被删除，可能是模型质量问题
2. **GGUF 模型删除** (Mungert): 多个 GGUF 版本被删除
3. **低质量模型清理**: 发布者为 "Unknown" 的模型可能是测试数据或爬虫错误

---

## 已知问题

1. ⚠️ 部分 PaddleOCR-VL 模型的发布者显示为 "Unknown"，可能影响模型识别准确性
2. ℹ️ 已删除模型的下载量数据仅为最后记录时的值，不代表累计总量

---

## 未来改进建议

1. 添加已删除模型的趋势图（按删除日期统计）
2. 导出已删除模型列表到 Excel
3. 添加邮件通知功能，当检测到重要模型被删除时发送警报
4. 定期清理 "Unknown" 发布者的低质量数据

---

## 验证清单

- [x] PaddleOCR-VL Tab 显示累计衍生模型统计
- [x] ERNIE-4.5 Tab 显示已删除模型列表
- [x] PaddleOCR-VL Tab 显示已删除模型列表
- [x] 已删除模型包含最后出现日期
- [x] 已删除模型包含最后下载量
- [x] 功能测试通过
- [x] 代码文档完善
