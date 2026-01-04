# 模型分类错误修复总结

**修复日期**: 2026-01-04
**修复人**: Claude Code

## 问题描述

数据库中存在衍生模型分类错误的情况：

### 错误案例
- `RysOCR`: base_model 是 `PaddlePaddle/PaddleOCR-VL`，但被错误分类为 `ernie-4.5`
- `polish-ocr-lora-broken`: base_model 是 `PaddlePaddle/PaddleOCR-VL`，但被错误分类为 `ernie-4.5`
- 共计 **106 条记录** 被错误分类

## 根本原因

代码逻辑时序问题：

1. `fetch_model_detail()` 函数从 API 提取 base_model → `base_from_api` (可能为空)
2. 使用 `base_from_api` (None) 进行分类 → **错误分类为 ernie-4.5**
3. 通过 Model Tree 查询后，在 `add_record()` 时才传入正确的 `base_model`
4. 最终：数据库中 `base_model` 字段正确，但 `model_category` 已经错了

**问题位置**: `ernie_tracker/fetchers/fetchers_modeltree.py`:660-716

## 修复方案

### 1. 数据库修复

**执行脚本**: `scripts/fix_model_classification.py`

**修复策略**:
```sql
-- 情况1：base_model 是 PaddleOCR-VL → 应分类为 paddleocr-vl
UPDATE model_downloads
SET model_category = 'paddleocr-vl'
WHERE base_model LIKE '%PaddleOCR-VL%'
AND model_category = 'ernie-4.5'

-- 情况2：base_model 是 ERNIE，且模型名不含 PaddleOCR → 应分类为 ernie-4.5
UPDATE model_downloads
SET model_category = 'ernie-4.5'
WHERE (base_model LIKE '%ERNIE%' OR base_model LIKE '%ernie%')
AND base_model NOT LIKE '%PaddleOCR%'
AND model_category = 'paddleocr-vl'
AND model_name NOT LIKE '%PaddleOCR%'
AND model_name NOT LIKE '%paddleocr%'
```

**修复结果**: 106 条记录被成功修复

### 2. 代码逻辑修复

**修改文件**: `ernie_tracker/fetchers/fetchers_modeltree.py`

#### 修改点1: Model Tree 衍生模型分类 (第668-690行)

```python
# 🔧 修复：使用 Model Tree 提供的 base_model 重新分类
deriv_detail['model_category'] = classify_model(
    deriv['id'],
    deriv_detail['publisher'],
    model_id  # 使用 Model Tree 的 base_model，而不是 base_from_api
)
```

#### 修改点2: 关键词搜索结果分类 (第699-705行)

```python
# 🔧 修复：使用当前基座的 model_id 重新分类
detail['model_category'] = classify_model(
    model.id,
    detail['publisher'],
    model_id  # 使用当前基座的 model_id
)
```

## 修复效果验证

### 数据对比

**修复前**:
- ERNIE-4.5: 224 个模型 (包含 11 个错误归类)
- PaddleOCR-VL: 21 个模型 (缺少 11 个)

**修复后**:
- ERNIE-4.5: 214 个模型 ✅
- PaddleOCR-VL: 36 个模型 ✅

### 具体案例验证

| 模型名 | base_model | 修复前 | 修复后 |
|--------|-----------|--------|--------|
| RysOCR | PaddlePaddle/PaddleOCR-VL | ernie-4.5 ❌ | paddleocr-vl ✅ |
| polish-ocr-lora-broken | PaddlePaddle/PaddleOCR-VL | ernie-4.5 ❌ | paddleocr-vl ✅ |
| PaddleOCR-VL-half-GGUF-pured | PaddlePaddle/PaddleOCR-VL | paddleocr-vl ✅ | paddleocr-vl ✅ |

### 测试验证

所有测试通过 (`scripts/test_classification.py`):
- RysOCR: ✅ 正确
- polish-ocr-lora-broken: ✅ 正确
- PaddleOCR-VL-half-GGUF-pured: ✅ 正确
- PaddleOCR-VL-MLX: ✅ 正确

## 备份信息

**备份文件**: `ernie_downloads.db.backup_20260104_142146`
**备份大小**: 2.2M
**备份时间**: 2026-01-04 14:21:46

## 相关文件

- 分析脚本: `scripts/analyze_derivative_growth.py`
- 测试脚本: `scripts/test_classification.py`, `scripts/test_fixed_classification.py`
- 修复脚本: `scripts/fix_model_classification.py`
- 核心代码: `ernie_tracker/fetchers/fetchers_modeltree.py`

## 结论

✅ 数据库中所有错误分类已修复
✅ 代码逻辑已更新，防止未来出现类似问题
✅ 所有测试验证通过
✅ 数据已备份，可安全回滚
