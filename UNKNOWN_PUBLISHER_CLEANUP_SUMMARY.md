# Unknown Publisher 数据清理总结

**清理日期**: 2026-01-04
**问题类型**: 数据质量问题 - API 异常导致的重复记录

---

## 问题背景

### 发现过程

用户提出两个问题：
1. 修复前是否有 PaddleOCR-VL 模型被计入 ERNIE-4.5？
2. 已删除模型列表中，Publisher 为 Unknown 的是数据问题还是真删除？

### 问题根源

**2025-12-08 和 2025-12-11 两天，HuggingFace API 返回的 publisher 信息异常**，导致一些模型被记录为 `publisher='Unknown'`。

**典型案例** (AILAND 模型):
```
2026-01-02: publisher=HMOOZX, downloads=0     ✅ 正常
2025-12-26: publisher=HMOOZX, downloads=0     ✅ 正常
2025-12-11: publisher=Unknown, downloads=0    ❌ API 异常
2025-12-08: publisher=Unknown, downloads=0    ❌ API 异常
2025-12-05: publisher=HMOOZX, downloads=0     ✅ 正常
```

**结果**: 回填模式下，`(Unknown, AILAND)` 和 `(HMOOZX, AILAND)` 被视为两个不同的模型，造成重复计数。

---

## 影响范围

### 受影响的模型数量

**总计**: 20 个模型 × 2 个日期 = **40 条错误记录**

**分布**:
- ERNIE-4.5: 10 个模型
- PaddleOCR-VL: 10 个模型

### 受影响模型列表

#### ERNIE-4.5 (10 个)

| 模型名 | 正确 Publisher | Unknown 日期 |
|--------|---------------|-------------|
| Darwin-gpt-ernie-20b | Seawolf2357 | 2025-12-08, 2025-12-11 |
| Hexo-0.5-Nano-Experimental | Legokeeper | 2025-12-08, 2025-12-11 |
| bai-ming-reinit-550m-zero | Nqzfaizal77Ai | 2025-12-08, 2025-12-11 |
| elementor-ernie45-custom | Nikskic | 2025-12-08, 2025-12-11 |
| vl-lora | Meister1378 | 2025-12-08, 2025-12-11 |
| eUp_NMT_10-36-40_22-09-2025 | Bunbohue | 2025-12-08, 2025-12-11 |
| eUp_NMT_10-57-55_19-09-2025 | Bunbohue | 2025-12-08, 2025-12-11 |
| food_nutrition_coach | Siddhu217256 | 2025-12-08, 2025-12-11 |
| jarvis | Linuxk205 | 2025-12-08, 2025-12-11 |
| Owl | Anthropicallyhuggingfaces | 2025-12-08, 2025-12-11 |

#### PaddleOCR-VL (10 个)

| 模型名 | 正确 Publisher | Unknown 日期 |
|--------|---------------|-------------|
| Proton | Urislam777 | 2025-12-08, 2025-12-11 |
| Trouter-20b | Opentrouter | 2025-12-08, 2025-12-11 |
| anpr-models | Josephinekmj | 2025-12-08, 2025-12-11 |
| tfjs-mobilenet-692 | Re2906 | 2025-12-08, 2025-12-11 |
| ProVerBs_Law_777 | Solomon7890 | 2025-12-08, 2025-12-11 |
| Neopulse | Alvessweet | 2025-12-08, 2025-12-11 |
| Mark | Mark227p | 2025-12-08, 2025-12-11 |
| AILAND | HMOOZX | 2025-12-08, 2025-12-11 |
| Elonmusk | Stanley647 | 2025-12-08, 2025-12-11 |
| MODEL-ONE-1 | Unyil17 | 2025-12-08, 2025-12-11 |

---

## 清理方案

### 方案选择

**用户建议**: 直接修复数据，而不是修改代码逻辑 ✅

**理由**:
- 这是数据质量问题，应该在数据层面解决
- 代码逻辑保持简洁，不引入复杂的去重规则
- 一次性清理，彻底解决问题

### 清理脚本

**文件**: `scripts/cleanup_unknown_publisher_duplicates.py`

**逻辑**:
1. 备份数据库
2. 识别 20 个受影响的模型
3. 删除这些模型在 2025-12-08 和 2025-12-11 的 Unknown 记录
4. 验证清理结果

### 执行结果

```
✅ 备份完成: backups/ernie_downloads_backup_20260104_162413.db
✅ 已删除 40 条 Unknown Publisher 重复记录
✅ 所有 Unknown 重复记录已成功清理
```

---

## 清理效果

### 数据统计对比

#### ERNIE-4.5

| 指标 | 清理前 | 清理后 |
|------|--------|--------|
| 累计衍生模型 | 204 | 194 ✅ |
| 当前日期衍生模型 | 187 | 187 |
| 已删除/隐藏模型 | 17 | 7 ✅ |
| Unknown publisher 模型 | 10 | 0 ✅ |
| **数学验证** | 187 + 17 = 204 | **187 + 7 = 194 ✅** |

**差异**: 减少 10 个重复（全部是 Unknown 假删除）

#### PaddleOCR-VL

| 指标 | 清理前 | 清理后 |
|------|--------|--------|
| 累计衍生模型 | 45 | 35 ✅ |
| 当前日期衍生模型 | 35 | 35 |
| 已删除/隐藏模型 | 10 | 0 ✅ |
| Unknown publisher 模型 | 10 | 0 ✅ |
| **数学验证** | 35 + 10 = 45 | **35 + 0 = 35 ✅** |

**差异**: 减少 10 个重复（全部是 Unknown 假删除）

### 真正被删除的模型

清理后，只有 **7 个 ERNIE-4.5 模型真正被删除**，PaddleOCR-VL 无删除模型。

#### ERNIE-4.5 真删除模型 (7 个)

1. **Mungert/ERNIE-4.5-21B-A3B-Thinking-GGUF** (quantized)
   - 最后出现: 2025-12-19
   - 最后下载量: 14,587

2. **Mungert/ERNIE-4.5-21B-A3B-PT-GGUF** (quantized)
   - 最后出现: 2025-12-19
   - 最后下载量: 4,003

3. **Hsuwill000/ERNIE-4.5-0.3B-PT_int4_ov** (quantized)
   - 最后出现: 2025-12-19
   - 最后下载量: 38

4. **Wasirmen123/ERNIE-4.5-21B-A3B-PT-Q3_K_L-GGUF** (quantized)
   - 最后出现: 2025-12-17
   - 最后下载量: 2,578

5. **Wasirmen123/ERNIE-4.5-21B-A3B-PT-Q4_K_M-GGUF** (quantized)
   - 最后出现: 2025-12-17
   - 最后下载量: 55

6. **Hon9Kon9Ize/ERINE-4.5-0.3B-Yue** (finetune)
   - 最后出现: 2025-12-02
   - 最后下载量: 28

7. **Fixart/ERNIE-4.5-21B-A3B-Thinking-mlx-3Bit** (quantized)
   - 最后出现: 2025-11-28
   - 最后下载量: 147

**特点**: 主要是量化（GGUF）模型被删除，可能是因为模型质量或维护问题。

#### PaddleOCR-VL 真删除模型

**0 个** - 所有模型都仍然存在，之前显示的 10 个"删除"全部是 Unknown 重复。

---

## 回答用户问题

### Q1: 修复前是否有 PaddleOCR-VL 模型被计入 ERNIE-4.5？

**答：是的！**

- **27 个独特的 PaddleOCR-VL 模型**（共 106 条历史记录）被错误分类为 ERNIE-4.5
- 这些模型的 `base_model` 都是 `PaddlePaddle/PaddleOCR-VL`
- 已在之前的分类修复中全部纠正

**结论**: 修复前看到的 ERNIE-4.5 数字偏大是因为包含了这些本应属于 PaddleOCR-VL 的模型。

### Q2: Unknown Publisher 模型是数据问题还是真删除？是否重复计算？

**答：全部都是数据获取问题，且正在被重复计算！**

- **20 个模型**（10 ERNIE + 10 PaddleOCR）在 2025-12-08 和 2025-12-11 被错误记录为 Unknown
- 同一模型在其他日期都有正确的 publisher
- 回填模式下被视为两个不同的模型，造成重复计数
- **100% 都不是真删除**

**清理结果**:
- ERNIE-4.5: 17 → 7 真删除（减少 10 个假删除）
- PaddleOCR-VL: 10 → 0 真删除（减少 10 个假删除）

---

## 技术洞察

### ⚠️ 重要：模型唯一性标识

**模型的唯一键必须是: `(repo, publisher, model_name)`**

这三个字段缺一不可：
- `repo`: 平台（Hugging Face, ModelScope 等）
- `publisher`: 发布者（用户/组织名）
- `model_name`: 模型名称

**错误示例**:
```
❌ 使用 (repo, model_name) 作为唯一键
   问题：不同 publisher 可能发布同名模型
   例如：bartowski/ERNIE-4.5-GGUF 和 lmstudio/ERNIE-4.5-GGUF 是两个不同的模型
```

**本次清理的特殊性**:

本次清理之所以可以删除 Unknown publisher 记录，是因为：

1. ✅ **这些模型在其他日期都有正确的 publisher**
   - 例如：`(Hugging Face, HMOOZX, AILAND)` 在 2026-01-02 存在
   - 而 `(Hugging Face, Unknown, AILAND)` 只在 2025-12-08/11 出现

2. ✅ **经过验证，这是同一个模型的重复记录**
   - 通过检查历史记录，确认 AILAND 模型始终属于 HMOOZX
   - Unknown 版本是 API 异常导致的错误数据

3. ⚠️ **这是特例，不代表一般规则**
   - 正常情况下，`(repo, Unknown, model_name)` 和 `(repo, publisher_A, model_name)` 应该被视为两个不同的模型
   - 只有在确认是 API 异常导致的重复时，才能删除 Unknown 记录

**正确的唯一性判断**:
```python
# ✅ 正确：使用完整的三元组
unique_key = (repo, publisher, model_name)

# ❌ 错误：缺少 publisher
unique_key = (repo, model_name)  # 会导致不同 publisher 的同名模型被误判为重复
```

### API 异常日期

**2025-12-08 和 2025-12-11** 这两天 HuggingFace API 返回的 publisher 信息不完整，可能原因：
- HuggingFace API 服务异常
- 网络问题导致数据不完整
- 特定时间段的 API 版本变更

### 数据完整性策略

**现有策略**: 保存所有原始数据，在查询时去重

**改进建议**:
1. ✅ 定期检查 `publisher='Unknown'` 的记录
2. ✅ 对于 Unknown 记录，验证同一模型在其他日期是否有正确的 publisher
3. ✅ 发现异常日期后，批量清理错误数据
4. 考虑在数据获取时增加重试机制
5. ⚠️ **务必使用 `(repo, publisher, model_name)` 作为模型唯一标识**

---

## 相关文件

**清理脚本**: `scripts/cleanup_unknown_publisher_duplicates.py`

**备份文件**: `backups/ernie_downloads_backup_20260104_162413.db`

**相关文档**:
- `CLASSIFICATION_FIX_SUMMARY.md`: 模型分类错误修复
- `STANDARDIZATION_FIX_SUMMARY.md`: 数据标准化一致性修复

---

## 总结

### 清理成果

✅ 删除 40 条错误的 Unknown publisher 记录
✅ ERNIE-4.5 累计数从 204 降至 194（更准确）
✅ PaddleOCR-VL 累计数从 45 降至 35（更准确）
✅ 消除所有 Unknown publisher 的重复计数
✅ 所有数学关系验证通过

### 数据质量提升

- **ERNIE-4.5 真删除模型**: 7 个（主要是量化模型）
- **PaddleOCR-VL 真删除模型**: 0 个（全部仍存在）
- **Unknown publisher 记录**: 0 个（已全部清理）

### 系统改进

- 数据更准确，反映真实的模型生态
- 消除了因 API 异常导致的数据污染
- 为未来的数据质量监控提供了参考案例
