# 衍生模型统计修复说明

## 问题描述

在统计衍生模型的下载量时，系统只根据模型名称（`model_name`）进行统计，导致不同 Publisher 发布的同名衍生模型会被统计到一起。

### 具体案例

以 `ERNIE-4.5-21B-A3B-PT-GGUF` 模型为例，数据库中有多个不同 publisher 发布的同名模型：

| Publisher | Hugging Face 下载量 |
|-----------|-------------------|
| unsloth | 28,228 |
| lmstudio-community | 21,376 |
| Mungert | 3,746 |
| mradermacher | 786 |
| dengcao | 49 |
| **总计** | **54,185** |

旧的统计方式会将这些全部合并为一个模型，无法区分各 publisher 的贡献。

## 问题根源

在 `analysis.py` 的 `create_pivot_table` 函数中：

```python
# 旧代码 - 只使用 model_name 作为索引
pivot_df = pd.pivot_table(
    data,
    values='download_count',
    index='model_name',        # 🔴 问题所在
    columns='repo',
    aggfunc='sum',             # 相同 model_name 会被求和
    fill_value=0
)
```

## 修复方案

### 1. 修改 `create_pivot_table` 函数

添加 `group_by_publisher` 参数，用于衍生模型统计：

```python
def create_pivot_table(data, repo_order=None, model_order=None, group_by_publisher=False):
    """
    Args:
        group_by_publisher: 是否按 publisher 分组（用于衍生模型）。
                           如果为 True，索引为 (model_name, publisher)；
                           如果为 False，索引仅为 model_name（用于官方模型）
    """
    if group_by_publisher:
        # 衍生模型：使用 (model_name, publisher) 作为索引
        pivot_df = pd.pivot_table(
            data,
            values='download_count',
            index=['model_name', 'publisher'],  # ✅ 多层索引
            columns='repo',
            aggfunc='sum',
            fill_value=0
        )
    else:
        # 官方模型：使用 model_name 作为索引
        pivot_df = pd.pivot_table(
            data,
            values='download_count',
            index='model_name',
            columns='repo',
            aggfunc='sum',
            fill_value=0
        )
```

### 2. 更新衍生模型统计调用

在 `calculate_weekly_report` 函数中：

```python
# 衍生模型数据使用 group_by_publisher=True
current_derivative_pivot = create_pivot_table(
    current_derivative_data,
    model_order=None,
    group_by_publisher=True  # ✅ 区分不同 publisher
)
previous_derivative_pivot = create_pivot_table(
    previous_derivative_data,
    model_order=None,
    group_by_publisher=True  # ✅ 区分不同 publisher
)
```

### 3. 更新 `_get_top_models` 函数

支持多层索引的处理：

```python
# 检查 pivot 索引是否为多层索引（衍生模型）
has_multiindex = isinstance(current_pivot.index, pd.MultiIndex)

# 下载量最高
top_download_idx = current_pivot[repo].idxmax()

if has_multiindex:
    # 多层索引：(model_name, publisher)
    top_download_model, top_download_publisher = top_download_idx
else:
    # 单层索引：model_name
    top_download_model = top_download_idx
    top_download_publisher = ...  # 从数据源查找
```

## 测试结果

运行 `test_fix.py` 的测试结果：

```
旧方式 Hugging Face 总下载量: 54,185  （合并所有 publisher）

新方式 Hugging Face 总下载量: 54,185  （分开统计，总数相同）

各 publisher 的下载量分布:
  - Mungert: 3,746
  - dengcao: 49
  - lmstudio-community: 21,376
  - mradermacher: 786
  - unsloth: 28,228
```

## 修改文件

1. `analysis.py` - 主要修改文件
   - `create_pivot_table` 函数：添加 `group_by_publisher` 参数
   - `calculate_weekly_report` 函数：衍生模型统计时使用 `group_by_publisher=True`
   - `_get_top_models` 函数：支持多层索引处理

2. `test_fix.py` - 测试文件（新增）

## 影响范围

- ✅ 官方模型统计：**不受影响**（继续使用单层索引）
- ✅ 衍生模型统计：**修复完成**（使用多层索引区分 publisher）
- ✅ 周报生成：**正常工作**（自动处理单层和多层索引）

## 后续建议

1. 在周报中可以考虑展示"下载量最高的衍生模型及其 publisher"
2. 可以统计每个 publisher 对衍生模型生态的贡献度
3. 监控哪些 publisher 最活跃（发布最多衍生模型）

---

修复时间: 2025-11-10
修复人: Claude Code
