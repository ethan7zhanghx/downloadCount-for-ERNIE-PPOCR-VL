# ModelScope Model Tree 功能使用说明

## 功能概述

ModelScope Model Tree 功能可以自动获取 ModelScope 平台上官方模型的所有衍生模型（微调、量化、重组等）。

## 核心特性

1. **自动发现官方模型**: 从数据库中自动发现所有 ModelScope 平台的官方模型
2. **侧边栏标签切换**: 打开侧边栏后，通过点击标签切换不同类型的衍生模型
3. **批量处理**: 一次性获取所有官方模型的衍生模型
4. **智能去重**: 自动跳过重复的模型
5. **完整信息**: 通过 API 获取下载量、时间戳等完整信息

## 使用方法

### 方法1: 自动发现所有官方模型（推荐）

```python
from ernie_tracker.fetchers.fetchers_modeltree import update_modelscope_model_tree

# 自动发现数据库中所有 ModelScope 官方模型并获取其 Model Tree
df, count = update_modelscope_model_tree(
    save_to_db=True,      # 保存到数据库
    auto_discover=True     # 自动发现官方模型
)

print(f"获取到 {count} 个衍生模型")
```

### 方法2: 指定特定模型

```python
from ernie_tracker.fetchers.fetchers_modeltree import update_modelscope_model_tree

# 只获取指定模型的 Model Tree
df, count = update_modelscope_model_tree(
    save_to_db=True,
    base_models=[
        'PaddlePaddle/PaddleOCR-VL',
        'PaddlePaddle/ERNIE-4.5-21B-A3B-PT',
        # ... 更多模型
    ],
    auto_discover=False    # 不自动发现
)
```

### 方法3: 直接使用核心函数

```python
from ernie_tracker.fetchers.fetchers_modeltree import get_modelscope_model_tree_children

# 获取单个模型的 Model Tree
derivatives = get_modelscope_model_tree_children('PaddlePaddle/PaddleOCR-VL')

for deriv in derivatives:
    print(f"模型: {deriv['id']}")
    print(f"  类型: {deriv['name_zh']} ({deriv['name_en']})")
    print(f"  下载量: {deriv['downloads']}")
    print(f"  发布者: {deriv['author']}")
```

## 数据输出格式

每条记录包含以下字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `date` | 抓取日期 | `2026-01-16` |
| `repo` | 平台名称 | `ModelScope` |
| `model_name` | 模型名称 | `PaddleOCR-VL-Receipt` |
| `publisher` | 发布者 | `megemini` |
| `download_count` | 下载量 | `196` |
| `model_category` | 模型分类 | `paddleocr-vl`, `ernie-4.5` |
| `model_type` | 衍生类型 | `finetunes`, `repackaged`, `quantized` |
| `base_model` | 基础模型 | `PaddlePaddle/PaddleOCR-VL` |
| `data_source` | 数据来源 | `model_tree` |
| `likes` | 点赞数 | `23` |
| `created_at` | 创建时间 | `2026-01-15` |
| `last_modified` | 最后修改时间 | `2026-01-15` |

## 官方模型识别规则

自动发现时，以下发布者被视为官方：
- `百度`
- `baidu`
- `Paddle`
- `PaddlePaddle`
- `yiyan`
- `一言`
- 包含上述关键词的发布者（如 `PaddleTeam`）

## 技术实现

### 点击策略

通过测试发现，正确的点击元素是：
- **目标元素**: `div.acss-hd4erf`（包含中文标题的div）
- **不要点击**: `span.antd5-tree-node-content-wrapper`（外层wrapper）

### 侧边栏标签切换

1. 点击第一个衍生类型打开侧边栏
2. 查找侧边栏内的标签元素：`div.acss-xqwyei`
3. 依次点击每个标签切换不同类型
4. 提取每个标签下的模型卡片

### 优势

相比之前的逐个点击方案：
- ✅ 更快：只需打开一次侧边栏
- ✅ 更可靠：避免重复打开/关闭侧边栏的问题
- ✅ 更完整：能获取所有类型的衍生模型

## 性能考虑

- 单个模型处理时间：约 10-15 秒
- 批量处理建议：分批处理，避免一次性处理过多模型
- 复用 Driver：多个模型共用一个 Selenium Driver 实例

## 测试示例

```bash
# 测试单个模型
python3 -c "
from ernie_tracker.fetchers.fetchers_modeltree import get_modelscope_model_tree_children
derivatives = get_modelscope_model_tree_children('PaddlePaddle/PaddleOCR-VL')
print(f'获取到 {len(derivatives)} 个衍生模型')
"

# 测试批量获取（自动发现）
python3 tests/test_batch_modelscope.py
```

## 已知问题

1. **数据库连接**: 某些环境可能无法连接数据库，会回退到默认模型列表
2. **点击遮挡**: 某些衍生类型可能被遮挡无法点击，会跳过该类型
3. **API限流**: 大量API调用可能触发限流，建议添加适当延迟

## 后续优化

- [ ] 添加断点续传功能（记录已处理的模型）
- [ ] 并发处理多个模型
- [ ] 添加更详细的进度显示
- [ ] 支持从配置文件读取模型列表
