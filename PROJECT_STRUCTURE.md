# 项目目录结构

本文档说明项目的目录结构和组织方式。

## 📁 目录结构

```
DownloadData/
├── app.py                      # Streamlit主应用入口
├── start.sh                    # 应用启动脚本
├── requirements.txt            # Python依赖列表
├── README.md                   # 项目说明文档
├── CLAUDE.md                   # Claude Code使用指南
├── PROJECT_STRUCTURE.md        # 本文件 - 目录结构说明
├── .gitignore                  # Git忽略规则
│
├── ernie_tracker/              # 核心功能包
│   ├── __init__.py
│   ├── config.py               # 配置文件（数据库路径、平台配置等）
│   ├── db.py                   # 数据库核心操作
│   ├── db_manager.py           # 数据库管理（备份、恢复等）
│   ├── analysis.py             # 数据分析功能
│   ├── model_analysis.py       # 模型分析功能
│   ├── utils.py                # 工具函数
│   └── fetchers/               # 数据获取模块
│       ├── __init__.py
│       ├── base_fetcher.py             # 基础获取器抽象类
│       ├── fetchers_api.py             # API方式获取
│       ├── fetchers_fixed_links.py     # 固定链接获取
│       ├── fetchers_modeltree.py       # Model Tree功能
│       ├── fetchers_unified.py         # 统一获取器入口
│       └── selenium.py                 # Selenium爬虫获取
│
├── scripts/                    # 工具脚本目录
│   ├── export_db.py            # 数据库导出（到data/exports/）
│   ├── import_excel.py         # 从Excel导入数据
│   ├── cleanup_db.py           # 数据库清理
│   ├── backfill_model_category.py      # 回填模型分类
│   ├── backfill_model_category_all.py  # 批量回填
│   ├── fetch_ernie_model_tree.py       # 获取ERNIE模型树
│   ├── fetch_qwen_model_tree.py        # 获取Qwen模型树
│   ├── fetch_qwen_model_tree_v2.py     # 获取Qwen模型树v2
│   ├── analyze_derivative_growth.py    # 分析衍生模型增长
│   ├── analyze_qwen_model_tree.py      # 分析Qwen模型树
│   ├── fix_base_model_and_stats.py     # 修复base_model和统计
│   ├── fix_created_at.py               # 修复创建时间
│   ├── fix_model_classification.py     # 修复模型分类
│   ├── fix_model_tree_tags.py          # 修复模型树标签
│   ├── reclassify_by_base_model.py     # 按base_model重新分类
│   ├── reclassify_quantized.py         # 重新分类量化模型
│   ├── run_gitcode_fetcher.py          # 运行GitCode获取器
│   ├── check_sheets.py                 # 检查sheets
│   ├── cleanup_unknown_publisher_duplicates.py  # 清理未知发布者重复
│   ├── debug_missing_models.py         # 调试缺失模型
│   └── paddle_attribution/             # PaddlePaddle使用归因分析
│       └── paddle_attribution.py
│
├── data/                       # 数据目录（gitignored）
│   ├── .gitkeep                # 保持目录结构
│   ├── ernie_downloads.db      # 主数据库文件
│   ├── backups/                # 数据库备份目录
│   │   └── *.db                # 备份文件
│   └── exports/                # 数据导出目录
│       └── *.xlsx              # Excel导出文件
│
├── logs/                       # 日志目录（gitignored）
│   ├── .gitkeep                # 保持目录结构
│   └── *.log                   # 应用和爬虫日志
│
└── temp/                       # 临时文件目录（gitignored）
    ├── .gitkeep                # 保持目录结构
    ├── test_*.py               # 临时测试脚本
    └── BUGFIX_*.md             # 临时bug修复文档
```

## 📝 目录说明

### 核心文件
- **app.py**: Streamlit应用主入口
- **start.sh**: 应用启动脚本（带依赖检查）
- **requirements.txt**: Python包依赖列表

### ernie_tracker/ - 核心功能包
包含所有核心业务逻辑：
- **config.py**: 统一配置管理（数据库路径、平台配置等）
- **db.py**: 数据库操作（初始化、查询、保存）
- **db_manager.py**: 数据库管理（备份、恢复、删除）
- **analysis.py**: 周报生成、数据分析
- **model_analysis.py**: 模型分类、生态分析
- **fetchers/**: 各平台数据获取器

### scripts/ - 工具脚本
辅助工具和数据修复脚本：
- **export_db.py**: 导出数据库到Excel（自动保存到`data/exports/`）
- **import_excel.py**: 从Excel导入数据
- **cleanup_db.py**: 清理数据库
- **backfill_*.py**: 数据回填脚本
- **fix_*.py**: 数据修复脚本
- **fetch_*.py**: 模型树获取脚本
- **analyze_*.py**: 数据分析脚本

### data/ - 数据目录
**重要**: 此目录已加入`.gitignore`，不会被提交到Git。
- **ernie_downloads.db**: SQLite主数据库
- **backups/**: 数据库自动备份
- **exports/**: 数据导出文件（Excel格式）

### logs/ - 日志目录
**重要**: 此目录已加入`.gitignore`，不会被提交到Git。
- 存放应用日志、爬虫日志等

### temp/ - 临时文件目录
**重要**: 此目录已加入`.gitignore`，不会被提交到Git。
- 临时测试脚本
- 临时文档（BUGFIX.md等）

## 🔧 配置说明

### 数据库路径配置
数据库路径在`ernie_tracker/config.py`中配置：
```python
DB_PATH = "data/ernie_downloads.db"
```

### 导出路径配置
导出脚本（`scripts/export_db.py`）会自动将文件保存到：
```
data/exports/database_export_YYYY-MM-DD.xlsx
```

### 备份路径配置
数据库管理器（`ernie_tracker/db_manager.py`）会自动将备份保存到：
```
data/backups/ernie_downloads_backup_YYYYMMDD_HHMMSS.db
```

## 🚀 使用建议

### 首次运行
1. 确保已安装依赖：`pip3 install -r requirements.txt`
2. 运行应用：`./start.sh` 或 `python3 -m streamlit run app.py`
3. 数据库会自动在`data/`目录创建

### 数据备份
定期备份数据库：
```bash
python3 -c "from ernie_tracker.db_manager import backup_database; backup_database()"
```

### 数据导出
导出当前数据到Excel：
```bash
python3 scripts/export_db.py
```
文件会保存到`data/exports/`目录

### 添加新脚本
1. 将脚本放入`scripts/`目录
2. 如果是临时测试脚本，放入`temp/`目录
3. 如果需要操作数据库，使用`ernie_tracker/config.py`中的`DB_PATH`

## 📌 注意事项

1. **不要提交数据文件**: `data/`、`logs/`、`temp/`目录已在`.gitignore`中
2. **统一数据库路径**: 新脚本应从`ernie_tracker.config`导入`DB_PATH`
3. **导出文件位置**: 所有导出文件应保存到`data/exports/`
4. **备份文件位置**: 所有备份文件应保存到`data/backups/`
5. **日志文件位置**: 所有日志文件应保存到`logs/`

## 🔄 迁移说明

如果你有旧版本的项目，需要迁移到新目录结构：

1. **创建新目录**:
   ```bash
   mkdir -p data/backups data/exports logs temp
   touch data/.gitkeep logs/.gitkeep temp/.gitkeep
   ```

2. **移动数据库文件**:
   ```bash
   mv ernie_downloads.db data/
   mv *.db.backup* data/backups/
   ```

3. **移动日志文件**:
   ```bash
   mv *.log logs/
   ```

4. **移动临时文件**:
   ```bash
   mv test_*.py temp/
   mv BUGFIX_*.md temp/
   ```

5. **移动导出文件**:
   ```bash
   mv exports/* data/exports/
   mv backups/* data/backups/
   ```

6. **更新配置**: 已自动更新`ernie_tracker/config.py`中的`DB_PATH`

## 📚 参考文档

- **README.md**: 项目整体说明
- **CLAUDE.md**: Claude Code开发指南
- **requirements.txt**: 依赖列表

---

最后更新：2026-01-15
