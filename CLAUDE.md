# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ERNIE model download statistics tracking system built with Streamlit. It collects download data for PaddlePaddle/ERNIE models from multiple platforms (Hugging Face, ModelScope, AI Studio, GitCode, CAICT, Modelers, Gitee) and stores them in a SQLite database for analysis.

## Quick Start Commands

### Run the Application

```bash
# Start the Streamlit app (recommended)
./start.sh

# Or manually
python3 -m streamlit run app.py
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Run Tests

Tests can be run directly as Python scripts (pytest is not installed):

```bash
# Test HuggingFace API
python3 tests/test_hf_api.py

# Test model search
python3 tests/test_search_models.py

# Test data entry
python3 tests/test_data_entry.py
```

### Database Operations

```bash
# Export database to CSV
python3 scripts/export_db.py

# Import data from Excel
python3 scripts/import_excel.py

# Clean up database
python3 scripts/cleanup_db.py

# Check database schema
sqlite3 ernie_downloads.db ".schema"

# Query data
sqlite3 ernie_downloads.db "SELECT * FROM model_downloads LIMIT 10"
```

## Architecture

### Core Package: `ernie_tracker/`

**Configuration (`config.py`)**
- `DB_PATH`: Database file path (ernie_downloads.db)
- `DATA_TABLE`: Main table name (model_downloads)
- `STATS_TABLE`: Platform statistics table (platform_stats)
- `PLATFORM_NAMES`: Platform name mappings
- `GITCODE_MODEL_LINKS`, `CAICT_MODEL_LINKS`: Fixed model URLs for platforms without search APIs

**Database Layer**
- `db.py`: Core database operations (init, save, load)
  - `init_database()`: Initialize tables with schema migration support
  - `save_to_db()`: Insert data without deduplication (raw data preserved)
  - `load_data_from_db()`: Load and deduplicate data dynamically
- `db_manager.py`: Management operations (backup, restore, delete by date)

**Data Fetching Architecture**

The system uses a two-tier fetcher architecture:

1. **Base Fetcher (`fetchers/base_fetcher.py`)**
   - Abstract base class `BaseFetcher` that all platform fetchers inherit from
   - Defines standard interface: `fetch(progress_callback, progress_total)`
   - Returns tuple: `(DataFrame, total_count)`

2. **Unified Fetchers (`fetchers/fetchers_unified.py`)**
   - `UNIFIED_PLATFORM_FETCHERS`: Central registry mapping platform names to fetcher functions
   - `fetch_hugging_face_data_unified()`: HuggingFace fetcher with model tree support
   - `fetch_all_paddlepaddle_data()`: Orchestrates all platform fetches

3. **Model Tree Feature (`fetchers/fetchers_modeltree.py`)**
   - `get_all_ernie_derivatives()`: Recursively discovers derivative models (finetuned, quantized, etc.) using HuggingFace's base_model relationships
   - `classify_model()`: Categorizes models into 'ernie-4.5' or 'paddleocr-vl'
   - `classify_model_type()`: Identifies model type (original, finetune, quantized, adapter, lora, merge)
   - Uses tag-based detection (`base_model:quantized:`, `base_model:finetune:`, etc.) for reliable classification

4. **Platform-Specific Fetchers**
   - `fetchers_api.py`: API-based platforms (ModelScope)
   - `fetchers_fixed_links.py`: Fixed URL list platforms (GitCode, CAICT)
   - `selenium.py`: Selenium-based scraping for platforms without APIs

**Analysis Module (`analysis.py`)**
- `calculate_weekly_report()`: Generates weekly comparison reports for ERNIE-4.5 or PaddleOCR-VL
- `mark_official_models()`: Identifies official models by publisher (百度, baidu, Paddle, etc.)
- `analyze_derivative_models_all_platforms()`: Cross-platform derivative model analysis
- Uses backfill logic (`last_value_per_model=True`) for counting models across time
- Enforces deduplication and standardization (publisher name normalization, model name cleanup)

**Model Analysis (`model_analysis.py`)**
- `OFFICIAL_MODEL_GROUPS`: Defines official ERNIE model families and their variants
- `normalize_base_models()`: Standardizes base_model references and fixes misclassifications
- `analyze_derivative_ecosystem()`: HuggingFace-specific derivative analysis with base_model grouping

### Main Application (`app.py`)

Streamlit-based web interface with multiple analysis pages:

**Pages:**
1. **📥 数据更新**: Fetch data from all platforms (supports parallel/sequential execution)
2. **📊 ERNIE-4.5 分析**: Weekly report comparing two dates for ERNIE-4.5 models
3. **📊 PaddleOCR-VL 分析**: Weekly report for PaddleOCR-VL models
4. **🌳 衍生模型生态**: Cross-platform derivative model ecosystem analysis
5. **🌲 Model Tree 统计**: HuggingFace model tree exploration
6. **🗄️ 数据库管理**: Database backup, restore, and data entry

**Key Functions:**
- `fetch_platform_data_only()`: Core fetching logic with progress tracking
- Parallel fetching support using `concurrent.futures`
- Progress callbacks for UI updates
- Automatic database saving with deduplication at query time

### Scripts Directory (`scripts/`)

Utility scripts for data management and fixes:
- `backfill_model_category.py`: Populate `model_category` field using `search_keyword` and name matching
- `fetch_*_model_tree.py`: Fetch model trees for specific model families
- `reclassify_*.py`: Fix model classifications
- `fix_*.py`: Various data repair scripts
- `run_gitcode_fetcher.py`: Standalone GitCode fetcher
- `export_db.py`: Export database to Excel
- `import_excel.py`: Import data from Excel
- `cleanup_db.py`: Database cleanup operations
- `paddle_attribution/`: PaddlePaddle usage attribution analysis

## Database Schema

**Table: `model_downloads`**
- Core fields: `date`, `repo`, `model_name`, `publisher`, `download_count`
- Classification: `model_type`, `model_category`, `tags`, `base_model`
- Metadata: `data_source`, `likes`, `library_name`, `pipeline_tag`, `search_keyword`
- Timestamps: `created_at`, `last_modified`, `fetched_at`
- API data: `base_model_from_api`

**Table: `platform_stats`**
- Tracks last known model count per platform for progress estimation
- Fields: `platform`, `last_model_count`, `last_updated`

**Data Strategy**: Raw data is saved without deduplication. Deduplication and max value selection happens at query time in `load_data_from_db()`.

**Critical: `model_category` Field**
- All records must have `model_category` filled (ernie-4.5, paddleocr-vl, other-ernie, or other)
- If adding new data without this field, run `python3 scripts/backfill_model_category.py` to populate it
- The backfill script uses `search_keyword` (if available) and model name matching
- This field is essential for cross-platform analysis

## Key Implementation Details

### Model Classification

**Model Categories** (stored in `model_category` field):
- `ernie-4.5`: ERNIE 4.5 models and related derivatives
- `paddleocr-vl`: PaddleOCR-VL models
- `other-ernie`: Other ERNIE family models
- `other`: Non-ERNIE models

**Classification Logic:**
1. **Priority**: Use `search_keyword` field if available (records which search term found the model)
2. **Fallback**: Model name matching (case-insensitive, supports Chinese characters like "文心")
3. All platforms now have `model_category` populated via backfill script

**Model Types** (HuggingFace only, detected using structured tags):
- Priority: HF tags > model card data > name-based heuristics
- Tag patterns: `base_model:quantized:*`, `base_model:finetune:*`, etc.
- Types: `original`, `quantized`, `finetune`, `adapter`, `lora`, `merge`, `other`

### Progress Tracking

Fetchers use a two-phase progress system:
1. **With reference count** (subsequent runs): Use stored `last_model_count` from database
2. **Without reference** (first run): Use discovered total during fetch
3. Auto-updates reference count when discovering more models

### Selenium Configuration

Set `SELENIUM_HEADLESS = False` in `config.py` by default. Change to `True` for headless operation.

### HuggingFace Model Tree

When `use_model_tree=True`:
- Fetches ERNIE-4.5 and PaddleOCR-VL base models
- Recursively discovers all derivatives using HuggingFace's base_model tags
- More comprehensive than keyword search alone
- Automatically classifies derivative types
- Sets `data_source` field to indicate discovery method (`search`, `model_tree`, `both`)

### Official vs Derivative Model Detection

**Official Models** are identified by publisher containing (case-insensitive):
- '百度', 'baidu', 'Paddle', 'yiyan', '一言'

**Derivative Models** are all non-official models (used in cross-platform ecosystem analysis)

The `mark_official_models()` function adds an `is_official` column to DataFrames for filtering.

## Development Notes

### When Adding New Platforms

1. Create fetcher function in appropriate `fetchers/fetchers_*.py` file
2. Ensure fetcher records `search_keyword` parameter in `create_record()` calls
3. Add to `UNIFIED_PLATFORM_FETCHERS` in `fetchers_unified.py`
4. Add platform name to `PLATFORM_NAMES` in `config.py`
5. Follow `BaseFetcher` interface: return `(DataFrame, total_count)`
6. Add official publisher pattern to `OFFICIAL_RULES` in `analysis.py` if needed

### When Adding Database Fields

1. Add column to schema in `db.py::init_database()`
2. Add migration logic in the column check section (lines 36-62)
3. Update DataFrame creation in relevant fetchers
4. If the field should be populated for existing data, create a backfill script in `scripts/` (see `backfill_model_category.py` as example)

### When Fixing Data Issues

1. Always backup first using `db_manager.backup_database()`
2. Write scripts in `scripts/` directory for reproducibility
3. Update classification logic in `model_analysis.py` or `fetchers_modeltree.py` if needed
4. After schema changes or data cleanup, clear Python cache: `find . -name "__pycache__" -type d -exec rm -rf {} +`

### Debugging Fetchers

Individual platform fetchers can be tested by:
```python
from ernie_tracker.fetchers.fetchers_unified import UNIFIED_PLATFORM_FETCHERS
df, count = UNIFIED_PLATFORM_FETCHERS['huggingface']()
print(f"Found {count} models")
```

## Configuration Files

- `requirements.txt`: Python dependencies (Streamlit, Selenium, HuggingFace Hub, etc.)
- `start.sh`: Startup script with dependency checking
- `.gitignore`: Excludes `ernie_downloads.db`, exports, backups, and Python cache files
