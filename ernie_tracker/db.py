"""数据库操作模块"""
import sqlite3
import pandas as pd
import re
from datetime import date, datetime
from .config import DB_PATH, DATA_TABLE, STATS_TABLE

CUSTOM_MODELS_TABLE = "custom_models"
MODEL_FIELD_OVERRIDES_TABLE = "model_field_overrides"
OVERRIDABLE_FIELDS = ["model_category", "model_type", "base_model", "tags"]


def _normalize_override_key(value):
    """标准化覆盖规则的匹配键（用于大小写无关匹配）"""
    if value is None:
        return ""
    return str(value).strip().lower()


def _clean_override_value(value):
    """清理覆盖字段值，空值统一为 None"""
    if value is None:
        return None
    value_str = str(value).strip()
    return None if value_str.lower() in {"", "none", "nan"} else value_str


def _construct_huggingface_url(publisher, model_name):
    """根据 publisher/model_name 构造 Hugging Face 模型 URL"""
    publisher_val = str(publisher or "").strip().strip("/")
    model_name_val = str(model_name or "").strip().strip("/")
    if not publisher_val or not model_name_val:
        return None

    # 避免 model_name 已包含 publisher 前缀时出现重复路径
    prefix = f"{publisher_val}/"
    if model_name_val.lower().startswith(prefix.lower()):
        model_name_val = model_name_val[len(prefix):]
    if not model_name_val:
        return None

    return f"https://huggingface.co/{publisher_val}/{model_name_val}"


def _ensure_huggingface_urls(df):
    """
    补全 Hugging Face 缺失 URL：
    https://huggingface.co/{publisher}/{model_name}
    """
    if df is None or df.empty:
        return df

    required_cols = {"repo", "publisher", "model_name"}
    if not required_cols.issubset(df.columns):
        return df

    fixed_df = df.copy()
    if "url" not in fixed_df.columns:
        fixed_df["url"] = None

    repo_mask = fixed_df["repo"].astype(str).str.strip().str.lower() == "hugging face"
    missing_url_mask = (
        fixed_df["url"].isna()
        | fixed_df["url"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])
    )
    has_id_mask = (
        fixed_df["publisher"].notna()
        & fixed_df["model_name"].notna()
        & fixed_df["publisher"].astype(str).str.strip().str.lower().ne("")
        & fixed_df["publisher"].astype(str).str.strip().str.lower().ne("none")
        & fixed_df["publisher"].astype(str).str.strip().str.lower().ne("nan")
        & fixed_df["model_name"].astype(str).str.strip().str.lower().ne("")
        & fixed_df["model_name"].astype(str).str.strip().str.lower().ne("none")
        & fixed_df["model_name"].astype(str).str.strip().str.lower().ne("nan")
    )

    target_mask = repo_mask & missing_url_mask & has_id_mask
    if target_mask.any():
        fixed_df.loc[target_mask, "url"] = fixed_df.loc[target_mask].apply(
            lambda row: _construct_huggingface_url(row["publisher"], row["model_name"]),
            axis=1
        )

    return fixed_df


def init_database():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)

    # 创建模型下载数据表（扩展版本，支持模型类型和标签）
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATA_TABLE} (
            date TEXT,
            repo TEXT,
            model_name TEXT,
            publisher TEXT,
            download_count TEXT,
            model_type TEXT,
            model_category TEXT,
            tags TEXT,
            base_model TEXT,
            data_source TEXT,
            likes TEXT,
            library_name TEXT,
            pipeline_tag TEXT,
            created_at TEXT,
            last_modified TEXT,
            fetched_at TEXT,
            base_model_from_api TEXT,
            search_keyword TEXT
        )
    """)

    # 检查并添加新列（如果表已存在）
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({DATA_TABLE})")
        columns = [column[1] for column in cursor.fetchall()]

        # 添加缺失的列
        if 'model_type' not in columns:
            conn.execute(f"ALTER TABLE {DATA_TABLE} ADD COLUMN model_type TEXT")
        if 'model_category' not in columns:
            conn.execute(f"ALTER TABLE {DATA_TABLE} ADD COLUMN model_category TEXT")
        if 'tags' not in columns:
            conn.execute(f"ALTER TABLE {DATA_TABLE} ADD COLUMN tags TEXT")
        if 'base_model' not in columns:
            conn.execute(f"ALTER TABLE {DATA_TABLE} ADD COLUMN base_model TEXT")
        for missing in [
            'data_source',
            'likes',
            'library_name',
            'pipeline_tag',
            'created_at',
            'last_modified',
            'fetched_at',
            'base_model_from_api',
            'search_keyword',
            'url',  # 新增：模型详情页URL
        ]:
            if missing not in columns:
                conn.execute(f"ALTER TABLE {DATA_TABLE} ADD COLUMN {missing} TEXT")

        conn.commit()
    except Exception as e:
        print(f"更新数据库结构时出错: {e}")

    # 创建平台统计表
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {STATS_TABLE} (
            platform TEXT PRIMARY KEY,
            last_model_count INTEGER,
            last_updated TEXT
        )
    """)

    # 创建自定义模型表（手动添加的模型白名单）
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {CUSTOM_MODELS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            model_id TEXT NOT NULL,
            url TEXT NOT NULL,
            added_at TEXT NOT NULL,
            publisher TEXT,
            model_name TEXT
        )
    """)

    # 检查并添加 custom_models 表的新列
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({CUSTOM_MODELS_TABLE})")
        custom_columns = [column[1] for column in cursor.fetchall()]

        if 'publisher' not in custom_columns:
            conn.execute(f"ALTER TABLE {CUSTOM_MODELS_TABLE} ADD COLUMN publisher TEXT")
        if 'model_name' not in custom_columns:
            conn.execute(f"ALTER TABLE {CUSTOM_MODELS_TABLE} ADD COLUMN model_name TEXT")
        if 'model_category' not in custom_columns:
            conn.execute(f"ALTER TABLE {CUSTOM_MODELS_TABLE} ADD COLUMN model_category TEXT DEFAULT 'ernie-4.5'")
    except Exception as e:
        print(f"更新 custom_models 表结构时出错: {e}")

    # 模型字段覆盖规则表：用于手动修正后续抓取入库字段
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {MODEL_FIELD_OVERRIDES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT NOT NULL,
            publisher TEXT NOT NULL,
            model_name TEXT NOT NULL,
            repo_key TEXT NOT NULL,
            publisher_key TEXT NOT NULL,
            model_name_key TEXT NOT NULL,
            model_category TEXT,
            model_type TEXT,
            base_model TEXT,
            tags TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(repo_key, publisher_key, model_name_key)
        )
    """)

    # 检查并添加 model_field_overrides 表缺失列（兼容已有库）
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({MODEL_FIELD_OVERRIDES_TABLE})")
        override_columns = [column[1] for column in cursor.fetchall()]

        for missing in [
            'repo',
            'publisher',
            'model_name',
            'repo_key',
            'publisher_key',
            'model_name_key',
            'model_category',
            'model_type',
            'base_model',
            'tags',
            'updated_at',
        ]:
            if missing not in override_columns:
                conn.execute(f"ALTER TABLE {MODEL_FIELD_OVERRIDES_TABLE} ADD COLUMN {missing} TEXT")

        # 尝试创建唯一索引（如果已存在会被忽略）
        conn.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{MODEL_FIELD_OVERRIDES_TABLE}_key
            ON {MODEL_FIELD_OVERRIDES_TABLE}(repo_key, publisher_key, model_name_key)
        """)
    except Exception as e:
        print(f"更新 {MODEL_FIELD_OVERRIDES_TABLE} 表结构时出错: {e}")

    conn.commit()
    conn.close()


def get_model_field_overrides(limit=500):
    """
    获取模型字段覆盖规则列表（按更新时间倒序）

    Args:
        limit: 返回条数上限

    Returns:
        DataFrame: 覆盖规则
    """
    try:
        init_database()
        conn = sqlite3.connect(DB_PATH)
        query = f"""
            SELECT id, repo, publisher, model_name, model_category, model_type, base_model, tags, updated_at
            FROM {MODEL_FIELD_OVERRIDES_TABLE}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(int(limit),))
        conn.close()
        return df
    except Exception as e:
        print(f"读取覆盖规则失败: {e}")
        return pd.DataFrame()


def upsert_model_field_override(
    repo,
    publisher,
    model_name,
    model_category=None,
    model_type=None,
    base_model=None,
    tags=None
):
    """
    新增/更新模型字段覆盖规则（后续抓取入库自动应用）

    匹配键：repo + publisher + model_name（大小写无关）
    覆盖字段：model_category / model_type / base_model / tags

    Returns:
        tuple: (success, message)
    """
    try:
        init_database()

        repo_val = str(repo).strip() if repo is not None else ""
        publisher_val = str(publisher).strip() if publisher is not None else ""
        model_name_val = str(model_name).strip() if model_name is not None else ""

        if not repo_val or not publisher_val or not model_name_val:
            return False, "保存覆盖规则失败：平台、发布者、模型名称不能为空"

        payload = {
            "model_category": _clean_override_value(model_category),
            "model_type": _clean_override_value(model_type),
            "base_model": _clean_override_value(base_model),
            "tags": _clean_override_value(tags),
        }

        if not any(value is not None for value in payload.values()):
            return False, "保存覆盖规则失败：至少需要指定一个静态字段（分类/类型/base_model/tags）"

        repo_key = _normalize_override_key(repo_val)
        publisher_key = _normalize_override_key(publisher_val)
        model_name_key = _normalize_override_key(model_name_val)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id
            FROM {MODEL_FIELD_OVERRIDES_TABLE}
            WHERE repo_key = ? AND publisher_key = ? AND model_name_key = ?
        """, (repo_key, publisher_key, model_name_key))
        existing = cursor.fetchone()

        if existing:
            cursor.execute(f"""
                UPDATE {MODEL_FIELD_OVERRIDES_TABLE}
                SET repo = ?, publisher = ?, model_name = ?,
                    model_category = ?, model_type = ?, base_model = ?, tags = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                repo_val, publisher_val, model_name_val,
                payload["model_category"], payload["model_type"], payload["base_model"], payload["tags"],
                now_str,
                existing[0]
            ))
            action = "更新"
        else:
            cursor.execute(f"""
                INSERT INTO {MODEL_FIELD_OVERRIDES_TABLE}
                (
                    repo, publisher, model_name,
                    repo_key, publisher_key, model_name_key,
                    model_category, model_type, base_model, tags, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                repo_val, publisher_val, model_name_val,
                repo_key, publisher_key, model_name_key,
                payload["model_category"], payload["model_type"], payload["base_model"], payload["tags"],
                now_str
            ))
            action = "新增"

        conn.commit()
        conn.close()
        return True, f"已{action}覆盖规则：{repo_val} / {publisher_val} / {model_name_val}"

    except Exception as e:
        return False, f"保存覆盖规则失败: {str(e)}"


def delete_model_field_override(override_id):
    """
    删除模型字段覆盖规则

    Args:
        override_id: 覆盖规则 ID

    Returns:
        tuple: (success, message)
    """
    try:
        init_database()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {MODEL_FIELD_OVERRIDES_TABLE} WHERE id = ?", (override_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        if affected > 0:
            return True, f"已删除覆盖规则 (id={override_id})"
        return False, f"未找到覆盖规则 (id={override_id})"
    except Exception as e:
        return False, f"删除覆盖规则失败: {str(e)}"


def apply_model_field_overrides(df, db_path=DB_PATH):
    """
    对待入库数据应用模型字段覆盖规则

    仅覆盖静态字段：model_category / model_type / base_model / tags
    不覆盖时变字段（如 download_count/date 等）。
    """
    if df is None or df.empty:
        return df

    required_cols = {'repo', 'publisher', 'model_name'}
    if not required_cols.issubset(df.columns):
        return df

    conn = sqlite3.connect(db_path)
    try:
        overrides_df = pd.read_sql_query(f"""
            SELECT
                repo_key, publisher_key, model_name_key,
                model_category, model_type, base_model, tags
            FROM {MODEL_FIELD_OVERRIDES_TABLE}
        """, conn)
    except Exception:
        conn.close()
        return df
    conn.close()

    if overrides_df.empty:
        return df

    result_df = df.copy()
    result_df['_ov_repo_key'] = result_df['repo'].apply(_normalize_override_key)
    result_df['_ov_publisher_key'] = result_df['publisher'].apply(_normalize_override_key)
    result_df['_ov_model_name_key'] = result_df['model_name'].apply(_normalize_override_key)

    overrides_for_merge = overrides_df.rename(columns={
        'repo_key': '_ov_repo_key',
        'publisher_key': '_ov_publisher_key',
        'model_name_key': '_ov_model_name_key',
        'model_category': '_ov_model_category',
        'model_type': '_ov_model_type',
        'base_model': '_ov_base_model',
        'tags': '_ov_tags',
    })

    merged_df = result_df.merge(
        overrides_for_merge,
        on=['_ov_repo_key', '_ov_publisher_key', '_ov_model_name_key'],
        how='left'
    )

    for field in OVERRIDABLE_FIELDS:
        override_col = f"_ov_{field}"
        if field not in merged_df.columns:
            merged_df[field] = None
        if override_col in merged_df.columns:
            has_override = merged_df[override_col].notna() & merged_df[override_col].astype(str).str.strip().ne('')
            merged_df.loc[has_override, field] = merged_df.loc[has_override, override_col]

    drop_cols = [
        '_ov_repo_key', '_ov_publisher_key', '_ov_model_name_key',
        '_ov_model_category', '_ov_model_type', '_ov_base_model', '_ov_tags'
    ]
    merged_df = merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns])

    return merged_df


def save_to_db(df, db_path=DB_PATH):
    """
    保存数据到数据库（保留所有原始记录，不做去重）

    策略：
    - 所有数据直接入库，保持完整性
    - 入库前自动应用模型字段覆盖规则（model_category / model_type / base_model / tags）
    - 去重和取最大值在查询时动态处理（load_data_from_db）
    - 这样既保留了历史数据，又避免复杂的合并逻辑

    Args:
        df: 要保存的 DataFrame
        db_path: 数据库路径
    """
    if df is None or len(df) == 0:
        print("没有可保存的数据")
        return

    init_database()

    df_to_save = apply_model_field_overrides(df, db_path=db_path)
    df_to_save = _ensure_huggingface_urls(df_to_save)
    conn = sqlite3.connect(db_path)

    # 直接插入所有数据，不做去重
    df_to_save.to_sql(DATA_TABLE, conn, if_exists="append", index=False)
    print(f"成功保存 {len(df_to_save)} 条记录到数据库（已应用覆盖规则，原始数据未去重）")

    conn.close()


def get_last_model_count(platform):
    """获取平台上次记录的模型数量"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT last_model_count FROM {STATS_TABLE} WHERE platform=?", (platform,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def update_last_model_count(platform, count):
    """更新平台的模型数量记录"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        INSERT INTO {STATS_TABLE} (platform, last_model_count, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(platform) DO UPDATE SET
            last_model_count=excluded.last_model_count,
            last_updated=excluded.last_updated
    """, (platform, count, date.today().isoformat()))
    conn.commit()
    conn.close()


def get_previous_week_model_count(platform, days_ago=7):
    """
    获取平台上周（或指定天数前）的模型数量作为进度参考

    Args:
        platform: 平台名称
        days_ago: 往前推算的天数，默认7天

    Returns:
        int: 该日期的模型数量，如果没有数据则返回 None
    """
    from datetime import timedelta

    init_database()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 计算目标日期
    target_date = (date.today() - timedelta(days=days_ago)).isoformat()

    # 查询该日期该平台的唯一模型数量
    cur.execute(f"""
        SELECT COUNT(DISTINCT model_name)
        FROM {DATA_TABLE}
        WHERE repo=? AND date=?
    """, (platform, target_date))

    row = cur.fetchone()
    conn.close()

    return row[0] if row and row[0] > 0 else None


def load_data_from_db(date_filter=None, platform_filter=None, last_value_per_model=False):
    """
    从数据库中读取数据

    默认行为：按 (date, repo, publisher, model_name) 取同日最大下载量。
    last_value_per_model=True 时：按 (repo, publisher, model_name) 取**指定日期及之前**最后一个有值的记录，
    用于“取最后一个有值的节点”场景，避免仅使用单个时点的抓取结果。

    策略：
    - 先在同一 (date, repo, publisher, model_name) 内按优先级选最佳记录
    - last_value_per_model=True 时，再按 repo/publisher/model_name 选最近日期（<= date_filter）

    Args:
        date_filter: 日期过滤器，格式为 'YYYY-MM-DD'。在 last_value_per_model 模式下作为“截止日期”。
        platform_filter: 平台过滤器列表
        last_value_per_model: 是否按模型取“最后一个有值的节点”

    Returns:
        DataFrame: 查询结果（已去重）
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 优先顺序：
        # 1) 有 base_model 的记录
        # 2) data_source 优先级 both > model_tree > search > other
        # 3) download_count 较大
        # 4) 最新 rowid
        conditions = []
        params = []

        if date_filter:
            if last_value_per_model:
                conditions.append("DATE(date) <= ?")
            else:
                conditions.append("DATE(date) = ?")
            params.append(date_filter)

        if platform_filter and len(platform_filter) > 0:
            platform_placeholders = ','.join(['?' for _ in platform_filter])
            conditions.append(f"repo IN ({platform_placeholders})")
            params.extend(platform_filter)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # 构建基础去重（同日同模型取最优记录）
        # 使用 LOWER() 归一化 publisher 和 model_name，避免大小写不一致导致重复
        base_cte = f"""
            WITH ranked AS (
                SELECT
                    *,
                    rowid AS _rowid_,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, repo, LOWER(publisher), LOWER(model_name)
                        ORDER BY
                            (COALESCE(base_model, base_model_from_api) IS NOT NULL
                             AND TRIM(COALESCE(base_model, base_model_from_api)) != ''
                             AND LOWER(COALESCE(base_model, base_model_from_api)) NOT IN ('none', 'nan')) DESC,
                            CASE data_source
                                WHEN 'both' THEN 3
                                WHEN 'model_tree' THEN 2
                                WHEN 'search' THEN 1
                                ELSE 0
                            END DESC,
                            CAST(download_count AS REAL) DESC,
                            _rowid_ DESC
                    ) AS rn
                FROM {DATA_TABLE}
                {where_clause}
            )
        """

        if last_value_per_model:
            # 先选出每日最佳，再按 repo/publisher/model_name 取最近一条有值的记录（<= date_filter）
            query = base_cte + """
            , best_per_day AS (
                SELECT * FROM ranked WHERE rn = 1
            ),
            latest_per_model AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY repo, LOWER(publisher), LOWER(model_name)
                        ORDER BY DATE(date) DESC, _rowid_ DESC
                    ) AS rn_last
                FROM best_per_day
                WHERE download_count IS NOT NULL
                  AND LOWER(TRIM(download_count)) NOT IN ('', 'none', 'nan')
            )
            SELECT * FROM latest_per_model WHERE rn_last = 1
            """
        else:
            query = base_cte + "SELECT * FROM ranked WHERE rn = 1"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # 在“最后有效值”模式下，使用指定的 date_filter 作为快照日期，避免后续按 date 精确筛选时丢失记录
        if last_value_per_model and date_filter and not df.empty:
            df['date'] = date_filter

        if not df.empty and 'base_model' in df.columns and 'base_model_from_api' in df.columns:
            df['base_model'] = df.apply(
                lambda row: row['base_model_from_api']
                if (pd.isna(row['base_model']) or str(row['base_model']).strip().lower() in ['', 'none', 'nan'])
                else row['base_model'],
                axis=1
            )
        if not df.empty and 'base_model' in df.columns:
            df['base_model'] = df['base_model'].apply(
                lambda v: None if str(v).strip().lower() in ['', 'none', 'nan'] else v
            )

        # 动态补全历史数据中的 Hugging Face 缺失 URL
        df = _ensure_huggingface_urls(df)

        return df

    except Exception as e:
        print(f"读取数据库数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ========== 自定义模型管理 ==========

def parse_model_url(url):
    """
    解析模型URL，提取平台和模型ID

    支持的URL格式：
    - Hugging Face: https://huggingface.co/{publisher}/{model_name}
    - ModelScope: https://modelscope.cn/models/{publisher}/{model_name}
    - AI Studio: https://aistudio.baidu.com/modelsdetail/{model_id} 或
                 https://aistudio.baidu.com/modeldetail/{model_id}
    - GitCode: https://gitcode.com/{publisher}/{model_name}

    Returns:
        tuple: (platform, model_id) 或 (None, None) 如果无法解析
    """
    import re

    url = url.strip()

    # Hugging Face
    if 'huggingface.co/' in url:
        match = re.search(r'huggingface\.co/([^/]+)/([^/]+)', url)
        if match:
            publisher, model_name = match.groups()
            return 'Hugging Face', f"{publisher}/{model_name}"

    # ModelScope
    elif 'modelscope.cn/models/' in url:
        match = re.search(r'modelscope\.cn/models/([^/]+)/([^/]+)', url)
        if match:
            publisher, model_name = match.groups()
            return 'ModelScope', f"{publisher}/{model_name}"

    # AI Studio (支持 modelsdetail 和 modeldetail 两种格式)
    elif 'aistudio.baidu.com/' in url and ('modelsdetail/' in url or 'modeldetail/' in url):
        # AI Studio URL 格式: https://aistudio.baidu.com/modelsdetail/{id}/intro
        # 无法从 URL 直接解析 publisher/model_name，需要在 fetch 时从页面获取
        return 'AI Studio', url

    # GitCode
    elif 'gitcode.com/' in url:
        match = re.search(r'gitcode\.com/([^/]+)/([^/]+)', url)
        if match:
            publisher, model_name = match.groups()
            return 'GitCode', f"{publisher}/{model_name}"

    return None, None


def add_custom_model(url, model_category=None):
    """
    添加自定义模型到跟踪列表

    Args:
        url: 模型URL
        model_category: 模型分类（可选，如不提供则自动推断）

    Returns:
        dict: {'success': bool, 'message': str, 'id': int}
    """
    init_database()
    platform, model_id = parse_model_url(url)

    if not platform:
        return {'success': False, 'message': f'无法解析URL: {url}'}

    # 从 model_id 中解析 publisher 和 model_name（适用于 HF/ModelScope 格式）
    publisher = None
    model_name = None
    if '/' in model_id and platform != 'AI Studio':
        parts = model_id.split('/', 1)
        if len(parts) == 2:
            publisher, model_name = parts

    # 自动推断 model_category（如果未提供）
    if model_category is None:
        # 使用 backfill 脚本的逻辑
        name_lower = str(model_name or model_id).lower()
        name_compact = re.sub(r'[^a-z0-9]+', '', name_lower)
        if (
            'paddleocr-vl' in name_lower
            or 'paddleocrvl' in name_compact
            or ('paddleocr' in name_compact and 'vl' in name_compact)
        ):
            model_category = 'paddleocr-vl'
        elif 'ernie' in name_lower or '文心' in name_lower:
            model_category = 'ernie-4.5'  # 所有 ERNIE 相关都归入 ernie-4.5
        else:
            model_category = 'other'

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 检查是否已存在
    cursor.execute(f"SELECT id FROM {CUSTOM_MODELS_TABLE} WHERE url=?", (url,))
    if cursor.fetchone():
        conn.close()
        return {'success': False, 'message': '该模型已存在'}

    # 插入新记录
    cursor.execute(f"""
        INSERT INTO {CUSTOM_MODELS_TABLE} (platform, model_id, url, added_at, publisher, model_name, model_category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (platform, model_id, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), publisher, model_name, model_category))

    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {'success': True, 'message': '添加成功', 'id': new_id, 'platform': platform, 'model_id': model_id}


def remove_custom_model(model_id):
    """
    从跟踪列表中删除自定义模型

    Args:
        model_id: 数据库中的ID

    Returns:
        bool: 是否删除成功
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM {CUSTOM_MODELS_TABLE} WHERE id=?", (model_id,))
    affected = cursor.rowcount

    conn.commit()
    conn.close()

    return affected > 0


def get_custom_models():
    """
    获取所有自定义模型列表

    Returns:
        list: 字典列表，每个包含 id, platform, model_id, url, added_at, publisher, model_name, model_category
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT id, platform, model_id, url, added_at, publisher, model_name, model_category
        FROM {CUSTOM_MODELS_TABLE}
        ORDER BY added_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            'id': row[0],
            'platform': row[1],
            'model_id': row[2],
            'url': row[3],
            'added_at': row[4],
            'publisher': row[5],
            'model_name': row[6],
            'model_category': row[7] if len(row) > 7 else 'ernie-4.5'
        }
        for row in rows
    ]


def add_custom_model_with_info(url, platform, model_name, publisher, model_category='ernie-4.5'):
    """
    添加自定义模型到跟踪列表（支持手动指定模型信息）

    主要用于 AI Studio 等无法从 URL 自动解析模型信息的平台

    Args:
        url: 模型URL
        platform: 平台名称
        model_name: 模型名称
        publisher: 发布者
        model_category: 模型分类（默认为 'ernie-4.5'）

    Returns:
        dict: {'success': bool, 'message': str, 'id': int}
    """
    init_database()

    # 生成 model_id
    model_id = f"{publisher}/{model_name}"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 检查是否已存在
    cursor.execute(f"SELECT id FROM {CUSTOM_MODELS_TABLE} WHERE url=?", (url,))
    if cursor.fetchone():
        conn.close()
        return {'success': False, 'message': '该模型已存在'}

    # 插入新记录
    cursor.execute(f"""
        INSERT INTO {CUSTOM_MODELS_TABLE} (platform, model_id, url, added_at, publisher, model_name, model_category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (platform, model_id, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), publisher, model_name, model_category))

    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {'success': True, 'message': '添加成功', 'id': new_id, 'platform': platform, 'model_id': model_id}
