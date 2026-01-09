"""数据库操作模块"""
import sqlite3
import pandas as pd
from datetime import date
from .config import DB_PATH, DATA_TABLE, STATS_TABLE


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

    conn.commit()
    conn.close()


def save_to_db(df, db_path=DB_PATH):
    """
    保存数据到数据库（保留所有原始记录，不做去重）

    策略：
    - 所有数据直接入库，保持完整性
    - 去重和取最大值在查询时动态处理（load_data_from_db）
    - 这样既保留了历史数据，又避免复杂的合并逻辑

    Args:
        df: 要保存的 DataFrame
        db_path: 数据库路径
    """
    conn = sqlite3.connect(db_path)

    # 直接插入所有数据，不做去重
    df.to_sql(DATA_TABLE, conn, if_exists="append", index=False)
    print(f"成功保存 {len(df)} 条记录到数据库（原始数据，未去重）")

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
        base_cte = f"""
            WITH ranked AS (
                SELECT
                    *,
                    rowid AS _rowid_,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, repo, publisher, model_name
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
                        PARTITION BY repo, publisher, model_name
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

        return df

    except Exception as e:
        print(f"读取数据库数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
