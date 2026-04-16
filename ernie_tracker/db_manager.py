"""
数据库管理模块 - 提供备份、删除、查看等管理功能
"""
import sqlite3
import pandas as pd
import shutil
import os
from datetime import datetime, date
from .config import DB_PATH, DATA_TABLE, STATS_TABLE


def backup_database(backup_dir="backups"):
    """
    备份数据库

    Args:
        backup_dir: 备份目录

    Returns:
        tuple: (success, backup_path or error_message)
    """
    try:
        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)

        # 生成备份文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"ernie_downloads_backup_{timestamp}.db"
        backup_path = os.path.join(backup_dir, backup_filename)

        # 复制数据库文件
        shutil.copy2(DB_PATH, backup_path)

        return True, backup_path

    except Exception as e:
        return False, str(e)


def restore_database(backup_path):
    """
    从备份恢复数据库

    Args:
        backup_path: 备份文件路径

    Returns:
        tuple: (success, message)
    """
    try:
        if not os.path.exists(backup_path):
            return False, f"备份文件不存在: {backup_path}"

        # 先备份当前数据库
        success, current_backup = backup_database()
        if not success:
            return False, f"无法备份当前数据库: {current_backup}"

        # 恢复备份
        shutil.copy2(backup_path, DB_PATH)

        return True, f"数据库已恢复，当前数据库已备份到: {current_backup}"

    except Exception as e:
        return False, str(e)


def delete_data_by_date(target_date):
    """
    删除指定日期的数据

    Args:
        target_date: 日期字符串，格式为 'YYYY-MM-DD'

    Returns:
        tuple: (success, message, deleted_count)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 先查询要删除的记录数
        cursor.execute(f"SELECT COUNT(*) FROM {DATA_TABLE} WHERE date = ?", (target_date,))
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return True, f"日期 {target_date} 没有数据", 0

        # 删除数据
        cursor.execute(f"DELETE FROM {DATA_TABLE} WHERE date = ?", (target_date,))
        conn.commit()
        conn.close()

        return True, f"成功删除 {count} 条记录", count

    except Exception as e:
        return False, str(e), 0


def delete_data_by_platform(platform, target_date=None):
    """
    删除指定平台的数据

    Args:
        platform: 平台名称
        target_date: 可选的日期过滤

    Returns:
        tuple: (success, message, deleted_count)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 构建查询
        if target_date:
            query = f"SELECT COUNT(*) FROM {DATA_TABLE} WHERE repo = ? AND date = ?"
            params = (platform, target_date)
        else:
            query = f"SELECT COUNT(*) FROM {DATA_TABLE} WHERE repo = ?"
            params = (platform,)

        cursor.execute(query, params)
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return True, f"平台 {platform} 没有数据", 0

        # 删除数据
        if target_date:
            cursor.execute(f"DELETE FROM {DATA_TABLE} WHERE repo = ? AND date = ?", params)
        else:
            cursor.execute(f"DELETE FROM {DATA_TABLE} WHERE repo = ?", params)

        conn.commit()
        conn.close()

        return True, f"成功删除 {count} 条记录", count

    except Exception as e:
        return False, str(e), 0


def get_database_stats():
    """
    获取数据库统计信息

    Returns:
        dict: 统计信息
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 总记录数
        total_records = pd.read_sql(f"SELECT COUNT(*) as count FROM {DATA_TABLE}", conn).iloc[0]['count']

        # 按日期统计
        date_stats = pd.read_sql(
            f"SELECT date, COUNT(*) as count FROM {DATA_TABLE} GROUP BY date ORDER BY date DESC",
            conn
        )

        # 按平台统计
        platform_stats = pd.read_sql(
            f"SELECT repo, COUNT(*) as count FROM {DATA_TABLE} GROUP BY repo ORDER BY count DESC",
            conn
        )

        # 数据库文件大小
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        db_size_mb = db_size / (1024 * 1024)

        # 日期范围
        date_range = pd.read_sql(
            f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {DATA_TABLE}",
            conn
        )

        conn.close()

        return {
            'total_records': int(total_records),
            'date_stats': date_stats,
            'platform_stats': platform_stats,
            'db_size_mb': round(db_size_mb, 2),
            'min_date': date_range.iloc[0]['min_date'],
            'max_date': date_range.iloc[0]['max_date']
        }

    except Exception as e:
        return {
            'error': str(e),
            'total_records': 0,
            'date_stats': pd.DataFrame(),
            'platform_stats': pd.DataFrame(),
            'db_size_mb': 0,
            'min_date': None,
            'max_date': None
        }


def get_available_backups(backup_dir="backups"):
    """
    获取所有可用的备份文件

    Args:
        backup_dir: 备份目录

    Returns:
        list: 备份文件信息列表
    """
    try:
        if not os.path.exists(backup_dir):
            return []

        backups = []
        for filename in os.listdir(backup_dir):
            if filename.endswith('.db') and filename.startswith('ernie_downloads_backup_'):
                filepath = os.path.join(backup_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))

                backups.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size_mb': round(file_size, 2),
                    'created_time': file_time.strftime("%Y-%m-%d %H:%M:%S")
                })

        # 按创建时间倒序排列
        backups.sort(key=lambda x: x['created_time'], reverse=True)

        return backups

    except Exception as e:
        print(f"获取备份列表失败: {e}")
        return []


def delete_backup(backup_path):
    """
    删除备份文件

    Args:
        backup_path: 备份文件路径

    Returns:
        tuple: (success, message)
    """
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            return True, "备份文件已删除"
        else:
            return False, "备份文件不存在"

    except Exception as e:
        return False, str(e)


def vacuum_database():
    """
    清理数据库，回收空间

    Returns:
        tuple: (success, message)
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 记录清理前的大小
        before_size = os.path.getsize(DB_PATH) / (1024 * 1024)

        # 执行 VACUUM
        conn.execute("VACUUM")
        conn.close()

        # 记录清理后的大小
        after_size = os.path.getsize(DB_PATH) / (1024 * 1024)
        saved = before_size - after_size

        return True, f"数据库已优化，节省了 {saved:.2f} MB 空间"

    except Exception as e:
        return False, str(e)


def export_database_to_excel(output_path, date_filter=None):
    """
    导出数据库到 Excel（自动去重，取最大下载量）

    策略：
    - 对于相同的 (date, repo, publisher, model_name)，取 download_count 最大的记录
    - 与 load_data_from_db() 使用相同的去重逻辑

    Args:
        output_path: 输出文件路径
        date_filter: 可选的日期过滤

    Returns:
        tuple: (success, message)
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 🔧 修复：使用与 load_data_from_db() 相同的去重逻辑
        # 按 (date, repo, publisher, model_name) 分组，取最大下载量
        if date_filter:
            query = f"""
                SELECT t1.*
                FROM {DATA_TABLE} t1
                INNER JOIN (
                    SELECT
                        date, repo, publisher, model_name,
                        MAX(CAST(download_count AS REAL)) as max_download
                    FROM {DATA_TABLE}
                    WHERE date = ?
                    GROUP BY date, repo, publisher, model_name
                ) t2
                ON t1.date = t2.date
                   AND t1.repo = t2.repo
                   AND t1.publisher = t2.publisher
                   AND t1.model_name = t2.model_name
                   AND CAST(t1.download_count AS REAL) = t2.max_download
                ORDER BY repo, model_name
            """
            df = pd.read_sql(query, conn, params=(date_filter,))
        else:
            query = f"""
                SELECT t1.*
                FROM {DATA_TABLE} t1
                INNER JOIN (
                    SELECT
                        date, repo, publisher, model_name,
                        MAX(CAST(download_count AS REAL)) as max_download
                    FROM {DATA_TABLE}
                    GROUP BY date, repo, publisher, model_name
                ) t2
                ON t1.date = t2.date
                   AND t1.repo = t2.repo
                   AND t1.publisher = t2.publisher
                   AND t1.model_name = t2.model_name
                   AND CAST(t1.download_count AS REAL) = t2.max_download
                ORDER BY date DESC, repo, model_name
            """
            df = pd.read_sql(query, conn)

        if df.empty:
            conn.close()
            return False, "没有数据可导出"

        # 为导出结果补充“首次被抓取时间”
        # 口径：按 (repo, publisher, model_name) 大小写无关匹配，
        # 优先使用 fetched_at；若为空则回退到 date 00:00:00
        first_fetch_query = f"""
            SELECT
                repo,
                LOWER(TRIM(publisher)) AS publisher_key,
                LOWER(TRIM(model_name)) AS model_name_key,
                MIN(
                    CASE
                        WHEN fetched_at IS NOT NULL AND TRIM(fetched_at) != '' AND LOWER(TRIM(fetched_at)) NOT IN ('none', 'nan')
                        THEN fetched_at
                        ELSE date || ' 00:00:00'
                    END
                ) AS first_fetched_at
            FROM {DATA_TABLE}
            GROUP BY repo, LOWER(TRIM(publisher)), LOWER(TRIM(model_name))
        """
        first_fetch_df = pd.read_sql(first_fetch_query, conn)
        conn.close()

        df['_publisher_key'] = df['publisher'].astype(str).str.strip().str.lower()
        df['_model_name_key'] = df['model_name'].astype(str).str.strip().str.lower()
        df = df.merge(
            first_fetch_df,
            left_on=['repo', '_publisher_key', '_model_name_key'],
            right_on=['repo', 'publisher_key', 'model_name_key'],
            how='left'
        )
        df = df.drop(columns=['_publisher_key', '_model_name_key', 'publisher_key', 'model_name_key'], errors='ignore')

        # 重新排列列顺序，把重要的model tree信息放在前面
        # 确保这些列存在
        base_columns = ['date', 'repo', 'model_name', 'publisher', 'download_count']

        if 'first_fetched_at' in df.columns:
            base_columns.append('first_fetched_at')

        # Model Tree 相关列
        model_tree_columns = []
        if 'base_model' in df.columns:
            model_tree_columns.append('base_model')
        if 'model_type' in df.columns:
            model_tree_columns.append('model_type')
        if 'model_category' in df.columns:
            model_tree_columns.append('model_category')

        # 其他列
        other_columns = [col for col in df.columns if col not in base_columns + model_tree_columns]

        # 重新排列列顺序：基础列 -> Model Tree列 -> 其他列
        ordered_columns = base_columns + model_tree_columns + other_columns
        ordered_columns = [col for col in ordered_columns if col in df.columns]

        df = df[ordered_columns]

        # 添加一个更直观的"来源"列（中文描述），方便识别
        if 'data_source' in df.columns:
            # 数据库已有 data_source 字段，添加中文描述列
            def get_source_cn(row):
                source = row.get('data_source')
                if pd.notna(source):
                    if source == 'search':
                        return '搜索发现'
                    elif source == 'model_tree':
                        return f'Model Tree (衍生自 {row.get("base_model", "未知")})'
                    elif source == 'both':
                        return f'搜索+Model Tree (衍生自 {row.get("base_model", "未知")})'
                    else:
                        return source
                else:
                    # 历史数据兼容：根据 base_model 推断
                    if pd.notna(row.get('base_model')) and row.get('base_model'):
                        return f'Model Tree (衍生自 {row["base_model"]})'
                    else:
                        return '直接搜索 (历史数据)'

            df['data_source_cn'] = df.apply(get_source_cn, axis=1)
            # 把中文列放在 data_source 列后面
            if 'data_source' in df.columns:
                source_col_idx = df.columns.get_loc('data_source')
                cols = df.columns.tolist()
                cols.insert(source_col_idx + 1, cols.pop(cols.index('data_source_cn')))
                df = df[cols]
        elif 'base_model' in df.columns:
            # 兼容老版本：没有 data_source 字段，根据 base_model 创建
            def get_source_legacy(row):
                if pd.notna(row.get('base_model')) and row.get('base_model'):
                    return f"Model Tree (衍生自 {row['base_model']})"
                else:
                    return "直接搜索"

            df.insert(5, 'data_source_cn', df.apply(get_source_legacy, axis=1))

        # 翻译 model_type 和 model_category 为更易读的中文
        if 'model_type' in df.columns:
            type_mapping = {
                'original': '原始模型',
                'finetune': 'Finetune微调',
                'adapter': 'Adapter适配器',
                'lora': 'LoRA',
                'other': '其他'
            }
            df['model_type_cn'] = df['model_type'].map(type_mapping).fillna(df['model_type'])
            # 把中文列放在英文列后面
            type_col_idx = df.columns.get_loc('model_type')
            cols = df.columns.tolist()
            cols.insert(type_col_idx + 1, cols.pop(cols.index('model_type_cn')))
            df = df[cols]

        if 'model_category' in df.columns:
            category_mapping = {
                'ernie-4.5': 'ERNIE-4.5系列',
                'paddleocr-vl': 'PaddleOCR-VL系列',
                'other-ernie': '其他ERNIE系列',
                'other': '其他'
            }
            df['model_category_cn'] = df['model_category'].map(category_mapping).fillna(df['model_category'])
            # 把中文列放在英文列后面
            category_col_idx = df.columns.get_loc('model_category')
            cols = df.columns.tolist()
            cols.insert(category_col_idx + 1, cols.pop(cols.index('model_category_cn')))
            df = df[cols]

        # 导出到 Excel
        df.to_excel(output_path, index=False, engine='openpyxl')

        return True, f"成功导出 {len(df)} 条记录到 {output_path}"

    except Exception as e:
        return False, str(e)


def get_duplicate_records():
    """
    查找重复记录

    Returns:
        DataFrame: 重复记录
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 查找重复的记录（相同的日期、平台、发布者、模型名称）
        query = f"""
        SELECT date, repo, publisher, model_name, COUNT(*) as count
        FROM {DATA_TABLE}
        GROUP BY date, repo, publisher, model_name
        HAVING count > 1
        ORDER BY count DESC, date DESC
        """

        duplicates = pd.read_sql(query, conn)
        conn.close()

        return duplicates

    except Exception as e:
        print(f"查找重复记录失败: {e}")
        return pd.DataFrame()


def remove_duplicate_records():
    """
    删除重复记录，保留最新的一条

    Returns:
        tuple: (success, message, deleted_count)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 删除重复记录，保留 rowid 最大的（最新的）
        cursor.execute(f"""
        DELETE FROM {DATA_TABLE}
        WHERE rowid NOT IN (
            SELECT MAX(rowid)
            FROM {DATA_TABLE}
            GROUP BY date, repo, publisher, model_name
        )
        """)

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return True, f"成功删除 {deleted_count} 条重复记录", deleted_count

    except Exception as e:
        return False, str(e), 0


def insert_single_record(date, repo, model_name, publisher, download_count,
                        base_model=None, model_type=None, model_category=None):
    """
    插入单条记录到数据库

    Args:
        date: 日期 (YYYY-MM-DD)
        repo: 平台名称
        model_name: 模型名称
        publisher: 发布者
        download_count: 下载量
        base_model: 基础模型（可选，用于衍生模型）
        model_type: 模型类型（可选：original, finetune, adapter, lora, other）
        model_category: 模型分类（可选：ernie-4.5, paddleocr-vl, other-ernie, other）

    Returns:
        tuple: (success, message)
    """
    try:
        # 验证必填字段
        if not all([date, repo, model_name, publisher]):
            return False, "日期、平台、模型名称、发布者不能为空"

        # 验证下载量是否为数字
        try:
            download_count = int(download_count)
            if download_count < 0:
                return False, "下载量不能为负数"
        except (ValueError, TypeError):
            return False, "下载量必须是数字"

        # 验证日期格式
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return False, "日期格式错误，应为 YYYY-MM-DD"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 检查是否已存在相同记录
        cursor.execute(f"""
            SELECT COUNT(*) FROM {DATA_TABLE}
            WHERE date = ? AND repo = ? AND publisher = ? AND model_name = ?
        """, (date, repo, publisher, model_name))

        if cursor.fetchone()[0] > 0:
            conn.close()
            return False, f"该记录已存在（日期: {date}, 平台: {repo}, 模型: {model_name}, 发布者: {publisher}）"

        # 插入记录
        cursor.execute(f"""
            INSERT INTO {DATA_TABLE}
            (date, repo, model_name, publisher, download_count, base_model, model_type, model_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, repo, model_name, publisher, download_count, base_model, model_type, model_category))

        conn.commit()
        conn.close()

        return True, f"成功插入记录：{model_name} ({publisher}) - {download_count:,} 次下载"

    except Exception as e:
        return False, f"插入失败: {str(e)}"


def import_from_excel(file_path, skip_duplicates=True):
    """
    从 Excel 文件批量导入数据

    Args:
        file_path: Excel 文件路径或文件对象
        skip_duplicates: 是否跳过重复记录（True）或覆盖（False）

    Returns:
        tuple: (success, message, stats_dict)
        stats_dict 包含: total, inserted, skipped, errors
    """
    try:
        # 读取 Excel 文件
        if isinstance(file_path, str):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            # 支持 BytesIO 对象（用于 Streamlit 上传）
            df = pd.read_excel(file_path, engine='openpyxl')

        if df.empty:
            return False, "Excel 文件为空", {'total': 0, 'inserted': 0, 'skipped': 0, 'errors': 0}

        # 验证必需的列
        required_columns = ['date', 'repo', 'model_name', 'publisher', 'download_count']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Excel 文件缺少必需的列: {', '.join(missing_columns)}", \
                   {'total': len(df), 'inserted': 0, 'skipped': 0, 'errors': 0}

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        stats = {
            'total': len(df),
            'inserted': 0,
            'skipped': 0,
            'errors': 0
        }

        error_details = []

        for idx, row in df.iterrows():
            try:
                # 提取数据
                date = str(row['date']) if pd.notna(row['date']) else None
                repo = str(row['repo']) if pd.notna(row['repo']) else None
                model_name = str(row['model_name']) if pd.notna(row['model_name']) else None
                publisher = str(row['publisher']) if pd.notna(row['publisher']) else None
                download_count = int(row['download_count']) if pd.notna(row['download_count']) else 0

                # 可选字段
                base_model = str(row['base_model']) if 'base_model' in row and pd.notna(row['base_model']) else None
                model_type = str(row['model_type']) if 'model_type' in row and pd.notna(row['model_type']) else None
                model_category = str(row['model_category']) if 'model_category' in row and pd.notna(row['model_category']) else None

                # 验证必填字段
                if not all([date, repo, model_name, publisher]):
                    stats['errors'] += 1
                    error_details.append(f"第 {idx + 2} 行: 必填字段不能为空")
                    continue

                # 格式化日期
                if isinstance(date, str):
                    # 尝试解析日期
                    try:
                        # 如果是时间戳格式
                        if date.isdigit():
                            date = datetime.fromtimestamp(int(date) / 1000).strftime('%Y-%m-%d')
                        else:
                            # 尝试解析常见日期格式
                            parsed_date = pd.to_datetime(date)
                            date = parsed_date.strftime('%Y-%m-%d')
                    except:
                        pass

                # 检查是否已存在
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {DATA_TABLE}
                    WHERE date = ? AND repo = ? AND publisher = ? AND model_name = ?
                """, (date, repo, publisher, model_name))

                exists = cursor.fetchone()[0] > 0

                if exists:
                    if skip_duplicates:
                        stats['skipped'] += 1
                        continue
                    else:
                        # 覆盖模式：先删除旧记录
                        cursor.execute(f"""
                            DELETE FROM {DATA_TABLE}
                            WHERE date = ? AND repo = ? AND publisher = ? AND model_name = ?
                        """, (date, repo, publisher, model_name))

                # 插入记录
                cursor.execute(f"""
                    INSERT INTO {DATA_TABLE}
                    (date, repo, model_name, publisher, download_count, base_model, model_type, model_category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, repo, model_name, publisher, download_count, base_model, model_type, model_category))

                stats['inserted'] += 1

            except Exception as e:
                stats['errors'] += 1
                error_details.append(f"第 {idx + 2} 行: {str(e)}")

        conn.commit()
        conn.close()

        # 构建结果消息
        message = f"导入完成！\n"
        message += f"- 总记录数: {stats['total']}\n"
        message += f"- 成功插入: {stats['inserted']}\n"
        message += f"- 跳过重复: {stats['skipped']}\n"
        message += f"- 错误记录: {stats['errors']}"

        if error_details and len(error_details) <= 10:
            message += f"\n\n错误详情:\n" + "\n".join(error_details)
        elif error_details:
            message += f"\n\n错误详情（前10条）:\n" + "\n".join(error_details[:10])
            message += f"\n... 还有 {len(error_details) - 10} 条错误"

        success = stats['inserted'] > 0 or (stats['skipped'] > 0 and stats['errors'] == 0)

        return success, message, stats

    except Exception as e:
        return False, f"导入失败: {str(e)}", {'total': 0, 'inserted': 0, 'skipped': 0, 'errors': 0}


def search_records(date_filter=None, repo_filter=None, model_name_filter=None,
                   publisher_filter=None, limit=100):
    """
    搜索数据库记录

    Args:
        date_filter: 日期过滤（YYYY-MM-DD）
        repo_filter: 平台过滤
        model_name_filter: 模型名称过滤（支持模糊搜索）
        publisher_filter: 发布者过滤（支持模糊搜索）
        limit: 返回记录数上限

    Returns:
        DataFrame: 搜索结果，包含 rowid
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        # 构建查询
        query = f"SELECT rowid, * FROM {DATA_TABLE}"
        conditions = []
        params = []

        if date_filter:
            conditions.append("date = ?")
            params.append(date_filter)

        if repo_filter:
            conditions.append("repo = ?")
            params.append(repo_filter)

        if model_name_filter:
            conditions.append("model_name LIKE ?")
            params.append(f"%{model_name_filter}%")

        if publisher_filter:
            conditions.append("publisher LIKE ?")
            params.append(f"%{publisher_filter}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY date DESC, repo, model_name LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    except Exception as e:
        print(f"搜索记录失败: {e}")
        return pd.DataFrame()


def get_record_by_rowid(rowid):
    """
    根据 rowid 获取单条记录

    Args:
        rowid: 记录的 rowid

    Returns:
        dict: 记录数据，如果未找到返回 None
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"SELECT rowid, * FROM {DATA_TABLE} WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()

        if row:
            # 动态获取列名
            cursor.execute(f"PRAGMA table_info({DATA_TABLE})")
            table_columns = [col[1] for col in cursor.fetchall()]

            # 添加 rowid 到列名列表的开头
            columns = ['rowid'] + table_columns

            conn.close()
            return dict(zip(columns, row))

        conn.close()
        return None

    except Exception as e:
        print(f"获取记录失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_record(rowid, date=None, repo=None, model_name=None, publisher=None,
                 download_count=None, base_model=None, model_type=None,
                 model_category=None, tags=None):
    """
    更新数据库记录

    Args:
        rowid: 要更新的记录的 rowid
        date: 日期（可选）
        repo: 平台（可选）
        model_name: 模型名称（可选）
        publisher: 发布者（可选）
        download_count: 下载量（可选）
        base_model: 基础模型（可选）
        model_type: 模型类型（可选）
        model_category: 模型分类（可选）
        tags: 标签（可选）

    Returns:
        tuple: (success, message)
    """
    try:
        # 获取现有记录
        existing = get_record_by_rowid(rowid)
        if not existing:
            return False, f"未找到 rowid={rowid} 的记录"

        # 构建更新字段
        updates = []
        params = []

        if date is not None:
            # 验证日期格式
            try:
                datetime.strptime(date, '%Y-%m-%d')
                updates.append("date = ?")
                params.append(date)
            except ValueError:
                return False, "日期格式错误，应为 YYYY-MM-DD"

        if repo is not None:
            updates.append("repo = ?")
            params.append(repo)

        if model_name is not None:
            updates.append("model_name = ?")
            params.append(model_name)

        if publisher is not None:
            updates.append("publisher = ?")
            params.append(publisher)

        if download_count is not None:
            # 验证下载量
            try:
                download_count = int(download_count)
                if download_count < 0:
                    return False, "下载量不能为负数"
                updates.append("download_count = ?")
                params.append(download_count)
            except (ValueError, TypeError):
                return False, "下载量必须是数字"

        if base_model is not None:
            updates.append("base_model = ?")
            params.append(base_model if base_model else None)

        if model_type is not None:
            updates.append("model_type = ?")
            params.append(model_type if model_type else None)

        if model_category is not None:
            updates.append("model_category = ?")
            params.append(model_category if model_category else None)

        if tags is not None:
            updates.append("tags = ?")
            params.append(tags if tags else None)

        if not updates:
            return False, "没有需要更新的字段"

        # 执行更新
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = f"UPDATE {DATA_TABLE} SET {', '.join(updates)} WHERE rowid = ?"
        params.append(rowid)

        cursor.execute(query, params)
        conn.commit()
        conn.close()

        return True, f"成功更新记录 (rowid={rowid})"

    except Exception as e:
        return False, f"更新失败: {str(e)}"


def delete_record_by_rowid(rowid):
    """
    根据 rowid 删除记录

    Args:
        rowid: 要删除的记录的 rowid

    Returns:
        tuple: (success, message)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 检查记录是否存在
        cursor.execute(f"SELECT COUNT(*) FROM {DATA_TABLE} WHERE rowid = ?", (rowid,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return False, f"未找到 rowid={rowid} 的记录"

        # 删除记录
        cursor.execute(f"DELETE FROM {DATA_TABLE} WHERE rowid = ?", (rowid,))
        conn.commit()
        conn.close()

        return True, f"成功删除记录 (rowid={rowid})"

    except Exception as e:
        return False, f"删除失败: {str(e)}"
