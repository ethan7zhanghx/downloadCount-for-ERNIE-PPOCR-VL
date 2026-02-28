"""
数据分析模块 - 周报统计分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from .db import load_data_from_db
from .config import DB_PATH


# 百度官方模型识别规则
OFFICIAL_RULES = {
    'Hugging Face': 'baidu',
    'AI Studio': 'PaddlePaddle',
    'ModelScope': '飞桨PaddlePaddle',
    'GitCode': '飞桨PaddlePaddle',
    '鲸智': 'PaddlePaddle',
    '魔乐 Modelers': 'PaddlePaddle',
    'Gitee': 'PaddlePaddle'
}

# 模型顺序（按重要性排列）
MODEL_ORDER = [
    'ERNIE-4.5-VL-424B-A47B-Paddle',
    'ERNIE-4.5-VL-424B-A47B-PT',
    'ERNIE-4.5-VL-424B-A47B-Base-Paddle',
    'ERNIE-4.5-VL-424B-A47B-Base-PT',
    'ERNIE-4.5-300B-A47B-Paddle',
    'ERNIE-4.5-300B-A47B-PT',
    'ERNIE-4.5-300B-A47B-Base-Paddle',
    'ERNIE-4.5-300B-A47B-Base-PT',
    'ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle',
    'ERNIE-4.5-300B-A47B-FP8-Paddle',
    'ERNIE-4.5-300B-A47B-2Bits-Paddle',
    'ERNIE-4.5-300B-A47B-2Bits-TP2-Paddle',
    'ERNIE-4.5-300B-A47B-2Bits-TP4-Paddle',
    'ERNIE-4.5-VL-28B-A3B-Paddle',
    'ERNIE-4.5-VL-28B-A3B-PT',
    'ERNIE-4.5-VL-28B-A3B-Thinking',
    'ERNIE-4.5-VL-28B-A3B-Base-Paddle',
    'ERNIE-4.5-VL-28B-A3B-Base-PT',
    'ERNIE-4.5-21B-A3B-Paddle',
    'ERNIE-4.5-21B-A3B-PT',
    'ERNIE-4.5-21B-A3B-Thinking',
    'ERNIE-4.5-21B-A3B-Base-Paddle',
    'ERNIE-4.5-21B-A3B-Base-PT',
    'ERNIE-4.5-0.3B-Paddle',
    'ERNIE-4.5-0.3B-PT',
    'ERNIE-4.5-0.3B-Base-Paddle',
    'ERNIE-4.5-0.3B-Base-PT'
]

PADDLEOCR_VL_MODEL_ORDER = [
    'PaddleOCR-VL'
]

# 平台顺序
REPO_ORDER = ['Hugging Face', 'AI Studio', 'ModelScope', 'GitCode', '其他']

# 详细平台顺序（不合并"其他"）
REPO_ORDER_DETAILED = ['Hugging Face', 'AI Studio', 'ModelScope', 'GitCode', '魔乐 Modelers', '鲸智', 'Gitee']


def get_last_friday(current_date=None):
    """
    获取上周五的日期

    Args:
        current_date: 当前日期，默认为今天

    Returns:
        str: 上周五的日期字符串 (YYYY-MM-DD)
    """
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d')

    # 获取当前是星期几 (0=Monday, 6=Sunday)
    current_weekday = current_date.weekday()

    # 计算到上周五的天数
    # 如果今天是周一(0)，上周五是3天前
    # 如果今天是周五(4)，上周五是7天前
    if current_weekday >= 4:  # 周五、周六、周日
        days_to_last_friday = current_weekday - 4 + 7
    else:  # 周一到周四
        days_to_last_friday = current_weekday + 3

    last_friday = current_date - timedelta(days=days_to_last_friday)
    return last_friday.strftime('%Y-%m-%d')


def get_available_dates():
    """
    获取数据库中所有可用的日期

    Returns:
        list: 日期列表
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT date FROM model_downloads ORDER BY date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['date'].tolist()


def normalize_model_names(data):
    """
    标准化模型名称：移除 model_name 中的 publisher 前缀

    例如：'paddlepaddle/ERNIE-4.5-0.3B-PT' -> 'ERNIE-4.5-0.3B-PT'

    这确保了即使数据库中存储的模型名称格式不一致，
    在分析时也能正确匹配和比较。
    """
    data = data.copy()

    def remove_publisher_prefix(row):
        model_name = str(row['model_name']).strip()
        publisher = str(row['publisher']).strip()

        # 如果模型名称以 "publisher/" 开头（忽略大小写），移除前缀
        if publisher and publisher.lower() != 'nan' and '/' in model_name:
            parts = model_name.split('/', 1)
            if len(parts) == 2 and parts[0].lower() == publisher.lower():
                return parts[1]

        return model_name

    data['model_name'] = data.apply(remove_publisher_prefix, axis=1)
    return data


def mark_official_models(data):
    """
    标记官方模型。
    如果 publisher 包含 '百度', 'baidu', 或 'Paddle' (不区分大小写)，则视为官方模型。
    """
    data = data.copy()
    # 确保 publisher 列是字符串类型，以便进行文本操作
    data['publisher'] = data['publisher'].astype(str)

    keywords = ['百度', 'baidu', 'Paddle', 'yiyan', '一言']
    # 创建一个正则表达式，用 | (OR) 连接关键字
    pattern = '|'.join(keywords)

    # 使用 str.contains 进行不区分大小写的匹配
    data['is_official'] = data['publisher'].str.contains(pattern, case=False, na=False)

    return data


def create_pivot_table(data, repo_order=None, model_order=None, group_by_publisher=False, merge_other=True):
    """
    创建数据透视表

    Args:
        data: DataFrame
        repo_order: 平台顺序列表
        model_order: 模型顺序列表. 如果为 None, 则不按特定模型顺序处理.
        group_by_publisher: 是否按 publisher 分组（用于衍生模型）。
                           如果为 True，索引为 (model_name, publisher)；
                           如果为 False，索引仅为 model_name（用于官方模型）
        merge_other: 是否合并魔乐 Modelers、鲸智、Gitee为"其他"（默认True）

    Returns:
        DataFrame: 透视表
    """
    if repo_order is None:
        repo_order = REPO_ORDER if merge_other else REPO_ORDER_DETAILED

    # 确保 download_count 是数值类型
    data = data.copy()
    data['download_count'] = pd.to_numeric(data['download_count'], errors='coerce').fillna(0)

    # 合并平台（仅当 merge_other=True 时）
    if merge_other:
        data['repo'] = data['repo'].replace(['魔乐 Modelers', '鲸智', 'Gitee'], '其他')

    # 根据 group_by_publisher 决定索引
    if group_by_publisher:
        # 衍生模型：使用 (model_name, publisher) 作为索引
        pivot_df = pd.pivot_table(
            data,
            values='download_count',
            index=['model_name', 'publisher'],
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

    # 确保所有平台都在列中
    for repo in repo_order:
        if repo not in pivot_df.columns:
            pivot_df[repo] = 0

    # 按指定顺序排列列
    available_repos = [repo for repo in repo_order if repo in pivot_df.columns]
    pivot_df = pivot_df[available_repos]

    # 如果提供了 model_order，则按其处理
    if model_order:
        # 确保所有指定模型都在索引中
        for model in model_order:
            if model not in pivot_df.index:
                pivot_df.loc[model] = [0] * len(available_repos)

        # 按指定顺序排列行
        available_models = [model for model in model_order if model in pivot_df.index]
        pivot_df = pivot_df.reindex(available_models)

    return pivot_df


def get_all_new_models(current_date, previous_date, model_series='ERNIE-4.5'):
    """
    获取本周新增的所有模型（完整列表）

    🔧 修复：直接从数据库加载数据，确保与 get_weekly_new_finetune_adapters() 使用相同的筛选逻辑

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        previous_date: 对比日期 (YYYY-MM-DD)
        model_series: 模型系列 ('ERNIE-4.5' 或 'PaddleOCR-VL')

    Returns:
        dict: 包含新增模型信息的字典
    """
    try:
        # 从数据库加载数据（与 get_weekly_new_finetune_adapters 使用相同的加载方式）
        current_data = load_data_from_db(date_filter=current_date)
        previous_data = load_data_from_db(date_filter=previous_date)

        if current_data.empty:
            return {
                'new_models_list': [],
                'total_new': 0,
                'summary': f'本周没有新增{model_series}模型'
            }

        # 🔧 修复：使用与 get_weekly_new_finetune_adapters() 相同的筛选逻辑
        # 根据 model_series 确定要筛选的 model_category
        if model_series == 'ERNIE-4.5':
            target_category = 'ernie-4.5'
        elif model_series == 'PaddleOCR-VL':
            target_category = 'paddleocr-vl'
        else:
            target_category = 'ernie-4.5'

        # 🔴 关键修复：先按 model_category 筛选模型系列，再判断新增
        # 新增判断只看 (repo, publisher, model_name) 三元组，不受 model_category 缺失影响
        # 筛选策略：model_category 正确 OR model_name 包含关键词
        if model_series == 'ERNIE-4.5':
            name_pattern = 'ERNIE-4.5'
        else:  # PaddleOCR-VL
            name_pattern = 'PaddleOCR-VL'

        # 使用 model_category OR model_name 筛选，确保不遗漏因 model_category 缺失的模型
        hf_current = current_data[
            (current_data['repo'] == 'Hugging Face') & (
                (current_data['model_category'] == target_category) |
                (current_data['model_name'].str.contains(name_pattern, case=False, na=False))
            )
        ].copy()

        if previous_data.empty:
            hf_previous = pd.DataFrame()
        else:
            # 🔴 关键修复：previous_date 使用相同的筛选逻辑
            # 这样即使之前的数据 model_category 为空，也能通过 model_name 匹配识别已存在的模型
            hf_previous = previous_data[
                (previous_data['repo'] == 'Hugging Face') & (
                    (previous_data['model_category'] == target_category) |
                    (previous_data['model_name'].str.contains(name_pattern, case=False, na=False))
                )
            ].copy()

        # 找出在当前数据中但不在对比数据中的模型（按 publisher+model_name 去重）
        if hf_previous.empty:
            new_models = hf_current.copy()
        else:
            previous_keys = set(zip(hf_previous['publisher'], hf_previous['model_name']))
            current_keys = set(zip(hf_current['publisher'], hf_current['model_name']))
            new_keys = current_keys - previous_keys
            new_models = hf_current[hf_current.apply(lambda r: (r['publisher'], r['model_name']) in new_keys, axis=1)].copy()

        if new_models.empty:
            return {
                'new_models_list': [],
                'total_new': 0,
                'summary': f'本周没有新增{model_series}模型'
            }

        # 按平台分组，每个模型只保留一条记录（选择下载量最大的平台）
        new_models_dedup = new_models.sort_values('download_count', ascending=False).drop_duplicates(
            subset=['publisher', 'model_name'], keep='first'
        )

        # 格式化模型列表，包含更多信息
        models_list = []
        for _, row in new_models_dedup.iterrows():
            model_info = {
                'model_name': row['model_name'],
                'publisher': row['publisher'],
                'repo': row['repo'],
                'download_count': int(row['download_count']),
            }

            # 添加可选字段
            if 'model_type' in row and pd.notna(row['model_type']):
                model_info['model_type'] = row['model_type']
            if 'model_category' in row and pd.notna(row['model_category']):
                model_info['model_category'] = row['model_category']
            if 'base_model' in row and pd.notna(row['base_model']):
                model_info['base_model'] = row['base_model']

            models_list.append(model_info)

        # 按下载量降序排序
        models_list = sorted(models_list, key=lambda x: x['download_count'], reverse=True)

        return {
            'new_models_list': models_list,
            'total_new': len(models_list),
            'summary': f'本周共发现 {len(models_list)} 个新增{model_series}模型'
        }

    except Exception as e:
        print(f"获取本周新增模型完整列表失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'new_models_list': [],
            'total_new': 0,
            'summary': f'获取数据时出错: {e}'
        }


def calculate_weekly_report(current_date=None, previous_date=None, model_order=None, model_series='ERNIE-4.5'):
    """
    计算周报数据

    Args:
        current_date: 当前日期 (YYYY-MM-DD)，默认为今天
        previous_date: 对比日期 (YYYY-MM-DD)，默认为上周五
        model_order: 模型顺序列表
        model_series: 模型系列 ('ERNIE-4.5' 或 'PaddleOCR-VL')

    Returns:
        dict: 包含各种统计数据的字典
    """
    if model_order is None:
        model_order = MODEL_ORDER if model_series == 'ERNIE-4.5' else PADDLEOCR_VL_MODEL_ORDER

    # 设置日期
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
    if previous_date is None:
        previous_date = get_last_friday(current_date)

    # 🔧 修复：使用 load_data_from_db() 获取去重后的数据
    # 这确保了重复记录只取最大下载量，避免重复计算
    # 官方/非官方的当日统计都应使用当天记录，不做“取最近有值”回填
    current_data = load_data_from_db(date_filter=current_date, last_value_per_model=False)
    previous_data = load_data_from_db(date_filter=previous_date, last_value_per_model=False)

    # 负增长检测使用真实的当日记录（不带 last_value_per_model），单独加载
    warn_current_raw = load_data_from_db(date_filter=current_date, last_value_per_model=False)
    warn_previous_raw = load_data_from_db(date_filter=previous_date, last_value_per_model=False)

    # 🔴 关键修复：在合并和进一步处理之前，对数据进行强制标准化和二次去重
    # 确保即使数据库中存在不一致，也能在分析时得到修正
    def enforce_deduplication_and_standardization(df):
        if df.empty:
            return df

        # 1. 标准化模型名称（移除 publisher 前缀）
        df = normalize_model_names(df)

        # 2. 创建小写 key 用于去重，但保留原始大小写
        df['_dedup_publisher'] = df['publisher'].astype(str).str.lower()
        df['_dedup_model_name'] = df['model_name'].astype(str).str.lower()

        # 3. 再次去重，确保同一 (date, repo, publisher, model_name) 只有一条记录，且下载量最大
        df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
        df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
            subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
        )
        return df

    current_data = enforce_deduplication_and_standardization(current_data)
    previous_data = enforce_deduplication_and_standardization(previous_data)
    warn_current_raw = enforce_deduplication_and_standardization(warn_current_raw)
    warn_previous_raw = enforce_deduplication_and_standardization(warn_previous_raw)

    # 合并两个日期的数据
    data = pd.concat([current_data, previous_data], ignore_index=True)

    if data.empty:
        return None

    # 🔴 重要：标准化模型名称（移除 publisher 前缀）
    # 这一步在 enforce_deduplication_and_standardization 中已经完成，此处可以移除或保留作为冗余检查
    # 为了避免重复处理，此处不再重复调用 normalize_model_names
    # data = normalize_model_names(data)

    def filter_by_series(df):
        """按系列过滤数据，用于官方与衍生共用的筛选逻辑。"""
        if df.empty:
            return df
        if model_series == 'ERNIE-4.5':
            if 'model_category' in df.columns:
                condition = (
                    (df['model_category'] == 'ernie-4.5') |
                    (df['model_name'].str.contains('ERNIE-4.5', case=False, na=False))
                )
                return df[condition].copy()
            return df[df['model_name'].str.contains('ERNIE-4.5', case=False, na=False)].copy()
        if model_series == 'PaddleOCR-VL':
            if 'model_category' in df.columns:
                condition = (
                    (df['model_category'] == 'paddleocr-vl') |
                    (df['model_name'].str.contains('PaddleOCR-VL', case=False, na=False))
                )
                return df[condition].copy()
            return df[df['model_name'].str.contains('PaddleOCR-VL', case=False, na=False)].copy()
        return df

    # 🔧 修复：根据 model_series 使用 model_category 字段 **或** 模型名称筛选
    # 这样既能包含正确分类的衍生模型，也能包含其他平台的官方模型
    data = filter_by_series(data)

    if data.empty:
        print(f"警告: 在选定日期内未找到 {model_series} 系列的模型数据。")
        return None

    # 确保 'download_count' 是数值类型
    data['download_count'] = pd.to_numeric(data['download_count'], errors='coerce').fillna(0)

    # 标记官方模型
    data = mark_official_models(data)
    # 负增长检测用的原始当日数据也需要官方标记
    warn_current_raw = mark_official_models(warn_current_raw)
    warn_previous_raw = mark_official_models(warn_previous_raw)

    # 筛选官方模型
    official_data = data[data['is_official'] == True].copy()

    if official_data.empty:
        print("警告: 在选定日期内未找到符合条件的官方模型数据。")
        return None

    # --- 全量数据透视 (用于平台总览和详细数据) ---
    all_current_data = data[data['date'] == current_date]
    all_previous_data = data[data['date'] == previous_date]
    current_pivot = create_pivot_table(all_current_data, model_order=model_order, merge_other=True)
    previous_pivot = create_pivot_table(all_previous_data, model_order=model_order, merge_other=True)
    growth_pivot = current_pivot - previous_pivot

    # --- 官方模型数据透视（详细平台，不合并"其他"） ---
    # 用于显示详细的各平台模型下载量详情表格
    current_official_data = official_data[official_data['date'] == current_date]
    previous_official_data = official_data[official_data['date'] == previous_date]
    current_official_pivot = create_pivot_table(current_official_data, model_order=model_order, merge_other=False, repo_order=REPO_ORDER_DETAILED)
    previous_official_pivot = create_pivot_table(previous_official_data, model_order=model_order, merge_other=False, repo_order=REPO_ORDER_DETAILED)
    growth_official_pivot = current_official_pivot - previous_official_pivot

    # 计算官方模型的总计 (用于Top N排名)
    current_totals = current_official_pivot.sum(axis=1).sort_values(ascending=False)
    growth_totals = growth_official_pivot.sum(axis=1).sort_values(ascending=False)

    # Top 5 增长最高的模型
    top5_growth = growth_totals.head(5)

    # Top 3 总下载量最高的模型
    top3_downloads = current_totals.head(3)

    # --- 衍生模型数据 ---
    derivative_data = data[data['is_official'] == False].copy()
    current_derivative_data = derivative_data[derivative_data['date'] == current_date]
    previous_derivative_data = derivative_data[derivative_data['date'] == previous_date]
    # 注意：此处 model_order=None，以包含所有衍生模型
    # 🔴 重要：使用 group_by_publisher=True 来区分不同 publisher 的同名模型
    current_derivative_pivot = create_pivot_table(current_derivative_data, model_order=None, group_by_publisher=True)
    previous_derivative_pivot = create_pivot_table(previous_derivative_data, model_order=None, group_by_publisher=True)
    
    # 确保两个透视表有相同的索引和列，以便相减
    all_derivative_models = current_derivative_pivot.index.union(previous_derivative_pivot.index)
    current_derivative_pivot = current_derivative_pivot.reindex(index=all_derivative_models, columns=REPO_ORDER, fill_value=0)
    previous_derivative_pivot = previous_derivative_pivot.reindex(index=all_derivative_models, columns=REPO_ORDER, fill_value=0)
    
    growth_derivative_pivot = current_derivative_pivot - previous_derivative_pivot

    # 各平台下载量最高和增长最高的模型 (详细版)
    def _get_top_models(repo, current_pivot, growth_pivot, data_source):
        """辅助函数，用于获取指定平台和数据类型的顶尖模型"""
        if repo not in current_pivot.columns or current_pivot[repo].sum() == 0:
            return {
                'top_download_model': 'N/A', 'top_download_publisher': '', 'top_download_count': 0, 'top_download_growth': 0,
                'top_growth_model': 'N/A', 'top_growth_publisher': '', 'top_growth_count': 0, 'top_growth_current': 0,
            }

        # 检查 pivot 索引是否为多层索引（衍生模型）
        has_multiindex = isinstance(current_pivot.index, pd.MultiIndex)

        # 下载量最高
        top_download_idx = current_pivot[repo].idxmax()
        top_download_count = current_pivot.loc[top_download_idx, repo]
        top_download_growth = growth_pivot.loc[top_download_idx, repo] if repo in growth_pivot.columns and top_download_idx in growth_pivot.index else 0

        if has_multiindex:
            # 多层索引：(model_name, publisher)
            top_download_model, top_download_publisher = top_download_idx
        else:
            # 单层索引：model_name
            top_download_model = top_download_idx
            # 🔧 修复：从 data_source 中筛选出对应 repo 的数据再查找 publisher
            filtered_data_source = data_source[data_source['repo'] == repo]
            top_download_publisher = filtered_data_source.loc[filtered_data_source['model_name'] == top_download_model, 'publisher'].iloc[0] if not filtered_data_source.loc[filtered_data_source['model_name'] == top_download_model].empty else ''

        # 增长最高
        if repo not in growth_pivot.columns or growth_pivot[repo].sum() == 0:
            top_growth_model, top_growth_publisher, top_growth_count, top_growth_current = 'N/A', '', 0, 0
        else:
            top_growth_idx = growth_pivot[repo].idxmax()
            top_growth_count = growth_pivot.loc[top_growth_idx, repo]
            top_growth_current = current_pivot.loc[top_growth_idx, repo]

            if has_multiindex:
                # 多层索引：(model_name, publisher)
                top_growth_model, top_growth_publisher = top_growth_idx
            else:
                # 单层索引：model_name
                top_growth_model = top_growth_idx
                # 🔧 修复：从 data_source 中筛选出对应 repo 的数据再查找 publisher
                filtered_data_source = data_source[data_source['repo'] == repo]
                top_growth_publisher = filtered_data_source.loc[filtered_data_source['model_name'] == top_growth_model, 'publisher'].iloc[0] if not filtered_data_source.loc[filtered_data_source['model_name'] == top_growth_model].empty else ''

        return {
            'top_download_model': top_download_model, 'top_download_publisher': top_download_publisher,
            'top_download_count': int(top_download_count), 'top_download_growth': int(top_download_growth),
            'top_growth_model': top_growth_model, 'top_growth_publisher': top_growth_publisher,
            'top_growth_count': int(top_growth_count), 'top_growth_current': int(top_growth_current),
        }

    # --- 为“各平台榜首模型”准备合并后的数据 ---
    # 创建一个临时的 current_official_data，将 '魔乐 Modelers', '鲸智', 'Gitee' 合并到 '其他'
    temp_current_official_data_merged = current_official_data.copy()
    temp_current_official_data_merged['repo'] = temp_current_official_data_merged['repo'].replace(['魔乐 Modelers', '鲸智', 'Gitee'], '其他')
    current_official_pivot_merged = create_pivot_table(temp_current_official_data_merged, model_order=model_order, merge_other=True)

    # 创建一个临时的 previous_official_data，将 '魔乐 Modelers', '鲸智', 'Gitee' 合并到 '其他'
    temp_previous_official_data_merged = previous_official_data.copy()
    temp_previous_official_data_merged['repo'] = temp_previous_official_data_merged['repo'].replace(['魔乐 Modelers', '鲸智', 'Gitee'], '其他')
    previous_official_pivot_merged = create_pivot_table(temp_previous_official_data_merged, model_order=model_order, merge_other=True)

    growth_official_pivot_merged = current_official_pivot_merged - previous_official_pivot_merged

    platform_top_models = []
    
    for repo in REPO_ORDER: # 遍历 REPO_ORDER，包含 '其他'
        if repo in ['Hugging Face', 'AI Studio', 'ModelScope', 'GitCode']: # 这些平台保持独立
            official_tops = _get_top_models(repo, current_official_pivot, growth_official_pivot, current_official_data)
            # 只有 Hugging Face 和 ModelScope 有衍生模型
            derivative_tops = None
            if repo in ['Hugging Face', 'ModelScope']:
                derivative_tops = _get_top_models(repo, current_derivative_pivot, growth_derivative_pivot, current_derivative_data)
            platform_top_models.append({
                'platform': repo,
                'official_tops': official_tops,
                'derivative_tops': derivative_tops
            })
        elif repo == '其他': # '其他'平台使用合并后的数据
            official_tops = _get_top_models(repo, current_official_pivot_merged, growth_official_pivot_merged, temp_current_official_data_merged)
            # '其他'平台目前没有区分衍生模型
            platform_top_models.append({'platform': repo, 'official_tops': official_tops, 'derivative_tops': None})

    # 加载全量数据用于历史峰值计算
    full_data = load_data_from_db(date_filter=None, last_value_per_model=False)
    full_data = enforce_deduplication_and_standardization(full_data)
    full_data = filter_by_series(full_data)
    if not full_data.empty:
        full_data['download_count'] = pd.to_numeric(full_data['download_count'], errors='coerce').fillna(0)
        full_data = mark_official_models(full_data)
        # 便于日期比较，转换为 datetime
        full_data['date'] = pd.to_datetime(full_data['date'])
        current_dt = pd.to_datetime(current_date)
        previous_dt = pd.to_datetime(previous_date)

        def peak_total_by_type(df, cutoff_dt, is_official):
            """统计衍生模型的历史峰值总和"""
            subset = df[(df['is_official'] == is_official) & (df['date'] <= cutoff_dt)]
            if subset.empty:
                return 0
            peak_per_combo = subset.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
            return peak_per_combo.sum()

        def peak_total_by_platform(df, cutoff_dt, is_official):
            """统计衍生模型按平台的历史峰值"""
            subset = df[(df['is_official'] == is_official) & (df['date'] <= cutoff_dt)]
            if subset.empty:
                return pd.Series()
            peak_per_model = subset.groupby(['repo', 'publisher', 'model_name'])['download_count'].max()
            # 按平台汇总
            platform_totals = peak_per_model.groupby('repo').sum()
            return platform_totals

        # ========== 总体情况摘要 ==========
        # 官方模型：使用当日值
        official_current_total = official_data[official_data['date'] == current_date]['download_count'].sum()
        official_previous_total = official_data[official_data['date'] == previous_date]['download_count'].sum()
        official_growth = official_current_total - official_previous_total

        # 衍生模型：使用历史峰值
        derivative_current_total = peak_total_by_type(full_data, current_dt, is_official=False)
        derivative_previous_total = peak_total_by_type(full_data, previous_dt, is_official=False)
        derivative_growth = derivative_current_total - derivative_previous_total

        # 汇总总数（官方=当日值，衍生=历史峰值）
        all_current_total = official_current_total + derivative_current_total
        all_previous_total = official_previous_total + derivative_previous_total
        all_growth = all_current_total - all_previous_total

        # ========== 平台汇总（官方用当日值，衍生用历史峰值） ==========
        # 官方模型：使用当日值按平台统计
        current_official_data_by_date = official_data[official_data['date'] == current_date]
        previous_official_data_by_date = official_data[official_data['date'] == previous_date]
        current_official_platforms = current_official_data_by_date.groupby('repo')['download_count'].sum()
        previous_official_platforms = previous_official_data_by_date.groupby('repo')['download_count'].sum()

        # 衍生模型：使用历史峰值按平台统计
        current_derivative_platforms = peak_total_by_platform(full_data, current_dt, is_official=False)
        previous_derivative_platforms = peak_total_by_platform(full_data, previous_dt, is_official=False)

        # 合并官方和衍生模型
        current_platform_totals = current_official_platforms.add(current_derivative_platforms, fill_value=0)
        previous_platform_totals = previous_official_platforms.add(previous_derivative_platforms, fill_value=0)

        # 为了在平台汇总中显示"其他"的汇总数据，需要将 Gitee, Modelers, 鲸智合并为"其他"
        current_platform_totals = current_platform_totals.rename(index={'魔乐 Modelers': '其他', '鲸智': '其他', 'Gitee': '其他'})
        current_platform_totals = current_platform_totals.groupby(level=0).sum()

        previous_platform_totals = previous_platform_totals.rename(index={'魔乐 Modelers': '其他', '鲸智': '其他', 'Gitee': '其他'})
        previous_platform_totals = previous_platform_totals.groupby(level=0).sum()
    else:
        # 数据为空时的默认值
        official_current_total = 0
        official_previous_total = 0
        official_growth = 0
        derivative_current_total = 0
        derivative_previous_total = 0
        derivative_growth = 0
        all_current_total = 0
        all_previous_total = 0
        all_growth = 0
        current_platform_totals = pd.Series()
        previous_platform_totals = pd.Series()

    # 合并并确保数值类型，避免TypeError
    platform_summary = pd.DataFrame({
        'current_total': current_platform_totals,
        'previous_total': previous_platform_totals
    }).reindex(REPO_ORDER).fillna(0).astype(int) # 🔧 修复：reindex 使用 REPO_ORDER

    platform_summary['growth_total'] = platform_summary['current_total'] - platform_summary['previous_total']
    platform_summary = platform_summary[['current_total', 'growth_total']]
    # 衍生模型（按 HF、非官方、publisher+model_name 去重的新出现数量）
    # 🔴 修复：移除多余的日期筛选，因为 all_current_data/all_previous_data 已经只包含对应日期的数据
    hf_curr_non_official = all_current_data[
        (all_current_data['repo'] == 'Hugging Face') &
        (all_current_data['is_official'] == False)
    ]
    hf_prev_non_official = all_previous_data[
        (all_previous_data['repo'] == 'Hugging Face') &
        (all_previous_data['is_official'] == False)
    ]
    curr_deriv_keys = set(zip(hf_curr_non_official['publisher'], hf_curr_non_official['model_name']))
    prev_deriv_keys = set(zip(hf_prev_non_official['publisher'], hf_prev_non_official['model_name']))
    new_derivative_keys = curr_deriv_keys - prev_deriv_keys
    derivative_new_models = len(new_derivative_keys)
    # 列表明细（HF非官方新增差集）
    derivative_new_models_list = []
    if derivative_new_models > 0:
        # 按下载量降序，保持唯一
        hf_curr_non_official = hf_curr_non_official.sort_values('download_count', ascending=False)
        seen = set()
        for _, row in hf_curr_non_official.iterrows():
            key = (row['publisher'], row['model_name'])
            if key in new_derivative_keys and key not in seen:
                seen.add(key)
                derivative_new_models_list.append({
                    'model_name': row['model_name'],
                    'publisher': row['publisher'],
                    'download_count': int(row.get('download_count', 0) or 0),
                    'model_type': row.get('model_type'),
                    'model_category': row.get('model_category'),
                    'base_model': row.get('base_model'),
                    'repo': row.get('repo')
                })

    # 社区维度 & 模型维度
    platform_top_models_df = pd.DataFrame(platform_top_models)
    # 1. 社区维度：HF增长最高 (基于官方模型)
    hf_row = platform_top_models_df.loc[platform_top_models_df['platform'] == 'Hugging Face']
    if not hf_row.empty:
        hf_official_tops = hf_row['official_tops'].iloc[0]
        hf_top_growth_model_name = hf_official_tops['top_growth_model']
        hf_top_model_growth = hf_official_tops['top_growth_count']
    else:
        hf_top_growth_model_name = "N/A"
        hf_top_model_growth = 0

    # 2. 模型维度：下载总量前三
    top3_downloads_details = current_totals.head(3)

    # 3. 模型维度：增长最快前三
    top3_growth_details = growth_totals.head(3)

    community_summary = {
        'hf_top_model_name': hf_top_growth_model_name,
        'hf_top_model_growth': hf_top_model_growth,
        'top3_downloads_details': top3_downloads_details.to_dict(),
        'top3_growth_details': top3_growth_details.to_dict(),
    }

    # 获取本周新增的Finetune和Adapter模型
    try:
        from .fetchers.fetchers_modeltree import get_weekly_new_finetune_adapters
        # 🔧 修复：传递 model_series 参数以精确筛选模型系列
        new_models_info = get_weekly_new_finetune_adapters(current_date, previous_date, model_series=model_series)
    except Exception as e:
        print(f"获取新增Finetune/Adapter模型信息失败: {e}")
        new_models_info = {
            'new_finetune_models': [],
            'new_adapter_models': [],
            'new_lora_models': [],
            'total_new': 0,
            'summary': '获取新增模型信息时出错'
        }

    # 🆕 获取本周新增的所有模型（完整列表）
    # 🔧 修复：直接传递日期，让函数自己加载数据（与 get_weekly_new_finetune_adapters 保持一致）
    all_new_models_info = get_all_new_models(
        current_date=current_date,
        previous_date=previous_date,
        model_series=model_series
    )

    # 统计模型数量（按类别、按是否原始）——衍生模型计数采用回填（取当前日期及之前的最后一条）
    backfill_for_count = load_data_from_db(date_filter=current_date, last_value_per_model=True)
    backfill_for_count = enforce_deduplication_and_standardization(backfill_for_count)
    backfill_for_count = filter_by_series(backfill_for_count)
    backfill_for_count = mark_official_models(backfill_for_count)
    derivative_current_total_models = len(
        backfill_for_count[
            (backfill_for_count['model_category'] == ('ernie-4.5' if model_series == 'ERNIE-4.5' else 'paddleocr-vl')) &
            (backfill_for_count['model_type'] != 'original')
        ]
    )

    summary_stats = {
        'all_current_total': all_current_total,
        'all_growth': all_growth,
        'official_current_total': official_current_total,
        'official_growth': official_growth,
        'derivative_current_total': derivative_current_total,
        'derivative_growth': derivative_growth,
        'derivative_current_total_models': derivative_current_total_models,
        'derivative_new_models': derivative_new_models,
        'derivative_new_models_list': derivative_new_models_list,
    }

    # 🔴 负增长检测：检查所有平台和模型的增长情况
    negative_growth_warnings = []

    # 🔧 修复：基于原始数据检测负增长，而不是基于 pivot 表
    # 这样能捕获所有模型（包括不在 model_order 中的衍生模型）
    # 修复：使用 REPO_ORDER_DETAILED 来单独检测 '魔乐 Modelers', '鲸智', 'Gitee' 的负增长
    for repo in REPO_ORDER_DETAILED: # 修改为 REPO_ORDER_DETAILED
        # 获取该平台上周和本周的原始当日数据（不使用 last_value_per_model）
        prev_platform_data = warn_previous_raw[warn_previous_raw['repo'] == repo].copy()
        curr_platform_data = warn_current_raw[warn_current_raw['repo'] == repo].copy()

        # 按模型+发布者聚合下载量，避免不同发布者的同名模型被合并
        prev_by_model = prev_platform_data.groupby(['model_name', 'publisher'])['download_count'].sum()
        curr_by_model = curr_platform_data.groupby(['model_name', 'publisher'])['download_count'].sum()

        # 检查每个在上周存在的模型
        for model_name in prev_by_model.index:
            previous_val = prev_by_model[model_name]
            current_val = curr_by_model.get(model_name, 0)  # 如果不存在则为0
            growth_val = current_val - previous_val

            # 只报告负增长
            if growth_val < 0:
                # 判断是官方模型还是衍生模型
                model_name_only, publisher = model_name
                prev_rows = prev_platform_data[
                    (prev_platform_data['model_name'] == model_name_only) &
                    (prev_platform_data['publisher'] == publisher)
                ]
                curr_rows = curr_platform_data[
                    (curr_platform_data['model_name'] == model_name_only) &
                    (curr_platform_data['publisher'] == publisher)
                ]
                is_official_prev = (not prev_rows.empty) and prev_rows['is_official'].any()
                is_official_curr = (not curr_rows.empty) and curr_rows['is_official'].any()
                is_official = is_official_prev or is_official_curr
                model_type = '官方模型' if is_official else '衍生模型'

                negative_growth_warnings.append({
                    'platform': repo,
                    'model_name': model_name_only,
                    'publisher': publisher,
                    'model_type': model_type,
                    'previous': int(previous_val),
                    'current': int(current_val),
                    'growth': int(growth_val)
                })

    # 如果发现负增长，打印警告
    if negative_growth_warnings:
        print("\n" + "=" * 80)
        print("⚠️  警告：检测到负增长！")
        print("=" * 80)
        for warning in negative_growth_warnings:
            print(f"平台: {warning['platform']}")
            print(f"模型: {warning['model_name']} | 发布者: {warning['publisher']} | 类型: {warning['model_type']}")
            print(f"数据: {warning['previous']:,} → {warning['current']:,} (变化: {warning['growth']:,})")
            print(f"说明: 该模型在 {previous_date} 有数据，但在 {current_date} 数据减少或缺失")
            print("-" * 80)

    return {
        'current_date': current_date,
        'summary_stats': summary_stats,
        'community_summary': community_summary,
        'previous_date': previous_date,
        'current_pivot': current_official_pivot,  # 修改：使用官方模型透视表
        'previous_pivot': previous_official_pivot,  # 修改：使用官方模型透视表
        'growth_pivot': growth_official_pivot,  # 修改：使用官方模型透视表
        'top5_growth': top5_growth,
        'top3_downloads': top3_downloads,
        'platform_top_models': platform_top_models_df,
        'platform_summary': platform_summary,
        'new_models_info': new_models_info,  # 新增Finetune/Adapter/LoRA模型信息
        'all_new_models_info': all_new_models_info,  # 🆕 所有新增模型完整列表
        'negative_growth_warnings': negative_growth_warnings  # 负增长警告
    }


def format_report_tables(report_data):
    """
    格式化报表数据为可显示的表格

    Args:
        report_data: calculate_weekly_report 返回的数据

    Returns:
        dict: 包含格式化表格的字典
    """
    if report_data is None:
        return None

    tables = {}

    # 1. & 2. 合并下载量和增长量
    current_pivot = report_data['current_pivot'].astype(int)
    growth_pivot = report_data['growth_pivot'].astype(int)

    interleaved_df = pd.DataFrame(index=current_pivot.index)

    # 使用 REPO_ORDER_DETAILED 保证顺序（显示详细平台，不合并"其他"）
    for repo in REPO_ORDER_DETAILED:
        if repo in current_pivot.columns:
            interleaved_df[f'{repo} (总)'] = current_pivot[repo]
            interleaved_df[f'{repo} (周增)'] = growth_pivot[repo]

    tables['combined_downloads_growth'] = interleaved_df

    # 3. Top 5 增长最高的模型
    top5_df = pd.DataFrame({
        '模型名称': report_data['top5_growth'].index,
        '本周增长': report_data['top5_growth'].values.astype(int)
    }).reset_index(drop=True)
    top5_df.index = top5_df.index + 1
    tables['top5_growth'] = top5_df

    # 4. Top 3 总下载量最高的模型
    top3_df = pd.DataFrame({
        '模型名称': report_data['top3_downloads'].index,
        '总下载量': report_data['top3_downloads'].values.astype(int)
    }).reset_index(drop=True)
    top3_df.index = top3_df.index + 1
    tables['top3_downloads'] = top3_df

    # 5. 各平台榜首模型
    platform_top_df = report_data['platform_top_models'].copy()
    
    def format_model_info(tops, label, is_growth_col=False):
        if not tops:
            return ""
            
        model_name = tops['top_growth_model'] if is_growth_col else tops['top_download_model']
        if model_name == 'N/A':
            return ""

        publisher = tops['top_growth_publisher'] if is_growth_col else tops['top_download_publisher']
        total_downloads = tops['top_growth_current'] if is_growth_col else tops['top_download_count']
        weekly_growth = tops['top_growth_count'] if is_growth_col else tops['top_download_growth']

        return (
            f"{label}: {model_name} ({publisher})\n"
            f"  ({total_downloads:,}) (本周: +{weekly_growth:,})"
        )

    def format_combined_download(row):
        official_str = format_model_info(row['official_tops'], '官方', is_growth_col=False)
        derivative_str = format_model_info(row['derivative_tops'], '衍生', is_growth_col=False)
        return "\n".join(filter(None, [official_str, derivative_str]))

    def format_combined_growth(row):
        official_str = format_model_info(row['official_tops'], '官方', is_growth_col=True)
        derivative_str = format_model_info(row['derivative_tops'], '衍生', is_growth_col=True)
        return "\n".join(filter(None, [official_str, derivative_str]))

    platform_top_df['下载量最高模型'] = platform_top_df.apply(format_combined_download, axis=1)
    platform_top_df['增长最高模型'] = platform_top_df.apply(format_combined_growth, axis=1)
    
    # 筛选并重命名列
    tables['platform_top_models'] = platform_top_df[['platform', '下载量最高模型', '增长最高模型']]

    # 6. 各平台汇总
    summary_df = report_data['platform_summary'].copy()
    summary_df.columns = ['当前总下载量', '本周增长']
    summary_df = summary_df.astype(int)
    tables['platform_summary'] = summary_df

    # 7. 新增Finetune和Adapter模型表格
    new_models_info = report_data.get('new_models_info', {})

    # 格式化新增Finetune模型
    finetune_data = new_models_info.get('new_finetune_models', [])
    if finetune_data:
        finetune_df = pd.DataFrame(finetune_data)
        finetune_df.index = finetune_df.index + 1
        finetune_df.columns = ['模型名称', '发布者', '下载量']
        tables['new_finetune_models'] = finetune_df
    else:
        tables['new_finetune_models'] = pd.DataFrame(columns=['模型名称', '发布者', '下载量'])

    # 格式化新增Adapter模型
    adapter_data = new_models_info.get('new_adapter_models', [])
    if adapter_data:
        adapter_df = pd.DataFrame(adapter_data)
        adapter_df.index = adapter_df.index + 1
        adapter_df.columns = ['模型名称', '发布者', '下载量']
        tables['new_adapter_models'] = adapter_df
    else:
        tables['new_adapter_models'] = pd.DataFrame(columns=['模型名称', '发布者', '下载量'])

    # 格式化新增LoRA模型
    lora_data = new_models_info.get('new_lora_models', [])
    if lora_data:
        lora_df = pd.DataFrame(lora_data)
        lora_df.index = lora_df.index + 1
        lora_df.columns = ['模型名称', '发布者', '下载量']
        tables['new_lora_models'] = lora_df
    else:
        tables['new_lora_models'] = pd.DataFrame(columns=['模型名称', '发布者', '下载量'])

    # 新增模型汇总信息
    tables['new_models_summary'] = new_models_info.get('summary', '无新增模型信息')

    # 8. Model Tree 新增衍生模型表格
    # 移除 Model Tree 新增衍生模型模块
    tables['new_model_tree_models'] = pd.DataFrame(columns=['模型名称', '发布者', '下载量', '基础模型', '模型类型'])
    tables['model_tree_summary'] = '（已移除 Model Tree 新增模块）'

    # 9. 负增长警告表格
    negative_growth_warnings = report_data.get('negative_growth_warnings', [])
    if negative_growth_warnings:
        warnings_df = pd.DataFrame(negative_growth_warnings)
        warnings_df.index = warnings_df.index + 1
        warnings_df.columns = ['平台', '模型名称', '发布者', '模型类型', '上周下载量', '本周下载量', '周增长']
        tables['negative_growth_warnings'] = warnings_df
    else:
        tables['negative_growth_warnings'] = pd.DataFrame(columns=['平台', '模型名称', '发布者', '模型类型', '上周下载量', '本周下载量', '周增长'])

    # 10. 🆕 所有新增模型完整列表
    all_new_models_info = report_data.get('all_new_models_info', {})

    new_models_list = all_new_models_info.get('new_models_list', [])
    if new_models_list:
        all_new_df = pd.DataFrame(new_models_list)
        all_new_df.index = all_new_df.index + 1

        # 根据包含的列来设置列名
        column_mapping = {
            'model_name': '模型名称',
            'publisher': '发布者',
            'repo': '平台',
            'download_count': '下载量',
            'model_type': '模型类型',
            'model_category': '模型分类',
            'base_model': '基础模型'
        }

        # 只重命名存在的列
        rename_dict = {k: v for k, v in column_mapping.items() if k in all_new_df.columns}
        all_new_df = all_new_df.rename(columns=rename_dict)

        tables['all_new_models'] = all_new_df
    else:
        tables['all_new_models'] = pd.DataFrame(columns=['模型名称', '发布者', '平台', '下载量', '模型类型'])

    # 新增模型汇总信息
    tables['all_new_models_summary'] = all_new_models_info.get('summary', '无新增模型')

    # 11. HF 非官方新增衍生模型列表
    derivative_new_list = report_data['summary_stats'].get('derivative_new_models_list', [])
    if derivative_new_list:
        deriv_new_df = pd.DataFrame(derivative_new_list)
        deriv_new_df.index = deriv_new_df.index + 1
        rename_map = {
            'model_name': '模型名称',
            'publisher': '发布者',
            'download_count': '下载量',
            'model_type': '模型类型',
            'model_category': '模型系列',
            'base_model': '基础模型',
            'repo': '平台'
        }
        deriv_new_df = deriv_new_df.rename(columns={k: v for k, v in rename_map.items() if k in deriv_new_df.columns})
        tables['derivative_new_models'] = deriv_new_df
    else:
        tables['derivative_new_models'] = pd.DataFrame(columns=['模型名称', '发布者', '下载量', '模型类型', '模型系列', '基础模型', '平台'])

    return tables


def calculate_paddleocr_vl_weekly_report(current_date=None, previous_date=None):
    """
    计算 PaddleOCR-VL 的周报数据
    """
    return calculate_weekly_report(
        current_date,
        previous_date,
        model_order=PADDLEOCR_VL_MODEL_ORDER,
        model_series='PaddleOCR-VL'
    )


def get_deleted_or_hidden_models(current_date, model_series='ERNIE-4.5'):
    """
    检测已被删除或隐藏的模型

    逻辑：
    - 使用回填模式（last_value_per_model=True）获取截止到当前日期的所有历史模型
    - 使用正常模式（last_value_per_model=False）获取当前日期的实际数据
    - 对比两者，找出在历史中存在但当前日期不存在的模型

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        model_series: 模型系列 ('ERNIE-4.5' 或 'PaddleOCR-VL')

    Returns:
        list: 已删除/隐藏的模型列表，每个元素包含:
            - model_name: 模型名称
            - publisher: 发布者
            - model_type: 模型类型
            - model_category: 模型分类
            - base_model: 基础模型
            - last_seen_date: 最后出现日期
            - last_download_count: 最后记录的下载量
            - repo: 平台
    """
    try:
        # 1. 获取所有历史模型（回填模式）
        all_historical = load_data_from_db(date_filter=current_date, last_value_per_model=True)

        # 2. 获取当前日期的实际数据
        current_actual = load_data_from_db(date_filter=current_date, last_value_per_model=False)

        if all_historical.empty:
            return []

        # 3. 筛选目标系列的衍生模型
        target_category = 'ernie-4.5' if model_series == 'ERNIE-4.5' else 'paddleocr-vl'

        # 历史中的衍生模型
        historical_derivatives = all_historical[
            (all_historical['model_category'] == target_category) &
            (all_historical['model_type'] != 'original')
        ].copy()

        # 当前日期的衍生模型
        current_derivatives = current_actual[
            (current_actual['model_category'] == target_category) &
            (current_actual['model_type'] != 'original')
        ].copy()

        if historical_derivatives.empty:
            return []

        # 3.5. 应用与周报相同的标准化逻辑
        # 标准化模型名称（移除 publisher 前缀）
        historical_derivatives = normalize_model_names(historical_derivatives)
        if not current_derivatives.empty:
            current_derivatives = normalize_model_names(current_derivatives)

        # 创建小写 key 用于去重
        historical_derivatives['_dedup_publisher'] = historical_derivatives['publisher'].astype(str).str.lower()
        historical_derivatives['_dedup_model_name'] = historical_derivatives['model_name'].astype(str).str.lower()
        if not current_derivatives.empty:
            current_derivatives['_dedup_publisher'] = current_derivatives['publisher'].astype(str).str.lower()
            current_derivatives['_dedup_model_name'] = current_derivatives['model_name'].astype(str).str.lower()

        # 去重（按下载量降序，保留最高的）
        historical_derivatives['download_count'] = pd.to_numeric(
            historical_derivatives['download_count'], errors='coerce'
        ).fillna(0)
        historical_derivatives = historical_derivatives.sort_values(
            by='download_count', ascending=False
        ).drop_duplicates(
            subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
        )

        if not current_derivatives.empty:
            current_derivatives['download_count'] = pd.to_numeric(
                current_derivatives['download_count'], errors='coerce'
            ).fillna(0)
            current_derivatives = current_derivatives.sort_values(
                by='download_count', ascending=False
            ).drop_duplicates(
                subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
            )

        # 4. 创建模型唯一标识（用小写 key）
        historical_derivatives['model_key'] = (
            historical_derivatives['repo'] + '|||' +
            historical_derivatives['_dedup_publisher'] + '|||' +
            historical_derivatives['_dedup_model_name']
        )

        if not current_derivatives.empty:
            current_derivatives['model_key'] = (
                current_derivatives['repo'] + '|||' +
                current_derivatives['_dedup_publisher'] + '|||' +
                current_derivatives['_dedup_model_name']
            )
            current_keys = set(current_derivatives['model_key'].unique())
        else:
            current_keys = set()

        historical_keys = set(historical_derivatives['model_key'].unique())

        # 5. 找出已删除/隐藏的模型
        deleted_keys = historical_keys - current_keys

        if not deleted_keys:
            return []

        # 6. 获取已删除模型的详细信息
        deleted_models = historical_derivatives[
            historical_derivatives['model_key'].isin(deleted_keys)
        ].copy()

        # 7. 对于每个已删除的模型，找到它最后出现的日期
        deleted_models_info = []

        for _, row in deleted_models.iterrows():
            repo = row.get('repo', '')
            publisher = str(row.get('publisher', ''))
            model_name = str(row.get('model_name', ''))
            publisher_key = str(row.get('_dedup_publisher', publisher.lower()))
            model_name_key = str(row.get('_dedup_model_name', model_name.lower()))

            # 查询该模型在数据库中最后出现的日期
            # 使用 LOWER() 进行不区分大小写匹配，避免大小写差异导致查不到最后记录
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT date, download_count
                FROM model_downloads
                WHERE repo = ? AND LOWER(publisher) = LOWER(?) AND LOWER(model_name) = LOWER(?)
                ORDER BY date DESC
                LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=(repo, publisher_key, model_name_key))
            conn.close()

            if not result.empty:
                last_seen_date = result.iloc[0]['date']
                last_download_count = result.iloc[0]['download_count']
            else:
                last_seen_date = row.get('date', 'Unknown')
                last_download_count = row.get('download_count', 0)

            model_info = {
                'model_name': model_name,
                'publisher': publisher,
                'model_type': row.get('model_type', 'unknown'),
                'model_category': row.get('model_category', ''),
                'base_model': row.get('base_model', ''),
                'last_seen_date': last_seen_date,
                'last_download_count': int(last_download_count) if pd.notna(last_download_count) else 0,
                'repo': repo
            }

            deleted_models_info.append(model_info)

        # 8. 按最后出现日期降序排序
        deleted_models_info = sorted(
            deleted_models_info,
            key=lambda x: x['last_seen_date'],
            reverse=True
        )

        return deleted_models_info

    except Exception as e:
        print(f"检测已删除/隐藏模型失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_derivative_models_all_platforms(df, selected_series=None):
    """
    分析全平台的衍生模型生态（基于 is_official 标记）

    Args:
        df: 包含全平台数据的 DataFrame
        selected_series: 要分析的系列列表，如 ['ERNIE-4.5', 'PaddleOCR-VL']

    Returns:
        dict: 包含分析结果的字典
    """
    if df.empty:
        return {
            'total_models': 0,
            'total_derivative_models': 0,
            'total_official_models': 0,
            'derivative_rate': 0,
            'by_platform': {},
            'by_series': {},
            'derivative_models_df': pd.DataFrame()
        }

    # 确保必要的列存在
    df = df.copy()

    # 🔴 标准化和去重（与 calculate_weekly_report 保持一致）
    # 1. 标准化模型名称（移除 publisher 前缀）
    df = normalize_model_names(df)

    # 2. 创建小写 key 用于去重，但保留原始大小写
    df['_dedup_publisher'] = df['publisher'].astype(str).str.lower()
    df['_dedup_model_name'] = df['model_name'].astype(str).str.lower()

    # 3. 再次去重，确保同一 (date, repo, publisher, model_name) 只有一条记录，且下载量最大
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
    df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
        subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
    )

    # 标记官方模型（如果还没有 is_official 列）
    if 'is_official' not in df.columns:
        df = mark_official_models(df)

    # 按系列筛选（所有记录现在都有 model_category 字段）
    if selected_series:
        series_mapping = {
            "ERNIE-4.5": "ernie-4.5",
            "PaddleOCR-VL": "paddleocr-vl"
        }

        selected_categories = [series_mapping.get(s, s) for s in selected_series]
        df = df[df['model_category'].isin(selected_categories)].copy()

    # 统计总数
    total_models = len(df)
    official_models_df = df[df['is_official'] == True]
    derivative_models_df = df[df['is_official'] == False]

    total_official_models = len(official_models_df)
    total_derivative_models = len(derivative_models_df)
    derivative_rate = (total_derivative_models / total_models * 100) if total_models > 0 else 0

    # 按平台统计
    by_platform = {}
    for platform in df['repo'].unique():
        platform_df = df[df['repo'] == platform]
        platform_derivative_df = derivative_models_df[derivative_models_df['repo'] == platform]

        # 计算下载量（转换为数值）
        platform_derivative_df['download_count_num'] = pd.to_numeric(
            platform_derivative_df['download_count'], errors='coerce'
        ).fillna(0)

        total_downloads = int(platform_derivative_df['download_count_num'].sum())

        # 找出下载量最高的模型（Top 5）
        top_models = platform_derivative_df.nlargest(5, 'download_count_num')[
            ['model_name', 'publisher', 'download_count']
        ].to_dict('records')

        # 🔧 新增：按系列统计（如果选择了多个系列）
        by_series_stats = {}
        if selected_series and 'model_category' in platform_derivative_df.columns:
            series_mapping = {
                "ERNIE-4.5": "ernie-4.5",
                "PaddleOCR-VL": "paddleocr-vl"
            }

            for series in selected_series:
                category = series_mapping.get(series, series)
                series_df = platform_derivative_df[platform_derivative_df['model_category'] == category]
                series_downloads = int(series_df['download_count_num'].sum())

                by_series_stats[category] = {
                    'count': len(series_df),
                    'downloads': series_downloads
                }

        by_platform[platform] = {
            'total_models': len(platform_df),
            'derivative_models': len(platform_derivative_df),
            'official_models': len(platform_df[platform_df['is_official'] == True]),
            'total_downloads': total_downloads,
            'derivative_rate': (len(platform_derivative_df) / len(platform_df) * 100) if len(platform_df) > 0 else 0,
            'top_models': top_models,
            'by_series': by_series_stats  # 新增：按系列统计
        }

    # 按系列统计（如果有 model_category 字段）
    by_series = {}
    if 'model_category' in df.columns:
        for category in df['model_category'].dropna().unique():
            category_df = df[df['model_category'] == category]
            category_derivative_df = derivative_models_df[derivative_models_df['model_category'] == category]

            by_series[category] = {
                'total_models': len(category_df),
                'derivative_models': len(category_derivative_df),
                'official_models': len(category_df[category_df['is_official'] == True]),
                'derivative_rate': (len(category_derivative_df) / len(category_df) * 100) if len(category_df) > 0 else 0
            }

    return {
        'total_models': total_models,
        'total_derivative_models': total_derivative_models,
        'total_official_models': total_official_models,
        'derivative_rate': derivative_rate,
        'by_platform': by_platform,
        'by_series': by_series,
        'derivative_models_df': derivative_models_df
    }


def get_quarter_start_date(current_date):
    """
    获取当前日期所在季度的开始日期

    Args:
        current_date: 当前日期 (datetime 或 str)

    Returns:
        str: 季度开始日期 (YYYY-MM-DD)
    """
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d')

    year = current_date.year
    month = current_date.month

    # 计算季度
    if month <= 3:
        quarter_start = datetime(year, 1, 1)
    elif month <= 6:
        quarter_start = datetime(year, 4, 1)
    elif month <= 9:
        quarter_start = datetime(year, 7, 1)
    else:
        quarter_start = datetime(year, 10, 1)

    return quarter_start.strftime('%Y-%m-%d')


def get_current_quarter_name(current_date):
    """
    获取当前季度名称

    Args:
        current_date: 当前日期 (datetime 或 str)

    Returns:
        str: 季度名称，如 "2026Q1"
    """
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d')

    year = current_date.year
    month = current_date.month

    # 计算季度
    if month <= 3:
        quarter = 1
    elif month <= 6:
        quarter = 2
    elif month <= 9:
        quarter = 3
    else:
        quarter = 4

    return f"{year}Q{quarter}"


def calculate_periodic_stats(current_date, selected_series=None, base_date=None):
    """
    计算周期性统计数据（新增模型等）

    Args:
        current_date: 分析日期 (YYYY-MM-DD)，即终止日期
        selected_series: 模型系列列表，如 ['ERNIE-4.5', 'PaddleOCR-VL']
        base_date: 对比基准日期 (YYYY-MM-DD)，即起始日期。如果不提供则默认为 7 天前

    Returns:
        dict: 包含周期性统计的字典
    """
    # 计算时间点
    current_date_dt = datetime.strptime(current_date, '%Y-%m-%d')
    if base_date:
        last_week_date = base_date
    else:
        last_week_date = (current_date_dt - timedelta(days=7)).strftime('%Y-%m-%d')
    quarter_start_date = get_quarter_start_date(current_date)
    quarter_name = get_current_quarter_name(current_date)

    # 加载数据（使用回填逻辑）
    current_data = load_data_from_db(date_filter=current_date, last_value_per_model=True)
    last_week_data = load_data_from_db(date_filter=last_week_date, last_value_per_model=True)
    quarter_start_data = load_data_from_db(date_filter=quarter_start_date, last_value_per_model=True)

    # 标准化和去重
    def standardize(df):
        if df.empty:
            return df
        df = df.copy()
        df = normalize_model_names(df)
        df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
        # 用小写 key 去重，但保留原始大小写
        df['_dedup_publisher'] = df['publisher'].astype(str).str.lower()
        df['_dedup_model_name'] = df['model_name'].astype(str).str.lower()
        df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
            subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
        )
        return df

    current_data = standardize(current_data)
    last_week_data = standardize(last_week_data)
    quarter_start_data = standardize(quarter_start_data)

    # 标记官方模型
    current_data = mark_official_models(current_data)
    last_week_data = mark_official_models(last_week_data)
    quarter_start_data = mark_official_models(quarter_start_data)

    # 按系列筛选
    def filter_series(df):
        if df.empty or not selected_series:
            return df
        series_mapping = {
            "ERNIE-4.5": "ernie-4.5",
            "PaddleOCR-VL": "paddleocr-vl"
        }
        selected_categories = [series_mapping.get(s, s) for s in selected_series]

        # 🔴 关键修复：使用 model_category OR model_name 匹配，避免因 model_category 缺失导致假新增
        # 为每个系列创建筛选条件
        conditions = []
        for category in selected_categories:
            # 获取对应的 name_pattern
            if category == 'ernie-4.5':
                name_pattern = 'ERNIE-4.5'
            elif category == 'paddleocr-vl':
                name_pattern = 'PaddleOCR-VL'
            else:
                name_pattern = category

            # model_category 匹配 OR model_name 包含关键词
            condition = (
                (df['model_category'] == category) |
                (df['model_name'].str.contains(name_pattern, case=False, na=False))
            )
            conditions.append(condition)

        # 合并所有条件（OR 关系）
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            return df[combined_condition].copy()
        else:
            return df.copy()

    current_data = filter_series(current_data)
    last_week_data = filter_series(last_week_data)
    quarter_start_data = filter_series(quarter_start_data)

    # 获取衍生模型
    current_derivatives = current_data[current_data['is_official'] == False].copy()
    last_week_derivatives = last_week_data[last_week_data['is_official'] == False].copy()
    quarter_start_derivatives = quarter_start_data[quarter_start_data['is_official'] == False].copy()

    # 累计数量
    total_count = len(current_derivatives)

    # 用小写 key 做集合比较，避免大小写差异导致误判新增
    def make_keys(df):
        return set(zip(df['repo'], df['_dedup_publisher'], df['_dedup_model_name']))

    # 本周新增：在当前日期存在但上周不存在的模型
    current_keys = make_keys(current_derivatives)
    last_week_keys = make_keys(last_week_derivatives)
    weekly_new_keys = current_keys - last_week_keys
    weekly_new_count = len(weekly_new_keys)

    # 季度新增
    quarter_start_keys = make_keys(quarter_start_derivatives)
    quarter_new_keys = current_keys - quarter_start_keys
    quarter_new_count = len(quarter_new_keys)

    # 本周新增模型列表（用小写 key 匹配，但返回原始大小写的数据）
    weekly_new_models = []
    for repo, pub_lower, name_lower in weekly_new_keys:
        model_row = current_derivatives[
            (current_derivatives['repo'] == repo) &
            (current_derivatives['_dedup_publisher'] == pub_lower) &
            (current_derivatives['_dedup_model_name'] == name_lower)
        ].iloc[0]

        weekly_new_models.append({
            'repo': repo,
            'publisher': model_row['publisher'],
            'model_name': model_row['model_name'],
            'download_count': int(model_row.get('download_count', 0)),
            'model_category': model_row.get('model_category', ''),
            'model_type': model_row.get('model_type', ''),
            'base_model': model_row.get('base_model', ''),
            'url': model_row.get('url', '')
        })

    # 按下载量排序
    weekly_new_models = sorted(weekly_new_models, key=lambda x: x['download_count'], reverse=True)

    # 按系列统计
    stats_by_series = {}
    if 'model_category' in current_data.columns:
        for category in current_data['model_category'].dropna().unique():
            cat_current = current_derivatives[current_derivatives['model_category'] == category]
            cat_last_week = last_week_derivatives[last_week_derivatives['model_category'] == category]
            cat_quarter_start = quarter_start_derivatives[quarter_start_derivatives['model_category'] == category]

            cat_current_keys = make_keys(cat_current)
            cat_last_week_keys = make_keys(cat_last_week)
            cat_quarter_start_keys = make_keys(cat_quarter_start)

            stats_by_series[category] = {
                'total_count': len(cat_current),
                'weekly_new_count': len(cat_current_keys - cat_last_week_keys),
                'quarter_new_count': len(cat_current_keys - cat_quarter_start_keys)
            }

    return {
        'current_date': current_date,
        'base_date': last_week_date,
        'total_count': total_count,
        'base_total_count': len(last_week_derivatives),
        'weekly_new_count': weekly_new_count,
        'quarter_new_count': quarter_new_count,
        'quarter_name': quarter_name,
        'weekly_new_models': weekly_new_models,
        'stats_by_series': stats_by_series
    }


def get_deleted_derivative_models_all_platforms(current_date, selected_series=None):
    """
    检测全平台已删除的衍生模型（基于 is_official 标记）

    逻辑：
    - 使用回填模式获取截止到当前日期的所有历史模型
    - 使用正常模式获取当前日期的实际数据
    - 对比两者，找出在历史中存在但当前日期不存在的衍生模型

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        selected_series: 可选的系列列表，如 ['ERNIE-4.5', 'PaddleOCR-VL']

    Returns:
        list: 已删除的衍生模型列表，每个元素包含:
            - model_name: 模型名称
            - publisher: 发布者
            - model_category: 模型分类
            - last_seen_date: 最后出现日期
            - last_download_count: 最后记录的下载量
            - repo: 平台
    """
    try:
        # 1. 获取所有历史模型（回填模式）
        all_historical = load_data_from_db(date_filter=current_date, last_value_per_model=True)

        # 2. 获取当前日期的实际数据
        current_actual = load_data_from_db(date_filter=current_date, last_value_per_model=False)

        if all_historical.empty:
            return []

        # 3. 应用标准化和去重逻辑
        def standardize_and_deduplicate(df):
            if df.empty:
                return df
            df = df.copy()
            # 标准化模型名称（移除 publisher 前缀）
            df = normalize_model_names(df)
            # 创建小写 key 用于去重
            df['_dedup_publisher'] = df['publisher'].astype(str).str.lower()
            df['_dedup_model_name'] = df['model_name'].astype(str).str.lower()
            # 转换下载量为数字
            df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
            # 去重（按下载量降序，保留最高的）
            df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
                subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
            )
            return df

        all_historical = standardize_and_deduplicate(all_historical)
        current_actual = standardize_and_deduplicate(current_actual)

        # 4. 标记官方模型
        all_historical = mark_official_models(all_historical)
        current_actual = mark_official_models(current_actual)

        # 5. 筛选衍生模型（非官方）
        historical_derivatives = all_historical[all_historical['is_official'] == False].copy()
        current_derivatives = current_actual[current_actual['is_official'] == False].copy()

        # 6. 按系列筛选（如果指定）
        if selected_series:
            series_mapping = {"ERNIE-4.5": "ernie-4.5", "PaddleOCR-VL": "paddleocr-vl"}
            selected_categories = [series_mapping.get(s, s) for s in selected_series]
            historical_derivatives = historical_derivatives[
                historical_derivatives['model_category'].isin(selected_categories)
            ].copy()
            current_derivatives = current_derivatives[
                current_derivatives['model_category'].isin(selected_categories)
            ].copy()

        if historical_derivatives.empty:
            return []

        # 7. 创建模型唯一标识（用小写 key）
        historical_derivatives['model_key'] = (
            historical_derivatives['repo'] + '|||' +
            historical_derivatives['_dedup_publisher'] + '|||' +
            historical_derivatives['_dedup_model_name']
        )

        if not current_derivatives.empty:
            current_derivatives['model_key'] = (
                current_derivatives['repo'] + '|||' +
                current_derivatives['_dedup_publisher'] + '|||' +
                current_derivatives['_dedup_model_name']
            )
            current_keys = set(current_derivatives['model_key'].unique())
        else:
            current_keys = set()

        historical_keys = set(historical_derivatives['model_key'].unique())

        # 8. 找出已删除的模型
        deleted_keys = historical_keys - current_keys

        if not deleted_keys:
            return []

        # 9. 获取已删除模型的详细信息
        deleted_models = historical_derivatives[
            historical_derivatives['model_key'].isin(deleted_keys)
        ].copy()

        # 10. 对于每个已删除的模型，找到它最后出现的日期
        deleted_models_info = []

        for _, row in deleted_models.iterrows():
            repo = row.get('repo', '')
            publisher = str(row.get('publisher', ''))
            model_name = str(row.get('model_name', ''))
            publisher_key = str(row.get('_dedup_publisher', publisher.lower()))
            model_name_key = str(row.get('_dedup_model_name', model_name.lower()))

            # 查询该模型在数据库中最后出现的日期
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT date, download_count
                FROM model_downloads
                WHERE repo = ? AND LOWER(publisher) = LOWER(?) AND LOWER(model_name) = LOWER(?)
                ORDER BY date DESC
                LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=(repo, publisher_key, model_name_key))
            conn.close()

            if not result.empty:
                last_seen_date = result.iloc[0]['date']
                last_download_count = result.iloc[0]['download_count']
            else:
                last_seen_date = row.get('date', 'Unknown')
                last_download_count = row.get('download_count', 0)

            model_info = {
                'model_name': model_name,
                'publisher': publisher,
                'model_category': row.get('model_category', ''),
                'last_seen_date': last_seen_date,
                'last_download_count': int(last_download_count) if pd.notna(last_download_count) else 0,
                'repo': repo
            }

            deleted_models_info.append(model_info)

        # 11. 按最后出现日期降序排序
        deleted_models_info = sorted(
            deleted_models_info,
            key=lambda x: x['last_seen_date'],
            reverse=True
        )

        return deleted_models_info

    except Exception as e:
        print(f"检测已删除衍生模型失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_models_needing_backfill(current_date, selected_series=None):
    """
    检测需要回填的模型（最后一天下载量不是历史最大值）

    逻辑：
    - 对于当前日期存在的每个模型
    - 查询该模型的历史最大下载量
    - 如果当前日期的下载量 < 历史最大值，则该模型需要回填

    Args:
        current_date: 当前日期 (YYYY-MM-DD)
        selected_series: 可选的系列列表，如 ['ERNIE-4.5', 'PaddleOCR-VL']

    Returns:
        list: 需要回填的模型列表，每个元素包含:
            - model_name: 模型名称
            - publisher: 发布者
            - model_category: 模型分类
            - repo: 平台
            - current_download_count: 当前日期下载量
            - max_download_count: 历史最大下载量
            - max_download_date: 历史最大下载量的日期
    """
    try:
        # 1. 获取当前日期的实际数据
        current_data = load_data_from_db(date_filter=current_date, last_value_per_model=False)

        if current_data.empty:
            return []

        # 2. 应用标准化和去重
        current_data = current_data.copy()
        current_data = normalize_model_names(current_data)
        # 创建小写 key 用于去重
        current_data['_dedup_publisher'] = current_data['publisher'].astype(str).str.lower()
        current_data['_dedup_model_name'] = current_data['model_name'].astype(str).str.lower()
        current_data['download_count'] = pd.to_numeric(current_data['download_count'], errors='coerce').fillna(0)
        current_data = current_data.sort_values(by='download_count', ascending=False).drop_duplicates(
            subset=['date', 'repo', '_dedup_publisher', '_dedup_model_name'], keep='first'
        )

        # 3. 标记官方模型并筛选衍生模型
        current_data = mark_official_models(current_data)
        current_derivatives = current_data[current_data['is_official'] == False].copy()

        # 4. 按系列筛选（如果指定）
        if selected_series:
            series_mapping = {"ERNIE-4.5": "ernie-4.5", "PaddleOCR-VL": "paddleocr-vl"}
            selected_categories = [series_mapping.get(s, s) for s in selected_series]
            current_derivatives = current_derivatives[
                current_derivatives['model_category'].isin(selected_categories)
            ].copy()

        if current_derivatives.empty:
            return []

        # 5. 对于每个模型，查询历史最大下载量
        models_needing_backfill = []

        conn = sqlite3.connect(DB_PATH)

        for _, row in current_derivatives.iterrows():
            repo = row['repo']
            publisher = row['publisher']
            model_name = row['model_name']
            current_download = row['download_count']

            # 查询历史最大下载量
            query = """
                SELECT MAX(download_count) as max_count, date
                FROM model_downloads
                WHERE repo = ? AND LOWER(publisher) = LOWER(?) AND LOWER(model_name) = LOWER(?)
                GROUP BY repo, publisher, model_name
                ORDER BY max_count DESC
                LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=(repo, publisher, model_name))

            if not result.empty:
                max_download = pd.to_numeric(result.iloc[0]['max_count'], errors='coerce')
                if pd.notna(max_download) and max_download > 0:
                    # 如果当前下载量 < 历史最大值，则需要回填
                    if current_download < max_download:
                        # 查询最大下载量的日期
                        date_query = """
                            SELECT date
                            FROM model_downloads
                            WHERE repo = ? AND LOWER(publisher) = LOWER(?)
                                  AND LOWER(model_name) = LOWER(?) AND download_count = ?
                            ORDER BY date DESC
                            LIMIT 1
                        """
                        date_result = pd.read_sql_query(
                            date_query, conn,
                            params=(repo, publisher, model_name, max_download)
                        )
                        max_date = date_result.iloc[0]['date'] if not date_result.empty else 'Unknown'

                        model_info = {
                            'model_name': model_name,
                            'publisher': publisher,
                            'model_category': row.get('model_category', ''),
                            'repo': repo,
                            'current_download_count': int(current_download),
                            'max_download_count': int(max_download),
                            'max_download_date': max_date
                        }

                        models_needing_backfill.append(model_info)

        conn.close()

        # 6. 按差值排序（差值越大排在越前面）
        models_needing_backfill = sorted(
            models_needing_backfill,
            key=lambda x: x['max_download_count'] - x['current_download_count'],
            reverse=True
        )

        return models_needing_backfill

    except Exception as e:
        print(f"检测需要回填的模型失败: {e}")
        import traceback
        traceback.print_exc()
        return []
