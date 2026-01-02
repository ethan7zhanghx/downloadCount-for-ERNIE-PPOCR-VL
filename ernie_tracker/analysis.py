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

        # 筛选Hugging Face平台的指定系列模型（使用 model_category 字段）
        hf_current = current_data[
            (current_data['repo'] == 'Hugging Face') &
            (current_data['model_category'] == target_category)
        ].copy()

        if previous_data.empty:
            hf_previous = pd.DataFrame()
        else:
            hf_previous = previous_data[
                (previous_data['repo'] == 'Hugging Face') &
                (previous_data['model_category'] == target_category)
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
    current_data = load_data_from_db(date_filter=current_date, last_value_per_model=True)
    previous_data = load_data_from_db(date_filter=previous_date, last_value_per_model=True)

    # 负增长检测使用真实的当日记录（不带 last_value_per_model），单独加载
    warn_current_raw = load_data_from_db(date_filter=current_date, last_value_per_model=False)
    warn_previous_raw = load_data_from_db(date_filter=previous_date, last_value_per_model=False)

    # 🔴 关键修复：在合并和进一步处理之前，对数据进行强制标准化和二次去重
    # 确保即使数据库中存在不一致，也能在分析时得到修正
    def enforce_deduplication_and_standardization(df):
        if df.empty:
            return df
        
        # 1. 标准化 publisher 名称（统一大小写）
        df['publisher'] = df['publisher'].astype(str).apply(lambda x: x.title() if x.lower() != 'nan' else x)
        
        # 2. 标准化模型名称（移除 publisher 前缀）
        df = normalize_model_names(df)
        
        # 3. 再次去重，确保同一 (date, repo, publisher, model_name) 只有一条记录，且下载量最大
        # 按照 download_count 降序排序，然后保留每个分组的第一个
        df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0)
        df = df.sort_values(by='download_count', ascending=False).drop_duplicates(
            subset=['date', 'repo', 'publisher', 'model_name'], keep='first'
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

    # 🔧 修复：根据 model_series 使用 model_category 字段 **或** 模型名称筛选
    # 这样既能包含正确分类的衍生模型，也能包含其他平台的官方模型
    if model_series == 'ERNIE-4.5':
        # 筛选条件：model_category == 'ernie-4.5' 或 model_name 包含 'ERNIE-4.5'
        if 'model_category' in data.columns:
            condition = (
                (data['model_category'] == 'ernie-4.5') |  # HuggingFace 正确分类的模型
                (data['model_name'].str.contains('ERNIE-4.5', case=False, na=False))  # 其他平台的官方模型
            )
            data = data[condition].copy()
        else:
            # 回退到名称搜索（兼容旧数据）
            data = data[data['model_name'].str.contains('ERNIE-4.5', case=False, na=False)].copy()
    elif model_series == 'PaddleOCR-VL':
        # 筛选条件：model_category == 'paddleocr-vl' 或 model_name 包含 'PaddleOCR-VL'
        if 'model_category' in data.columns:
            condition = (
                (data['model_category'] == 'paddleocr-vl') |  # HuggingFace 正确分类的模型
                (data['model_name'].str.contains('PaddleOCR-VL', case=False, na=False))  # 其他平台的官方模型
            )
            data = data[condition].copy()
        else:
            # 回退到名称搜索（兼容旧数据）
            data = data[data['model_name'].str.contains('PaddleOCR-VL', case=False, na=False)].copy()

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

    # 各平台总下载量和增长 (基于全量数据)
    # 🔧 修复：为了在平台汇总中显示“其他”的汇总数据，需要将 Gitee, Modelers, 鲸智合并为“其他”
    all_current_data_with_other = data[data['date'] == current_date].copy()
    all_current_data_with_other['repo'] = all_current_data_with_other['repo'].replace(['魔乐 Modelers', '鲸智', 'Gitee'], '其他')
    current_platform_totals = all_current_data_with_other.groupby('repo')['download_count'].sum()

    all_previous_data_with_other = data[data['date'] == previous_date].copy()
    all_previous_data_with_other['repo'] = all_previous_data_with_other['repo'].replace(['魔乐 Modelers', '鲸智', 'Gitee'], '其他')
    previous_platform_totals = all_previous_data_with_other.groupby('repo')['download_count'].sum()

    # 合并并确保数值类型，避免TypeError
    platform_summary = pd.DataFrame({
        'current_total': current_platform_totals,
        'previous_total': previous_platform_totals
    }).reindex(REPO_ORDER).fillna(0).astype(int) # 🔧 修复：reindex 使用 REPO_ORDER

    platform_summary['growth_total'] = platform_summary['current_total'] - platform_summary['previous_total']
    platform_summary = platform_summary[['current_total', 'growth_total']]

    # 增加总体统计
    # 官方模型
    official_current_total = official_data[official_data['date'] == current_date]['download_count'].sum()
    official_previous_total = official_data[official_data['date'] == previous_date]['download_count'].sum()
    official_growth = official_current_total - official_previous_total

    # 所有模型（包括衍生）
    all_current_data = data[data['date'] == current_date]
    all_previous_data = data[data['date'] == previous_date]
    all_current_total = all_current_data['download_count'].sum()
    all_previous_total = all_previous_data['download_count'].sum()
    all_growth = all_current_total - all_previous_total

    # 衍生模型
    derivative_current_total = all_current_total - official_current_total
    derivative_growth = all_growth - official_growth
    # 衍生模型（按 HF、非官方、publisher+model_name 去重的新出现数量）
    hf_curr_non_official = all_current_data[
        (all_current_data['date'] == current_date) &
        (all_current_data['repo'] == 'Hugging Face') &
        (all_current_data['is_official'] == False)
    ]
    hf_prev_non_official = all_previous_data[
        (all_previous_data['date'] == previous_date) &
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

    # 统计模型数量（按类别、按是否原始）
    derivative_current_total_models = len(
        all_current_data[
            (all_current_data['model_category'] == ('ernie-4.5' if model_series == 'ERNIE-4.5' else 'paddleocr-vl')) &
            (all_current_data['model_type'] != 'original')
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
