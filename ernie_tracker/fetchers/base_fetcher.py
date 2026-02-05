"""
爬虫基类和具体实现
"""
from abc import ABC, abstractmethod
from datetime import date, datetime
import pandas as pd
from ..config import SEARCH_QUERY


def classify_model_category(model_name, search_keyword=None):
    """
    根据 model_name 和 search_keyword 自动推断 model_category

    Args:
        model_name: 模型名称
        search_keyword: 搜索关键词（可选）

    Returns:
        str: 'ernie-4.5', 'paddleocr-vl', 或 'other'
    """
    model_name_lower = str(model_name).lower()

    # 1. 优先使用 search_keyword
    if search_keyword:
        sk_upper = str(search_keyword).upper()
        if 'ERNIE-4.5' in sk_upper or sk_upper == 'ERNIE-4.5':
            return 'ernie-4.5'
        elif 'PADDLEOCR-VL' in sk_upper or sk_upper == 'PaddleOCR-VL':
            return 'paddleocr-vl'

    # 2. 使用模型名称判断
    if 'paddleocr-vl' in model_name_lower or 'paddleocrvl' in model_name_lower:
        return 'paddleocr-vl'
    elif 'ernie' in model_name_lower or '文心' in model_name_lower:
        return 'ernie-4.5'  # 所有 ERNIE 相关都归入 ernie-4.5
    else:
        return 'other'


class BaseFetcher(ABC):
    """爬虫基类"""

    def __init__(self, platform_name):
        self.platform_name = platform_name
        self.today = date.today().isoformat()
        self.fetched_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.results = []

    @abstractmethod
    def fetch(self, progress_callback=None, progress_total=None):
        """
        抓取数据的抽象方法

        Args:
            progress_callback: 进度回调函数
            progress_total: 总数（用于进度条）

        Returns:
            tuple: (DataFrame, total_count)
        """
        pass

    def create_record(self, model_name, publisher, download_count, search_keyword=None,
                     created_at=None, last_modified=None, url=None, model_category=None):
        """
        创建一条记录

        Args:
            model_name: 模型名称
            publisher: 发布者
            download_count: 下载量
            search_keyword: 搜索关键词（可选）
            created_at: 创建时间（可选）
            last_modified: 最后修改时间（可选）
            url: 模型详情页URL（可选）
            model_category: 模型分类（可选，如不提供则自动推断）

        Returns:
            dict: 记录字典
        """
        # 自动推断 model_category（如果未提供）
        if model_category is None:
            model_category = classify_model_category(model_name, search_keyword)

        record = {
            "date": self.today,
            "repo": self.platform_name,
            "model_name": model_name,
            "publisher": publisher,
            "download_count": download_count,
            "model_category": model_category,
            "fetched_at": self.fetched_at  # 入库时间(日期时间)
        }
        if search_keyword:
            record["search_keyword"] = search_keyword
        if created_at:
            record["created_at"] = created_at
        if last_modified:
            record["last_modified"] = last_modified
        if url:
            record["url"] = url
        return record

    def to_dataframe(self):
        """将结果转换为 DataFrame"""
        if not self.results:
            return pd.DataFrame()

        # 动态生成列名，包含所有可能的字段
        all_columns = set()
        for record in self.results:
            all_columns.update(record.keys())

        # 确保基础列在前
        base_columns = ["date", "repo", "model_name", "publisher", "download_count"]
        optional_columns = ["fetched_at", "search_keyword", "created_at", "last_modified", "url"]
        ordered_columns = base_columns + [col for col in optional_columns if col in all_columns]

        # 添加其他可能出现的列
        for col in all_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        return pd.DataFrame(self.results, columns=ordered_columns)

    def __call__(self, progress_callback=None, progress_total=None):
        """使实例可调用"""
        return self.fetch(progress_callback, progress_total)
