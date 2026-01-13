"""
爬虫基类和具体实现
"""
from abc import ABC, abstractmethod
from datetime import date
import pandas as pd
from ..config import SEARCH_QUERY


class BaseFetcher(ABC):
    """爬虫基类"""

    def __init__(self, platform_name):
        self.platform_name = platform_name
        self.today = date.today().isoformat()
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
                     created_at=None, last_modified=None):
        """
        创建一条记录

        Args:
            model_name: 模型名称
            publisher: 发布者
            download_count: 下载量
            search_keyword: 搜索关键词（可选）
            created_at: 创建时间（可选）
            last_modified: 最后修改时间（可选）

        Returns:
            dict: 记录字典
        """
        record = {
            "date": self.today,
            "repo": self.platform_name,
            "model_name": model_name,
            "publisher": publisher,
            "download_count": download_count
        }
        if search_keyword:
            record["search_keyword"] = search_keyword
        if created_at:
            record["created_at"] = created_at
        if last_modified:
            record["last_modified"] = last_modified
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
        optional_columns = ["search_keyword", "created_at", "last_modified"]
        ordered_columns = base_columns + [col for col in optional_columns if col in all_columns]

        # 添加其他可能出现的列
        for col in all_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        return pd.DataFrame(self.results, columns=ordered_columns)

    def __call__(self, progress_callback=None, progress_total=None):
        """使实例可调用"""
        return self.fetch(progress_callback, progress_total)
