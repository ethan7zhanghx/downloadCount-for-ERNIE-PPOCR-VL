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

    def create_record(self, model_name, publisher, download_count):
        """
        创建一条记录

        Args:
            model_name: 模型名称
            publisher: 发布者
            download_count: 下载量

        Returns:
            dict: 记录字典
        """
        return {
            "date": self.today,
            "repo": self.platform_name,
            "model_name": model_name,
            "publisher": publisher,
            "download_count": download_count
        }

    def to_dataframe(self):
        """将结果转换为 DataFrame"""
        return pd.DataFrame(
            self.results,
            columns=["date", "repo", "model_name", "publisher", "download_count"]
        )

    def __call__(self, progress_callback=None, progress_total=None):
        """使实例可调用"""
        return self.fetch(progress_callback, progress_total)
