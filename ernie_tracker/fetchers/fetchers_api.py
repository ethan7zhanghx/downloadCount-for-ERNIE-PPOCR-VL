"""
Hugging Face 和 ModelScope 爬虫实现
这两个平台使用 API，不需要 Selenium
"""
from .base_fetcher import BaseFetcher
from ..config import SEARCH_QUERY


class HuggingFaceFetcher(BaseFetcher):
    """Hugging Face 爬虫"""

    def __init__(self):
        super().__init__("Hugging Face")

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 Hugging Face 数据"""
        from huggingface_hub import list_models, model_info

        models = list(list_models(search=SEARCH_QUERY, full=True))
        total_count = len(models)

        for i, m in enumerate(models, start=1):
            try:
                info = model_info(m.id, expand=["downloadsAllTime"])
                self.results.append(self.create_record(
                    model_name=m.id,
                    publisher=m.id.split("/")[0],
                    download_count=getattr(info, 'downloads_all_time', None)
                ))
            except Exception as e:
                print(f"获取 {m.id} 失败: {e}")

            if progress_callback:
                progress_callback(i, discovered_total=total_count)

        return self.to_dataframe(), total_count


class ModelScopeFetcher(BaseFetcher):
    """ModelScope 爬虫"""

    def __init__(self):
        super().__init__("ModelScope")

    def _get_model_ids(self):
        """获取所有模型 ID"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
        from ..utils import create_chrome_driver

        driver = create_chrome_driver()
        wait = WebDriverWait(driver, 20)
        model_ids = []
        page = 1

        while True:
            url = f"https://modelscope.cn/search?page={page}&search={SEARCH_QUERY}&type=model"
            print(f"[ModelScope] 爬取页面: {url}")
            driver.get(url)

            try:
                wait.until(EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "#normal_tab_model .antd5-row a")
                ))
            except TimeoutException:
                print("页面加载失败，已到最后一页")
                break

            cards = driver.find_elements(By.CSS_SELECTOR, "#normal_tab_model .antd5-row a")
            if not cards:
                break

            for link in cards:
                href = link.get_attribute("href")
                if "/models/" in href:
                    model_id = href.split("/models/")[-1]
                    model_ids.append(model_id)

            page += 1

        driver.quit()
        return list(set(model_ids))

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 ModelScope 数据"""
        from modelscope.hub.api import HubApi

        model_ids = self._get_model_ids()
        total_count = len(model_ids)
        api = HubApi()

        for i, model_id in enumerate(model_ids, start=1):
            try:
                info = api.get_model(model_id, revision="master")
                downloads = info.get("Downloads", 0)

                # 🔧 新增：获取时间字段
                from datetime import datetime
                created_at = None
                last_modified = None

                if info.get("CreatedTime"):
                    try:
                        created_at = datetime.fromtimestamp(info["CreatedTime"]).strftime('%Y-%m-%d')
                    except:
                        created_at = None

                if info.get("LastUpdatedTime"):
                    try:
                        last_modified = datetime.fromtimestamp(info["LastUpdatedTime"]).strftime('%Y-%m-%d')
                    except:
                        last_modified = None

                self.results.append(self.create_record(
                    model_name=model_id,
                    publisher=model_id.split("/")[0],
                    download_count=downloads,
                    created_at=created_at,
                    last_modified=last_modified
                ))
            except Exception as e:
                print(f"获取 {model_id} 失败: {e}")

            if progress_callback:
                progress_callback(i, discovered_total=total_count)

        return self.to_dataframe(), total_count
