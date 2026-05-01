"""固定链接爬虫实现 - GitCode 和 CAICT（鲸智）"""
import time
import requests
from .base_fetcher import BaseFetcher
from ..utils import create_chrome_driver
from ..config import GITCODE_MODEL_LINKS, CAICT_MODEL_LINKS, SELENIUM_TIMEOUT
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

GITCODE_API_BASE = "https://web-api.gitcode.com"
GITCODE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://ai.gitcode.com/",
    "Accept": "application/json",
}


class GitCodeFetcher(BaseFetcher):
    """GitCode 爬虫（通过 API 获取下载量，无需 Selenium）"""

    def __init__(self):
        super().__init__("GitCode")

    def _get_repo_id(self, namespace, repo_name, session):
        """通过 namespace/repo_name 获取数字 repo ID"""
        url = f"{GITCODE_API_BASE}/api/v2/projects/{namespace}%2F{repo_name}"
        resp = session.get(url, headers=GITCODE_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()["id"]

    def _get_download_count(self, repo_id, session):
        """通过 repo ID 获取总下载量"""
        url = f"{GITCODE_API_BASE}/api/v2/projects/{repo_id}/repository/download_statistics"
        resp = session.get(url, headers=GITCODE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("total_dl_cnt", 0)
        return 0

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 GitCode 数据（通过 API）"""
        total_count = len(GITCODE_MODEL_LINKS)
        session = requests.Session()

        for i, model_link in enumerate(GITCODE_MODEL_LINKS, start=1):
            try:
                # 从 URL 解析 namespace 和模型名
                # 格式: https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-0.3B-PT
                url_parts = model_link.rstrip('/').split('/')
                model_name = url_parts[-1]
                namespace = url_parts[-2]

                repo_id = self._get_repo_id(namespace, model_name, session)
                download_count = self._get_download_count(repo_id, session)

                self.results.append(self.create_record(
                    model_name=model_name,
                    publisher="飞桨PaddlePaddle",
                    download_count=download_count
                ))
                print(f"[GitCode] {i}/{total_count} {model_name}: {download_count}")

            except Exception as e:
                print(f"获取 {model_link} 失败: {e}")

            if progress_callback:
                progress_callback(i, discovered_total=total_count)

        return self.to_dataframe(), total_count


class CAICTFetcher(BaseFetcher):
    """鲸智 CAICT 爬虫"""

    def __init__(self):
        super().__init__("鲸智")

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取鲸智数据"""
        driver = create_chrome_driver()
        wait = WebDriverWait(driver, SELENIUM_TIMEOUT)
        total_models = len(CAICT_MODEL_LINKS)

        for idx, model_link in enumerate(CAICT_MODEL_LINKS, start=1):
            print(f"[鲸智] 正在处理 {idx}/{total_models}：{model_link}")
            driver.get(model_link)

            try:
                model_name = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                        "#community-app > div > div:nth-child(2) > "
                        "div.w-full.bg-\\[\\#FCFCFD\\].pt-9.pb-\\[60px\\].xl\\:px-10.md\\:px-0.md\\:pb-6.md\\:h-auto > "
                        "div > div.flex.flex-col.gap-\\[16px\\].flex-wrap.mb-\\[8px\\].text-lg.text-\\[\\#606266\\]."
                        "font-semibold.md\\:px-5 > div > a"))
                ).text.strip()

                downloads = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                        "#pane-summary > div > div.w-\\[40\\%\\].sm\\:w-\\[100\\%\\].border-l.border-\\[\\#EBEEF5\\]."
                        "md\\:border-l-0.md\\:border-b.md\\:w-full.md\\:pl-0 > div > "
                        "div.text-\\[\\#303133\\].text-base.font-semibold.leading-6.mt-1.md\\:pl-0"))
                ).text.strip().replace(',', '')

                self.results.append(self.create_record(
                    model_name=model_name,
                    publisher="PaddlePaddle",
                    download_count=downloads
                ))

            except Exception as e:
                print(f"处理 {model_link} 时失败，原因：{e}")
                continue

            if progress_callback:
                progress_callback(idx, discovered_total=total_models)

        driver.quit()
        return self.to_dataframe(), total_models
