"""固定链接爬虫实现 - GitCode 和 CAICT（鲸智）"""
import time
from .base_fetcher import BaseFetcher
from ..utils import create_chrome_driver
from ..config import GITCODE_MODEL_LINKS, CAICT_MODEL_LINKS, SELENIUM_TIMEOUT
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class GitCodeFetcher(BaseFetcher):
    """GitCode 爬虫"""

    def __init__(self):
        super().__init__("GitCode")

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 GitCode 数据"""
        driver = create_chrome_driver()
        wait = WebDriverWait(driver, SELENIUM_TIMEOUT)
        total_count = len(GITCODE_MODEL_LINKS)

        for i, model_link in enumerate(GITCODE_MODEL_LINKS, start=1):
            try:
                driver.get(model_link)

                model_name = wait.until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR,
                        "#repo-banner-box > div > div.repo-info.h-full.ai-hub > div > "
                        "div:nth-child(1) > div > div > div.info-item.project-name > "
                        "div.project-text > div > p > a > span"))
                ).text.strip()

                downloads_element = wait.until(
                    EC.presence_of_element_located((By.XPATH,
                        '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/'
                        'div[1]/div[1]/div/div[2]'))
                )

                # 等待下载量加载完成
                last_val = ""
                for _ in range(5):
                    val = downloads_element.text.strip().replace(',', '')
                    if val and val != last_val:
                        last_val = val
                        time.sleep(1)
                    else:
                        break

                self.results.append(self.create_record(
                    model_name=model_name,
                    publisher="飞桨PaddlePaddle",
                    download_count=last_val
                ))

            except Exception as e:
                print(f"获取 {model_link} 失败: {e}")

            if progress_callback:
                progress_callback(i, discovered_total=total_count)

        driver.quit()
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
