"""
单个模型重新获取模块
复用现有fetcher的create_record方法和逻辑
"""
from datetime import date
import pandas as pd
from huggingface_hub import model_info
from modelscope.hub.api import HubApi
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from ..config import DB_PATH
from ..utils import create_chrome_driver, extract_numbers


class SingleModelFetcher:
    """单模型重新获取器 - 复用BaseFetcher的create_record逻辑"""

    def __init__(self, platform_name, target_date=None):
        self.platform_name = platform_name
        self.target_date = target_date if target_date else date.today().isoformat()

    def create_record(self, model_name, publisher, download_count, search_keyword=None,
                     created_at=None, last_modified=None, url=None):
        """
        创建一条记录（与BaseFetcher完全一致）
        """
        record = {
            "date": self.target_date,  # 使用target_date而不是today
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
        if url:
            record["url"] = url
        return record

    def get_url_from_db(self, model_name, publisher):
        """从数据库获取模型URL（忽略publisher大小写）"""
        import sqlite3
        try:
            conn = sqlite3.connect(DB_PATH)
            # 使用LOWER()忽略publisher大小写匹配
            query = """
                SELECT url FROM model_downloads
                WHERE repo = ? AND model_name = ? AND LOWER(publisher) = LOWER(?)
                AND url IS NOT NULL AND url != ''
                ORDER BY date DESC
                LIMIT 1
            """
            cursor = conn.execute(query, (self.platform_name, model_name, publisher))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"  [{self.platform_name}] 从数据库获取URL失败: {e}")
            return None

    def refetch(self, model_name, publisher):
        """
        重新获取单个模型的下载量

        Returns:
            dict: record字典，失败返回None
        """
        raise NotImplementedError("子类必须实现refetch方法")


class HuggingFaceSingleFetcher(SingleModelFetcher):
    """Hugging Face单模型获取器"""

    def __init__(self, target_date=None):
        super().__init__("Hugging Face", target_date)

    def refetch(self, model_name, publisher):
        """从Hugging Face API重新获取单个模型"""
        try:
            model_id = f"{publisher}/{model_name}"
            info = model_info(model_id, expand=["downloadsAllTime"])
            downloads = getattr(info, 'downloads_all_time', 0)
            url = f"https://huggingface.co/{model_id}"

            print(f"  [HF] {model_name}: {downloads:,}")
            return self.create_record(
                model_name=model_name,
                publisher=publisher,
                download_count=downloads,
                url=url
            )
        except Exception as e:
            print(f"  [HF] 获取失败: {e}")
            return None


class ModelScopeSingleFetcher(SingleModelFetcher):
    """ModelScope单模型获取器"""

    def __init__(self, target_date=None):
        super().__init__("ModelScope", target_date)

    def refetch(self, model_name, publisher):
        """从ModelScope API重新获取单个模型"""
        try:
            model_id = f"{publisher}/{model_name}"
            api = HubApi()
            info = api.get_model(model_id, revision="master")
            downloads = info.get("Downloads", 0)
            url = f"https://modelscope.cn/models/{model_id}"

            print(f"  [ModelScope] {model_name}: {downloads:,}")
            return self.create_record(
                model_name=model_name,
                publisher=publisher,
                download_count=downloads,
                url=url
            )
        except Exception as e:
            print(f"  [ModelScope] 获取失败: {e}")
            return None


class AIStudioSingleFetcher(SingleModelFetcher):
    """AI Studio单模型获取器 - 使用Selenium访问详情页"""

    def __init__(self, target_date=None):
        super().__init__("AI Studio", target_date)

    def refetch(self, model_name, publisher):
        """从AI Studio详情页重新获取单个模型"""
        url = self.get_url_from_db(model_name, publisher)
        if not url:
            print(f"  [AI Studio] 没有URL，跳过")
            return None

        driver = None
        try:
            driver = create_chrome_driver()
            wait = WebDriverWait(driver, 40)

            print(f"  [AI Studio] 访问: {url}")
            driver.get(url)
            time.sleep(8)  # 增加等待时间，确保JavaScript加载完成

            # 使用正确的XPath：查找包含"使用量"的元素
            downloads_xpath = "//*[contains(text(), '使用量')]"

            try:
                element = driver.find_element(By.XPATH, downloads_xpath)
                downloads_text = element.text.strip()  # 例如："使用量 72458"

                # 提取数字（"使用量 72458" → 72458）
                number = extract_numbers(downloads_text)
                if number is not None:
                    downloads = str(number)
                    print(f"  [AI Studio] 找到使用量: {downloads}")
                else:
                    downloads = "0"
                    print(f"  [AI Studio] 无法提取数字，原始文本: {downloads_text}")
            except Exception as e:
                print(f"  [AI Studio] 获取使用量元素失败: {e}")
                downloads = "0"

            return self.create_record(
                model_name=model_name,
                publisher=publisher,
                download_count=downloads,
                url=url
            )
        except Exception as e:
            print(f"  [AI Studio] 获取失败: {e}")
            return None
        finally:
            if driver:
                driver.quit()


class GitCodeSingleFetcher(SingleModelFetcher):
    """GitCode单模型获取器 - 使用Selenium访问详情页"""

    def __init__(self, target_date=None):
        super().__init__("GitCode", target_date)

    def refetch(self, model_name, publisher):
        """从GitCode详情页重新获取单个模型"""
        url = self.get_url_from_db(model_name, publisher)
        if not url:
            print(f"  [GitCode] 没有URL，跳过")
            return None

        driver = None
        try:
            driver = create_chrome_driver()
            wait = WebDriverWait(driver, 40)

            print(f"  [GitCode] 访问: {url}")
            driver.get(url)
            time.sleep(3)

            # 使用与现有fetcher一致的XPath
            downloads_xpath = '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[1]/div[1]/div/div[2]'
            downloads = "0"

            try:
                downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                # 等待数值稳定
                last_val = ""
                for _ in range(5):
                    downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                    val = downloads_element.text.strip().replace(',', '')
                    if val and val != last_val:
                        last_val = val
                        time.sleep(1)
                    else:
                        break
                downloads = last_val
            except Exception as e:
                print(f"  [GitCode] 获取下载量元素失败: {e}")

            print(f"  [GitCode] {model_name}: {downloads}")
            return self.create_record(
                model_name=model_name,
                publisher=publisher,
                download_count=downloads,
                url=url
            )
        except Exception as e:
            print(f"  [GitCode] 获取失败: {e}")
            return None
        finally:
            if driver:
                driver.quit()


# 单模型fetcher工厂
SINGLE_MODEL_FETCHERS = {
    "Hugging Face": HuggingFaceSingleFetcher,
    "ModelScope": ModelScopeSingleFetcher,
    "AI Studio": AIStudioSingleFetcher,
    "GitCode": GitCodeSingleFetcher,
}


def refetch_single_model(repo, model_name, publisher, target_date=None):
    """
    重新获取单个模型的下载量

    Args:
        repo: 平台名称
        model_name: 模型名称
        publisher: 发布者
        target_date: 目标日期（保存到数据库的日期）

    Returns:
        dict: record字典，失败返回None
    """
    fetcher_class = SINGLE_MODEL_FETCHERS.get(repo)
    if not fetcher_class:
        print(f"  ⚠️ 平台 {repo} 暂不支持重新获取")
        return None

    fetcher = fetcher_class(target_date=target_date)
    return fetcher.refetch(model_name, publisher)


def refetch_models_batch(negative_growth_list, target_date=None):
    """
    批量重新获取负增长模型的下载量

    Args:
        negative_growth_list: 负增长模型列表，每个元素为字典，包含:
            - platform: 平台
            - model_name: 模型名称
            - publisher: 发布者
            - current: 当前下载量（用于对比）
        target_date: 目标日期（保存到数据库的日期）

    Returns:
        tuple: (成功更新的列表, 失败的列表)
    """
    success_list = []
    failed_list = []

    for item in negative_growth_list:
        repo = item['platform']
        model_name = item['model_name']
        publisher = item['publisher']
        old_count = item['current']

        record = refetch_single_model(repo, model_name, publisher, target_date=target_date)

        if record:
            try:
                # 正确处理download_count（转换为整数）
                raw_count = record['download_count']
                try:
                    new_count = int(raw_count)
                except (ValueError, TypeError):
                    new_count = 0

                change = new_count - old_count

                result = {
                    'platform': repo,
                    'model_name': model_name,
                    'publisher': publisher,
                    'old_count': old_count,
                    'new_count': new_count,
                    'change': change,
                    'record': record
                }

                success_list.append(result)
            except Exception as e:
                print(f"  ❌ 处理结果失败: {e}")
                import traceback
                traceback.print_exc()
                failed_list.append(item)
        else:
            failed_list.append(item)

    return success_list, failed_list
