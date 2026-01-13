"""基于 Selenium 的爬虫实现 - AI Studio, Gitee, Modelers"""
import time
import re
from .base_fetcher import BaseFetcher
from ..utils import create_chrome_driver, is_simplified_count, extract_numbers
from ..config import SELENIUM_TIMEOUT
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException


class AIStudioFetcher(BaseFetcher):
    """AI Studio 爬虫"""

    def __init__(self):
        super().__init__("AI Studio")

    def _get_detailed_download_count(self, driver, card_index):
        """获取详细下载量（点击进入详情页）"""
        try:
            current_url = driver.current_url
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")

            if card_index >= len(cards):
                return None

            card = cards[card_index]
            card.click()
            time.sleep(1)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)

            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH,
                    "//*[@id='main']/div[1]/div[2]/div/div/div[1]/div/div[1]/div[4]/div[2]"))
            )
            detailed_count = extract_numbers(element.text)

            driver.back()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
            )

            return detailed_count

        except Exception as e:
            print(f"获取详情页下载量失败: {e}")
            try:
                if driver.current_url != current_url:
                    driver.back()
            except:
                pass
            return None

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 AI Studio 数据"""
        max_retries = 3
        for attempt in range(max_retries):
            driver = None
            try:
                print(f"[AI Studio] 尝试第 {attempt + 1} 次爬取...")
                driver = create_chrome_driver()
                wait = WebDriverWait(driver, SELENIUM_TIMEOUT)

                processed_count = 0

                # 使用ERNIE-4.5和PaddleOCR-VL作为搜索词
                search_terms = ["ERNIE-4.5", "PaddleOCR-VL"]

                for search_term in search_terms:
                    print(f"[AI Studio] 搜索 {search_term} 相关模型...")
                    url = f"https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q={search_term}"
                    driver.get(url)

                    # 等待页面加载并检查是否成功
                    try:
                        wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                        ))
                        print(f"[AI Studio] {search_term} 页面加载成功")
                    except TimeoutException:
                        print(f"[AI Studio] {search_term} 页面加载超时，可能是网络问题或页面结构变化")
                        if attempt < max_retries - 1:
                            continue
                        raise

                    # 开始爬取数据
                    while True:
                        wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                        ))
                        old_container = driver.find_element(By.CSS_SELECTOR, "div.ai-model-list-wapper")

                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                        print(f"[AI Studio] {search_term} 当前页面有 {len(cards)} 个卡片")

                        for i in range(len(cards)):
                            try:
                                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                if i >= len(cards):
                                    continue

                                card = cards[i]

                                # 🔧 新增：点击卡片进入详情页获取URL
                                model_url = None
                                try:
                                    # 记录当前URL
                                    current_url = driver.current_url

                                    # 点击卡片
                                    card.click()
                                    time.sleep(2)  # 等待页面加载

                                    # 获取详情页URL
                                    model_url = driver.current_url

                                    # 返回搜索页
                                    driver.back()
                                    time.sleep(1)  # 等待返回

                                    # 重新获取cards引用（返回后可能失效）
                                    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")

                                except Exception as e:
                                    print(f"  获取URL失败: {e}")
                                    # 尝试确保返回搜索页
                                    try:
                                        if "modelsoverview" not in driver.current_url:
                                            driver.back()
                                            time.sleep(1)
                                    except:
                                        pass
                                    # 重新获取cards
                                    try:
                                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                    except:
                                        pass

                                full_model_name = card.find_element(
                                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                                ).text.strip()

                                # 🔧 修改：获取所有detail元素，提取下载量和时间
                                detail_items = card.find_elements(
                                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-detail-one-item-tip"
                                )

                                # 获取下载量（第1个tip）
                                usage_count = detail_items[0].find_element(
                                    By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
                                ).text.strip()

                                # 🔧 新增：获取更新时间（第3个tip）
                                last_modified = None
                                if len(detail_items) >= 3:
                                    try:
                                        last_modified = detail_items[2].find_element(
                                            By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
                                        ).text.strip()
                                    except Exception as e:
                                        print(f"获取更新时间失败: {e}")
                                        last_modified = None

                                publisher = card.find_element(
                                    By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-publisher"
                                ).text.strip()

                                final_usage_count = usage_count
                                if is_simplified_count(usage_count):
                                    final_usage_count = self._get_detailed_download_count(driver, i)

                                # 处理模型名称和发布者
                                model_name = full_model_name

                                # 如果模型名称以PaddlePaddle/开头，提取出真正的模型名
                                if model_name.startswith("PaddlePaddle/"):
                                    model_name = model_name[len("PaddlePaddle/"):]
                                    # 确保publisher是PaddlePaddle
                                    if publisher not in ["PaddlePaddle", "PaddleOCR-VL"]:
                                        publisher = "PaddlePaddle"

                                # 修复重复的PaddlePaddle路径问题
                                if publisher.startswith("PaddlePaddle/PaddlePaddle/"):
                                    publisher = publisher.replace("PaddlePaddle/PaddlePaddle/", "PaddlePaddle/")

                                # 确保只包含ERNIE-4.5和PaddleOCR-VL相关模型
                                if ("ERNIE-4.5" in model_name or "PaddleOCR-VL" in model_name or
                                    "ernie-4.5" in model_name or "paddleocr-vl" in model_name):

                                    self.results.append(self.create_record(
                                        model_name=model_name,
                                        publisher=publisher,
                                        download_count=final_usage_count,
                                        search_keyword=search_term,
                                        last_modified=last_modified,  # 更新时间
                                        url=model_url  # 新增：模型详情页URL
                                    ))

                                    processed_count += 1
                                    if progress_callback:
                                        progress_callback(processed_count)

                            except Exception as e:
                                print(f"处理卡片 {i} 时出错: {e}")
                                continue

                        try:
                            next_page_button = driver.find_element(
                                By.CSS_SELECTOR, "li.ant-pagination-next button"
                            )
                            if not next_page_button.is_enabled():
                                print(f"[AI Studio] {search_term} 最后一页")
                                break

                            next_page_button.click()
                            wait.until(EC.staleness_of(old_container))
                        except Exception as e:
                            print(f"[AI Studio] {search_term} 翻页时出错: {e}")
                            break

                print(f"[AI Studio] 第 {attempt + 1} 次爬取成功，共处理 {processed_count} 个模型")
                break  # 成功完成，跳出重试循环

            except WebDriverException as e:
                print(f"[AI Studio] WebDriver异常: {e}")
                if "Message: Stacktrace" in str(e):
                    print("[AI Studio] ChromeDriver崩溃，准备重试...")
                if attempt < max_retries - 1:
                    continue
                raise
            except Exception as e:
                print(f"[AI Studio] 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    print(f"[AI Studio] 将在5秒后进行第 {attempt + 2} 次尝试...")
                    time.sleep(5)
                else:
                    print("[AI Studio] 所有尝试均失败")
                    raise

            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass

        return self.to_dataframe(), processed_count


class GiteeFetcher(BaseFetcher):
    """Gitee AI 爬虫"""

    def __init__(self):
        super().__init__("Gitee")

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取 Gitee AI 数据"""
        driver = create_chrome_driver()
        total_count = 0
        search_terms = ["ERNIE-4.5", "PaddleOCR-VL"]
        seen = set()

        try:
            for search_term in search_terms:
                page = 1
                while True:
                    url = f"https://ai.gitee.com/models?q={search_term}&p={page}"
                    driver.get(url)
                    time.sleep(2)

                    model_links = driver.find_elements(
                        By.CSS_SELECTOR, "main section div.relative > div > a"
                    )
                    if not model_links:
                        break

                    for link in model_links:
                        try:
                            publisher = link.find_element(
                                By.CSS_SELECTOR,
                                "div div.flex.flex-col.items-start.gap-1.self-stretch.overflow-hidden "
                                "div.flex.items-center.gap-2.self-stretch div span:nth-child(1)"
                            ).text.strip().rstrip('/')

                            model_name = link.find_element(
                                By.CSS_SELECTOR,
                                "div.line-clamp-1.break-all.text-lg.font-medium.leading-7.text-slate-auto-900"
                            ).get_attribute('title').strip()

                            key = (publisher, model_name)
                            if key in seen:
                                continue
                            seen.add(key)

                            download_count = link.find_element(
                                By.CSS_SELECTOR,
                                "div.flex.items-center.gap-2.self-stretch.pt-2.md\\:gap-3 > div:nth-child(2) > div"
                            ).text.strip()

                            self.results.append(self.create_record(
                                model_name=model_name,
                                publisher=publisher,
                                download_count=download_count,
                                search_keyword=search_term
                            ))

                            total_count += 1
                            if progress_callback:
                                progress_callback(total_count, discovered_total=progress_total)

                        except Exception as e:
                            print(f"[Gitee] 处理模型时出错: {e}")
                            continue

                    page += 1
        finally:
            driver.quit()

        return self.to_dataframe(), total_count


class ModelersFetcher(BaseFetcher):
    """魔乐 Modelers 爬虫"""

    def __init__(self):
        super().__init__("魔乐 Modelers")

    def fetch(self, progress_callback=None, progress_total=None):
        """抓取魔乐数据"""
        driver = create_chrome_driver()
        wait = WebDriverWait(driver, 20)
        total_models = 0
        search_terms = ["ERNIE-4.5", "PaddleOCR-VL"]
        seen = set()

        try:
            for search_term in search_terms:
                page = 1
                while True:
                    url = f"https://modelers.cn/models?name={search_term}&page={page}&size=64"
                    driver.get(url)

                    try:
                        wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.cards-content")
                        ))
                        cards = driver.find_elements(By.CSS_SELECTOR, "div.cards-content > a")
                    except:
                        print(f"[魔乐 Modelers] {search_term} 到达最后一页 (page {page})")
                        break

                    if not cards:
                        break

                    for card in cards:
                        try:
                            full_title = card.find_element(
                                By.CSS_SELECTOR, "div.title"
                            ).get_attribute('title').strip()

                            download_count = card.find_element(
                                By.CSS_SELECTOR, "div.repo-card-footer-right span.value"
                            ).text.strip()

                            publisher, model_name = "N/A", full_title
                            if " / " in full_title:
                                parts = full_title.split(' / ', 1)
                                publisher = parts[0]
                                model_name = parts[1]

                            key = (publisher, model_name)
                            if key in seen:
                                continue
                            seen.add(key)

                            self.results.append(self.create_record(
                                model_name=model_name,
                                publisher=publisher,
                                download_count=download_count,
                                search_keyword=search_term
                            ))

                            total_models += 1
                            if progress_callback:
                                progress_callback(total_models, discovered_total=progress_total)

                        except Exception as e:
                            print(f"处理模型卡片失败 - {e}")
                            continue

                    page += 1
        finally:
            driver.quit()

        return self.to_dataframe(), total_models
