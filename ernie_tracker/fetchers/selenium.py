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

    def _get_detailed_info(self, driver, card, card_index):
        """获取详情页信息（URL和详细下载量）- 点击并返回

        Args:
            driver: WebDriver instance
            card: WebElement of the model card
            card_index: Index of the card (for debugging)

        Returns:
            tuple: (detailed_count, model_url) 或 (None, None)

        注意：点击卡片返回后，AI Studio会回到第一页（URL不变），调用方需要处理
        """
        try:
            print(f"    [详情页] 点击卡片 {card_index} 进入详情页...")

            # 尝试关闭横幅广告
            self._close_banner(driver)

            # 滚动到卡片位置，确保可见
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
            time.sleep(0.5)

            # 使用JavaScript点击，避免被遮挡
            try:
                driver.execute_script("arguments[0].click();", card)
            except Exception as e:
                print(f"    [详情页] JavaScript点击失败，尝试普通点击: {e}")
                card.click()

            time.sleep(1)

            # 等待详情页加载
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)

            # 获取详情页URL和下载量
            model_url = driver.current_url
            print(f"    [详情页] ✅ 获取详情页URL: {model_url}")

            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH,
                    "//*[@id='main']/div[1]/div[2]/div/div/div[1]/div/div[1]/div[4]/div[2]"))
            )
            detailed_count = extract_numbers(element.text)
            print(f"    [详情页] ✅ 获取详细下载量: {detailed_count}")

            # 返回搜索页
            print(f"    [详情页] 返回搜索页...")
            driver.back()
            time.sleep(1)

            # 等待搜索页加载
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
                )
            except:
                print(f"    [详情页] ⚠️  等待搜索页超时")

            time.sleep(0.5)

            print(f"    [详情页] ✅ 已返回搜索页")

            return detailed_count, model_url

        except Exception as e:
            print(f"    [详情页] ❌ 获取详情页信息失败: {e}")
            import traceback
            traceback.print_exc()
            # 尝试返回
            try:
                driver.back()
                time.sleep(1)
            except:
                pass
            return None, None

    def _close_banner(self, driver):
        """尝试关闭横幅广告"""
        try:
            # 优先使用准确的关闭按钮选择器
            close_button_selectors = [
                "#main > div.a-s-6th-footer-banner-wrapper > a > span",  # 用户提供的准确路径
                "div.a-s-6th-footer-banner-wrapper > a > span",  # 简化版本
                ".a-s-6th-footer-banner-wrapper a span",  # 更宽松的选择器
            ]

            for selector in close_button_selectors:
                try:
                    close_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    if close_buttons:
                        close_buttons[0].click()
                        print(f"    [关闭横幅] 已点击关闭按钮: {selector}")
                        time.sleep(0.5)
                        return True
                except:
                    continue

            # 如果找不到关闭按钮，使用JavaScript移除整个横幅
            try:
                driver.execute_script("""
                    var bannerWrapper = document.querySelector('div.a-s-6th-footer-banner-wrapper');
                    if (bannerWrapper) {
                        bannerWrapper.style.display = 'none';
                    }
                """)
                print(f"    [关闭横幅] 已使用JavaScript隐藏横幅wrapper")
                return True
            except:
                pass

        except Exception as e:
            pass

        return False

    def _restore_to_page(self, driver, page_first_model):
        """在回到第一页后，重新翻页到目标页

        Args:
            driver: WebDriver instance
            page_first_model: 目标页第一个模型的名称

        Returns:
            bool: 是否成功恢复到目标页
        """
        try:
            print(f"    [恢复页] 检测到回到第一页，正在重新翻页到目标页...")
            print(f"    [恢复页] 目标页标识（第一个模型）: {page_first_model}")

            max_page_clicks = 50  # 最多点击50页，防止无限循环
            page_clicks = 0

            while page_clicks < max_page_clicks:
                # 获取当前页第一个模型
                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                if len(cards) == 0:
                    print(f"    [恢复页] ❌ 没有找到卡片")
                    return False

                current_first = cards[0].find_element(
                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                ).text.strip()

                if current_first == page_first_model:
                    print(f"    [恢复页] ✅ 已恢复到目标页")
                    return True

                # 点击下一页
                try:
                    # 尝试关闭横幅广告
                    self._close_banner(driver)

                    next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
                    if not next_button.is_enabled():
                        print(f"    [恢复页] ❌ 已到最后一页，但未找到目标页")
                        return False

                    print(f"    [恢复页] 点击下一页 ({page_clicks + 1}/{max_page_clicks})...")

                    # 使用JavaScript点击
                    try:
                        driver.execute_script("arguments[0].click();", next_button)
                    except:
                        next_button.click()

                    time.sleep(1)
                except:
                    print(f"    [恢复页] ❌ 无法找到或点击下一页按钮")
                    return False

                page_clicks += 1

            print(f"    [恢复页] ❌ 超过最大翻页次数，未找到目标页")
            return False

        except Exception as e:
            print(f"    [恢复页] ❌ 恢复页失败: {e}")
            import traceback
            traceback.print_exc()
            return False

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
                processed_models = set()  # 记录已处理模型的名称（用于去重）

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
                    page_num = 1
                    while True:
                        wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                        ))

                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                        print(f"[AI Studio] {search_term} 第{page_num}页，有 {len(cards)} 个卡片")

                        # 记录当前页第一个模型（用于恢复）
                        if len(cards) > 0:
                            page_first_model = cards[0].find_element(
                                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                            ).text.strip()
                            print(f"[AI Studio] 当前页标识（第一个模型）: {page_first_model}")
                        else:
                            page_first_model = None
                            print(f"[AI Studio] ⚠️  当前页没有卡片，跳过")
                            break

                        for i in range(len(cards)):
                            try:
                                print(f"[AI Studio] ========== 处理卡片 {i}/{len(cards)-1} ==========")

                                # 重新获取cards（因为可能已过时）
                                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                if i >= len(cards):
                                    print(f"[AI Studio] ⚠️  i={i} 超出cards范围({len(cards)})，跳过")
                                    break

                                card = cards[i]
                                full_model_name = card.find_element(
                                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                                ).text.strip()
                                print(f"[AI Studio] 模型名称: {full_model_name}")

                                # 检查是否已处理过
                                if full_model_name in processed_models:
                                    print(f"[AI Studio] ⏭️  模型已处理过，跳过: {full_model_name}")
                                    continue

                                # 获取下载量和时间
                                detail_items = card.find_elements(
                                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-detail-one-item-tip"
                                )

                                # 获取下载量（第1个tip）
                                usage_count = detail_items[0].find_element(
                                    By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
                                ).text.strip()

                                # 获取更新时间（第3个tip）
                                last_modified = None
                                if len(detail_items) >= 3:
                                    try:
                                        last_modified = detail_items[2].find_element(
                                            By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-like"
                                        ).text.strip()
                                    except Exception as e:
                                        print(f"获取更新时间失败: {e}")

                                publisher = card.find_element(
                                    By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-publisher"
                                ).text.strip()
                                print(f"[AI Studio] 发布者: {publisher}, 下载量: {usage_count}")

                                # 点击获取URL和详细下载量
                                final_usage_count, model_url = self._get_detailed_info(driver, card, i)

                                # 检查是否回到了第一页，如果是则恢复到目标页
                                if page_first_model:
                                    # 等待页面稳定，重新获取cards
                                    time.sleep(0.5)
                                    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")

                                    # 如果没有卡片，刷新页面重新加载
                                    retry_count = 0
                                    while len(cards) == 0 and retry_count < 2:
                                        print(f"[AI Studio] ⚠️  返回后页面没有卡片，刷新页面... ({retry_count + 1}/2)")
                                        driver.refresh()
                                        time.sleep(1.5)
                                        wait.until(EC.presence_of_element_located(
                                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                                        ))
                                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                        retry_count += 1

                                    if len(cards) == 0:
                                        print(f"[AI Studio] ❌ 刷新后仍没有卡片，跳过剩余卡片")
                                        break

                                    if len(cards) > 0:
                                        current_first = cards[0].find_element(
                                            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                                        ).text.strip()

                                        if current_first != page_first_model:
                                            print(f"[AI Studio] ⚠️  检测到回到第一页，正在恢复...")
                                            print(f"[AI Studio]   目标页第一个模型: {page_first_model}")
                                            print(f"[AI Studio]   当前页第一个模型: {current_first}")

                                            if not self._restore_to_page(driver, page_first_model):
                                                print(f"[AI Studio] ❌ 恢复页失败，跳过剩余卡片")
                                                break

                                            # 恢复后重新获取cards
                                            time.sleep(0.5)
                                            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                            if i >= len(cards):
                                                print(f"[AI Studio] ⚠️  恢复后i={i}超出范围，跳过")
                                                break

                                print(f"[AI Studio] ✅ 获取详情页结果: 下载量={final_usage_count}, URL={model_url}")

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
                                        last_modified=last_modified,
                                        url=model_url  # 新增：模型详情页URL
                                    ))

                                    # 记录已处理
                                    processed_models.add(full_model_name)
                                    processed_count += 1
                                    if progress_callback:
                                        progress_callback(processed_count)

                            except Exception as e:
                                print(f"[AI Studio] ❌ 处理卡片 {i} 时出错: {e}")
                                import traceback
                                traceback.print_exc()
                                continue

                        print(f"[AI Studio] ===== 当前页所有卡片处理完成，准备翻页 =====")

                        try:
                            # 尝试关闭横幅广告
                            self._close_banner(driver)

                            next_page_button = driver.find_element(
                                By.CSS_SELECTOR, "li.ant-pagination-next button"
                            )
                            if not next_page_button.is_enabled():
                                print(f"[AI Studio] ✓  {search_term} 到达最后一页")
                                break

                            print(f"[AI Studio] 点击翻页按钮...")

                            # 使用JavaScript点击翻页按钮，避免被遮挡
                            try:
                                driver.execute_script("arguments[0].click();", next_page_button)
                            except Exception as e:
                                print(f"[AI Studio] JavaScript点击翻页失败，尝试普通点击: {e}")
                                next_page_button.click()

                            # 等待翻页：等待新页面的卡片容器出现
                            time.sleep(1.5)

                            # 等待新页面加载
                            wait.until(EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                            ))

                            page_num += 1
                            print(f"[AI Studio] ✓  翻页完成，当前第{page_num}页")
                        except Exception as e:
                            print(f"[AI Studio] ❌ 翻页时出错: {e}")
                            import traceback
                            traceback.print_exc()
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
                    print(f"[AI Studio] 将在2秒后进行第 {attempt + 2} 次尝试...")
                    time.sleep(2)
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
