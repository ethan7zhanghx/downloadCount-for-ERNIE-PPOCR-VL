"""基于 Selenium 的爬虫实现 - AI Studio, Gitee, Modelers"""
import time
import re
import logging
from datetime import datetime
from .base_fetcher import BaseFetcher
from ..utils import create_chrome_driver, is_simplified_count, extract_numbers
from ..config import SELENIUM_TIMEOUT
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# 配置详细的日志记录器
def setup_detailed_logger(name):
    """设置带时间戳的详细日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler - 记录所有详细信息
        file_handler = logging.FileHandler(f'aistudio_crawl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 控制台handler - 只显示重要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


class AIStudioFetcher(BaseFetcher):
    """AI Studio 爬虫"""

    def __init__(self, test_mode=False, enable_detailed_log=False):
        """
        Args:
            test_mode: 测试模式，每个页面只获取第一个和最后一个模型
            enable_detailed_log: 启用详细日志（仅用于调试，默认关闭）
        """
        super().__init__("AI Studio")
        self.test_mode = test_mode
        self.enable_detailed_log = enable_detailed_log
        self.logger = None

        if self.enable_detailed_log:
            self.logger = setup_detailed_logger("AIStudioFetcher")
            self._log_info("=" * 80)
            self._log_info(f"AIStudioFetcher 初始化完成 | 测试模式: {self.test_mode} | 详细日志: {self.enable_detailed_log}")
            self._log_info("=" * 80)

    def _log(self, level, message):
        """记录日志（如果启用了详细日志）"""
        if self.logger:
            # 在消息前添加时间戳
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # 毫秒级
            message_with_time = f"[{timestamp}] {message}"
            getattr(self.logger, level)(message_with_time)

    def _log_debug(self, message):
        self._log('debug', message)

    def _log_info(self, message):
        self._log('info', message)

    def _log_warning(self, message):
        self._log('warning', message)

    def _log_error(self, message):
        self._log('error', message)

    def _is_simplified_count(self, count_str):
        """判断下载量是否为简化形式（如1k+, 7.2w, 7.3千等），需要获取精确值

        Args:
            count_str: 下载量字符串，如 "72456", "72.4k", "7.2w", "1k+"

        Returns:
            bool: 是否为简化形式
        """
        if not count_str:
            return False

        count_str = str(count_str).strip()

        # 纯数字（可能包含逗号）视为精确值
        if count_str.replace(',', '').replace('.', '').replace('+', '').isdigit():
            # 检查是否包含小数点（如 "7.3" 不是简化格式）
            if '.' in count_str:
                # 有小数点的数字，可能被截断了，需要获取完整值
                return True
            return False

        # 检查是否包含简化标记
        count_str_upper = count_str.upper()
        simplified_markers = ['K', 'W', 'M', '+', '千', '万', '亿']
        return any(marker in count_str_upper for marker in simplified_markers)

    def _parse_download_count(self, count_str):
        """解析下载量字符串，转换为数字

        Args:
            count_str: 下载量字符串，可能是 "72456", "72.4k", "7.2w" 等

        Returns:
            int: 解析后的数字
        """
        if not count_str:
            return 0

        count_str = str(count_str).strip().upper()

        # 移除逗号等分隔符
        count_str = count_str.replace(',', '')

        # 处理 k/K (千)
        if 'K' in count_str:
            num = float(count_str.replace('K', '').replace('K', ''))
            return int(num * 1000)

        # 处理 w/W (万)
        if 'W' in count_str:
            num = float(count_str.replace('W', '').replace('W', ''))
            return int(num * 10000)

        # 纯数字
        try:
            return int(count_str)
        except:
            return 0

    def _validate_download_count(self, list_count_str, detail_count_int):
        """验证列表页和详情页下载量是否匹配

        Args:
            list_count_str: 列表页显示的下载量（可能是 "72.4k"）
            detail_count_int: 详情页的下载量（可能是 72456）

        Returns:
            tuple: (is_valid, reason)
                - is_valid: 是否匹配
                - reason: 不匹配的原因
        """
        list_count = self._parse_download_count(list_count_str)

        # 允许 10% 的误差（因为可能有人在下载数据）
        tolerance = 0.1
        min_expected = list_count * (1 - tolerance)
        max_expected = list_count * (1 + tolerance)

        if detail_count_int < min_expected:
            return False, f"详情页下载量({detail_count_int}) < 列表页({list_count_str}={list_count})的{1-tolerance:.0%}"

        if detail_count_int > max_expected:
            return False, f"详情页下载量({detail_count_int}) > 列表页({list_count_str}={list_count})的{1+tolerance:.0%}，可能不是同一时间点的数据"

        return True, "OK"

    def _get_detailed_info(self, driver, card, card_index, list_usage_count=None):
        """获取详情页信息（URL和详细下载量）- 点击并返回

        Args:
            driver: WebDriver instance
            card: WebElement of the model card
            card_index: Index of the card (for debugging)
            list_usage_count: 列表页的下载量（用于核对）

        Returns:
            tuple: (detailed_count, model_url) 或 (None, None)

        注意：点击卡片返回后，AI Studio会回到第一页（URL不变），调用方需要处理
        """
        start_time = time.time()
        self._log_debug(f"  [详情页 #{card_index}] 开始获取详情页信息")
        if list_usage_count:
            self._log_debug(f"  [详情页 #{card_index}] 列表页下载量: {list_usage_count}")

        try:
            self._log_debug(f"  [详情页 #{card_index}] 滚动到卡片位置")
            scroll_start = time.time()
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
            time.sleep(0.5)
            self._log_debug(f"  [详情页 #{card_index}] 滚动完成 (耗时: {(time.time() - scroll_start)*1000:.2f}ms)")

            # 使用JavaScript点击，避免被遮挡
            click_start = time.time()
            try:
                self._log_debug(f"  [详情页 #{card_index}] 尝试JavaScript点击")
                driver.execute_script("arguments[0].click();", card)
                self._log_debug(f"  [详情页 #{card_index}] JavaScript点击成功 (耗时: {(time.time() - click_start)*1000:.2f}ms)")
            except Exception as e:
                self._log_warning(f"  [详情页 #{card_index}] JavaScript点击失败: {e}，尝试普通点击")
                card.click()
                self._log_debug(f"  [详情页 #{card_index}] 普通点击完成 (耗时: {(time.time() - click_start)*1000:.2f}ms)")

            sleep_start = time.time()
            time.sleep(1)
            self._log_debug(f"  [详情页 #{card_index}] 点击后等待完成 (耗时: {(time.time() - sleep_start)*1000:.2f}ms)")

            # 等待详情页加载
            self._log_debug(f"  [详情页 #{card_index}] 等待详情页body元素出现")
            wait_start = time.time()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self._log_debug(f"  [详情页 #{card_index}] body元素已出现 (耗时: {(time.time() - wait_start)*1000:.2f}ms)")

            time.sleep(1)

            # 获取详情页URL和下载量
            url_start = time.time()
            model_url = driver.current_url
            self._log_info(f"  [详情页 #{card_index}] ✅ 获取URL: {model_url} (耗时: {(time.time() - url_start)*1000:.2f}ms)")

            # 获取下载量（带重试机制）
            max_retries = 3
            detailed_count = None

            for retry_count in range(max_retries):
                self._log_debug(f"  [详情页 #{card_index}] 第 {retry_count + 1} 次尝试获取下载量")

                self._log_debug(f"  [详情页 #{card_index}] 等待下载量元素出现")
                element_wait_start = time.time()
                # 使用更灵活的XPath：查找包含"使用量"的元素
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(text(), '使用量')]"))
                )
                self._log_debug(f"  [详情页 #{card_index}] 下载量元素已出现 (耗时: {(time.time() - element_wait_start)*1000:.2f}ms)")

                extract_start = time.time()
                element_text = element.text.strip()
                self._log_debug(f"  [详情页 #{card_index}] 原始文本: {element_text}")

                # 检查是否为简化格式
                if self._is_simplified_count(element_text):
                    self._log_warning(f"  [详情页 #{card_index}] ⚠️  详情页显示简化格式: {element_text}")

                    # 尝试等待更长时间，看是否会更新为精确值
                    if retry_count < max_retries - 1:
                        self._log_info(f"  [详情页 #{card_index}] 等待3秒，看是否会更新为精确值...")
                        time.sleep(3)
                        element_text = element.text.strip()
                        self._log_debug(f"  [详情页 #{card_index}] 等待后文本: {element_text}")

                        # 如果仍是简化格式，刷新页面重试
                        if self._is_simplified_count(element_text):
                            self._log_info(f"  [详情页 #{card_index}] 仍为简化格式，刷新页面重试...")
                            driver.refresh()
                            time.sleep(2)
                            continue
                    else:
                        # 最后一次重试仍然失败
                        self._log_error(f"  [详情页 #{card_index}] ❌ 详情页始终显示简化格式，无法获取精确值: {element_text}")
                        self._log_error(f"  [详情页 #{card_index}] 跳过此模型，不保存估算值")
                        return None, None  # 返回 None 表示失败
                else:
                    # 是精确值，提取数字
                    detailed_count = extract_numbers(element_text)
                    self._log_info(f"  [详情页 #{card_index}] ✅ 获取精确下载量: {detailed_count} (提取耗时: {(time.time() - extract_start)*1000:.2f}ms)")
                    break  # 成功获取精确值，退出重试循环

                # 核对列表页和详情页下载量
                if list_usage_count:
                    try:
                        is_valid, reason = self._validate_download_count(list_usage_count, detailed_count)
                        if is_valid:
                            self._log_info(f"  [详情页 #{card_index}] ✅ 下载量核对通过: 列表页={list_usage_count}, 详情页={detailed_count}")
                            break  # 核对通过，退出重试循环
                        else:
                            self._log_warning(f"  [详情页 #{card_index}] ⚠️  下载量核对失败: {reason}")

                            # 如果是最后一次重试，不再重试
                            if retry_count == max_retries - 1:
                                self._log_error(f"  [详情页 #{card_index}] ❌ 已达到最大重试次数({max_retries})，使用详情页数据: {detailed_count}")
                                break

                            # 等待并刷新页面重试
                            self._log_info(f"  [详情页 #{card_index}] 等待2秒后刷新页面重试...")
                            time.sleep(2)
                            driver.refresh()
                            self._log_debug(f"  [详情页 #{card_index}] 页面已刷新，等待重新加载...")
                            time.sleep(2)

                    except Exception as e:
                        self._log_warning(f"  [详情页 #{card_index}] ⚠️  下载量核对异常: {e}")
                        if retry_count == max_retries - 1:
                            break
                else:
                    # 没有列表页数据用于核对，直接使用详情页数据
                    break

            # 返回搜索页
            self._log_debug(f"  [详情页 #{card_index}] 准备返回搜索页")
            back_start = time.time()
            driver.back()
            self._log_debug(f"  [详情页 #{card_index}] driver.back()调用完成 (耗时: {(time.time() - back_start)*1000:.2f}ms)")

            time.sleep(1)

            # 等待搜索页加载
            try:
                self._log_debug(f"  [详情页 #{card_index}] 等待搜索页容器出现")
                search_wait_start = time.time()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
                )
                self._log_debug(f"  [详情页 #{card_index}] 搜索页容器已出现 (耗时: {(time.time() - search_wait_start)*1000:.2f}ms)")
            except:
                self._log_warning(f"  [详情页 #{card_index}] ⚠️  等待搜索页超时")

            time.sleep(0.5)

            total_time = time.time() - start_time
            self._log_info(f"  [详情页 #{card_index}] ✅ 详情页处理完成 (总耗时: {total_time*1000:.2f}ms)")

            return detailed_count, model_url

        except Exception as e:
            self._log_error(f"  [详情页 #{card_index}] ❌ 获取详情页失败: {e} (耗时: {(time.time() - start_time)*1000:.2f}ms)")
            import traceback
            self._log_debug(f"  [详情页 #{card_index}] 异常堆栈:\n{traceback.format_exc()}")
            # 尝试返回
            try:
                self._log_debug(f"  [详情页 #{card_index}] 异常后尝试返回搜索页")
                driver.back()
                time.sleep(1)
            except:
                self._log_error(f"  [详情页 #{card_index}] 返回搜索页也失败了")
            return None, None

    def _close_banner(self, driver):
        """尝试关闭横幅广告"""
        start_time = time.time()
        self._log_debug(f"    [关闭横幅] 尝试关闭横幅广告")

        try:
            # 优先使用准确的关闭按钮选择器
            close_button_selectors = [
                "#main > div.a-s-6th-footer-banner-wrapper > a > span",  # 用户提供的准确路径
                "div.a-s-6th-footer-banner-wrapper > a > span",  # 简化版本
                ".a-s-6th-footer-banner-wrapper a span",  # 更宽松的选择器
            ]

            for idx, selector in enumerate(close_button_selectors):
                try:
                    self._log_debug(f"    [关闭横幅] 尝试选择器 #{idx+1}: {selector}")
                    close_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    if close_buttons:
                        self._log_debug(f"    [关闭横幅] 找到 {len(close_buttons)} 个关闭按钮")
                        close_buttons[0].click()
                        self._log_info(f"    [关闭横幅] ✅ 已点击关闭按钮 (选择器: #{idx+1}, 耗时: {(time.time() - start_time)*1000:.2f}ms)")
                        time.sleep(0.5)
                        return True
                    else:
                        self._log_debug(f"    [关闭横幅] 选择器 #{idx+1} 未找到元素")
                except Exception as e:
                    self._log_debug(f"    [关闭横幅] 选择器 #{idx+1} 失败: {e}")
                    continue

            # 如果找不到关闭按钮，使用JavaScript移除整个横幅
            try:
                self._log_debug(f"    [关闭横幅] 尝试使用JavaScript隐藏横幅")
                driver.execute_script("""
                    var bannerWrapper = document.querySelector('div.a-s-6th-footer-banner-wrapper');
                    if (bannerWrapper) {
                        bannerWrapper.style.display = 'none';
                    }
                """)
                self._log_info(f"    [关闭横幅] ✅ 已使用JavaScript隐藏横幅wrapper (耗时: {(time.time() - start_time)*1000:.2f}ms)")
                return True
            except Exception as e:
                self._log_debug(f"    [关闭横幅] JavaScript隐藏失败: {e}")

        except Exception as e:
            self._log_warning(f"    [关闭横幅] ⚠️  关闭横幅过程异常: {e}")

        self._log_debug(f"    [关闭横幅] 未找到或无法关闭横幅 (耗时: {(time.time() - start_time)*1000:.2f}ms)")
        return False

    def _restore_to_page(self, driver, page_first_model):
        """在回到第一页后，重新翻页到目标页

        Args:
            driver: WebDriver instance
            page_first_model: 目标页第一个模型的名称

        Returns:
            bool: 是否成功恢复到目标页
        """
        start_time = time.time()
        self._log_warning(f"    [恢复页] 检测到回到第一页，开始恢复到目标页")
        self._log_info(f"    [恢复页] 目标页标识（第一个模型）: {page_first_model}")

        try:
            max_page_clicks = 50  # 最多点击50页，防止无限循环
            page_clicks = 0

            while page_clicks < max_page_clicks:
                # 获取当前页第一个模型
                self._log_debug(f"    [恢复页] 第 {page_clicks + 1} 次尝试：获取当前页卡片")
                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                if len(cards) == 0:
                    self._log_error(f"    [恢复页] ❌ 没有找到卡片")
                    return False

                self._log_debug(f"    [恢复页] 找到 {len(cards)} 个卡片")
                current_first = cards[0].find_element(
                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                ).text.strip()

                self._log_debug(f"    [恢复页] 当前页第一个模型: {current_first}")

                if current_first == page_first_model:
                    self._log_info(f"    [恢复页] ✅ 已恢复到目标页 (点击次数: {page_clicks + 1}, 耗时: {(time.time() - start_time)*1000:.2f}ms)")
                    return True

                # 点击下一页
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
                    if not next_button.is_enabled():
                        self._log_error(f"    [恢复页] ❌ 已到最后一页，但未找到目标页 (已点击: {page_clicks + 1}次)")
                        return False

                    self._log_info(f"    [恢复页] 点击下一页 ({page_clicks + 1}/{max_page_clicks})...")

                    # 使用JavaScript点击
                    try:
                        click_start = time.time()
                        driver.execute_script("arguments[0].click();", next_button)
                        self._log_debug(f"    [恢复页] JavaScript点击成功 (耗时: {(time.time() - click_start)*1000:.2f}ms)")
                    except Exception as e:
                        self._log_warning(f"    [恢复页] JavaScript点击失败: {e}，尝试普通点击")
                        next_button.click()

                    time.sleep(1)
                except Exception as e:
                    self._log_error(f"    [恢复页] ❌ 无法找到或点击下一页按钮: {e}")
                    return False

                page_clicks += 1

            self._log_error(f"    [恢复页] ❌ 超过最大翻页次数 ({max_page_clicks})，未找到目标页 (耗时: {(time.time() - start_time)*1000:.2f}ms)")
            return False

        except Exception as e:
            self._log_error(f"    [恢复页] ❌ 恢复页失败: {e} (耗时: {(time.time() - start_time)*1000:.2f}ms)")
            import traceback
            self._log_debug(f"    [恢复页] 异常堆栈:\n{traceback.format_exc()}")
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

                # 加载已有URL的模型映射（用于获取已有的URL）
                from ..config import DB_PATH
                import sqlite3
                existing_models_with_url = {}
                try:
                    conn = sqlite3.connect(DB_PATH)
                    existing_query = """
                        SELECT DISTINCT publisher, model_name, url
                        FROM model_downloads
                        WHERE repo = 'AI Studio' AND url IS NOT NULL AND url != ''
                    """
                    import pandas as pd
                    existing_df = pd.read_sql_query(existing_query, conn)
                    conn.close()

                    if not existing_df.empty:
                        existing_models_with_url = {
                            f"{row['publisher']}/{row['model_name']}": row['url']
                            for _, row in existing_df.iterrows()
                        }
                        print(f"[AI Studio] 📚 数据库中已有 {len(existing_models_with_url)} 个模型带URL，将跳过URL获取")
                except Exception as e:
                    print(f"[AI Studio] ⚠️  无法加载已存在模型列表: {e}")

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
                        self._log_info(f"[AI Studio] {search_term} 页面加载成功")
                    except TimeoutException:
                        self._log_error(f"[AI Studio] {search_term} 页面加载超时，可能是网络问题或页面结构变化")
                        if attempt < max_retries - 1:
                            continue
                        raise

                    # 页面加载成功后，立即关闭横幅（每个搜索词只关闭一次）
                    self._log_info(f"[AI Studio] 尝试关闭横幅广告")
                    self._close_banner(driver)

                    # 开始爬取数据
                    page_num = 1
                    while True:
                        wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                        ))

                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                        self._log_info(f"[AI Studio] {search_term} 第{page_num}页，有 {len(cards)} 个卡片")

                        # 记录当前页第一个模型（用于恢复）
                        if len(cards) > 0:
                            page_first_model = cards[0].find_element(
                                By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                            ).text.strip()
                            self._log_info(f"[AI Studio] 当前页标识（第一个模型）: {page_first_model}")
                        else:
                            page_first_model = None
                            self._log_warning(f"[AI Studio] ⚠️  当前页没有卡片，跳过")
                            break

                        # 测试模式：只处理第一个和最后一个卡片
                        if self.test_mode and len(cards) > 2:
                            indices_to_process = [0, len(cards) - 1]
                            self._log_info(f"[AI Studio] 🧪 测试模式：只处理第1个和第{len(cards)}个卡片")
                        else:
                            indices_to_process = range(len(cards))

                        for i in indices_to_process:
                            try:
                                self._log_info(f"[AI Studio] ========== 处理卡片 {i}/{len(cards)-1} ==========")

                                # 重新获取cards（因为可能已过时）
                                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                if i >= len(cards):
                                    self._log_warning(f"[AI Studio] ⚠️  i={i} 超出cards范围({len(cards)})，跳过")
                                    break

                                card = cards[i]
                                card_start_time = time.time()
                                full_model_name = card.find_element(
                                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                                ).text.strip()
                                self._log_info(f"[AI Studio] 模型名称: {full_model_name}")

                                # 检查是否已处理过
                                if full_model_name in processed_models:
                                    self._log_info(f"[AI Studio] ⏭️  模型已处理过，跳过: {full_model_name}")
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
                                        self._log_debug(f"[AI Studio] 更新时间: {last_modified}")
                                    except Exception as e:
                                        self._log_debug(f"[AI Studio] 获取更新时间失败: {e}")

                                publisher = card.find_element(
                                    By.CSS_SELECTOR, "span.ai-model-list-wapper-card-right-detail-one-publisher"
                                ).text.strip()
                                self._log_info(f"[AI Studio] 发布者: {publisher}, 下载量: {usage_count}")

                                # 处理模型名称（用于检查是否已有URL）
                                model_name = full_model_name
                                if model_name.startswith("PaddlePaddle/"):
                                    model_name = model_name[len("PaddlePaddle/"):]

                                # 统一 publisher 大小写用于 URL 查找
                                lookup_publisher = "PaddlePaddle" if publisher.lower() == "paddlepaddle" else publisher
                                model_key = f"{lookup_publisher}/{model_name}"
                                existing_url = existing_models_with_url.get(model_key)
                                has_url = existing_url is not None
                                needs_precise_count = self._is_simplified_count(usage_count)

                                # 决策：是否需要点击
                                should_click = not has_url or needs_precise_count

                                if not should_click:
                                    # 已有URL且下载量精确，跳过点击
                                    self._log_info(f"[AI Studio] ⏭️  跳过点击（已有URL且下载量精确）: {model_key}")
                                    final_usage_count = usage_count
                                    model_url = existing_url  # 🔧 使用数据库中已有的URL
                                else:
                                    # 需要点击：获取URL 或 精确下载量
                                    if has_url and needs_precise_count:
                                        self._log_info(f"[AI Studio] 🔍 获取精确下载量（已有URL）: {usage_count}")
                                    elif not has_url:
                                        self._log_info(f"[AI Studio] 🔍 获取URL和精确下载量: {model_key}")

                                    # 点击获取URL和详细下载量（传入列表页的下载量用于核对）
                                    final_usage_count, model_url = self._get_detailed_info(driver, card, i, list_usage_count=usage_count)

                                # 检查是否回到了第一页，如果是则恢复到目标页（仅当点击了之后）
                                if should_click and page_first_model:
                                    # 等待页面稳定，重新获取cards
                                    time.sleep(0.5)
                                    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")

                                    # 如果没有卡片，刷新页面重新加载
                                    retry_count = 0
                                    while len(cards) == 0 and retry_count < 2:
                                        self._log_warning(f"[AI Studio] ⚠️  返回后页面没有卡片，刷新页面... ({retry_count + 1}/2)")
                                        driver.refresh()
                                        time.sleep(1.5)
                                        wait.until(EC.presence_of_element_located(
                                            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
                                        ))
                                        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                        retry_count += 1

                                    if len(cards) == 0:
                                        self._log_error(f"[AI Studio] ❌ 刷新后仍没有卡片，跳过剩余卡片")
                                        break

                                    if len(cards) > 0:
                                        current_first = cards[0].find_element(
                                            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                                        ).text.strip()

                                        if current_first != page_first_model:
                                            self._log_warning(f"[AI Studio] ⚠️  检测到回到第一页，正在恢复...")
                                            self._log_info(f"[AI Studio]   目标页第一个模型: {page_first_model}")
                                            self._log_info(f"[AI Studio]   当前页第一个模型: {current_first}")

                                            if not self._restore_to_page(driver, page_first_model):
                                                self._log_error(f"[AI Studio] ❌ 恢复页失败，跳过剩余卡片")
                                                break

                                            # 恢复后重新获取cards
                                            time.sleep(0.5)
                                            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                                            if i >= len(cards):
                                                self._log_warning(f"[AI Studio] ⚠️  恢复后i={i}超出范围，跳过")
                                                break

                                self._log_info(f"[AI Studio] ✅ 卡片处理完成 | 下载量={final_usage_count}, URL={model_url} (耗时: {(time.time() - card_start_time)*1000:.2f}ms)")

                                # 处理模型名称和发布者
                                model_name = full_model_name

                                # 如果模型名称以PaddlePaddle/开头，提取出真正的模型名
                                if model_name.startswith("PaddlePaddle/"):
                                    model_name = model_name[len("PaddlePaddle/"):]

                                # 统一 publisher 大小写：AI Studio 页面可能返回 "Paddlepaddle"
                                if publisher.lower() == "paddlepaddle":
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

                            # 获取模型详情页URL
                            model_url = link.get_attribute('href').strip()

                            self.results.append(self.create_record(
                                model_name=model_name,
                                publisher=publisher,
                                download_count=download_count,
                                search_keyword=search_term,
                                url=model_url  # 模型详情页URL
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

                            # 获取模型详情页URL
                            model_url = card.get_attribute('href').strip()

                            self.results.append(self.create_record(
                                model_name=model_name,
                                publisher=publisher,
                                download_count=download_count,
                                search_keyword=search_term,
                                url=model_url  # 模型详情页URL
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
