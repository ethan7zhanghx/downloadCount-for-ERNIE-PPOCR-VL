"""
工具函数模块
"""
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from .config import SELENIUM_TIMEOUT, SELENIUM_WINDOW_SIZE, SELENIUM_HEADLESS


def create_chrome_driver(headless=SELENIUM_HEADLESS):
    """
    创建 Chrome WebDriver 实例

    Args:
        headless: 是否使用无头模式

    Returns:
        WebDriver 实例
    """
    options = Options()

    # 通用稳定性参数（必须在最前面）
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={SELENIUM_WINDOW_SIZE}")

    if headless:
        options.add_argument("--headless=new")  # 使用新版headless模式
        options.add_argument("--disable-gpu")

    # 其他优化参数
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-features=TranslateUI")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-translate")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--no-first-run")
    options.add_argument("--safebrowsing-disable-auto-update")
    options.add_argument("--password-store=basic")
    options.add_argument("--use-mock-keychain")

    # 修复 data: 页面问题
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # 方案1：使用Selenium 4的自动driver管理（最可靠）
    try:
        print("尝试使用 Selenium 自动管理 ChromeDriver...")
        driver = webdriver.Chrome(options=options)

        # 设置更长的超时时间，避免某些网站加载慢
        driver.set_page_load_timeout(120)  # 增加到120秒
        driver.set_script_timeout(60)  # 增加到60秒

        # 设置隐式等待
        driver.implicitly_wait(10)

        # 添加 CDP 命令以避免检测
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        print("✅ ChromeDriver 启动成功！")
        return driver
    except Exception as e1:
        print(f"Selenium 自动管理失败: {e1}")

        # 方案2：使用webdriver-manager
        try:
            print("尝试使用 webdriver-manager...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(120)
            driver.set_script_timeout(60)
            driver.implicitly_wait(10)

            # CDP 命令
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            print("✅ ChromeDriver 启动成功！")
            return driver
        except Exception as e2:
            print(f"webdriver-manager 也失败: {e2}")
            raise Exception(
                f"无法初始化 ChromeDriver。\n"
                f"方案1错误: {e1}\n"
                f"方案2错误: {e2}\n"
                f"请确保:\n"
                f"1. 已安装 Chrome 浏览器\n"
                f"2. Chrome 版本与 ChromeDriver 兼容\n"
                f"3. 网络连接正常，可以下载 ChromeDriver"
            )


def extract_numbers(text):
    """
    从文本中提取数字

    Args:
        text: 包含数字的文本

    Returns:
        提取的第一个数字，如果没有则返回 None
    """
    numbers = re.findall(r'\d+', text.replace(',', ''))
    return numbers[0] if numbers else None


def safe_extract_text(element, selector, by_type="css", default=""):
    """
    安全地从元素中提取文本

    Args:
        element: WebElement
        selector: 选择器
        by_type: 选择器类型 ("css" 或 "xpath")
        default: 默认值

    Returns:
        提取的文本或默认值
    """
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import NoSuchElementException

    try:
        if by_type == "css":
            return element.find_element(By.CSS_SELECTOR, selector).text.strip()
        elif by_type == "xpath":
            return element.find_element(By.XPATH, selector).text.strip()
        else:
            return default
    except NoSuchElementException:
        return default


def retry_on_failure(func, max_retries=3, delay=2):
    """
    重试装饰器

    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试延迟（秒）

    Returns:
        函数执行结果
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"尝试 {attempt + 1} 失败: {e}，{delay}秒后重试...")
                time.sleep(delay)
            else:
                print(f"达到最大重试次数，失败: {e}")
                raise


def is_simplified_count(count_text):
    """
    判断是否为简化的计数文本（如 "1.2K"）

    Args:
        count_text: 计数文本

    Returns:
        bool: 是否为简化格式
    """
    count_text = count_text.strip()
    return not count_text.replace(' ', '').isdigit()
