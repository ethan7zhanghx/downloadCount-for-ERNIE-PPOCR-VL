"""测试AI Studio能否通过JavaScript直接跳转到指定页码"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ernie_tracker.utils import create_chrome_driver

def test_pagination_jump():
    """测试能否用JavaScript直接跳转到指定页码"""
    driver = create_chrome_driver()
    wait = WebDriverWait(driver, 20)

    try:
        # 打开搜索页面
        url = "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5"
        driver.get(url)

        wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
        ))

        print("=== 第1页 ===")
        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        first_model_page1 = cards[0].find_element(
            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
        ).text.strip()
        print(f"第一个模型: {first_model_page1}")

        # 方法1: 尝试直接修改URL参数
        print("\n=== 方法1: 尝试URL参数 ===")
        test_urls = [
            "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5&page=2",
            "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5&p=2",
            "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5&currentPage=2",
        ]
        for test_url in test_urls:
            print(f"尝试URL: {test_url}")
            driver.get(test_url)
            import time
            time.sleep(2)
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            if len(cards) > 0:
                first_model = cards[0].find_element(
                    By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                ).text.strip()
                print(f"  第一个模型: {first_model}")
                if first_model != first_model_page1:
                    print(f"  ✅ URL参数有效！")
                    break
                else:
                    print(f"  ❌ 还是第1页")
            else:
                print(f"  ❌ 没有卡片")

        # 方法2: 尝试通过JavaScript触发翻页事件
        print("\n=== 方法2: 尝试JavaScript触发翻页 ===")

        # 先回到第1页
        driver.get(url)
        time.sleep(2)
        print("当前在第1页")

        # 尝试查找分页组件
        try:
            pagination = driver.find_element(By.CSS_SELECTOR, "ul.ant-pagination")
            print(f"找到分页组件: {pagination.tag_name}")

            # 获取所有页码按钮
            page_items = driver.find_elements(By.CSS_SELECTOR, "li.ant-pagination-item")
            print(f"找到 {len(page_items)} 个页码按钮")

            # 尝试点击第5页（如果存在）
            if len(page_items) >= 5:
                print(f"尝试点击第5页按钮...")
                page_items[4].click()
                time.sleep(2)
                cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                if len(cards) > 0:
                    first_model = cards[0].find_element(
                        By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                    ).text.strip()
                    print(f"第5页第一个模型: {first_model}")
                    if first_model != first_model_page1:
                        print(f"✅ 点击页码按钮有效！")

        except Exception as e:
            print(f"❌ 找不到分页组件: {e}")

        # 方法3: 尝试找到React/Vue状态并修改
        print("\n=== 方法3: 尝试修改页面状态 ===")

        # 检查页面使用的框架
        framework = driver.execute_script("""
            return window.React ? 'React' : (window.Vue ? 'Vue' : 'Unknown');
        """)
        print(f"页面框架: {framework}")

        # 尝试查找分页组件的内部状态
        try:
            # 检查ant-pagination的属性
            next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
            button_attrs = driver.execute_script("""
                var items = {};
                var attrs = arguments[0].attributes;
                for (var i = 0; i < attrs.length; i++) {
                    items[attrs[i].name] = attrs[i].value;
                }
                return items;
            """, next_button)
            print(f"翻页按钮属性: {button_attrs}")
        except:
            pass

        # 方法4: 尝试多次点击"下一页"
        print("\n=== 方法4: 测试连续点击下一页 ===")
        driver.get(url)
        time.sleep(2)

        for page_num in range(2, 4):
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, "li.ant-pagination-next button")
                if next_button.is_enabled():
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(1)
                    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                    if len(cards) > 0:
                        first_model = cards[0].find_element(
                            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
                        ).text.strip()
                        print(f"第{page_num}页第一个模型: {first_model}")
                else:
                    print(f"翻页按钮已禁用")
                    break
            except Exception as e:
                print(f"翻页失败: {e}")
                break

        input("\n按回车退出...")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_pagination_jump()
