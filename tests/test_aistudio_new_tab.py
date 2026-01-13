"""测试AI Studio能否在新标签页打开详情页"""
import sys
sys.path.insert(0, '/Users/zhanghaoxin/Desktop/Baidu/DownloadData')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ernie_tracker.utils import create_chrome_driver
import time

def test_new_tab_open():
    """测试各种在新标签页打开详情页的方法"""
    driver = create_chrome_driver()
    wait = WebDriverWait(driver, 20)

    try:
        # 打开搜索页面
        url = "https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q=ERNIE-4.5"
        driver.get(url)

        wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.ai-model-list-wapper")
        ))

        # 获取第一个卡片
        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        card = cards[0]
        model_name = card.find_element(
            By.CSS_SELECTOR, "div.ai-model-list-wapper-card-right-desc"
        ).text.strip()
        print(f"测试卡片: {model_name}")
        print(f"当前窗口数: {len(driver.window_handles)}")

        # 方法1: 尝试找到卡片内的链接元素
        print("\n=== 方法1: 查找卡片内的链接 ===")
        try:
            links = card.find_elements(By.TAG_NAME, "a")
            print(f"卡片内找到 {len(links)} 个链接")
            for i, link in enumerate(links):
                href = link.get_attribute('href')
                print(f"  链接{i}: {href}")
                if href and "model" in href:
                    print(f"  -> 这是模型链接！")
                    # 尝试在新标签页打开
                    driver.execute_script("window.open(arguments[0], '_blank');", href)
                    time.sleep(2)
                    print(f"  打开后窗口数: {len(driver.window_handles)}")
                    if len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[-1])
                        print(f"  新标签页URL: {driver.current_url}")
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        break
        except Exception as e:
            print(f"❌ 没有找到链接: {e}")

        # 方法2: 尝试右键菜单打开
        print("\n=== 方法2: 尝试右键菜单 ===")
        try:
            from selenium.webdriver.common.action_chains import ActionChains
            from selenium.webdriver.common.keys import Keys

            # 重新获取卡片
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            card = cards[0]

            # 模拟右键点击
            actions = ActionChains(driver)
            actions.context_click(card).perform()
            time.sleep(1)

            # 尝试按W键（在某些浏览器中是"在新标签页打开"的快捷键）
            # 这个方法通常不可行，因为浏览器安全限制
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            print("  ❌ 右键菜单方法通常不可行")
        except Exception as e:
            print(f"❌ 右键菜单失败: {e}")

        # 方法3: 尝试点击时按住Shift/Ctrl/Cmd
        print("\n=== 方法3: 尝试Shift+Click ===")
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            card = cards[0]

            # 记录当前窗口数
            windows_before = len(driver.window_handles)

            # 尝试Shift+Click
            driver.execute_script("""
                arguments[0].click();
            """, card)

            # 或者用JavaScript模拟Shift+Click
            # driver.execute_script("""
            #     var event = new MouseEvent('click', {
            #         bubbles: true,
            #         cancelable: true,
            #         shiftKey: true
            #     });
            #     arguments[0].dispatchEvent(event);
            # """, card)

            time.sleep(2)
            windows_after = len(driver.window_handles)
            print(f"  点击前窗口数: {windows_before}")
            print(f"  点击后窗口数: {windows_after}")

            if windows_after > windows_before:
                print(f"  ✅ Shift+Click 成功打开新标签页！")
                driver.switch_to.window(driver.window_handles[-1])
                print(f"  新标签页URL: {driver.current_url}")
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            else:
                print(f"  ❌ Shift+Click 没有打开新标签页")

            # 如果打开了详情页，返回
            if driver.current_url != url:
                print(f"  在当前窗口打开了详情页，返回中...")
                driver.back()
                time.sleep(2)

        except Exception as e:
            print(f"❌ Shift+Click失败: {e}")

        # 方法4: 检查卡片的onclick属性
        print("\n=== 方法4: 检查卡片的点击处理 ===")
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            card = cards[0]

            # 获取onclick属性
            onclick = card.get_attribute('onclick')
            print(f"卡片onclick: {onclick}")

            # 检查卡片及其父元素的事件监听器
            event_listeners = driver.execute_script("""
                var element = arguments[0];
                var listeners = [];

                // 检查element本身
                if (element.onclick) {
                    listeners.push('element.onclick: ' + element.onclick);
                }

                // 检查addEventListener添加的监听器（无法直接访问）
                // 只能检查是否有data属性或其他线索

                return listeners;
            """, card)
            print(f"事件监听器: {event_listeners}")

            # 检查data属性
            data_attrs = driver.execute_script("""
                var items = {};
                for (var attr of arguments[0].attributes) {
                    if (attr.name.startsWith('data-')) {
                        items[attr.name] = attr.value;
                    }
                }
                return items;
            """, card)
            print(f"data属性: {data_attrs}")

        except Exception as e:
            print(f"❌ 检查失败: {e}")

        # 方法5: 尝试找到点击后跳转的URL
        print("\n=== 方法5: 拦截点击后的导航 ===")
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            card = cards[0]

            # 设置导航监听（需要通过JavaScript）
            driver.execute_script("""
                window.capturedUrl = null;
                // 注意：这只能在导航前设置，实际拦截比较复杂
            """)

            # 点击卡片
            card.click()
            time.sleep(2)

            # 获取当前URL
            detail_url = driver.current_url
            print(f"点击后URL: {detail_url}")

            if detail_url != url:
                print(f"✅ 成功打开详情页: {detail_url}")

                # 返回
                driver.back()
                time.sleep(2)

        except Exception as e:
            print(f"❌ 拦截失败: {e}")

        input("\n按回车退出...")

    finally:
        driver.quit()

if __name__ == "__main__":
    test_new_tab_open()
