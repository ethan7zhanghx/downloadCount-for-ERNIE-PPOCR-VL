import requests
from bs4 import BeautifulSoup
import time

import streamlit as st

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

from datetime import date
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import sqlite3
import re

def save_to_db(df, db_path="ernie_downloads.db"):
    conn = sqlite3.connect(db_path)
    
    try:
        # 尝试读取已有记录
        existing = pd.read_sql("SELECT * FROM model_downloads", conn)
    except pd.errors.DatabaseError:
        existing = pd.DataFrame()
    
    if not existing.empty:
        # 构建删除条件
        conditions = []
        params = []
        for _, row in df.iterrows():
            conditions.append("(date = ? AND repo = ? AND model_name = ?)")
            params.extend([row['date'], row['repo'], row['model_name']])
        
        if conditions:
            delete_sql = f"DELETE FROM model_downloads WHERE {' OR '.join(conditions)}"
            conn.execute(delete_sql, params)
            conn.commit()
    
    # 插入新数据
    df.to_sql("model_downloads", conn, if_exists="append", index=False)
    print(f"处理 {len(df)} 条记录（包含新增和覆盖）")
    
    conn.close()


# hugging face
from huggingface_hub import list_models, model_info

def fetch_hugging_face_data(search_query="ERNIE-4.5", progress_callback=None, progress_total=None):
    models = list(list_models(search=search_query, full=True))
    total_count = len(models)
    results = []

    for i, m in enumerate(models, start=1):
        try:
            info = model_info(m.id, expand=["downloadsAllTime"])
            results.append({
                "date": date.today().isoformat(),
                "repo": "Hugging Face",
                "model_name": m.id.split("/", 1)[1] if "/" in m.id else m.id,
                "publisher": m.id.split("/")[0],
                "download_count": getattr(info, 'downloads_all_time', None)
            })
        except Exception as e:
            print("error", e)

        if progress_callback:
            progress_callback(i, discovered_total=total_count)

    df = pd.DataFrame(
        results,
        columns=["date", "repo", "model_name", "publisher", "download_count"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count

#modelscope
from modelscope.hub.api import HubApi

def get_modelscope_ids(search_query="ERNIE-4.5"):
    today = date.today().isoformat()
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    page = 1
    wait = WebDriverWait(driver, 20)
    model_ids = []

    while True:
        url = f"https://modelscope.cn/search?page={page}&search={search_query}&type=model"
        print(f"[ModelScope] 爬取页面: {url}")
        driver.get(url)

        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#normal_tab_model .antd5-row a")))
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

def fetch_modelscope_data(search_query="ERNIE-4.5", progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    model_ids = list(get_modelscope_ids(search_query=search_query))  # 确保是 list
    total_count = len(model_ids)

    api = HubApi()
    records = []

    for i, model_id in enumerate(model_ids, start=1):
        try:
            info = api.get_model(model_id, revision="master")
            downloads = info.get("Downloads", 0)
            records.append({
                "date": today,
                "repo": "ModelScope",
                "model_name": model_id.split("/", 1)[1] if "/" in model_id else model_id,
                "publisher": model_id.split("/")[0],
                "download_count": downloads
            })
        except Exception as e:
            print(f"获取 {model_id} 失败: {e}")

        if progress_callback:
            progress_callback(i, discovered_total=total_count)

    df = pd.DataFrame(
        records,
        columns=["date", "repo", "model_name", "publisher", "download_count"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count


# AI Studio
def is_simplified_count(count_text):
    count_text = count_text.strip()
    return not count_text.replace(' ', '').isdigit()

def get_detailed_download_count(driver, card_index, model_name):
    try:
        current_url = driver.current_url
        
        # 重新获取卡片元素
        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        if card_index >= len(cards):
            return None
        
        card = cards[card_index]

        cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
        if card_index < len(cards):
            card = cards[card_index]
            # 直接点击卡片
            card.click()
            time.sleep(1)
        
        # 等待页面加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)
        # 获取下载量
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[@id='main']/div[1]/div[2]/div/div/div[1]/div/div[1]/div[4]/div[2]"))
            )
            text = element.text.strip()
            numbers = re.findall(r'\d+', text)
            detailed_count = numbers[0] if numbers else None
        except Exception as e:
            print(f"获取详情页数据失败: {e}")
            detailed_count = None
        
        # 返回原来的页面
        driver.back()
        # 等待页面加载完成
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
        )
        
        return detailed_count
    except Exception as e:
        print(f"获取 {model_name} 详情页下载量时出错: {e}")
        try:
            if driver.current_url != current_url:
                driver.back()
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.ai-model-list-wapper"))
                )
        except:
            pass
        return None

def fetch_aistudio_data(search_query="ERNIE-4.5", progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    results = []
    wait = WebDriverWait(driver, 40)
    url = f"https://aistudio.baidu.com/modelsoverview?sortBy=useCount&q={search_query}"
    driver.get(url)

    processed_count = 0

    try:
        while True:
            cards_container_selector = "div.ai-model-list-wapper"
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, cards_container_selector)))
            old_container_reference = driver.find_element(By.CSS_SELECTOR, cards_container_selector)

            cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
            card_count = len(cards)
            print(f"当前页面有 {card_count} 个卡片")

            for i in range(card_count):
                try:
                    cards = driver.find_elements(By.CSS_SELECTOR, "div.ai-model-list-wapper > div")
                    if i >= len(cards):
                        print(f"卡片索引 {i} 超出范围，跳过")
                        continue

                    card = cards[i]
                    name_selector = "div.ai-model-list-wapper-card-right-desc"
                    model_name = card.find_element(By.CSS_SELECTOR, name_selector).text.strip()

                    usage_selector = "span.ai-model-list-wapper-card-right-detail-one-like"
                    usage_count = card.find_element(By.CSS_SELECTOR, usage_selector).text.strip()

                    publisher_selector = "span.ai-model-list-wapper-card-right-detail-one-publisher"
                    publisher = card.find_element(By.CSS_SELECTOR, publisher_selector).text.strip()

                    final_usage_count = usage_count
                    if is_simplified_count(usage_count):
                        final_usage_count = get_detailed_download_count(driver, i, model_name)

                    # 如果 model_name 包含斜杠，则拆分
                    if "/" in model_name:
                        publisher_from_name, model_name_only = model_name.split('/', 1)
                        # 如果爬取的 publisher 为空或不一致，则使用从名称中提取的
                        if not publisher or publisher != publisher_from_name:
                            publisher = publisher_from_name
                        model_name = model_name_only

                    results.append({
                        "date": today,
                        "repo": "AI Studio",
                        "model_name": model_name,
                        "publisher": publisher,
                        "download_count": final_usage_count
                    })

                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count)

                except Exception as e:
                    print(f"处理卡片 {i} 时出错: {e}")
                    continue

            try:
                next_page_li_selector = "li.ant-pagination-next"
                next_page_li = driver.find_element(By.CSS_SELECTOR, next_page_li_selector)
                next_page_button = next_page_li.find_element(By.TAG_NAME, "button")

                if not next_page_button.is_enabled():
                    print("最后一页")
                    break

                next_page_button.click()
                wait.until(EC.staleness_of(old_container_reference))
            except Exception as e:
                print(f"翻页时出错: {e}")
                break

    finally:
        driver.quit()

    total_count = processed_count
    df = pd.DataFrame(
        results,
        columns=["date", "repo", "model_name", "publisher", "download_count"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count

def fetch_gitcode_data(progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 40)

    results = []
    model_links = [
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-424B-A47B-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-28B-A3B-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-424B-A47B-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-21B-A3B-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-21B-A3B-Base-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-21B-A3B-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-FP8-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-0.3B-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-0.3B-Base-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-0.3B-Base-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-0.3B-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-424B-A47B-Base-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-28B-A3B-Base-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-Base-PT",
        "https://ai.gitcode.com/paddlepaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle"
    ]

    total_count = len(model_links)

    for i, model_link in enumerate(model_links, start=1):
        try:
            driver.get(model_link)
            print(f"访问链接: {model_link}")
            # 等待 URL 变化或页面加载完成
            try:
                wait.until(EC.url_changes(model_link))
            except TimeoutException:
                print(f"URL 未变化，可能没有重定向或页面加载缓慢: {driver.current_url}")
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(1) # 额外等待，确保URL更新
            print(f"当前页面URL (加载后): {driver.current_url}")

            model_name_selector = "#repo-banner-box > div > div.repo-info.h-full.ai-hub > div > div:nth-child(1) > div > div > div.info-item.project-name > div.project-text > div > p > a > span"
            model_name = wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, model_name_selector))
            ).text.strip()

            # 检查是否重定向到 /model-inference 页面
            if "/model-inference" in driver.current_url:
                print(f"检测到重定向到 /model-inference 页面: {driver.current_url}")
                try:
                    # 点击“模型介绍”标签回到原始页面
                    model_intro_selector = "#repo-header-tab > div.nav-tabs-item.flex-1.w-\[100\%\].overflow-hidden > div > div.repo-header-tab-ul > a:nth-child(1) > div"
                    print(f"尝试点击元素: {model_intro_selector}")
                    model_intro_element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, model_intro_selector)))
                    model_intro_element.click()
                    print("已点击“模型介绍”标签，等待页面加载...")
                    # 等待 URL 变化回原始链接，并等待下载量元素重新出现
                    wait.until(EC.url_contains(model_link.split('?')[0]))
                    wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[1]/div[1]/div/div[2]')))
                    time.sleep(3) # 额外等待，确保页面稳定和元素可见
                    print(f"点击后当前页面URL: {driver.current_url}")
                except Exception as click_e:
                    print(f"点击“模型介绍”标签或等待页面加载失败: {click_e}")
                    downloads = "0" # 如果点击失败，则无法获取下载量
            
            downloads = "0" # 默认值
            if downloads == "0": # 如果之前点击失败导致 downloads 为 "0"，则不再尝试获取
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        time.sleep(2) # Add a small delay to allow the page to stabilize
                        # 尝试原始 XPath 获取下载量
                        downloads_xpath = '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[1]/div[1]/div/div[2]'
                        print(f"尝试获取下载量元素: {downloads_xpath}")
                        downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                        last_val = ""
                        for _ in range(5):
                            # Re-locate the element in each iteration to avoid StaleElementReferenceException
                            downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                            val = downloads_element.text.strip().replace(',', '')
                            if val and val != last_val:
                                last_val = val
                                time.sleep(1)
                            else:
                                break
                        downloads = last_val
                        print(f"获取到下载量: {downloads}")
                        break # If successful, break out of the retry loop
                    except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
                        print(f"在 {driver.current_url} 页面获取下载量失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                        downloads = "0" # 最终失败，设为0
                        if attempt == max_retries - 1: # If it's the last attempt, set downloads to "0"
                            downloads = "0"
                        else:
                            time.sleep(2) # Wait before retrying

            results.append({
                "date": today,
                "repo": "GitCode",
                "model_name": model_name,
                "publisher": "飞桨PaddlePaddle",
                "download_count": downloads
            })

        except Exception as e:
            print(f"获取 {model_link} 失败: {e}")

        if progress_callback:
            progress_callback(i, discovered_total=total_count)

    driver.quit()

    df = pd.DataFrame(
        results,
        columns=["date", "repo", "model_name", "publisher", "download_count"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count

def fetch_caict_data(progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 40)

    model_links = [
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-2Bits-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-0.3B-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-0.3B-PT",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-VL-28B-A3B-PT",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-Base-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-VL-28B-A3B-Base-PT",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-0.3B-Base-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-Base-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-21B-A3B-Base-PT",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-0.3B-Base-PT",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-0.3B-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-FP8-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Paddle",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-VL-424B-A47B-Base-PT",
        "https://aihub.caict.cn/models/yiyan/ERNIE-4.5-21B-A3B-Paddle",
        "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-21B-A3B-PT",
        "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle"
    ]

    results = []
    total_models = len(model_links)

    for idx, model_link in enumerate(model_links, start=1):
        print(f"[鲸智] 正在处理 {idx}/{total_models}：{model_link}")
        driver.get(model_link)
        
        try:
            model_name_selector = "#community-app > div > div:nth-child(2) > div.w-full.bg-\[\#FCFCFD\].pt-9.pb-\[60px\].xl\:px-10.md\:px-0.md\:pb-6.md\:h-auto > div > div.flex.flex-col.gap-\[16px\].flex-wrap.mb-\[8px\].text-lg.text-\[\#606266\].font-semibold.md\:px-5 > div > a"
            model_name = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, model_name_selector))).text.strip()
            
            downloads_selector = "#pane-summary > div > div.w-\[40\%\].sm\:w-\[100\%\].border-l.border-\[\#EBEEF5\].md\:border-l-0.md\:border-b.md\:w-full.md\:pl-0 > div > div.text-\[\#303133\].text-base.font-semibold.leading-6.mt-1.md\:pl-0"
            downloads_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, downloads_selector)))
            
            # 增加等待和重试机制，确保下载量刷新
            downloads = "0"
            for _ in range(5):  # 最多等待5秒
                downloads = downloads_element.text.strip().replace(',', '')
                if downloads and downloads != "0":
                    break
                time.sleep(1)

            results.append({
                "date": today,
                "repo": "鲸智",
                "model_name": model_name,
                "publisher": "PaddlePaddle",
                "download_count": downloads
            })

        except Exception as e:
            print(f"!! 错误：处理 {model_link} 时失败，原因：{e}")
            continue
        if progress_callback:
            progress_callback(idx, discovered_total=total_models)

    driver.quit()
    df = pd.DataFrame(results)
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_models

# 魔乐 Modelers
def fetch_modelers_data(search_query="ERNIE-4.5", progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    results = []
    page = 1
    total_models = 0  # 总数统计用

    while True:
        url = f"https://modelers.cn/models?name={search_query}&page={page}&size=64"
        driver.get(url)

        try:
            cards_container_selector = "div.cards-content"
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, cards_container_selector)))
            cards = driver.find_elements(By.CSS_SELECTOR, "div.cards-content > a")
        except:
            print(f"[魔乐 Modelers] 到达最后一页 (page {page})")
            break

        if not cards:
            break

        for idx, card in enumerate(cards, start=1):
            total_models += 1
            print(f"[魔乐 Modelers] 正在处理 {total_models}：", end=" ")

            try:
                name_element = card.find_element(By.CSS_SELECTOR, "div.title")
                full_title = name_element.get_attribute('title').strip()

                download_element = card.find_element(By.CSS_SELECTOR, "div.repo-card-footer-right span.value")
                download_count = download_element.text.strip()

                publisher, model_name = "N/A", full_title
                if " / " in full_title:
                    parts = full_title.split(' / ', 1)
                    publisher = parts[0]
                    model_name = parts[1]

                print(model_name)

                results.append({
                    "date": today,
                    "repo": "魔乐 Modelers",
                    "model_name": model_name,
                    "publisher": publisher,
                    "download_count": download_count
                })
            except Exception as e:
                print(f"!!错误：处理模型卡片失败 - {e}")
                continue
            if progress_callback:
                progress_callback(total_models, discovered_total=progress_total)

        page += 1

    driver.quit()
    df = pd.DataFrame(results)
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_models

def fetch_gitee_data(search_query="ERNIE-4.5", progress_callback=None, progress_total=None):
    today = date.today().isoformat()
    options = Options()
    # options.add_argument("--headless")  # 如需无界面可开启
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    results = []
    page = 1
    total_count = 0

    while True:
        url = f"https://ai.gitee.com/models?q={search_query}&p={page}"
        driver.get(url)
        time.sleep(2)  # 可换成显式等待

        model_links = driver.find_elements(By.CSS_SELECTOR, "main section div.relative > div > a")
        if not model_links:
            break
            
        for link in model_links:
            try:
                publisher_elem = link.find_element(By.CSS_SELECTOR, 
                    "div div.flex.flex-col.items-start.gap-1.self-stretch.overflow-hidden div.flex.items-center.gap-2.self-stretch div span:nth-child(1)")
                publisher = publisher_elem.text.strip().rstrip('/')

                model_name = link.find_element(By.CSS_SELECTOR,
                    "div.line-clamp-1.break-all.text-lg.font-medium.leading-7.text-slate-auto-900").get_attribute('title').strip()

                download_elem = link.find_element(By.CSS_SELECTOR,
                    "div.flex.items-center.gap-2.self-stretch.pt-2.md\\:gap-3 > div:nth-child(2) > div")
                download_count = download_elem.text.strip()
                
                # 移除 model_name 中的 publisher 前缀
                if model_name.startswith(publisher + '/'):
                    model_name = model_name.split('/', 1)[1]
                                
                results.append({
                    'date': today,
                    'repo': 'Gitee',
                    "model_name": model_name,
                    "publisher": publisher,
                    "download_count": download_count
                })

                total_count += 1

                # 更新进度条
                if progress_callback:
                    progress_callback(total_count, discovered_total=progress_total)

            except Exception as e:
                print(f"[Gitee] 处理模型时出错: {e}")
                continue

        page += 1
                
    driver.quit()
    df = pd.DataFrame(results)
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count


PLATFORM_FETCHERS = {
    "Hugging Face": fetch_hugging_face_data,
    "ModelScope": fetch_modelscope_data,
    "AI Studio": fetch_aistudio_data,
    "GitCode": fetch_gitcode_data,
    "鲸智": fetch_caict_data,
    "魔乐 Modelers": fetch_modelers_data,
    "Gitee": fetch_gitee_data,
}


def fetch_gitcode_paddleocr_vl_data(progress_callback=None, progress_total=None):
    """
    专门用于从 GitCode 获取 PaddleOCR-VL 模型数据的函数。
    """
    today = date.today().isoformat()
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 40)

    results = []
    model_link = "https://ai.gitcode.com/paddlepaddle/PaddleOCR-VL"
    total_count = 1

    try:
        driver.get(model_link)
        print(f"访问链接: {model_link}")
        # 等待 URL 变化或页面加载完成
        try:
            wait.until(EC.url_changes(model_link))
        except TimeoutException:
            print(f"URL 未变化，可能没有重定向或页面加载缓慢: {driver.current_url}")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1) # 额外等待，确保URL更新
        print(f"当前页面URL (加载后): {driver.current_url}")

        model_name_selector = "#repo-banner-box > div > div.repo-info.h-full.ai-hub > div > div:nth-child(1) > div > div > div.info-item.project-name > div.project-text > div > p > a > span"
        model_name = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, model_name_selector))
        ).text.strip()

        # 检查是否重定向到 /model-inference 页面
        if "/model-inference" in driver.current_url:
            print(f"检测到重定向到 /model-inference 页面: {driver.current_url}")
            try:
                # 点击“模型介绍”标签回到原始页面
                model_intro_selector = "#repo-header-tab > div.nav-tabs-item.flex-1.w-\[100\%\].overflow-hidden > div > div.repo-header-tab-ul > a:nth-child(1) > div"
                print(f"尝试点击元素: {model_intro_selector}")
                model_intro_element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, model_intro_selector)))
                model_intro_element.click()
                print("已点击“模型介绍”标签，等待页面加载...")
                # 等待 URL 变化回原始链接，并等待下载量元素重新出现
                wait.until(EC.url_contains(model_link.split('?')[0]))
                wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[1]/div[1]/div/div[2]')))
                time.sleep(3)
                print(f"点击后当前页面URL: {driver.current_url}")
            except Exception as click_e:
                print(f"点击“模型介绍”标签或等待页面加载失败: {click_e}")
                downloads = "0"
                return pd.DataFrame(columns=["date", "repo", "model_name", "publisher", "download_count"]), 0

        downloads = "0"
        if downloads == "0":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    time.sleep(2) # Add a small delay to allow the page to stabilize
                    # 尝试原始 XPath 获取下载量
                    downloads_xpath = '//*[@id="app"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[1]/div[1]/div/div[2]'
                    print(f"尝试获取下载量元素: {downloads_xpath}")
                    downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                    last_val = ""
                    for _ in range(5):
                        # Re-locate the element in each iteration to avoid StaleElementReferenceException
                        downloads_element = wait.until(EC.presence_of_element_located((By.XPATH, downloads_xpath)))
                        val = downloads_element.text.strip().replace(',', '')
                        if val and val != last_val:
                            last_val = val
                            time.sleep(1)
                        else:
                            break
                    downloads = last_val
                    print(f"获取到下载量: {downloads}")
                    break # If successful, break out of the retry loop
                except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
                    print(f"在 {driver.current_url} 页面获取下载量失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    downloads = "0" # 最终失败，设为0
                    if attempt == max_retries - 1: # If it's the last attempt, set downloads to "0"
                        downloads = "0"
                    else:
                        time.sleep(2) # Wait before retrying

        results.append({
            "date": today,
            "repo": "GitCode",
            "model_name": model_name,
            "publisher": "paddlepaddle",
            "download_count": downloads
        })

    except Exception as e:
        print(f"获取 {model_link} 失败: {e}")

    if progress_callback:
        progress_callback(1, discovered_total=total_count)

    driver.quit()

    df = pd.DataFrame(
        results,
        columns=["date", "repo", "model_name", "publisher", "download_count"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count

PLATFORM_FETCHERS_PADDLEOCR = {
    "Hugging Face": lambda **kwargs: fetch_hugging_face_data(search_query="PaddleOCR-VL", **kwargs),
    "ModelScope": lambda **kwargs: fetch_modelscope_data(search_query="PaddleOCR-VL", **kwargs),
    "AI Studio": lambda **kwargs: fetch_aistudio_data(search_query="PaddleOCR-VL", **kwargs),
    "Gitee": lambda **kwargs: fetch_gitee_data(search_query="PaddleOCR-VL", **kwargs),
    "魔乐 Modelers": lambda **kwargs: fetch_modelers_data(search_query="PaddleOCR-VL", **kwargs),
    "GitCode": fetch_gitcode_paddleocr_vl_data,
}


def fetch_paddleocr_vl_data(progress_callback=None, progress_total=None):
    """
    获取 PaddleOCR-VL 模型的数据
    """
    all_dfs = []
    total_count = 0

    for platform, fetcher in PLATFORM_FETCHERS_PADDLEOCR.items():
        df, count = fetcher(progress_callback=progress_callback)
        all_dfs.append(df)
        total_count += count

    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df, total_count
