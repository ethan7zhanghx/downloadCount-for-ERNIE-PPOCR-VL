"""
统一的数据获取模块 - 一次搜索获取所有PaddlePaddle模型
"""
import requests
from bs4 import BeautifulSoup
import time
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
from datetime import date
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from tqdm.notebook import tqdm

from ..config import SEARCH_QUERY, DB_PATH


# hugging face
from huggingface_hub import list_models, model_info

def fetch_hugging_face_data_unified(progress_callback=None, progress_total=None, use_model_tree: bool = True):
    """
    统一获取Hugging Face上的PaddlePaddle模型（包含model tree和搜索）

    Args:
        progress_callback: 进度回调函数
        progress_total: 总数参考
        use_model_tree: 是否使用model tree功能获取衍生模型

    Returns:
        tuple: (DataFrame, 总数量)
    """
    print(f"🤖 开始获取Hugging Face模型 (Model Tree: {use_model_tree})")

    all_models = []
    processed_ids = set()

    if use_model_tree:
        # Model Tree 模式：获取ERNIE-4.5和PaddleOCR-VL的完整模型
        print("🌳 获取ERNIE-4.5和PaddleOCR-VL完整模型树...")
        try:
            from .fetchers_modeltree import get_all_ernie_derivatives

            model_tree_df, tree_count = get_all_ernie_derivatives(include_paddleocr=True)

            if not model_tree_df.empty:
                # 过滤Hugging Face平台数据
                hf_tree_df = model_tree_df[model_tree_df['repo'] == 'Hugging Face'].copy()

                # 转换为标准格式（保留所有重要字段）
                if not hf_tree_df.empty:
                    # 确保包含所有必要的列
                    required_cols = ['date', 'repo', 'model_name', 'publisher', 'download_count']
                    optional_cols = ['model_type', 'model_category', 'tags', 'base_model', 'data_source']

                    # 选择存在的列
                    cols_to_keep = [col for col in required_cols + optional_cols if col in hf_tree_df.columns]
                    tree_results = hf_tree_df[cols_to_keep].to_dict('records')

                    all_models.extend(tree_results)
                    print(f"✅ Model Tree获取: {len(tree_results)} 个ERNIE/PaddleOCR相关模型")
                else:
                    print("⚠️ Model Tree未找到Hugging Face数据")
            else:
                print("⚠️ Model Tree未获取到任何数据")

        except Exception as e:
            print(f"⚠️ Model Tree获取失败: {e}")
    else:
        # 传统模式：只搜索ERNIE-4.5和PaddleOCR-VL，不查找model tree
        print("🔍 搜索ERNIE-4.5和PaddleOCR-VL模型...")
        try:
            search_terms = ['ERNIE-4.5', 'PaddleOCR-VL']
            search_results = []

            for search_term in search_terms:
                try:
                    term_models = list(list_models(search=search_term, full=True, limit=500))
                    print(f"  🔍 搜索 '{search_term}' 找到 {len(term_models)} 个模型")

                    for i, m in enumerate(term_models):
                        try:
                            info = model_info(m.id, expand=["downloadsAllTime"])

                            # 直接获取下载量并添加调试信息
                            downloads = getattr(info, 'downloads_all_time', 0)

                            # 添加调试信息
                            if i < 3:  # 只打印前3个模型的详细信息
                                print(f"  调试 {m.id}:")
                                print(f"    - downloads_all_time: {downloads}")
                                print(f"    - downloads (fallback): {getattr(info, 'downloads', 'N/A')}")
                                print(f"    - info对象属性: {[attr for attr in dir(info) if 'download' in attr.lower()]}")

                            model_data = {
                                "date": date.today().isoformat(),
                                "repo": "Hugging Face",
                                "model_name": m.id.split("/", 1)[1] if "/" in m.id else m.id,
                                "publisher": m.id.split("/")[0],
                                "download_count": downloads,
                                # 传统搜索模式不包含 model tree 信息
                                "model_type": None,
                                "model_category": None,
                                "tags": None,
                                "base_model": None,
                                "data_source": 'search',  # 标记为传统搜索模式
                                "search_keyword": search_term  # 记录搜索关键词
                            }
                            search_results.append(model_data)

                            if progress_callback:
                                progress_callback(len(search_results), discovered_total=None)

                        except Exception as e:
                            print(f"获取模型 {m.id} 详情失败: {e}")

                except Exception as e:
                    print(f"搜索 '{search_term}' 失败: {e}")

            # 添加搜索结果到总列表
            all_models.extend(search_results)
            print(f"✅ 传统搜索获取: {len(search_results)} 个模型")

        except Exception as e:
            print(f"❌ 传统搜索失败: {e}")

    # 3. 合并和最终处理
    if all_models:
        # 转换为DataFrame
        df = pd.DataFrame(all_models)

        # 确保数据类型正确
        df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)

        # 去重（基于publisher和model_name的组合）
        initial_count = len(df)
        df = df.drop_duplicates(subset=['publisher', 'model_name'], keep='first')
        final_count = len(df)

        print(f"📊 Hugging Face总计获取 {final_count} 个模型（去重前: {initial_count} 个）")

        # 按下载量排序
        df = df.sort_values('download_count', ascending=False).reset_index(drop=True)

        return df, final_count
    else:
        print("⚠️ 两种模式均未获取到数据")
        empty_df = pd.DataFrame(columns=["date", "repo", "model_name", "publisher", "download_count"])
        return empty_df, 0


# ModelScope
from modelscope.hub.api import HubApi

def get_modelscope_ids_unified():
    """获取ModelScope上的所有ERNIE-4.5和PaddleOCR-VL模型ID

    Returns:
        dict: {model_id: search_keyword} 记录每个模型通过哪个关键词搜索到的
    """
    driver = create_chrome_driver(headless=False)
    wait = WebDriverWait(driver, 20)
    model_id_to_keyword = {}  # 记录每个模型ID对应的搜索关键词

    # 搜索 ERNIE-4.5 和 PaddleOCR-VL
    search_terms = ["ERNIE-4.5", "PaddleOCR-VL"]

    for search_term in search_terms:
        print(f"[ModelScope] 搜索 {search_term}...")
        page = 1

        while True:
            url = f"https://modelscope.cn/search?page={page}&search={search_term}&type=model"
            print(f"[ModelScope] 爬取页面: {url}")
            driver.get(url)

            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#normal_tab_model .antd5-row a")))
            except TimeoutException:
                print(f"[ModelScope] {search_term} 页面加载失败，已到最后一页")
                break

            cards = driver.find_elements(By.CSS_SELECTOR, "#normal_tab_model .antd5-row a")
            if not cards:
                break

            for link in cards:
                href = link.get_attribute("href")
                if "/models/" in href:
                    model_id = href.split("/models/")[-1]
                    # 如果模型ID已存在，保持第一个搜索词（ERNIE-4.5优先）
                    if model_id not in model_id_to_keyword:
                        model_id_to_keyword[model_id] = search_term

            page += 1

    driver.quit()
    return model_id_to_keyword

def fetch_modelscope_data_unified(progress_callback=None, progress_total=None):
    """统一获取ModelScope上的PaddlePaddle模型"""
    from .fetchers_modeltree import classify_model
    today = date.today().isoformat()
    model_id_to_keyword = get_modelscope_ids_unified()  # 返回字典
    total_count = len(model_id_to_keyword)

    api = HubApi()
    records = []

    for i, (model_id, search_keyword) in enumerate(model_id_to_keyword.items(), start=1):
        try:
            info = api.get_model(model_id, revision="master")
            downloads = info.get("Downloads", 0)

            # 🔧 新增：获取时间字段
            from datetime import datetime
            created_at = None
            last_modified = None

            if "CreatedTime" in info and info["CreatedTime"]:
                try:
                    created_at = datetime.fromtimestamp(info["CreatedTime"]).strftime('%Y-%m-%d')
                except:
                    created_at = None

            if "LastUpdatedTime" in info and info["LastUpdatedTime"]:
                try:
                    last_modified = datetime.fromtimestamp(info["LastUpdatedTime"]).strftime('%Y-%m-%d')
                except:
                    last_modified = None

            # 🔧 新增：提取模型分类信息
            # 1. BaseModel (base_model)
            base_model = None
            if "BaseModel" in info and info["BaseModel"]:
                if isinstance(info["BaseModel"], list) and len(info["BaseModel"]) > 0:
                    base_model = info["BaseModel"][0]
                elif isinstance(info["BaseModel"], str):
                    base_model = info["BaseModel"]

            # 2. BaseModelRelation (model_type)
            model_type = None
            if "BaseModelRelation" in info and info["BaseModelRelation"]:
                model_type = info["BaseModelRelation"].lower()
                # 映射到标准类型名称
                type_mapping = {
                    'finetune': 'finetune',
                    'quantized': 'quantized',
                    'adapter': 'adapter',
                    'lora': 'lora',
                    'merge': 'merge'
                }
                if model_type not in type_mapping:
                    model_type = 'other' if model_type else None
            else:
                # 如果没有 BaseModelRelation，但也没有 base_model，则可能是 original
                if not base_model:
                    model_type = 'original'

            # 3. model_category - 使用 classify_model 函数根据名称、发布者和 base_model 推断
            publisher = model_id.split("/")[0] if "/" in model_id else 'Unknown'
            model_name = model_id.split("/", 1)[1] if "/" in model_id else model_id
            model_category = classify_model(model_name, publisher, base_model)

            records.append({
                "date": today,
                "repo": "ModelScope",
                "model_name": model_id.split("/", 1)[1] if "/" in model_id else model_id,
                "publisher": model_id.split("/")[0],
                "download_count": downloads,
                "search_keyword": search_keyword,
                "created_at": created_at,
                "last_modified": last_modified,
                "model_category": model_category,
                "model_type": model_type,
                "base_model": base_model,
                "base_model_from_api": base_model
            })
        except Exception as e:
            print(f"获取 {model_id} 失败: {e}")

        if progress_callback:
            progress_callback(i, discovered_total=total_count)

    df = pd.DataFrame(
        records,
        columns=["date", "repo", "model_name", "publisher", "download_count", "search_keyword",
                 "created_at", "last_modified", "model_category", "model_type", "base_model",
                 "base_model_from_api"]
    )
    df['download_count'] = pd.to_numeric(df['download_count'], errors='coerce').fillna(0).astype(int)
    return df, total_count


# AI Studio (使用修复后的selenium版本)
from ..utils import create_chrome_driver, is_simplified_count, extract_numbers

def fetch_aistudio_data_unified(progress_callback=None, progress_total=None):
    """统一获取AI Studio上的PaddlePaddle模型"""
    from .selenium import AIStudioFetcher

    fetcher = AIStudioFetcher()
    fetcher.name = "AI Studio"
    return fetcher.fetch(progress_callback=progress_callback, progress_total=progress_total)


# GitCode (包含ERNIE-4.5和PaddleOCR-VL)
def fetch_gitcode_data_unified(progress_callback=None, progress_total=None):
    """统一获取GitCode上的PaddlePaddle模型（包含ERNIE-4.5和PaddleOCR-VL）"""
    today = date.today().isoformat()
    driver = create_chrome_driver(headless=False)
    wait = WebDriverWait(driver, 40)

    results = []

    # ERNIE-4.5 模型列表
    ernie_model_links = [
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

    # PaddleOCR-VL 模型
    paddleocr_model_links = [
        "https://ai.gitcode.com/paddlepaddle/PaddleOCR-VL"
    ]

    # 合并所有模型链接
    all_model_links = ernie_model_links + paddleocr_model_links
    total_count = len(all_model_links)

    for i, model_link in enumerate(all_model_links, start=1):
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


# 鲸智 (CAICT)
def fetch_caict_data_unified(progress_callback=None, progress_total=None):
    """统一获取鲸智上的PaddlePaddle模型"""
    from ..config import CAICT_MODEL_LINKS

    today = date.today().isoformat()
    driver = create_chrome_driver(headless=False)
    wait = WebDriverWait(driver, 40)

    model_links = CAICT_MODEL_LINKS
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
                "publisher": "PaddlePaddle", # 🔧 修复：确保 publisher 始终为 "PaddlePaddle" (统一大小写)
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


# 魔乐 Modelers (使用修复后的selenium版本)
def fetch_modelers_data_unified(progress_callback=None, progress_total=None):
    """统一获取魔乐Modelers上的PaddlePaddle模型"""
    from .selenium import ModelersFetcher

    fetcher = ModelersFetcher()
    fetcher.name = "魔乐 Modelers"
    return fetcher.fetch(progress_callback=progress_callback, progress_total=progress_total)


# Gitee (使用修复后的selenium版本)
def fetch_gitee_data_unified(progress_callback=None, progress_total=None):
    """统一获取Gitee上的PaddlePaddle模型"""
    from .selenium import GiteeFetcher

    fetcher = GiteeFetcher()
    fetcher.name = "Gitee"
    return fetcher.fetch(progress_callback=progress_callback, progress_total=progress_total)


# 统一的平台抓取器字典
UNIFIED_PLATFORM_FETCHERS = {
    "Hugging Face": fetch_hugging_face_data_unified,
    "ModelScope": fetch_modelscope_data_unified,
    "AI Studio": fetch_aistudio_data_unified,
    "GitCode": fetch_gitcode_data_unified,
    "鲸智": fetch_caict_data_unified,
    "魔乐 Modelers": fetch_modelers_data_unified,
    "Gitee": fetch_gitee_data_unified,
}


def fetch_all_paddlepaddle_data(platforms=None, progress_callback=None, progress_total=None, enable_model_tree=True):
    """
    一次性获取所有平台的PaddlePaddle模型数据（包含ERNIE-4.5和PaddleOCR-VL）

    Args:
        platforms: 要抓取的平台列表，None表示所有平台
        progress_callback: 进度回调函数
        progress_total: 总数参考
        enable_model_tree: 是否启用AI Studio Model Tree功能

    Returns:
        tuple: (DataFrame, 总数量)
    """
    if platforms is None:
        platforms = list(UNIFIED_PLATFORM_FETCHERS.keys())

    all_dfs = []
    total_count = 0
    aistudio_included = "AI Studio" in platforms

    for platform in platforms:
        fetcher = UNIFIED_PLATFORM_FETCHERS.get(platform)
        if fetcher:
            try:
                df, count = fetcher(progress_callback=progress_callback, progress_total=progress_total)
                all_dfs.append(df)
                total_count += count
                print(f"✅ {platform} 完成：获取 {count} 个模型")
            except Exception as e:
                print(f"❌ {platform} 失败：{e}")
        else:
            print(f"⚠️ 找不到 {platform} 的抓取器")

    # 注意：AI Studio Model Tree 已移至 app.py 中调用，避免重复执行
    # 原因：app.py 中需要更精细的 UI 进度控制，且能避免两层调用导致的重复执行
    # 参考：app.py 的并行和串行执行模式中的 Model Tree 调用

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df, total_count
    else:
        return pd.DataFrame(), 0
