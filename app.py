"""
ERNIE 模型下载量统计系统
主界面 - 使用 Streamlit 构建
"""
import streamlit as st
import pandas as pd
import time
from datetime import date
import concurrent.futures
import threading
from enum import Enum
import re

from ernie_tracker.config import DB_PATH, PLATFORM_NAMES
from ernie_tracker.db import (
    save_to_db,
    get_last_model_count,
    update_last_model_count,
    load_data_from_db,
    init_database,
)
from ernie_tracker.fetchers.fetchers_unified import (
    UNIFIED_PLATFORM_FETCHERS,
    fetch_all_paddlepaddle_data,
    fetch_hugging_face_data_unified,
)
import sqlite3


# =============================================================================
# 日志系统（美化版）
# =============================================================================

class LogLevel(Enum):
    """日志级别枚举"""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class LogEntry:
    """日志条目类"""
    def __init__(self, level: LogLevel, message: str, platform: str = None, timestamp: str = None):
        self.level = level
        self.message = message
        self.platform = platform
        self.timestamp = timestamp or time.strftime('%H:%M:%S')

    def to_html(self) -> str:
        """转换为HTML格式（带样式）"""
        # 根据级别选择颜色和图标
        level_styles = {
            LogLevel.INFO: {
                'icon': 'ℹ️',
                'color': '#3498db',
                'bg_color': '#ebf5fb'
            },
            LogLevel.SUCCESS: {
                'icon': '✅',
                'color': '#27ae60',
                'bg_color': '#e8f8f5'
            },
            LogLevel.WARNING: {
                'icon': '⚠️',
                'color': '#f39c12',
                'bg_color': '#fef5e7'
            },
            LogLevel.ERROR: {
                'icon': '❌',
                'color': '#e74c3c',
                'bg_color': '#fdedec'
            },
            LogLevel.DEBUG: {
                'icon': '🔍',
                'color': '#95a5a6',
                'bg_color': '#f4f6f7'
            }
        }

        style = level_styles[self.level]

        # 平台标签
        platform_tag = f'<span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;">{self.platform}</span>' if self.platform else ''

        # 构建HTML
        html = f'''
        <div style="
            padding: 8px 12px;
            margin: 4px 0;
            background: {style['bg_color']};
            border-left: 4px solid {style['color']};
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        ">
            <span style="color: #7f8c8d; margin-right: 8px;">[{self.timestamp}]</span>
            <span style="color: {style['color']}; font-weight: bold; margin-right: 8px;">{style['icon']}</span>
            <span style="color: #2c3e50;">{self.message}</span>
            {platform_tag}
        </div>
        '''
        return html

    def to_text(self) -> str:
        """转换为纯文本格式"""
        platform_str = f"[{self.platform}] " if self.platform else ""
        return f"[{self.timestamp}] {self.level.value} {platform_str}{self.message}"


class Logger:
    """日志管理器（线程安全）"""
    def __init__(self, max_logs: int = 100):
        self.logs = []
        self.max_logs = max_logs
        self.lock = threading.Lock()

        # 统计信息
        self.stats = {
            LogLevel.INFO: 0,
            LogLevel.SUCCESS: 0,
            LogLevel.WARNING: 0,
            LogLevel.ERROR: 0,
            LogLevel.DEBUG: 0
        }

    def log(self, level: LogLevel, message: str, platform: str = None):
        """添加日志"""
        with self.lock:
            entry = LogEntry(level, message, platform)
            self.logs.append(entry)
            self.stats[level] += 1

            # 保留最近的日志
            if len(self.logs) > self.max_logs:
                removed = self.logs.pop(0)
                self.stats[removed.level] -= 1

    def info(self, message: str, platform: str = None):
        """记录信息日志"""
        self.log(LogLevel.INFO, message, platform)

    def success(self, message: str, platform: str = None):
        """记录成功日志"""
        self.log(LogLevel.SUCCESS, message, platform)

    def warning(self, message: str, platform: str = None):
        """记录警告日志"""
        self.log(LogLevel.WARNING, message, platform)

    def error(self, message: str, platform: str = None):
        """记录错误日志"""
        self.log(LogLevel.ERROR, message, platform)

    def debug(self, message: str, platform: str = None):
        """记录调试日志"""
        self.log(LogLevel.DEBUG, message, platform)

    def get_logs(self, level: LogLevel = None, limit: int = None) -> list:
        """获取日志"""
        with self.lock:
            if level:
                filtered = [log for log in self.logs if log.level == level]
            else:
                filtered = self.logs.copy()

            if limit:
                return filtered[-limit:]
            return filtered

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                'total': len(self.logs),
                'info': self.stats[LogLevel.INFO],
                'success': self.stats[LogLevel.SUCCESS],
                'warning': self.stats[LogLevel.WARNING],
                'error': self.stats[LogLevel.ERROR],
                'debug': self.stats[LogLevel.DEBUG]
            }

    def clear(self):
        """清空日志"""
        with self.lock:
            self.logs.clear()
            for level in self.stats:
                self.stats[level] = 0

    def render_html(self, level: LogLevel = None, limit: int = 50) -> str:
        """渲染为HTML"""
        logs = self.get_logs(level, limit)
        if not logs:
            return '<div style="padding: 20px; text-align: center; color: #95a5a6;">暂无日志</div>'

        html_parts = []
        for entry in logs:
            html_parts.append(entry.to_html())

        return ''.join(html_parts)


# =============================================================================
# Model Tree 辅助函数（重构：减少代码重复）
# =============================================================================

def get_official_model_count(repo: str) -> int:
    """
    获取指定平台的官方模型总数（带缓存）

    Args:
        repo: 平台名称（如 'AI Studio', 'ModelScope'）

    Returns:
        int: 官方模型总数，如果查询失败则返回1
    """
    cache_key = f"official_count_{repo}"

    # 从session_state缓存中读取
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT model_name)
                FROM model_downloads
                WHERE repo = ?
                AND (
                    publisher IN ('百度', 'baidu', 'Paddle', 'PaddlePaddle', 'yiyan', '一言')
                    OR publisher LIKE '%百度%'
                    OR publisher LIKE '%baidu%'
                    OR publisher LIKE '%Paddle%'
                )
            """, (repo,))
            count = cursor.fetchone()[0] or 1
            st.session_state[cache_key] = count
            return count
    except sqlite3.Error as e:
        st.warning(f"查询{repo}官方模型数量失败: {e}")
        return 1
    except Exception as e:
        st.warning(f"获取{repo}官方模型数量时出错: {e}")
        return 1


def run_model_tree_with_progress(
    platform_name: str,
    fetch_func,
    save_to_db: bool = False
) -> tuple:
    """
    通用的Model Tree执行函数（带进度显示）

    Args:
        platform_name: 平台名称（如 'AI Studio', 'ModelScope'）
        fetch_func: 抓取函数，接受progress_callback参数
        save_to_db: 是否保存到数据库

    Returns:
        tuple: (df, count, elapsed_time)
            - df: 获取的DataFrame（可能为None）
            - count: 模型数量
            - elapsed_time: 耗时（秒）
    """
    # 检查是否启用Model Tree
    if not st.session_state.get('use_model_tree', True):
        return None, 0, 0

    # 创建进度显示区域
    st.markdown(f"### 🌳 {platform_name} Model Tree 进度")
    status = st.empty()
    progress = st.progress(0)
    details = st.empty()

    start_time = time.time()

    try:
        status.info(f"🔄 正在获取 {platform_name} 衍生模型...")

        def progress_callback(processed, discovered_total=None):
            """Model Tree进度回调函数"""
            total_official = get_official_model_count(platform_name)
            progress_pct = min(processed / total_official, 1.0) if total_official > 0 else 0
            progress.progress(progress_pct)
            details.info(f"已处理 {processed} / {total_official} 个官方模型")

        # 执行Model Tree抓取
        model_tree_df, model_tree_count = fetch_func(progress_callback=progress_callback)

        elapsed = time.time() - start_time

        # 显示结果
        if model_tree_count > 0:
            status.success(f"✅ {platform_name} Model Tree 完成")
            progress.progress(1.0)
            details.success(f"获取 {model_tree_count} 个衍生模型，用时 {elapsed:.2f} 秒")
        else:
            status.info("ℹ️  未找到新的衍生模型")
            progress.progress(1.0)
            details.info(f"用时 {elapsed:.2f} 秒")

        return model_tree_df, model_tree_count, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        status.error(f"❌ Model Tree 失败")
        st.warning(f"⚠️  {platform_name} Model Tree 失败（不影响主流程）：{e}，用时 {elapsed:.2f} 秒")
        return None, 0, elapsed


# 页面配置
st.set_page_config(page_title="ERNIE模型下载数据统计", layout="wide")
st.title("📊 ERNIE模型下载数据统计")


def fetch_platform_data_only(platform_name, fetch_func, save_to_database=True, log_callback=None, progress_update_callback=None):
    """
    仅执行数据抓取（不包含UI操作，用于并行执行）

    Args:
        platform_name: 平台名称
        fetch_func: 抓取函数
        save_to_database: 是否保存到数据库
        log_callback: 日志回调函数（用于实时输出日志）
        progress_update_callback: 进度更新回调函数（用于实时更新进度条）

    Returns:
        tuple: (platform_name, DataFrame, success, elapsed_time, error_message, progress_updates)
    """
    # 获取上次记录的模型数量
    last_count = get_last_model_count(platform_name)

    # 进度更新信息列表
    progress_updates = []

    # 保存当前参考总数（使用字典避免闭包问题）
    ref = {"denom": last_count}

    def progress_callback(processed, discovered_total=None):
        """进度回调函数（收集进度信息并输出日志）"""
        if ref["denom"]:  # 有参考总数
            denom = ref["denom"]
            if processed > denom:
                if save_to_database:
                    update_last_model_count(platform_name, processed)
                ref["denom"] = processed
                denom = processed

            progress = min(processed / denom, 1.0)
            message = f"已处理 {processed} / 参考总数 {denom}"
            progress_data = {
                'processed': processed,
                'total': denom,
                'progress': progress,
                'message': message
            }
            progress_updates.append(progress_data)

            # 实时输出日志
            if log_callback:
                log_callback(f"[{platform_name}] {message}")

            # 实时更新进度条
            if progress_update_callback:
                progress_update_callback(progress_data)
        else:  # 首次运行
            if discovered_total:
                progress = processed / discovered_total
                message = f"已处理 {processed} / 实际总数 {discovered_total}"
                progress_data = {
                    'processed': processed,
                    'total': discovered_total,
                    'progress': progress,
                    'message': message
                }
                progress_updates.append(progress_data)
            else:
                message = f"已处理 {processed} （总数未知）"
                progress_data = {
                    'processed': processed,
                    'total': None,
                    'progress': None,
                    'message': message
                }
                progress_updates.append(progress_data)

            # 实时输出日志
            if log_callback:
                log_callback(f"[{platform_name}] {message}")

            # 实时更新进度条
            if progress_update_callback:
                progress_update_callback(progress_data)

    # 执行数据获取
    start_time = time.time()
    try:
        df, total_count = fetch_func(progress_callback=progress_callback, progress_total=ref["denom"])
        elapsed_time = time.time() - start_time

        # 保存到数据库
        if save_to_database:
            if total_count is not None and total_count != ref["denom"]:
                update_last_model_count(platform_name, total_count)
            save_to_db(df, DB_PATH)
            status_message = f"✅ 完成：共发现 {total_count} 个模型，已保存到数据库。"
        else:
            status_message = f"✅ 完成：共发现 {total_count} 个模型，仅获取数据。"

        progress_updates.append({
            'status': 'completed',
            'message': status_message
        })

        return platform_name, df, True, elapsed_time, None, progress_updates

    except Exception as e:
        error_message = f"❌ 爬取失败: {e}"
        progress_updates.append({
            'status': 'error',
            'message': error_message
        })
        return platform_name, None, False, time.time() - start_time, error_message, progress_updates


def run_platform_fetcher(platform_name, fetch_func, save_to_database=True, ui_container=None):
    """
    运行单个平台的数据抓取（包含UI更新，用于串行模式）

    Args:
        platform_name: 平台名称
        fetch_func: 抓取函数
        save_to_database: 是否保存到数据库
        ui_container: UI容器（兼容参数）

    Returns:
        DataFrame: 抓取的数据
    """
    if ui_container is None:
        # 兼容原有的独立UI模式
        st.subheader(platform_name)

    # 获取上次记录的模型数量
    last_count = get_last_model_count(platform_name)

    # 串行模式 - 原有UI显示方式
    st.write(
        f"上次记录的模型数量：{last_count if last_count is not None else '暂无记录（首次运行）'}"
    )
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    # 保存当前参考总数
    ref = {"denom": last_count}

    def progress_callback(processed, discovered_total=None):
        """进度回调函数"""
        if ref["denom"]:  # 有参考总数
            denom = ref["denom"]
            if processed > denom:
                if save_to_database:
                    update_last_model_count(platform_name, processed)
                ref["denom"] = processed
                denom = processed

            progress = min(processed / denom, 1.0)
            progress_bar.progress(progress)
            status_placeholder.text(
                f"已处理 {processed} / 参考总数 {denom}"
            )
        else:  # 首次运行
            if discovered_total:
                progress_bar.progress(processed / discovered_total)
                status_placeholder.text(
                    f"已处理 {processed} / 实际总数 {discovered_total}"
                )
            else:
                status_placeholder.text(f"已处理 {processed} （总数未知）")

    # 执行数据获取
    start_time = time.time()
    try:
        df, total_count = fetch_func(progress_callback=progress_callback, progress_total=last_count)
        elapsed_time = time.time() - start_time

        # 保存到数据库
        if save_to_database:
            if total_count is not None and total_count != last_count:
                update_last_model_count(platform_name, total_count)
            save_to_db(df, DB_PATH)
            status_message = f"完成：共发现 {total_count} 个模型，已保存到数据库。"
        else:
            status_message = f"完成：共发现 {total_count} 个模型，仅获取数据。"

        status_placeholder.text(status_message)
        progress_bar.progress(1.0)
        return df

    except Exception as e:
        st.error(f"{platform_name} 爬取失败: {e}")
        return None


def run_platforms_parallel(platforms, fetchers_to_use, save_to_database=True):
    """
    并行运行多个平台的数据抓取（修复版：实时进度显示）

    Args:
        platforms: 平台名称列表
        fetchers_to_use: 平台抓取函数字典
        save_to_database: 是否保存到数据库

    Returns:
        tuple: (DataFrame列表, 总用时)
    """
    # 支持Model Tree的平台列表
    model_tree_platforms = {"AI Studio", "ModelScope"}

    all_dfs = []
    total_start_time = time.time()

    # 创建UI容器 - 使用st.status来显示实时进度
    st.markdown("### ⏳ 并行更新进度")

    # 创建美化的日志系统
    logger = Logger(max_logs=200)

    # 共享的进度状态（线程安全）
    progress_state = {}
    for platform in platforms:
        progress_state[platform] = {
            'latest_update': None,
            'lock': threading.Lock()
        }
        # 为支持Model Tree的平台添加Model Tree进度状态
        if platform in model_tree_platforms:
            progress_state[f"{platform}_model_tree"] = {
                'latest_update': None,
                'lock': threading.Lock()
            }

    def log_callback_wrapper(message):
        """日志回调函数包装器（解析日志级别）"""
        # 解析日志级别
        level = LogLevel.INFO
        if message.startswith("✅") or "完成" in message or "成功" in message:
            level = LogLevel.SUCCESS
        elif message.startswith("❌") or "失败" in message or "错误" in message:
            level = LogLevel.ERROR
        elif message.startswith("⚠️") or "警告" in message:
            level = LogLevel.WARNING

        # 提取平台名称
        platform_match = re.match(r'\[(.*?)\]', message)
        platform = platform_match.group(1) if platform_match else None

        logger.log(level, message, platform)

    def update_progress(platform_name, progress_data):
        """线程安全的进度更新函数"""
        with progress_state[platform_name]['lock']:
            progress_state[platform_name]['latest_update'] = progress_data

    # 创建一个占位容器用于显示所有平台的状态
    status_container = st.container()

    with status_container:
        # 为每个平台创建状态显示区域
        platform_status = {}
        for platform in platforms:
            with st.expander(f"🔄 {platform}", expanded=True):
                platform_status[platform] = {
                    'status': st.empty(),
                    'progress': st.progress(0),
                    'details': st.empty(),
                    'time': st.empty()
                }
                platform_status[platform]['status'].info(f"🔄 {platform} 等待中...")

                # 为支持Model Tree的平台添加Model Tree进度显示
                if platform in model_tree_platforms:
                    st.markdown("---")
                    st.markdown(f"**🌳 {platform} Model Tree**")
                    platform_status[f"{platform}_model_tree"] = {
                        'progress': st.progress(0),
                        'details': st.empty()
                    }
                    platform_status[f"{platform}_model_tree"]['details'].info("等待Search完成...")

        # 添加美化后的日志输出区域
        st.markdown("---")

        # 日志控制栏
        log_control_col1, log_control_col2, log_control_col3 = st.columns([1, 1, 2])

        with log_control_col1:
            show_logs = st.checkbox("显示日志", value=True)

        with log_control_col2:
            log_level_filter = st.selectbox(
                "日志级别",
                ["全部", "INFO", "SUCCESS", "WARNING", "ERROR"],
                index=0
            )

        st.markdown("#### 📝 实时日志")
        log_stats_placeholder = st.empty()
        log_placeholder = st.empty()

    def fetch_platform_task(platform_name):
        """单个平台抓取任务（纯数据处理，不包含UI操作）"""
        try:
            fetch_func = fetchers_to_use.get(platform_name)
            if fetch_func:
                return fetch_platform_data_only(
                    platform_name,
                    fetch_func,
                    save_to_database,
                    log_callback=log_callback_wrapper,
                    progress_update_callback=lambda data: update_progress(platform_name, data)
                )
            return platform_name, None, False, 0, "抓取函数未找到", []
        except Exception as e:
            import traceback
            error_msg = f"任务执行异常: {str(e)}\n{traceback.format_exc()}"
            log_callback_wrapper(f"❌ [{platform_name}] {error_msg}")
            return platform_name, None, False, 0, error_msg, []

    def fetch_model_tree_task(platform_name):
        """单个平台的Model Tree任务（纯数据处理）"""
        try:
            # 获取官方模型数量作为参考总数
            official_count = get_official_model_count(platform_name)

            # 创建Model Tree进度回调函数
            def model_tree_progress_callback(p, **kwargs):
                # 处理两种类型的调用：
                # 1. 字符串 - 日志消息（来自 fetch_aistudio_model_tree 的 log() 函数）
                # 2. 整数 - 进度更新
                if isinstance(p, str):
                    # 字符串：仅输出日志
                    log_callback_wrapper(f"[{platform_name} Model Tree] {p}")
                else:
                    # 整数：输出日志并更新进度条
                    log_callback_wrapper(f"[{platform_name} Model Tree] 已处理 {p} 个官方模型")
                    update_progress(f"{platform_name}_model_tree", {
                        'processed': p,
                        'total': official_count,
                        'progress': min(p / official_count, 1.0) if official_count > 0 else 0,
                        'message': f"已处理 {p} / {official_count} 个官方模型"
                    })

            # 根据平台选择对应的Model Tree函数
            if platform_name == "AI Studio":
                from ernie_tracker.fetchers.fetchers_modeltree import fetch_aistudio_model_tree
                df, count = fetch_aistudio_model_tree(
                    progress_callback=model_tree_progress_callback,
                    save_to_db=save_to_database,
                    test_mode=False
                )
                return platform_name, df, count > 0, 0, None, []
            elif platform_name == "ModelScope":
                from ernie_tracker.fetchers.fetchers_modeltree import update_modelscope_model_tree
                df, count = update_modelscope_model_tree(
                    save_to_db=save_to_database,
                    auto_discover=True,
                    progress_callback=model_tree_progress_callback
                )
                return platform_name, df, count > 0, 0, None, []
            else:
                # 不支持Model Tree的平台
                return platform_name, None, False, 0, "该平台不支持Model Tree", []
        except Exception as e:
            import traceback
            error_msg = f"Model Tree执行异常: {str(e)}\n{traceback.format_exc()}"
            return platform_name, None, False, 0, error_msg, []

    # 使用线程池并行执行
    platforms_with_model_tree = [p for p in platforms if p in model_tree_platforms]
    platforms_without_model_tree = [p for p in platforms if p not in model_tree_platforms]

    # 统计任务总数（Search任务 + Model Tree任务）
    search_count = len(platforms)
    model_tree_count = len(platforms_with_model_tree) if st.session_state.get('use_model_tree', True) else 0
    total_tasks = search_count + model_tree_count

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(platforms) + model_tree_count, 6)) as executor:
        # 提交所有Search任务
        future_to_platform = {
            executor.submit(fetch_platform_task, platform): ('search', platform)
            for platform in platforms
        }

        completed_count = 0
        search_completed_count = 0

        # 总体进度显示
        overall_placeholder = st.empty()

        # 实时更新各平台状态
        while completed_count < total_tasks:
            # 先检查并更新所有平台的进度（包括未完成的）
            for platform in platforms:
                # 更新Search进度
                with progress_state[platform]['lock']:
                    latest = progress_state[platform]['latest_update']
                    if latest and 'progress' in latest:
                        try:
                            # 更新进度条
                            platform_status[platform]['progress'].progress(latest['progress'])
                            # 更新详细信息
                            if latest['message']:
                                platform_status[platform]['details'].info(latest['message'])
                        except Exception as e:
                            # 忽略UI更新错误，避免中断流程
                            pass

                # 更新Model Tree进度（如果支持）
                if platform in model_tree_platforms:
                    model_tree_key = f"{platform}_model_tree"
                    with progress_state[model_tree_key]['lock']:
                        latest_mt = progress_state[model_tree_key]['latest_update']
                        if latest_mt and 'progress' in latest_mt:
                            try:
                                # 更新Model Tree进度条
                                platform_status[model_tree_key]['progress'].progress(latest_mt['progress'])
                                # 更新Model Tree详细信息
                                if latest_mt['message']:
                                    platform_status[model_tree_key]['details'].info(latest_mt['message'])
                            except Exception as e:
                                # 忽略UI更新错误
                                pass

            # 检查已完成的任务
            for future in list(future_to_platform.keys()):
                if future.done():
                    task_type, platform_name = future_to_platform.pop(future)
                    completed_count += 1

                    try:
                        # 获取结果
                        _, df, success, elapsed_time, error_message, progress_updates = future.result()

                        if task_type == 'search':
                            # Search任务完成
                            search_completed_count += 1

                            # 更新该平台的Search状态
                            if success:
                                platform_status[platform_name]['status'].info(f"✅ {platform_name} Search完成")
                                final_message = progress_updates[-1]['message'] if progress_updates else "Search完成"
                                platform_status[platform_name]['details'].info(final_message)
                                platform_status[platform_name]['time'].info(f"⏱️ Search用时: {elapsed_time:.2f} 秒")

                                if df is not None:
                                    all_dfs.append(df)

                                # 如果该平台支持Model Tree且用户启用了Model Tree，立即提交Model Tree任务
                                if platform_name in model_tree_platforms and st.session_state.get('use_model_tree', True):
                                    platform_status[platform_name]['status'].info(f"🌳 {platform_name} 开始Model Tree...")
                                    # 更新Model Tree状态为运行中
                                    model_tree_key = f"{platform_name}_model_tree"
                                    platform_status[model_tree_key]['details'].info("🔄 Model Tree运行中...")
                                    future_to_platform[executor.submit(fetch_model_tree_task, platform_name)] = ('model_tree', platform_name)
                                    log_callback_wrapper(f"[{platform_name}] Search完成，开始Model Tree")
                                else:
                                    # 不支持Model Tree的平台，标记为完全完成
                                    platform_status[platform_name]['status'].success(f"✅ {platform_name} 完成")
                                    platform_status[platform_name]['progress'].progress(1.0)
                            else:
                                # Search失败
                                platform_status[platform_name]['status'].error(f"❌ {platform_name} Search失败")
                                platform_status[platform_name]['details'].error(error_message)
                                platform_status[platform_name]['time'].error(f"⏱️ 用时: {elapsed_time:.2f} 秒")

                        elif task_type == 'model_tree':
                            # Model Tree任务完成
                            model_tree_key = f"{platform_name}_model_tree"
                            if success:
                                platform_status[platform_name]['status'].success(f"✅ {platform_name} 完成（含Model Tree）")
                                platform_status[model_tree_key]['details'].success("✅ Model Tree完成")
                                platform_status[platform_name]['time'].success(f"⏱️ Model Tree用时: {elapsed_time:.2f} 秒")
                                platform_status[platform_name]['progress'].progress(1.0)
                                platform_status[model_tree_key]['progress'].progress(1.0)

                                if df is not None and not df.empty:
                                    all_dfs.append(df)
                            else:
                                # Model Tree失败（不影响Search的成功状态）
                                platform_status[platform_name]['status'].warning(f"⚠️ {platform_name} Search完成，Model Tree失败")
                                platform_status[model_tree_key]['details'].warning(f"❌ Model Tree失败: {error_message}")
                                platform_status[model_tree_key]['progress'].progress(1.0)

                    except Exception as e:
                        if task_type == 'search':
                            platform_status[platform_name]['status'].error(f"❌ {platform_name} 异常")
                            platform_status[platform_name]['details'].error(f"执行异常: {e}")
                        else:
                            platform_status[platform_name]['status'].warning(f"⚠️ {platform_name} Model Tree异常")
                            platform_status[platform_name]['details'].warning(f"Model Tree异常: {e}")

                    # 更新总体进度
                    overall_placeholder.info(f"🎯 总体进度：{completed_count}/{total_tasks} 个任务完成（Search: {search_completed_count}/{search_count}）")

            # 更新美化后的日志显示
            if show_logs:
                # 显示日志统计
                stats = logger.get_stats()
                stats_html = f"""
                <div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px;">
                    <strong>日志统计：</strong>
                    <span style="color: #3498db;">总计 {stats['total']}</span> |
                    <span style="color: #3498db;">ℹ️ INFO {stats['info']}</span> |
                    <span style="color: #27ae60;">✅ SUCCESS {stats['success']}</span> |
                    <span style="color: #f39c12;">⚠️ WARNING {stats['warning']}</span> |
                    <span style="color: #e74c3c;">❌ ERROR {stats['error']}</span>
                </div>
                """
                log_stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

                # 根据筛选条件渲染日志
                level_map = {
                    "INFO": LogLevel.INFO,
                    "SUCCESS": LogLevel.SUCCESS,
                    "WARNING": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR
                }
                filter_level = level_map.get(log_level_filter) if log_level_filter != "全部" else None

                logs_html = logger.render_html(level=filter_level, limit=100)
                log_placeholder.markdown(logs_html, unsafe_allow_html=True)

            # 短暂休眠避免过度占用CPU
            time.sleep(0.5)

    total_elapsed_time = time.time() - total_start_time

    # ========== 最终总结 ==========
    final_elapsed_time = time.time() - total_start_time

    # 统计Model Tree任务数量
    model_tree_tasks_count = len(platforms_with_model_tree) if st.session_state.get('use_model_tree', True) else 0

    if model_tree_tasks_count > 0:
        overall_placeholder.success(
            f"🎯 全部完成！总用时：{final_elapsed_time:.2f} 秒"
            f"（完成 {search_count} 个Search任务 + {model_tree_tasks_count} 个Model Tree任务）"
        )
        logger.success(f"全部完成！总用时：{final_elapsed_time:.2f} 秒", None)
    else:
        overall_placeholder.success(f"🎯 并行抓取完成！总用时：{total_elapsed_time:.2f} 秒")
        logger.success(f"并行抓取完成！总用时：{total_elapsed_time:.2f} 秒", None)

    # 显示最终日志统计
    if show_logs:
        final_stats = logger.get_stats()
        st.markdown("---")
        st.markdown("### 📊 日志统计摘要")

        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        with stat_col1:
            st.metric("总日志数", final_stats['total'])
        with stat_col2:
            st.metric("INFO", final_stats['info'], delta_color="normal")
        with stat_col3:
            st.metric("SUCCESS", final_stats['success'], delta_color="normal")
        with stat_col4:
            st.metric("WARNING", final_stats['warning'])
        with stat_col5:
            st.metric("ERROR", final_stats['error'])

    return all_dfs, total_elapsed_time


# 初始化数据库
init_database()

# 侧边栏导航
st.sidebar.title("🔧 功能选择")
page = st.sidebar.radio(
    " ",
    [
        "📥 数据更新",
        "📊 ERNIE-4.5 分析",
        "📊 PaddleOCR-VL 分析",
        "🌳 衍生模型生态",
        "🗄️ 数据库管理",
    ],
    index=0,
)


# ================= 数据更新模块 =================
if page == "📥 数据更新":
    from ernie_tracker.analysis import get_available_dates
    import os
    st.markdown("## 📥 数据更新")
    st.info("🚀 **优化更新模式**：现在一次更新即可获取所有PaddlePaddle模型数据（包含ERNIE-4.5和PaddleOCR-VL），无需分别选择！")

    # Model Tree 选项
    use_model_tree = st.checkbox(
        "🌳 使用 Model Tree 功能（获取衍生模型）",
        value=True,
        key='use_model_tree',
        help="启用后会获取ERNIE-4.5和PaddleOCR-VL的所有衍生模型，包括Finetune、Adapter等"
    )

    if use_model_tree:
        st.info("🔍 **Model Tree模式**：将获取ERNIE-4.5和PaddleOCR-VL的所有衍生模型，并自动分类识别Finetune、Adapter、LoRA等类型")

    # 使用统一的平台抓取器
    fetchers_to_use = UNIFIED_PLATFORM_FETCHERS.copy()

    # 根据Model Tree选项更新Hugging Face获取器
    fetchers_to_use["Hugging Face"] = lambda **kwargs: fetch_hugging_face_data_unified(
        progress_callback=kwargs.get('progress_callback'),
        progress_total=kwargs.get('progress_total'),
        use_model_tree=use_model_tree  # 传递用户的选择
    )

    platform_options = list(fetchers_to_use.keys())

    # 初始化 session_state
    if "select_all" not in st.session_state:
        st.session_state.select_all = False

    # 平台选择
    with st.container():
        toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1, 5, 1])

        with toolbar_col1:
            if st.button("✅ 全选 / 取消"):
                st.session_state.select_all = not st.session_state.select_all

        with toolbar_col2:
            platforms = st.multiselect(
                "选择需要更新的平台",
                platform_options,
                default=platform_options if st.session_state.select_all else [],
                label_visibility="collapsed"
            )

        with toolbar_col3:
            run_all = st.button("🚀 更新数据", use_container_width=True)

    # 数据保存选项
    st.markdown("### ⚙️ 数据保存设置")
    save_to_db_option = st.radio(
        "选择数据处理方式：",
        options=["保存到数据库", "仅获取数据（不保存）"],
        index=0,
        horizontal=True,
        help="选择是否将爬取的数据保存到数据库中"
    )

    save_to_database = (save_to_db_option == "保存到数据库")

    if save_to_database:
        st.info("💾 数据将保存到数据库，并更新平台统计信息")
    else:
        st.warning("⚠️ 数据仅用于预览，不会保存到数据库")

    # 执行模式选择
    st.markdown("### 🚀 执行模式")
    execution_mode = st.radio(
        "选择执行模式：",
        options=["🚀 并行执行（推荐）", "🔄 串行执行"],
        index=0,
        horizontal=True,
        help="并行执行可以大幅提升多平台抓取效率"
    )

    use_parallel = (execution_mode == "🚀 并行执行（推荐）")

    if use_parallel:
        st.info("⚡ 各平台将同时进行数据抓取，大幅提升效率")
    else:
        st.warning("🐌 各平台将依次进行数据抓取，耗时较长")

    st.markdown("---")

    # 执行更新
    if run_all:
        if not platforms:
            st.warning("请至少选择一个平台再更新。")
        else:
            all_dfs = []
            total_elapsed_time = 0
            total_start_time = time.time()  # 初始化总开始时间（用于计算最终用时）

            if use_parallel:
                # 并行执行模式
                all_dfs, total_elapsed_time = run_platforms_parallel(
                    platforms, fetchers_to_use, save_to_database
                )
            else:
                # 串行执行模式（改进逻辑：每个平台Search完成后立即执行Model Tree）
                st.markdown("### ⏳ 串行更新进度")
                progress_placeholder = st.empty()

                # 支持Model Tree的平台
                model_tree_platforms = {"AI Studio", "ModelScope"}

                for idx, platform in enumerate(platforms, start=1):
                    progress_placeholder.info(f"正在更新：**{platform}** ({idx}/{len(platforms)})")

                    # 步骤1: 调用平台Search抓取函数
                    fetch_func = fetchers_to_use.get(platform)
                    if fetch_func:
                        df = run_platform_fetcher(platform, fetch_func, save_to_database)
                        if df is not None:
                            all_dfs.append(df)

                        elapsed = time.time() - total_start_time
                        status_msg = "数据已保存" if save_to_database else "仅预览"
                        st.success(f"✅ {platform} Search完成，用时 {elapsed:.2f} 秒，{status_msg}")

                        # 步骤2: 如果该平台支持Model Tree且用户启用了Model Tree，立即执行
                        if platform in model_tree_platforms and st.session_state.get('use_model_tree', True):
                            st.info(f"🌳 开始执行 {platform} Model Tree...")

                            if platform == "AI Studio":
                                from ernie_tracker.fetchers.fetchers_modeltree import fetch_aistudio_model_tree
                                df_mt, count_mt, elapsed_mt = run_model_tree_with_progress(
                                    "AI Studio",
                                    lambda progress_callback: fetch_aistudio_model_tree(
                                        progress_callback=progress_callback,
                                        save_to_db=save_to_database,
                                        test_mode=False
                                    ),
                                    save_to_db=False
                                )
                            elif platform == "ModelScope":
                                from ernie_tracker.fetchers.fetchers_modeltree import update_modelscope_model_tree
                                df_mt, count_mt, elapsed_mt = run_model_tree_with_progress(
                                    "ModelScope",
                                    lambda progress_callback: update_modelscope_model_tree(
                                        save_to_db=save_to_database,
                                        auto_discover=True,
                                        progress_callback=progress_callback
                                    ),
                                    save_to_db=False
                                )

                            if df_mt is not None and not df_mt.empty:
                                all_dfs.append(df_mt)

                            total_elapsed = time.time() - total_start_time
                            st.success(f"✅ {platform} Model Tree完成，总用时 {total_elapsed:.2f} 秒")

                total_elapsed_time = time.time() - total_start_time
                st.info(f"🎯 串行抓取完成！总用时：{total_elapsed_time:.2f} 秒")

            # 数据预览
            st.markdown("### 📄 本次更新数据预览")
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                st.dataframe(final_df, use_container_width=True)

                # 下载按钮
                csv_data = final_df.to_csv(index=False).encode("utf-8-sig")
                download_label = "⬇️ 下载本次更新数据 (CSV)" if save_to_database else "⬇️ 下载获取的数据 (CSV)"

                st.download_button(
                    label=download_label,
                    data=csv_data,
                    file_name=f"paddlepaddle_models_downloads_{date.today().isoformat()}.csv",
                    mime="text/csv",
                    use_container_width=False
                )

    # 导出指定日期数据
    st.markdown("### 📁 导出指定日期数据到本地")

    # 获取可用日期
    available_dates_export = get_available_dates()

    if not available_dates_export:
        st.warning("⚠️ 数据库中暂无数据可供导出。")
    else:
        selected_date = st.selectbox(
            "选择要导出的日期",
            options=available_dates_export,
            index=0,
            key="export_date_selector",
            help="选择一个日期，将其数据导出为 Excel 文件到 Data 文件夹"
        )

        if st.button("💾 导出到 Data 文件夹"):
            with st.spinner(f"正在导出 {selected_date} 的数据..."):
                df_export = load_data_from_db(date_filter=selected_date)
                
                if df_export.empty:
                    st.error(f"❌ 未找到 {selected_date} 的数据。")
                else:
                    output_dir = "Data"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"data_{selected_date}.xlsx")
                    
                    try:
                        # 保存到 Excel
                        df_export.to_excel(output_path, index=False, engine='openpyxl')
                        st.success(f"✅ 数据成功导出到: `{output_path}`")
                        st.info(f"共导出 {len(df_export)} 条记录。")
                    except Exception as e:
                        st.error(f"导出文件时出错: {e}")


# ================= ERNIE-4.5 数据分析模块 =================
elif page == "📊 ERNIE-4.5 分析":
    from ernie_tracker.analysis import calculate_weekly_report, format_report_tables, get_available_dates, get_last_friday
    from datetime import datetime

    st.markdown("## 📈 周报分析")
    st.markdown("分析当前日期与对比日期之间的下载量增长情况")

    # 获取可用日期
    available_dates = get_available_dates()

    if not available_dates:
        st.warning("⚠️ 数据库中暂无数据，请先在「数据更新」页面抓取数据。")
    else:
        # 日期选择
        col1, col2 = st.columns(2)

        with col1:
            current_date = st.selectbox(
                "📅 当前日期",
                options=available_dates,
                index=0,
                help="选择要分析的当前日期"
            )

        with col2:
            # 默认为上周五
            default_previous = get_last_friday(current_date)
            if default_previous in available_dates:
                default_index = available_dates.index(default_previous)
            else:
                default_index = min(1, len(available_dates) - 1)

            previous_date = st.selectbox(
                "📅 对比日期",
                options=available_dates,
                index=default_index,
                help="选择要对比的日期（通常为上周五）"
            )

        if st.button("🔍 生成周报", type="primary"):
            with st.spinner("正在分析数据..."):
                report_data = calculate_weekly_report(current_date, previous_date, model_series='ERNIE-4.5')

            if report_data is None:
                st.error("❌ 无法生成周报，请检查选择的日期是否有数据。")
            else:
                # 保存到session_state
                st.session_state['report_data_ernie'] = report_data
                st.session_state['current_date'] = current_date
                st.session_state['previous_date'] = previous_date
                st.rerun()

        # 显示周报结果（从session_state或新生成的）
        report_data = st.session_state.get('report_data_ernie')

        if report_data is not None:
            tables = format_report_tables(report_data)

            # 获取保存的日期
            saved_current_date = st.session_state.get('current_date', current_date)
            saved_previous_date = st.session_state.get('previous_date', previous_date)

            st.success(f"✅ 周报生成成功！对比时间段：{saved_previous_date} → {saved_current_date}")

            # 检查并显示负增长警告
            warnings_df = tables.get('negative_growth_warnings')
            if warnings_df is not None and not warnings_df.empty:
                st.markdown("### ⚠️ 负增长警告")
                st.error(f"检测到 {len(warnings_df)} 个模型出现负增长！这可能表示数据采集问题或模型被下架。")
                st.dataframe(warnings_df, use_container_width=True)

                # 保存warnings_df到session_state
                st.session_state['warnings_df'] = warnings_df

                # 添加重新获取按钮
                with st.expander("🔄 重新获取负增长模型下载量", expanded=False):
                    st.info("💡 此功能将重新从平台API获取这些模型的最新下载量并更新到数据库。目前支持 Hugging Face 和 ModelScope 平台。")

                    if st.button("🚀 开始重新获取", type="primary", key="refetch_ernie"):
                        # 直接在按钮回调中执行，不要rerun
                        if 'warnings_df' in st.session_state:
                            warnings_data = st.session_state['warnings_df']

                            # 转换warnings_df为负增长模型列表
                            negative_list = []
                            for idx, row in warnings_data.iterrows():
                                negative_list.append({
                                    'platform': row['平台'],
                                    'model_name': row['模型名称'],
                                    'publisher': row['发布者'],
                                    'current': row['本周下载量']
                                })

                            # 获取current_date，用于保存数据
                            target_date = st.session_state.get('current_date', date.today().isoformat())

                            st.write(f"🔄 准备重新获取 {len(negative_list)} 个模型，将保存到日期: {target_date}")

                            # 执行重新获取
                            try:
                                from ernie_tracker.fetchers.fetchers_single_model import refetch_models_batch
                                from ernie_tracker.db import save_to_db

                                with st.spinner("正在重新获取模型下载量..."):
                                    success_list, failed_list = refetch_models_batch(negative_list, target_date=target_date)

                                # 直接保存成功的数据到数据库
                                if success_list:
                                    saved_count = 0
                                    for item in success_list:
                                        record = item['record']
                                        try:
                                            save_to_db(pd.DataFrame([record]), DB_PATH)
                                            saved_count += 1
                                        except Exception as e:
                                            st.error(f"❌ 保存 {item['model_name']} 失败: {e}")
                                    st.success(f"✅ 成功重新获取并保存 {saved_count} 条记录到数据库！")

                                # 显示结果
                                st.markdown("#### 📊 重新获取结果")

                                if success_list:
                                    st.info(f"✅ 成功重新获取 {len(success_list)} 个模型")
                                    success_df = pd.DataFrame(success_list)[['platform', 'model_name', 'old_count', 'new_count', 'change']]
                                    success_df.columns = ['平台', '模型名称', '原下载量', '新下载量', '变化']
                                    st.dataframe(success_df, use_container_width=True)

                                if failed_list:
                                    st.warning(f"⚠️ {len(failed_list)} 个模型获取失败")
                                    failed_df = pd.DataFrame(failed_list)[['platform', 'model_name', 'publisher']]
                                    failed_df.columns = ['平台', '模型名称', '发布者']
                                    st.dataframe(failed_df, use_container_width=True)

                                # 刷新页面以显示更新后的数据
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ 重新获取过程中出错: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                        else:
                            st.error("❌ 未找到 warnings_df，请重新生成周报")

                st.markdown("---")

            # 显示总体情况摘要
            st.markdown("### 📝 总体情况摘要")
            stats = report_data['summary_stats']

            # 格式化数字
            def format_num(n):
                return f"{n/10000:.2f}万"

            def format_percent(p):
                return f"{p:.2%}"

            # 计算百分比
            official_total_percent = stats['official_current_total'] / stats['all_current_total'] if stats['all_current_total'] else 0
            derivative_total_percent = stats['derivative_current_total'] / stats['all_current_total'] if stats['all_current_total'] else 0
            official_growth_percent = stats['official_growth'] / stats['all_growth'] if stats['all_growth'] else 0
            derivative_growth_percent = stats['derivative_growth'] / stats['all_growth'] if stats['all_growth'] else 0

            summary_text = f"""
            截至 **{saved_current_date}**，模型累计下载 **{format_num(stats['all_current_total'])}** 次
            （含官方模型 **{format_num(stats['official_current_total'])}** 次，占比 **{format_percent(official_total_percent)}**，
            衍生 **{format_num(stats['derivative_current_total'])}** 次，占比 **{format_percent(derivative_total_percent)}**），
            较上周增长 **{format_num(stats['all_growth'])}** 次
            （官方模型 **{format_num(stats['official_growth'])}** 次，占比 **{format_percent(official_growth_percent)}**，
            衍生模型增长 **{format_num(stats['derivative_growth'])}** 次，占比 **{format_percent(derivative_growth_percent)}**）。
            """
            st.markdown(summary_text)

            # 累计/本周新增衍生模型数量
            new_models_list_count = len(tables.get('all_new_models', pd.DataFrame()))
            st.info(
                f"累计衍生模型：{int(stats.get('derivative_current_total_models', 0) or 0)} 个｜"
                f"本周新增衍生（HF非官方差集）：{int(stats.get('derivative_new_models', 0) or 0)} 个｜"
                f"新增列表展示：{new_models_list_count} 个"
            )

            # 社区和模型维度摘要
            st.markdown("### 📈 社区与模型维度摘要")
            community_summary = report_data['community_summary']

            # 社区维度
            community_text = f"""
            - **社区维度**：Hugging Face下载量最高，**{community_summary['hf_top_model_name']}** 为本周HF平台下载最高模型，增长 **{community_summary['hf_top_model_growth']/10000:.2f}万** 次。
            """
            st.markdown(community_text)

            # 模型维度
            top3_downloads_str = " > ".join([f"{name}({int(val)})" for name, val in community_summary['top3_downloads_details'].items()])
            top3_growth_str = " > ".join([f"{name}({int(val)})" for name, val in community_summary['top3_growth_details'].items()])

            model_text = f"""
            - **模型维度**：
                - 模型（官方）下载总量前三位：{top3_downloads_str}
                - 本周（官方）增长最快前三位：{top3_growth_str}
            """
            st.markdown(model_text)

            # 显示汇总信息
            st.markdown("### 📊 平台汇总")
            st.dataframe(tables['platform_summary'], use_container_width=True)

            # Top榜单
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🏆 Top 5 增长最高的模型")
                st.dataframe(tables['top5_growth'], use_container_width=True)

            with col2:
                st.markdown("### 🥇 Top 3 总下载量最高的模型")
                st.dataframe(tables['top3_downloads'], use_container_width=True)

            # 各平台榜首
            st.markdown("### 🎯 各平台榜首模型")
            st.dataframe(
                tables['platform_top_models'],
                use_container_width=True,
                column_config={
                    "下载量最高模型": st.column_config.TextColumn(
                        "下载量最高模型",
                            help="各平台官方/衍生模型中，总下载量最高的模型",
                            width="large",
                        ),
                        "增长最高模型": st.column_config.TextColumn(
                            "增长最高模型",
                            help="各平台官方/衍生模型中，本周增长量最高的模型",
                            width="large",
                        ),
                    }
                )

                # 详细数据表格
            st.markdown("### 📋 各平台模型下载量详情 (总/周增)")
            st.dataframe(tables['combined_downloads_growth'], use_container_width=True)

            # 新增Finetune和Adapter模型展示
            st.markdown("### 🌟 本周新增Finetune和Adapter模型")

            # 显示汇总信息
            summary = tables.get('new_models_summary', '无新增模型信息')
            st.info(f"📊 {summary}")

            # 分列显示不同类型的新增模型
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🔧 新增Finetune模型")
                finetune_df = tables.get('new_finetune_models')
                if finetune_df is not None and not finetune_df.empty:
                    st.dataframe(finetune_df, use_container_width=True)
                else:
                    st.info("本周无新增Finetune模型")

            with col2:
                st.markdown("#### 🔌 新增Adapter模型")
                adapter_df = tables.get('new_adapter_models')
                if adapter_df is not None and not adapter_df.empty:
                    st.dataframe(adapter_df, use_container_width=True)
                else:
                    st.info("本周无新增Adapter模型")

            with col3:
                st.markdown("#### 🎯 新增LoRA模型")
                lora_df = tables.get('new_lora_models')
                if lora_df is not None and not lora_df.empty:
                    st.dataframe(lora_df, use_container_width=True)
                else:
                    st.info("本周无新增LoRA模型")

            # 🆕 所有新增模型完整列表
            st.markdown("### 📋 本周新增模型完整列表")

            # 显示汇总信息
            all_new_summary = tables.get('all_new_models_summary', '无新增模型')
            st.info(f"📊 {all_new_summary}")

            # 显示所有新增模型表格
            all_new_df = tables.get('all_new_models')
            if all_new_df is not None and not all_new_df.empty:
                st.dataframe(all_new_df, use_container_width=True, height=400)
            else:
                st.info("本周没有新增ERNIE-4.5模型")

            # 🆕 已删除/隐藏的模型列表
            st.markdown("### 🗑️ 已删除/隐藏的衍生模型")
            st.info("📌 这些模型在历史记录中存在，但在当前日期已不可见（可能被删除或隐藏）")

            from ernie_tracker.analysis import get_deleted_or_hidden_models
            deleted_models = get_deleted_or_hidden_models(current_date, model_series='ERNIE-4.5')

            if deleted_models:
                deleted_df = pd.DataFrame(deleted_models)
                deleted_df.index = deleted_df.index + 1

                # 重命名列
                column_mapping = {
                    'model_name': '模型名称',
                    'publisher': '发布者',
                    'repo': '平台',
                    'model_type': '模型类型',
                    'base_model': '基础模型',
                    'last_seen_date': '最后出现日期',
                    'last_download_count': '最后下载量'
                }
                deleted_df = deleted_df.rename(columns={k: v for k, v in column_mapping.items() if k in deleted_df.columns})

                st.warning(f"⚠️ 发现 {len(deleted_models)} 个模型已被删除或隐藏")
                st.dataframe(deleted_df, use_container_width=True, height=400)
            else:
                st.success("✅ 所有历史模型在当前日期仍然可见")

            # 导出功能
            st.markdown("### 💾 导出报表")

            # 合并所有表格为一个Excel
            from io import BytesIO

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                tables['platform_summary'].to_excel(writer, sheet_name='平台汇总')
                tables['top5_growth'].to_excel(writer, sheet_name='Top5增长')
                tables['top3_downloads'].to_excel(writer, sheet_name='Top3下载量')
                tables['platform_top_models'].to_excel(writer, sheet_name='各平台榜首', index=False)
                tables['combined_downloads_growth'].to_excel(writer, sheet_name='下载量详情')
                # 新增模型表格
                if not tables.get('new_finetune_models', pd.DataFrame()).empty:
                    tables['new_finetune_models'].to_excel(writer, sheet_name='新增Finetune模型')
                if not tables.get('new_adapter_models', pd.DataFrame()).empty:
                    tables['new_adapter_models'].to_excel(writer, sheet_name='新增Adapter模型')
                if not tables.get('new_lora_models', pd.DataFrame()).empty:
                    tables['new_lora_models'].to_excel(writer, sheet_name='新增LoRA模型')
                # 🆕 所有新增模型完整列表
                if not tables.get('all_new_models', pd.DataFrame()).empty:
                    tables['all_new_models'].to_excel(writer, sheet_name='所有新增模型')

            excel_data = output.getvalue()

            st.download_button(
                label="📥 下载完整周报 (Excel)",
                data=excel_data,
                file_name=f"ERNIE-4.5_周报_{previous_date}_to_{current_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ================= PaddleOCR-VL 数据分析模块 =================
elif page == "📊 PaddleOCR-VL 分析":
    from ernie_tracker.analysis import calculate_paddleocr_vl_weekly_report, format_report_tables, get_available_dates, get_last_friday
    from datetime import datetime

    st.markdown("## 📈 PaddleOCR-VL 周报分析")
    st.markdown("分析当前日期与对比日期之间的下载量增长情况")

    # 获取可用日期
    available_dates = get_available_dates()

    if not available_dates:
        st.warning("⚠️ 数据库中暂无数据，请先在「数据更新」页面抓取数据。")
    else:
        # 日期选择
        col1, col2 = st.columns(2)

        with col1:
            current_date = st.selectbox(
                "📅 当前日期 (PaddleOCR-VL)",
                options=available_dates,
                index=0,
                help="选择要分析的当前日期"
            )

        with col2:
            # 默认为上周五
            default_previous = get_last_friday(current_date)
            if default_previous in available_dates:
                default_index = available_dates.index(default_previous)
            else:
                default_index = min(1, len(available_dates) - 1)

            previous_date = st.selectbox(
                "📅 对比日期 (PaddleOCR-VL)",
                options=available_dates,
                index=default_index,
                help="选择要对比的日期（通常为上周五）"
            )

        if st.button("🔍 生成 PaddleOCR-VL 周报", type="primary"):
            with st.spinner("正在分析数据..."):
                report_data = calculate_paddleocr_vl_weekly_report(current_date, previous_date)

            if report_data is None:
                st.error("❌ 无法生成周报，请检查选择的日期是否有数据。")
            else:
                tables = format_report_tables(report_data)

                st.success(f"✅ 周报生成成功！对比时间段：{previous_date} → {current_date}")

                # 保存关键数据到session_state，用于重新获取功能
                st.session_state['current_date'] = current_date
                st.session_state['previous_date'] = previous_date

                # 检查并显示负增长警告
                warnings_df = tables.get('negative_growth_warnings')
                if warnings_df is not None and not warnings_df.empty:
                    st.markdown("### ⚠️ 负增长警告")
                    st.error(f"检测到 {len(warnings_df)} 个模型出现负增长！这可能表示数据采集问题或模型被下架。")
                    st.dataframe(warnings_df, use_container_width=True)

                    # 保存warnings_df到session_state
                    st.session_state['warnings_df'] = warnings_df

                    # 添加重新获取按钮
                    with st.expander("🔄 重新获取负增长模型下载量", expanded=False):
                        st.info("💡 此功能将重新从平台API获取这些模型的最新下载量并更新到数据库。目前支持 Hugging Face 和 ModelScope 平台。")

                        if st.button("🚀 开始重新获取", type="primary", key="refetch_ernie"):
                            # 从session_state获取warnings_df
                            if 'warnings_df' in st.session_state:
                                warnings_data = st.session_state['warnings_df']

                                # 转换warnings_df为负增长模型列表
                                negative_list = []
                                for idx, row in warnings_data.iterrows():
                                    negative_list.append({
                                        'platform': row['平台'],
                                        'model_name': row['模型名称'],
                                        'publisher': row['发布者'],
                                        'current': row['本周下载量']
                                    })

                                # 执行重新获取
                                with st.spinner("正在重新获取模型下载量..."):
                                    from ernie_tracker.fetchers.fetchers_single_model import refetch_models_batch
                                    from ernie_tracker.db import save_to_db

                                    success_list, failed_list, unsupported_list = refetch_models_batch(negative_list)

                                    # 保存结果到session_state
                                    st.session_state['refetch_success'] = success_list
                                    st.session_state['refetch_failed'] = failed_list
                                    st.session_state['refetch_unsupported'] = unsupported_list
                                    st.session_state['refetch_done'] = True

                                    # 重新运行页面以显示结果
                                    st.rerun()

                    # 显示重新获取结果（如果已执行）
                    if st.session_state.get('refetch_done', False):
                        st.markdown("#### 📊 重新获取结果")

                        success_list = st.session_state.get('refetch_success', [])
                        failed_list = st.session_state.get('refetch_failed', [])
                        unsupported_list = st.session_state.get('refetch_unsupported', [])

                        if success_list:
                            st.success(f"✅ 成功重新获取 {len(success_list)} 个模型")
                            success_df = pd.DataFrame(success_list)[['platform', 'model_name', 'old_count', 'new_count', 'change']]
                            success_df.columns = ['平台', '模型名称', '原下载量', '新下载量', '变化']
                            st.dataframe(success_df, use_container_width=True)

                            # 保存到数据库
                            if st.button("💾 保存更新到数据库", key="save_ernie"):
                                saved_count = 0
                                for item in success_list:
                                    record = item['record']
                                    try:
                                        save_to_db(pd.DataFrame([record]), DB_PATH, DATA_TABLE)
                                        saved_count += 1
                                    except Exception as e:
                                        st.error(f"保存 {item['model_name']} 失败: {e}")
                                st.success(f"✅ 已保存 {saved_count} 条记录到数据库！")
                                # 清除session_state
                                st.session_state['refetch_done'] = False
                                st.rerun()

                        if failed_list:
                            st.warning(f"⚠️ {len(failed_list)} 个模型获取失败")
                            failed_df = pd.DataFrame(failed_list)[['platform', 'model_name', 'publisher']]
                            failed_df.columns = ['平台', '模型名称', '发布者']
                            st.dataframe(failed_df, use_container_width=True)

                        if unsupported_list:
                            st.info(f"ℹ️ {len(unsupported_list)} 个模型的平台暂不支持自动重新获取")
                            unsupported_df = pd.DataFrame(unsupported_list)[['platform', 'model_name', 'publisher']]
                            unsupported_df.columns = ['平台', '模型名称', '发布者']
                            st.dataframe(unsupported_df, use_container_width=True)

                            # 显示手动检查建议
                            st.markdown("#### 🔍 手动检查建议")
                            for item in unsupported_list:
                                repo = item['platform']
                                model_name = item['model_name']
                                publisher = item['publisher']

                                url = None
                                if repo == "AI Studio":
                                    # AI Studio模型URL需要根据实际情况构造
                                    url = f"https://aistudio.baidu.com/modeldetail/{model_name}"
                                elif repo == "GitCode":
                                    from ernie_tracker.config import GITCODE_MODEL_LINKS
                                    for link in GITCODE_MODEL_LINKS:
                                        if model_name in link:
                                            url = link
                                            break

                                if url:
                                    st.markdown(f"- **{repo} | {model_name}**: [打开模型页面]({url})")

                        # 清除按钮
                        if st.button("🗑️ 清除结果", key="clear_ernie"):
                            st.session_state['refetch_done'] = False
                            st.rerun()

                    st.markdown("---")

                # 显示总体情况摘要
                st.markdown("### 📝 总体情况摘要")
                stats = report_data['summary_stats']
                
                # 格式化数字
                def format_num(n):
                    return f"{n/10000:.2f}万"

                def format_percent(p):
                    return f"{p:.2%}"

                # 计算百分比
                official_total_percent = stats['official_current_total'] / stats['all_current_total'] if stats['all_current_total'] else 0
                derivative_total_percent = stats['derivative_current_total'] / stats['all_current_total'] if stats['all_current_total'] else 0
                official_growth_percent = stats['official_growth'] / stats['all_growth'] if stats['all_growth'] else 0
                derivative_growth_percent = stats['derivative_growth'] / stats['all_growth'] if stats['all_growth'] else 0

                summary_text = f"""
                截至 **{current_date}**，模型累计下载 **{format_num(stats['all_current_total'])}** 次
                （含官方模型 **{format_num(stats['official_current_total'])}** 次，占比 **{format_percent(official_total_percent)}**，
                衍生 **{format_num(stats['derivative_current_total'])}** 次，占比 **{format_percent(derivative_total_percent)}**），
                较上周增长 **{format_num(stats['all_growth'])}** 次
                （官方模型 **{format_num(stats['official_growth'])}** 次，占比 **{format_percent(official_growth_percent)}**，
                衍生模型增长 **{format_num(stats['derivative_growth'])}** 次，占比 **{format_percent(derivative_growth_percent)}**）。
                """
                st.markdown(summary_text)

                # 累计/本周新增衍生模型数量
                new_models_list_count = len(tables.get('all_new_models', pd.DataFrame()))
                st.info(
                    f"累计衍生模型：{int(stats.get('derivative_current_total_models', 0) or 0)} 个｜"
                    f"本周新增衍生（HF非官方差集）：{int(stats.get('derivative_new_models', 0) or 0)} 个｜"
                    f"新增列表展示：{new_models_list_count} 个"
                )

                # 社区和模型维度摘要
                st.markdown("### 📈 社区与模型维度摘要")
                community_summary = report_data['community_summary']
                
                # 社区维度
                community_text = f"""
                - **社区维度**：Hugging Face下载量最高，**{community_summary['hf_top_model_name']}** 为本周HF平台下载最高模型，增长 **{community_summary['hf_top_model_growth']/10000:.2f}万** 次。
                """
                st.markdown(community_text)

                # 模型维度
                top3_downloads_str = " > ".join([f"{name}({int(val)})" for name, val in community_summary['top3_downloads_details'].items()])
                top3_growth_str = " > ".join([f"{name}({int(val)})" for name, val in community_summary['top3_growth_details'].items()])
                
                model_text = f"""
                - **模型维度**：
                    - 模型（官方）下载总量前三位：{top3_downloads_str}
                    - 本周（官方）增长最快前三位：{top3_growth_str}
                """
                st.markdown(model_text)

                # 显示汇总信息
                st.markdown("### 📊 平台汇总")
                st.dataframe(tables['platform_summary'], use_container_width=True)

                # Top榜单
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🏆 Top 5 增长最高的模型")
                    st.dataframe(tables['top5_growth'], use_container_width=True)

                with col2:
                    st.markdown("### 🥇 Top 3 总下载量最高的模型")
                    st.dataframe(tables['top3_downloads'], use_container_width=True)

                # 各平台榜首
                st.markdown("### 🎯 各平台榜首模型")
                st.dataframe(
                    tables['platform_top_models'],
                    use_container_width=True,
                    column_config={
                        "下载量最高模型": st.column_config.TextColumn(
                            "下载量最高模型",
                            help="各平台官方/衍生模型中，总下载量最高的模型",
                            width="large",
                        ),
                        "增长最高模型": st.column_config.TextColumn(
                            "增长最高模型",
                            help="各平台官方/衍生模型中，本周增长量最高的模型",
                            width="large",
                        ),
                    }
                )

                # 详细数据表格
                st.markdown("### 📋 各平台模型下载量详情 (总/周增)")
                st.dataframe(tables['combined_downloads_growth'], use_container_width=True)

                # 🔧 新增：PaddleOCR-VL 的 Finetune 和 Adapter 模型展示
                st.markdown("### 🌟 本周新增Finetune和Adapter模型")

                # 显示汇总信息
                summary = tables.get('new_models_summary', '无新增模型信息')
                st.info(f"📊 {summary}")

                # 分列显示不同类型的新增模型
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🔧 新增Finetune模型")
                    finetune_df = tables.get('new_finetune_models')
                    if finetune_df is not None and not finetune_df.empty:
                        st.dataframe(finetune_df, use_container_width=True)
                    else:
                        st.info("本周无新增Finetune模型")

                with col2:
                    st.markdown("#### 🔌 新增Adapter模型")
                    adapter_df = tables.get('new_adapter_models')
                    if adapter_df is not None and not adapter_df.empty:
                        st.dataframe(adapter_df, use_container_width=True)
                    else:
                        st.info("本周无新增Adapter模型")

                with col3:
                    st.markdown("#### 🎯 新增LoRA模型")
                    lora_df = tables.get('new_lora_models')
                    if lora_df is not None and not lora_df.empty:
                        st.dataframe(lora_df, use_container_width=True)
                    else:
                        st.info("本周无新增LoRA模型")

                # 🆕 所有新增模型完整列表
                st.markdown("### 📋 本周新增模型完整列表")

                # 显示汇总信息
                all_new_summary = tables.get('all_new_models_summary', '无新增模型')
                st.info(f"📊 {all_new_summary}")

                # 显示所有新增模型表格
                all_new_df = tables.get('all_new_models')
                if all_new_df is not None and not all_new_df.empty:
                    st.dataframe(all_new_df, use_container_width=True, height=400)
                else:
                    st.info("本周没有新增PaddleOCR-VL模型")

                # 🆕 已删除/隐藏的模型列表
                st.markdown("### 🗑️ 已删除/隐藏的衍生模型")
                st.info("📌 这些模型在历史记录中存在，但在当前日期已不可见（可能被删除或隐藏）")

                from ernie_tracker.analysis import get_deleted_or_hidden_models
                deleted_models = get_deleted_or_hidden_models(current_date, model_series='PaddleOCR-VL')

                if deleted_models:
                    deleted_df = pd.DataFrame(deleted_models)
                    deleted_df.index = deleted_df.index + 1

                    # 重命名列
                    column_mapping = {
                        'model_name': '模型名称',
                        'publisher': '发布者',
                        'repo': '平台',
                        'model_type': '模型类型',
                        'base_model': '基础模型',
                        'last_seen_date': '最后出现日期',
                        'last_download_count': '最后下载量'
                    }
                    deleted_df = deleted_df.rename(columns={k: v for k, v in column_mapping.items() if k in deleted_df.columns})

                    st.warning(f"⚠️ 发现 {len(deleted_models)} 个模型已被删除或隐藏")
                    st.dataframe(deleted_df, use_container_width=True, height=400)
                else:
                    st.success("✅ 所有历史模型在当前日期仍然可见")

                # 导出功能
                st.markdown("### 💾 导出报表")

                # 合并所有表格为一个Excel
                from io import BytesIO

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    tables['platform_summary'].to_excel(writer, sheet_name='平台汇总')
                    tables['top5_growth'].to_excel(writer, sheet_name='Top5增长')
                    tables['top3_downloads'].to_excel(writer, sheet_name='Top3下载量')
                    tables['platform_top_models'].to_excel(writer, sheet_name='各平台榜首', index=False)
                    tables['combined_downloads_growth'].to_excel(writer, sheet_name='下载量详情')
                    # 🔧 新增：导出新增模型表格
                    if not tables.get('new_finetune_models', pd.DataFrame()).empty:
                        tables['new_finetune_models'].to_excel(writer, sheet_name='新增Finetune模型')
                    if not tables.get('new_adapter_models', pd.DataFrame()).empty:
                        tables['new_adapter_models'].to_excel(writer, sheet_name='新增Adapter模型')
                    if not tables.get('new_lora_models', pd.DataFrame()).empty:
                        tables['new_lora_models'].to_excel(writer, sheet_name='新增LoRA模型')
                    if not tables.get('new_model_tree_models', pd.DataFrame()).empty:
                        tables['new_model_tree_models'].to_excel(writer, sheet_name='ModelTree新增模型')
                    # 🆕 所有新增模型完整列表
                    if not tables.get('all_new_models', pd.DataFrame()).empty:
                        tables['all_new_models'].to_excel(writer, sheet_name='所有新增模型')

                excel_data = output.getvalue()

        st.download_button(
            label="📥 下载 PaddleOCR-VL 完整周报 (Excel)",
            data=excel_data,
            file_name=f"PaddleOCR-VL_周报_{previous_date}_to_{current_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ================= 数据库管理模块 =================
elif page == "🗄️ 数据库管理":
    from ernie_tracker.db_manager import (
        backup_database, restore_database, delete_data_by_date,
        delete_data_by_platform, get_database_stats, get_available_backups,
        delete_backup, vacuum_database, export_database_to_excel,
        get_duplicate_records, remove_duplicate_records, insert_single_record,
        import_from_excel
    )
    from ernie_tracker.analysis import get_available_dates
    from io import BytesIO
    import os

    st.markdown("## 🗄️ 数据库管理")
    st.info("💡 提供数据库备份、恢复、删除、优化等管理功能")

    # 创建标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 数据库概览",
        "💾 备份与恢复",
        "🗑️ 数据删除",
        "🔧 数据维护",
        "📤 数据导出",
        "📝 数据录入",
        "✏️ 数据编辑"
    ])
    
    # ========== Tab 1: 数据库概览 ==========
    with tab1:
        st.markdown("### 📊 数据库统计信息")
        
        if st.button("🔄 刷新统计", key="refresh_stats"):
            st.rerun()
        
        stats = get_database_stats()
        
        if 'error' in stats:
            st.error(f"获取统计信息失败: {stats['error']}")
        else:
            # 显示关键指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总记录数", f"{stats['total_records']:,}")
            
            with col2:
                st.metric("数据库大小", f"{stats['db_size_mb']} MB")
            
            with col3:
                st.metric("最早日期", stats['min_date'] or "无数据")
            
            with col4:
                st.metric("最新日期", stats['max_date'] or "无数据")
            
            st.markdown("---")
            
            # 按日期统计
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 按日期统计")
                if not stats['date_stats'].empty:
                    st.dataframe(
                        stats['date_stats'].rename(columns={'date': '日期', 'count': '记录数'}),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.info("暂无数据")
            
            with col2:
                st.markdown("#### 🌐 按平台统计")
                if not stats['platform_stats'].empty:
                    st.dataframe(
                        stats['platform_stats'].rename(columns={'repo': '平台', 'count': '记录数'}),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.info("暂无数据")
    
    # ========== Tab 2: 备份与恢复 ==========
    with tab2:
        st.markdown("### 💾 数据库备份")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            backup_dir = st.text_input(
                "备份目录",
                value="backups",
                help="数据库备份文件将保存到这个目录"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("📦 立即备份", type="primary", use_container_width=True):
                with st.spinner("正在备份数据库..."):
                    success, result = backup_database(backup_dir)
                
                if success:
                    st.success(f"✅ 备份成功！\n文件路径: `{result}`")
                else:
                    st.error(f"❌ 备份失败: {result}")
        
        st.markdown("---")
        st.markdown("### 📂 已有备份")
        
        backups = get_available_backups(backup_dir)
        
        if not backups:
            st.info("暂无备份文件")
        else:
            st.write(f"共找到 **{len(backups)}** 个备份文件:")
            
            for backup in backups:
                with st.expander(f"📁 {backup['filename']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**创建时间**: {backup['created_time']}")
                        st.write(f"**文件大小**: {backup['size_mb']} MB")
                        st.write(f"**文件路径**: `{backup['filepath']}`")
                    
                    with col2:
                        if st.button("🔄 恢复此备份", key=f"restore_{backup['filename']}"):
                            if st.session_state.get(f"confirm_restore_{backup['filename']}", False):
                                with st.spinner("正在恢复数据库..."):
                                    success, message = restore_database(backup['filepath'])
                                
                                if success:
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ 恢复失败: {message}")
                                
                                st.session_state[f"confirm_restore_{backup['filename']}"] = False
                            else:
                                st.warning("⚠️ 请再次点击确认恢复")
                                st.session_state[f"confirm_restore_{backup['filename']}"] = True
                    
                    with col3:
                        if st.button("🗑️ 删除备份", key=f"delete_{backup['filename']}"):
                            success, message = delete_backup(backup['filepath'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(f"删除失败: {message}")
    
    # ========== Tab 3: 数据删除 ==========
    with tab3:
        st.markdown("### 🗑️ 数据删除")
        st.warning("⚠️ **警告**: 删除操作不可逆，建议先备份数据库！")
        
        # 删除指定日期的数据
        st.markdown("#### 🗓️ 按日期删除")
        
        available_dates = get_available_dates()
        
        if not available_dates:
            st.info("数据库中暂无数据")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                delete_date = st.selectbox(
                    "选择要删除的日期",
                    options=available_dates,
                    key="delete_date_selector"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("🗑️ 删除该日期数据", key="delete_by_date", use_container_width=True):
                    if st.session_state.get("confirm_delete_date", False):
                        with st.spinner(f"正在删除 {delete_date} 的数据..."):
                            success, message, count = delete_data_by_date(delete_date)
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ 删除失败: {message}")
                        
                        st.session_state["confirm_delete_date"] = False
                    else:
                        st.warning(f"⚠️ 确认删除 {delete_date} 的所有数据？请再次点击确认！")
                        st.session_state["confirm_delete_date"] = True
        
        st.markdown("---")
        
        # 删除指定平台的数据
        st.markdown("#### 🌐 按平台删除")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            delete_platform = st.selectbox(
                "选择平台",
                options=list(PLATFORM_NAMES.values()),
                key="delete_platform_selector"
            )
        
        with col2:
            delete_platform_date = st.selectbox(
                "选择日期（可选）",
                options=["全部日期"] + (available_dates if available_dates else []),
                key="delete_platform_date_selector"
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("🗑️ 删除平台数据", key="delete_by_platform", use_container_width=True):
                if st.session_state.get("confirm_delete_platform", False):
                    target_date = None if delete_platform_date == "全部日期" else delete_platform_date
                    
                    with st.spinner(f"正在删除 {delete_platform} 的数据..."):
                        success, message, count = delete_data_by_platform(delete_platform, target_date)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ 删除失败: {message}")
                    
                    st.session_state["confirm_delete_platform"] = False
                else:
                    date_info = f" ({delete_platform_date})" if delete_platform_date != "全部日期" else ""
                    st.warning(f"⚠️ 确认删除 {delete_platform}{date_info} 的数据？请再次点击确认！")
                    st.session_state["confirm_delete_platform"] = True
    
    # ========== Tab 4: 数据维护 ==========
    with tab4:
        st.markdown("### 🔧 数据维护")
        
        # 检查重复记录
        st.markdown("#### 🔍 重复记录检测")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("检查并清理数据库中的重复记录（相同的日期、平台、发布者、模型名称）")

        with col2:
            if st.button("🔎 检查重复记录", key="check_duplicates", use_container_width=True):
                with st.spinner("正在检查重复记录..."):
                    duplicates = get_duplicate_records()

                if duplicates.empty:
                    st.success("✅ 没有发现重复记录")
                else:
                    total_duplicates = duplicates['count'].sum() - len(duplicates)
                    st.session_state['duplicates_found'] = duplicates
                    st.session_state['duplicate_count'] = total_duplicates
                    st.rerun()

        # 显示检查结果
        if 'duplicates_found' in st.session_state and not st.session_state['duplicates_found'].empty:
            duplicates = st.session_state['duplicates_found']
            total_duplicates = st.session_state['duplicate_count']

            st.warning(f"⚠️ 发现 {len(duplicates)} 组重复记录，共 {total_duplicates} 条重复数据需要清理")
            st.dataframe(
                duplicates.rename(columns={
                    'date': '日期',
                    'repo': '平台',
                    'publisher': '发布者',
                    'model_name': '模型名称',
                    'count': '重复次数'
                }),
                use_container_width=True
            )

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🧹 清除重复记录", key="remove_duplicates", type="primary", use_container_width=True):
                    with st.spinner("正在清除重复记录..."):
                        success, message, count = remove_duplicate_records()

                    if success:
                        st.success(f"✅ {message}")
                        # 清除session state
                        if 'duplicates_found' in st.session_state:
                            del st.session_state['duplicates_found']
                        if 'duplicate_count' in st.session_state:
                            del st.session_state['duplicate_count']
                        st.rerun()
                    else:
                        st.error(f"❌ 清除失败: {message}")
        
        st.markdown("---")
        
        # 数据库优化
        st.markdown("#### ⚡ 数据库优化")
        st.info("数据库优化（VACUUM）可以回收删除数据后的空间，减小数据库文件大小")
        
        if st.button("⚡ 优化数据库", key="vacuum_db"):
            with st.spinner("正在优化数据库..."):
                success, message = vacuum_database()
            
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ 优化失败: {message}")
    
    # ========== Tab 5: 数据导出 ==========
    with tab5:
        st.markdown("### 📤 数据导出")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            export_date = st.selectbox(
                "选择导出日期",
                options=["全部日期"] + (available_dates if available_dates else []),
                key="export_date_selector"
            )
        
        with col2:
            export_filename = st.text_input(
                "文件名",
                value=f"database_export_{date.today().isoformat()}.xlsx",
                key="export_filename"
            )
        
        if st.button("📥 导出到 Excel", type="primary", key="export_excel"):
            output_path = os.path.join("exports", export_filename)
            os.makedirs("exports", exist_ok=True)
            
            target_date = None if export_date == "全部日期" else export_date
            
            with st.spinner("正在导出数据..."):
                success, message = export_database_to_excel(output_path, target_date)
            
            if success:
                st.success(f"✅ {message}")
                
                # 提供下载
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ 下载导出文件",
                        data=f.read(),
                        file_name=export_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error(f"❌ {message}")

    
    # ========== Tab 6: 数据录入 ==========
    with tab6:
        st.markdown("### 📝 数据录入")
        st.info("💡 支持单条数据录入和 Excel 批量导入")
        
        # 创建子标签
        subtab1, subtab2 = st.tabs(["➕ 单条录入", "📄 批量导入"])
        
        # ========== 子Tab 1: 单条录入 ==========
        with subtab1:
            st.markdown("#### ➕ 单条数据录入")
            st.markdown("---")
            
            # 必填字段
            col1, col2 = st.columns(2)
            
            with col1:
                input_date = st.date_input(
                    "日期 *",
                    value=date.today(),
                    help="数据采集的日期"
                )
                input_date_str = input_date.strftime('%Y-%m-%d')
                
                input_repo = st.selectbox(
                    "平台 *",
                    options=list(PLATFORM_NAMES.values()),
                    help="模型所在的平台"
                )
            
            with col2:
                input_model_name = st.text_input(
                    "模型名称 *",
                    help="模型的完整名称"
                )
                
                input_publisher = st.text_input(
                    "发布者 *",
                    help="模型的发布者/作者"
                )
            
            input_download_count = st.number_input(
                "下载量 *",
                min_value=0,
                value=0,
                step=1,
                help="模型的下载次数"
            )
            
            st.markdown("---")
            st.markdown("#### 可选字段（Model Tree 相关）")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                input_base_model = st.text_input(
                    "基础模型",
                    help="衍生模型的基础模型名称"
                )
            
            with col4:
                input_model_type = st.selectbox(
                    "模型类型",
                    options=["", "original", "finetune", "adapter", "lora", "other"],
                    help="模型的类型分类"
                )
            
            with col5:
                input_model_category = st.selectbox(
                    "模型分类",
                    options=["", "ernie-4.5", "paddleocr-vl", "other-ernie", "other"],
                    help="模型的系列分类"
                )
            
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns([3, 1])
            
            with col_btn2:
                if st.button("💾 保存数据", type="primary", use_container_width=True, key="insert_single"):
                    # 转换空字符串为 None
                    base_model = input_base_model if input_base_model else None
                    model_type = input_model_type if input_model_type else None
                    model_category = input_model_category if input_model_category else None
                    
                    with st.spinner("正在保存数据..."):
                        success, message = insert_single_record(
                            date=input_date_str,
                            repo=input_repo,
                            model_name=input_model_name,
                            publisher=input_publisher,
                            download_count=input_download_count,
                            base_model=base_model,
                            model_type=model_type,
                            model_category=model_category
                        )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
        
        # ========== 子Tab 2: 批量导入 ==========
        with subtab2:
            st.markdown("#### 📄 Excel 批量导入")
            st.markdown("---")
            
            # 说明信息
            st.info("""
            📋 **Excel 文件格式要求：**
            
            **必需列**（缺一不可）：
            - `date`: 日期（格式：YYYY-MM-DD）
            - `repo`: 平台名称
            - `model_name`: 模型名称
            - `publisher`: 发布者
            - `download_count`: 下载量（数字）
            
            **可选列**：
            - `base_model`: 基础模型（用于衍生模型）
            - `model_type`: 模型类型（original, finetune, adapter, lora, other）
            - `model_category`: 模型分类（ernie-4.5, paddleocr-vl, other-ernie, other）
            """)
            
            # 下载模板
            template_data = {
                'date': ['2025-01-01', '2025-01-01'],
                'repo': ['Hugging Face', 'ModelScope'],
                'model_name': ['示例模型1', '示例模型2'],
                'publisher': ['示例发布者1', '示例发布者2'],
                'download_count': [1000, 2000],
                'base_model': ['', ''],
                'model_type': ['', ''],
                'model_category': ['', '']
            }
            template_df = pd.DataFrame(template_data)
            
            template_buffer = BytesIO()
            with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='模型数据')
            
            st.download_button(
                label="📥 下载 Excel 模板",
                data=template_buffer.getvalue(),
                file_name="导入模板.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="下载包含示例数据的 Excel 模板"
            )
            
            st.markdown("---")
            
            # 文件上传
            uploaded_file = st.file_uploader(
                "选择 Excel 文件",
                type=['xlsx', 'xls'],
                help="上传包含模型数据的 Excel 文件"
            )
            
            if uploaded_file is not None:
                # 预览上传的文件
                st.markdown("##### 📊 文件预览")
                try:
                    preview_df = pd.read_excel(uploaded_file, engine='openpyxl')
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    st.info(f"文件包含 {len(preview_df)} 行数据")
                    
                    # 重置文件指针
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"无法读取文件: {e}")
                    uploaded_file = None
            
            if uploaded_file is not None:
                st.markdown("---")
                
                # 导入选项
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    skip_duplicates = st.radio(
                        "遇到重复记录时",
                        options=[True, False],
                        format_func=lambda x: "跳过（推荐）" if x else "覆盖",
                        help="选择如何处理与数据库中已存在记录相同的数据"
                    )
                
                col_import1, col_import2 = st.columns([3, 1])
                
                with col_import2:
                    if st.button("📤 开始导入", type="primary", use_container_width=True, key="import_excel"):
                        with st.spinner("正在导入数据..."):
                            # 重置文件指针
                            uploaded_file.seek(0)
                            success, message, stats = import_from_excel(uploaded_file, skip_duplicates)
                        
                        if success:
                            st.success("✅ 导入完成！")

                            # 显示统计信息
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                            with col_stat1:
                                st.metric("总记录数", stats['total'])

                            with col_stat2:
                                st.metric("成功插入", stats['inserted'], delta=stats['inserted'])

                            with col_stat3:
                                st.metric("跳过重复", stats['skipped'])

                            with col_stat4:
                                st.metric("错误记录", stats['errors'], delta=-stats['errors'] if stats['errors'] > 0 else 0)

                            # 显示详细信息
                            with st.expander("📋 详细信息"):
                                st.text(message)

                            if stats['inserted'] > 0:
                                st.balloons()
                        else:
                            st.error(f"❌ 导入失败")
                            st.error(message)

    # ========== Tab 7: 数据编辑 ==========
    with tab7:
        from ernie_tracker.db_manager import search_records, get_record_by_rowid, update_record, delete_record_by_rowid

        st.markdown("### ✏️ 数据编辑")
        st.info("💡 搜索并编辑数据库中的记录")

        # 搜索区域
        st.markdown("#### 🔍 搜索记录")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            search_date = st.selectbox(
                "日期",
                options=["全部"] + (available_dates if available_dates else []),
                key="search_date"
            )

        with col2:
            search_repo = st.selectbox(
                "平台",
                options=["全部"] + list(PLATFORM_NAMES.values()),
                key="search_repo"
            )

        with col3:
            search_model_name = st.text_input(
                "模型名称（支持模糊搜索）",
                key="search_model_name"
            )

        with col4:
            search_publisher = st.text_input(
                "发布者（支持模糊搜索）",
                key="search_publisher"
            )

        # 搜索按钮
        col_search1, col_search2 = st.columns([3, 1])
        with col_search2:
            search_button = st.button("🔎 搜索", type="primary", use_container_width=True, key="search_btn")

        # 执行搜索
        if search_button or 'search_results' in st.session_state:
            # 构建搜索参数
            search_params = {}

            if search_date != "全部":
                search_params['date_filter'] = search_date

            if search_repo != "全部":
                search_params['repo_filter'] = search_repo

            if search_model_name:
                search_params['model_name_filter'] = search_model_name

            if search_publisher:
                search_params['publisher_filter'] = search_publisher

            # 执行搜索
            if search_button:
                with st.spinner("正在搜索..."):
                    results = search_records(**search_params)
                    st.session_state['search_results'] = results
            else:
                results = st.session_state.get('search_results', pd.DataFrame())

            # 显示搜索结果
            st.markdown("---")
            st.markdown("#### 📋 搜索结果")

            if results.empty:
                st.info("未找到匹配的记录")
            else:
                st.success(f"找到 {len(results)} 条记录")

                # 显示搜索结果表格（可选择）
                # 选择要编辑的记录
                st.markdown("##### 选择要编辑的记录：")

                # 创建一个更友好的显示格式
                display_df = results.copy()
                display_df['选择'] = False

                # 重新排列列顺序，把 rowid 放在前面
                cols = ['rowid', 'date', 'repo', 'model_name', 'publisher', 'download_count']
                optional_cols = ['base_model', 'model_type', 'model_category', 'tags']

                for col in optional_cols:
                    if col in display_df.columns:
                        cols.append(col)

                display_df = display_df[cols]

                # 使用 data_editor 显示可选择的表格
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=300
                )

                # 输入要编辑的记录 rowid
                st.markdown("---")
                st.markdown("#### ✏️ 编辑记录")

                col_edit1, col_edit2 = st.columns([2, 2])

                with col_edit1:
                    edit_rowid = st.number_input(
                        "输入要编辑的记录 rowid",
                        min_value=1,
                        value=int(results.iloc[0]['rowid']) if not results.empty else 1,
                        step=1,
                        key="edit_rowid"
                    )

                with col_edit2:
                    st.write("")
                    st.write("")
                    load_button = st.button("📥 加载记录", use_container_width=True, key="load_record")

                # 加载记录进行编辑
                if load_button or 'editing_record' in st.session_state:
                    if load_button:
                        record = get_record_by_rowid(edit_rowid)
                        if record:
                            st.session_state['editing_record'] = record
                            st.session_state['editing_rowid'] = edit_rowid
                        else:
                            st.error(f"未找到 rowid={edit_rowid} 的记录")
                            if 'editing_record' in st.session_state:
                                del st.session_state['editing_record']

                    if 'editing_record' in st.session_state:
                        record = st.session_state['editing_record']

                        st.markdown(f"##### 正在编辑 rowid={st.session_state['editing_rowid']} 的记录")
                        st.markdown("---")

                        # 编辑表单
                        col_form1, col_form2 = st.columns(2)

                        with col_form1:
                            edit_date = st.date_input(
                                "日期 *",
                                value=pd.to_datetime(record['date']).date() if record['date'] else date.today(),
                                key="edit_date_input"
                            )
                            edit_date_str = edit_date.strftime('%Y-%m-%d')

                            edit_repo = st.selectbox(
                                "平台 *",
                                options=list(PLATFORM_NAMES.values()),
                                index=list(PLATFORM_NAMES.values()).index(record['repo']) if record['repo'] in list(PLATFORM_NAMES.values()) else 0,
                                key="edit_repo_input"
                            )

                            edit_model_name = st.text_input(
                                "模型名称 *",
                                value=record['model_name'] or "",
                                key="edit_model_name_input"
                            )

                        with col_form2:
                            edit_publisher = st.text_input(
                                "发布者 *",
                                value=record['publisher'] or "",
                                key="edit_publisher_input"
                            )

                            edit_download_count = st.number_input(
                                "下载量 *",
                                min_value=0,
                                value=int(record['download_count']) if record['download_count'] else 0,
                                step=1,
                                key="edit_download_count_input"
                            )

                        st.markdown("##### Model Tree 相关字段（可选）")

                        col_form3, col_form4, col_form5 = st.columns(3)

                        with col_form3:
                            edit_base_model = st.text_input(
                                "基础模型",
                                value=record['base_model'] or "",
                                key="edit_base_model_input"
                            )

                        with col_form4:
                            model_type_options = ["", "original", "finetune", "adapter", "lora", "other"]
                            current_type = record['model_type'] or ""
                            edit_model_type = st.selectbox(
                                "模型类型",
                                options=model_type_options,
                                index=model_type_options.index(current_type) if current_type in model_type_options else 0,
                                key="edit_model_type_input"
                            )

                        with col_form5:
                            category_options = ["", "ernie-4.5", "paddleocr-vl", "other-ernie", "other"]
                            current_category = record['model_category'] or ""
                            edit_model_category = st.selectbox(
                                "模型分类",
                                options=category_options,
                                index=category_options.index(current_category) if current_category in category_options else 0,
                                key="edit_model_category_input"
                            )

                        edit_tags = st.text_input(
                            "标签",
                            value=record['tags'] or "",
                            key="edit_tags_input"
                        )

                        st.markdown("---")

                        # 操作按钮
                        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

                        with col_btn2:
                            if st.button("💾 保存更改", type="primary", use_container_width=True, key="save_edit"):
                                # 转换空字符串为 None
                                base_model_value = edit_base_model if edit_base_model else None
                                model_type_value = edit_model_type if edit_model_type else None
                                model_category_value = edit_model_category if edit_model_category else None
                                tags_value = edit_tags if edit_tags else None

                                with st.spinner("正在保存..."):
                                    success, message = update_record(
                                        rowid=st.session_state['editing_rowid'],
                                        date=edit_date_str,
                                        repo=edit_repo,
                                        model_name=edit_model_name,
                                        publisher=edit_publisher,
                                        download_count=edit_download_count,
                                        base_model=base_model_value,
                                        model_type=model_type_value,
                                        model_category=model_category_value,
                                        tags=tags_value
                                    )

                                if success:
                                    st.success(f"✅ {message}")
                                    # 清除编辑状态
                                    if 'editing_record' in st.session_state:
                                        del st.session_state['editing_record']
                                    if 'editing_rowid' in st.session_state:
                                        del st.session_state['editing_rowid']
                                    # 重新搜索
                                    results = search_records(**search_params)
                                    st.session_state['search_results'] = results
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")

                        with col_btn3:
                            if st.button("🗑️ 删除记录", use_container_width=True, key="delete_edit"):
                                if st.session_state.get("confirm_delete_edit", False):
                                    with st.spinner("正在删除..."):
                                        success, message = delete_record_by_rowid(st.session_state['editing_rowid'])

                                    if success:
                                        st.success(f"✅ {message}")
                                        # 清除编辑状态
                                        if 'editing_record' in st.session_state:
                                            del st.session_state['editing_record']
                                        if 'editing_rowid' in st.session_state:
                                            del st.session_state['editing_rowid']
                                        st.session_state["confirm_delete_edit"] = False
                                        # 重新搜索
                                        results = search_records(**search_params)
                                        st.session_state['search_results'] = results
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {message}")
                                        st.session_state["confirm_delete_edit"] = False
                                else:
                                    st.warning("⚠️ 确认删除？请再次点击确认！")
                                    st.session_state["confirm_delete_edit"] = True


# ================= 衍生模型生态分析模块 =================
elif page == "🌳 衍生模型生态":
    from ernie_tracker.analysis import (
        get_available_dates,
        analyze_derivative_models_all_platforms,
        calculate_periodic_stats,
        get_deleted_derivative_models_all_platforms,
        get_models_needing_backfill
    )
    import plotly.express as px
    import plotly.graph_objects as go
    from io import BytesIO

    st.markdown("## 🌳 衍生模型生态分析（全平台）")
    st.info("📊 分析全平台（Hugging Face、ModelScope、AI Studio、GitCode、鲸智、魔乐、Gitee）的衍生模型生态。衍生模型定义：非官方发布者发布的模型。")

    # 获取可用日期
    available_dates = get_available_dates()

    if not available_dates:
        st.warning("⚠️ 数据库中暂无数据，请先在「数据更新」页面抓取数据。")
    else:
        # 配置选项
        col_config1, col_config2 = st.columns([2, 2])

        with col_config1:
            # 日期选择
            selected_date = st.selectbox(
                "📅 选择分析日期",
                options=available_dates,
                index=0,
                help="选择要分析的数据日期"
            )

        with col_config2:
            # 模型系列筛选
            selected_series = st.multiselect(
                "🎯 模型系列筛选",
                options=["ERNIE-4.5", "PaddleOCR-VL"],
                default=["ERNIE-4.5", "PaddleOCR-VL"],
                help="可以选择一个或多个模型系列进行分析"
            )

        if not selected_series:
            st.warning("⚠️ 请至少选择一个模型系列")
            st.stop()

        # 显示筛选说明
        series_info = "、".join(selected_series)
        st.info(f"📊 **分析系列**: {series_info} | **数据范围**: 所有平台 | **衍生模型定义**: 非官方发布者发布的模型")

        if st.button("🔍 开始分析", type="primary"):
            with st.spinner("正在分析衍生模型生态..."):
                # 加载数据（使用回填逻辑）
                df = load_data_from_db(date_filter=selected_date, last_value_per_model=True)

                if df.empty:
                    st.error(f"❌ {selected_date} 没有数据")
                else:
                    st.success(f"✅ 加载了 {len(df)} 条记录")

                    # 使用新的分析函数
                    analysis_result = analyze_derivative_models_all_platforms(df, selected_series=selected_series)

                    if analysis_result['total_models'] == 0:
                        st.warning(f"⚠️ 没有找到符合选择的模型系列（{series_info}）的数据")
                        st.stop()

                    st.success(f"✅ 分析完成！分析日期：{selected_date}")

                    # ========== 1. 总体概览 ==========
                    st.markdown("### 📊 总体概览")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("总模型数", f"{analysis_result['total_models']:,}")

                    with col2:
                        st.metric("衍生模型数", f"{analysis_result['total_derivative_models']:,}")

                    with col3:
                        st.metric("衍生率", f"{analysis_result['derivative_rate']:.1f}%")

                    st.markdown("---")

                    # ========== 2. 周期性统计（本周新增、当前季度新增） ==========
                    st.markdown("### 📅 周期性统计")

                    with st.spinner("正在计算周期性统计和检测删除模型..."):
                        # 先获取已删除模型列表，用于统计
                        deleted_models = get_deleted_derivative_models_all_platforms(
                            selected_date,
                            selected_series=selected_series
                        )

                        # 按系列统计已删除模型数量
                        deleted_by_category = {}
                        for model in deleted_models:
                            cat = model.get('model_category', '')
                            deleted_by_category[cat] = deleted_by_category.get(cat, 0) + 1

                        # 计算周期性统计
                        periodic_stats = calculate_periodic_stats(selected_date, selected_series=selected_series)

                        if periodic_stats:
                            # 显示总体统计摘要
                            total_deleted = len(deleted_models)
                            current_available = periodic_stats['total_count'] - total_deleted

                            if total_deleted > 0:
                                summary_text = f"""
                                **截止 {selected_date}**，累计出现过衍生模型 **{periodic_stats['total_count']:,}** 个（当前可用 **{current_available:,}** 个，已删除 **{total_deleted:,}** 个），
                                本周新增 **{periodic_stats['weekly_new_count']:,}** 个，
                                {periodic_stats['quarter_name']} 新增 **{periodic_stats['quarter_new_count']:,}** 个
                                """
                            else:
                                summary_text = f"""
                                **截止 {selected_date}**，累计衍生模型 **{periodic_stats['total_count']:,}** 个，
                                本周新增 **{periodic_stats['weekly_new_count']:,}** 个，
                                {periodic_stats['quarter_name']} 新增 **{periodic_stats['quarter_new_count']:,}** 个
                                """

                            st.markdown(summary_text)

                            # 按系列详细统计
                            if periodic_stats['stats_by_series']:
                                st.markdown("#### 📊 分系列统计")
                                for category, stats in periodic_stats['stats_by_series'].items():
                                    deleted_count = deleted_by_category.get(category, 0)
                                    available_count = stats['total_count'] - deleted_count

                                    if deleted_count > 0:
                                        detail_text = f"（当前可用 **{available_count:,}** 个，已删除 **{deleted_count:,}** 个）"
                                    else:
                                        detail_text = ""

                                    st.markdown(f"""
                                    - **{category}** 衍生模型累计 **{stats['total_count']:,}** 个{detail_text}，
                                      本周新增 **{stats['weekly_new_count']:,}** 个，
                                      {periodic_stats['quarter_name']} 新增 **{stats['quarter_new_count']:,}** 个
                                    """)

                            # 本周新增模型列表
                            if periodic_stats['weekly_new_models']:
                                with st.expander(f"📋 本周新增模型列表 ({periodic_stats['weekly_new_count']} 个)", expanded=False):
                                    weekly_new_df = pd.DataFrame(periodic_stats['weekly_new_models'])
                                    weekly_new_df['download_count'] = pd.to_numeric(
                                        weekly_new_df['download_count'], errors='coerce'
                                    ).fillna(0).astype(int)
                                    weekly_new_df = weekly_new_df.sort_values('download_count', ascending=False)

                                    # 选择要显示的列
                                    weekly_display_cols = ['model_name', 'publisher', 'repo', 'download_count']
                                    if 'model_category' in weekly_new_df.columns:
                                        weekly_display_cols.append('model_category')
                                    if 'model_type' in weekly_new_df.columns:
                                        weekly_display_cols.append('model_type')
                                    if 'base_model' in weekly_new_df.columns:
                                        weekly_display_cols.append('base_model')
                                    if 'url' in weekly_new_df.columns:
                                        weekly_display_cols.append('url')

                                    # 确保所有列都存在
                                    weekly_display_cols = [col for col in weekly_display_cols if col in weekly_new_df.columns]

                                    # 重命名列以便更好地显示
                                    rename_dict = {
                                        'model_name': '模型名称',
                                        'publisher': '发布者',
                                        'repo': '平台',
                                        'download_count': '下载量',
                                        'model_category': '模型系列',
                                        'model_type': '模型类型',
                                        'base_model': 'Base Model',
                                        'url': '模型URL'
                                    }
                                    display_df = weekly_new_df[weekly_display_cols].copy()
                                    display_df = display_df.rename(columns=rename_dict)

                                    st.dataframe(display_df, use_container_width=True, height=300)
                            else:
                                st.info("✅ 本周暂无新增衍生模型")

                    st.markdown("---")

                    # ========== 3. 已删除模型检测 ==========
                    st.markdown("### 🗑️ 已删除模型")

                    # 使用之前已经获取的 deleted_models
                    if deleted_models:
                        st.warning(f"⚠️ 检测到 {len(deleted_models)} 个模型已被删除或隐藏")
                        with st.expander(f"📋 已删除模型列表 ({len(deleted_models)} 个)", expanded=False):
                            deleted_df = pd.DataFrame(deleted_models)
                            deleted_df['last_download_count'] = pd.to_numeric(
                                deleted_df['last_download_count'], errors='coerce'
                            ).fillna(0).astype(int)
                            st.dataframe(deleted_df, use_container_width=True, height=300)
                    else:
                        st.success("✅ 未检测到已删除的模型")

                    st.markdown("---")

                    # ========== 4. 需要回填的模型 ==========
                    st.markdown("### 🔄 需要回填的模型")

                    with st.spinner("正在检测需要回填的模型..."):
                        models_needing_backfill = get_models_needing_backfill(
                            selected_date,
                            selected_series=selected_series
                        )

                        if models_needing_backfill:
                            st.info(f"📊 检测到 {len(models_needing_backfill)} 个模型的当前下载量低于历史最大值")
                            with st.expander(f"📋 需要回填的模型列表 ({len(models_needing_backfill)} 个)", expanded=False):
                                backfill_df = pd.DataFrame(models_needing_backfill)
                                backfill_df['差值'] = backfill_df['max_download_count'] - backfill_df['current_download_count']
                                backfill_df = backfill_df.rename(columns={
                                    'model_name': '模型名称',
                                    'publisher': '发布者',
                                    'model_category': '模型系列',
                                    'repo': '平台',
                                    'current_download_count': '当前下载量',
                                    'max_download_count': '历史最大下载量',
                                    'max_download_date': '最大下载量日期'
                                })
                                st.dataframe(backfill_df, use_container_width=True, height=300)
                        else:
                            st.success("✅ 所有模型的下载量均为历史最大值，无需回填")

                    st.markdown("---")

                    # ========== 5. 按平台统计 ==========
                    st.markdown("### 🌍 按平台统计")

                    if analysis_result['by_platform']:
                        # 创建平台统计表格
                        platform_data = []

                        # 判断是否选择了多个系列
                        is_multi_series = len(selected_series) > 1

                        for platform, stats in analysis_result['by_platform'].items():
                            row_data = {
                                '平台': platform,
                                '衍生模型总数': stats['derivative_models'],
                                '衍生模型总下载量': f"{stats['total_downloads']:,}"
                            }

                            # 如果选择了多个系列，添加分系列统计
                            if is_multi_series and 'by_series' in stats and stats['by_series']:
                                series_mapping = {
                                    "ernie-4.5": "ERNIE-4.5",
                                    "paddleocr-vl": "PaddleOCR-VL",
                                    "other-ernie": "其他ERNIE"
                                }

                                for category, category_stats in stats['by_series'].items():
                                    display_name = series_mapping.get(category, category)
                                    row_data[f'{display_name}衍生模型数'] = category_stats['count']
                                    row_data[f'{display_name}衍生模型下载量'] = f"{category_stats['downloads']:,}"

                            platform_data.append(row_data)

                        platform_df = pd.DataFrame(platform_data)

                        # 排序列：优先按衍生模型总数排序
                        if is_multi_series:
                            platform_df = platform_df.sort_values('衍生模型总数', ascending=False)
                        else:
                            # 单系列时保持原有排序逻辑
                            platform_df = platform_df.sort_values('衍生模型总数', ascending=False)

                        # 展示表格
                        st.dataframe(platform_df, use_container_width=True, height=300)

                        # 可视化：衍生模型数量对比
                        col_chart1, col_chart2 = st.columns(2)

                        with col_chart1:
                            fig_platform = px.bar(
                                platform_df,
                                x='平台',
                                y='衍生模型总数',
                                title="各平台衍生模型数量",
                                text='衍生模型总数'
                            )
                            fig_platform.update_traces(texttemplate='%{text}', textposition='outside')
                            fig_platform.update_layout(showlegend=False)
                            st.plotly_chart(fig_platform, use_container_width=True)

                        with col_chart2:
                            # 重新计算衍生率数据
                            rate_data = []
                            for platform, stats in analysis_result['by_platform'].items():
                                rate_data.append({
                                    '平台': platform,
                                    '衍生率': stats['derivative_rate']
                                })

                            rate_df = pd.DataFrame(rate_data)

                            fig_rate = px.bar(
                                rate_df,
                                x='平台',
                                y='衍生率',
                                title="各平台衍生率",
                                labels={'y': '衍生率 (%)'},
                                text=rate_df['衍生率'].apply(lambda x: f'{x:.1f}%')
                            )
                            fig_rate.update_traces(texttemplate='%{text}', textposition='outside')
                            fig_rate.update_layout(showlegend=False)
                            st.plotly_chart(fig_rate, use_container_width=True)

                        # ========== 6. 各平台Top模型 ==========
                        st.markdown("### 🏆 各平台下载量Top模型")

                        for platform, stats in analysis_result['by_platform'].items():
                            if stats['derivative_models'] > 0 and stats['top_models']:
                                with st.expander(f"📊 {platform} (衍生模型: {stats['derivative_models']} 个)", expanded=False):
                                    top_models_df = pd.DataFrame(stats['top_models'])
                                    if not top_models_df.empty:
                                        top_models_df['download_count'] = pd.to_numeric(
                                            top_models_df['download_count'], errors='coerce'
                                        ).fillna(0).astype(int)
                                        st.dataframe(top_models_df, use_container_width=True)
                                    else:
                                        st.info("暂无数据")

                    st.markdown("---")

                    # ========== 7. 按系列统计 ==========
                    if analysis_result['by_series']:
                        st.markdown("### 📈 按模型系列统计")

                        series_data = []
                        for series, stats in analysis_result['by_series'].items():
                            series_data.append({
                                '模型系列': series,
                                '总模型数': stats['total_models'],
                                '官方模型': stats['official_models'],
                                '衍生模型': stats['derivative_models'],
                                '衍生率': f"{stats['derivative_rate']:.1f}%"
                            })

                        series_df = pd.DataFrame(series_data)
                        st.dataframe(series_df, use_container_width=True)

                        st.markdown("---")

                    # ========== 8. 衍生模型详细列表 ==========
                    st.markdown("### 📋 衍生模型详细列表")

                    derivative_models_df = analysis_result['derivative_models_df']

                    if not derivative_models_df.empty:
                        # 筛选器
                        col_filter1, col_filter2 = st.columns(2)

                        with col_filter1:
                            platform_options = ['全部'] + sorted(derivative_models_df['repo'].unique().tolist())
                            selected_platform = st.selectbox("筛选平台", platform_options, key="filter_platform")

                        with col_filter2:
                            if 'model_category' in derivative_models_df.columns:
                                category_options = ['全部'] + sorted(
                                    derivative_models_df['model_category'].dropna().unique().tolist()
                                )
                                selected_category = st.selectbox("筛选模型系列", category_options, key="filter_category")
                            else:
                                selected_category = '全部'

                        # 应用筛选
                        filtered_derivatives = derivative_models_df.copy()

                        if selected_platform != '全部':
                            filtered_derivatives = filtered_derivatives[
                                filtered_derivatives['repo'] == selected_platform
                            ]

                        if selected_category != '全部' and 'model_category' in filtered_derivatives.columns:
                            filtered_derivatives = filtered_derivatives[
                                filtered_derivatives['model_category'] == selected_category
                            ]

                        st.info(f"📊 共 {len(filtered_derivatives)} 个衍生模型符合筛选条件")

                        # 从数据库获取每个模型的首次入库日期（使用原始数据，不是回填后的）
                        from ernie_tracker.db import DB_PATH, DATA_TABLE
                        import sqlite3

                        # 获取当前筛选结果中的模型唯一标识
                        if not filtered_derivatives.empty:
                            model_keys = filtered_derivatives[['repo', 'publisher', 'model_name']].drop_duplicates()

                            # 构建 SQL 查询，获取每个模型首次出现的日期
                            first_seen_dates = {}
                            conn = sqlite3.connect(DB_PATH)
                            for _, row in model_keys.iterrows():
                                repo = row['repo']
                                publisher = row['publisher']
                                model_name = row['model_name']

                                # 查询该模型在数据库中最早出现的日期
                                query = f"""
                                SELECT MIN(date) as first_date
                                FROM {DATA_TABLE}
                                WHERE repo = ? AND publisher = ? AND model_name = ?
                                """
                                cursor = conn.execute(query, (repo, publisher, model_name))
                                result = cursor.fetchone()
                                if result and result[0]:
                                    first_seen_dates[(repo, publisher, model_name)] = result[0]
                            conn.close()

                            # 添加首次入库日期列
                            filtered_derivatives['first_seen_date'] = filtered_derivatives.apply(
                                lambda row: first_seen_dates.get((row['repo'], row['publisher'], row['model_name']), ''),
                                axis=1
                            )

                        # 定义显示字段（移除大量缺失的字段）
                        all_possible_cols = [
                            'model_name', 'publisher', 'repo', 'download_count',
                            'model_category', 'model_type', 'base_model',
                            'data_source', 'url', 'first_seen_date'
                        ]

                        # 只显示存在的列
                        display_cols = [col for col in all_possible_cols if col in filtered_derivatives.columns]

                        # 转换下载量为数值类型用于排序
                        filtered_derivatives['download_count_num'] = pd.to_numeric(
                            filtered_derivatives['download_count'], errors='coerce'
                        ).fillna(0)

                        # 按下载量降序排序，显示所有字段
                        display_df = filtered_derivatives.sort_values('download_count_num', ascending=False)[display_cols].reset_index(drop=True)

                        # 显示所有模型
                        st.dataframe(display_df, use_container_width=True, height=500)

                        # 导出功能
                        st.markdown("### 📥 导出报告")

                        if st.button("生成Excel报告", type="secondary"):
                            from openpyxl import Workbook
                            output = BytesIO()

                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Sheet 1: 总体概览
                                overview_data = {
                                    '指标': ['总模型数', '衍生模型数', '官方模型数', '衍生率'],
                                    '数值': [
                                        analysis_result['total_models'],
                                        analysis_result['total_derivative_models'],
                                        analysis_result['total_official_models'],
                                        f"{analysis_result['derivative_rate']:.1f}%"
                                    ]
                                }
                                pd.DataFrame(overview_data).to_excel(writer, sheet_name='总体概览', index=False)

                                # Sheet 2: 平台统计
                                platform_df.to_excel(writer, sheet_name='平台统计', index=False)

                                # Sheet 3: 系列统计
                                if analysis_result['by_series']:
                                    series_df.to_excel(writer, sheet_name='系列统计', index=False)

                                # Sheet 4: 衍生模型列表（导出当前筛选结果，包含所有字段）
                                export_df = display_df.copy()
                                # 移除临时排序列
                                if 'download_count_num' in export_df.columns:
                                    export_df = export_df.drop(columns=['download_count_num'])
                                export_df.to_excel(writer, sheet_name='衍生模型列表', index=False)

                            excel_data = output.getvalue()

                            st.download_button(
                                label="📥 下载完整报告",
                                data=excel_data,
                                file_name=f"衍生模型生态分析_{selected_date}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    else:
                        st.info("该日期没有衍生模型数据")
