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


# 页面配置
st.set_page_config(page_title="ERNIE模型下载数据统计", layout="wide")
st.title("📊 ERNIE模型下载数据统计")


def fetch_platform_data_only(platform_name, fetch_func, save_to_database=True):
    """
    仅执行数据抓取（不包含UI操作，用于并行执行）

    Args:
        platform_name: 平台名称
        fetch_func: 抓取函数
        save_to_database: 是否保存到数据库

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
        """进度回调函数（仅收集进度信息，不更新UI）"""
        if ref["denom"]:  # 有参考总数
            denom = ref["denom"]
            if processed > denom:
                if save_to_database:
                    update_last_model_count(platform_name, processed)
                ref["denom"] = processed
                denom = processed

            progress = min(processed / denom, 1.0)
            progress_updates.append({
                'processed': processed,
                'total': denom,
                'progress': progress,
                'message': f"已处理 {processed} / 参考总数 {denom}"
            })
        else:  # 首次运行
            if discovered_total:
                progress = processed / discovered_total
                progress_updates.append({
                    'processed': processed,
                    'total': discovered_total,
                    'progress': progress,
                    'message': f"已处理 {processed} / 实际总数 {discovered_total}"
                })
            else:
                progress_updates.append({
                    'processed': processed,
                    'total': None,
                    'progress': None,
                    'message': f"已处理 {processed} （总数未知）"
                })

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
    并行运行多个平台的数据抓取（修复版：避免在线程中调用Streamlit API）

    Args:
        platforms: 平台名称列表
        fetchers_to_use: 平台抓取函数字典
        save_to_database: 是否保存到数据库

    Returns:
        tuple: (DataFrame列表, 总用时)
    """
    all_dfs = []
    total_start_time = time.time()

    # 创建UI容器
    st.markdown("### ⏳ 并行更新进度")
    overall_progress = st.empty()

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

    def fetch_platform_task(platform_name):
        """单个平台抓取任务（纯数据处理，不包含UI操作）"""
        fetch_func = fetchers_to_use.get(platform_name)
        if fetch_func:
            return fetch_platform_data_only(platform_name, fetch_func, save_to_database)
        return platform_name, None, False, 0, "抓取函数未找到", []

    # 使用线程池并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(platforms), 4)) as executor:
        # 提交所有任务
        future_to_platform = {
            executor.submit(fetch_platform_task, platform): platform
            for platform in platforms
        }

        completed_count = 0
        total_count = len(platforms)

        # 实时更新各平台状态
        while completed_count < total_count:
            # 检查已完成的任务
            for future in list(future_to_platform.keys()):
                if future.done():
                    platform_name = future_to_platform.pop(future)
                    completed_count += 1

                    try:
                        # 获取结果
                        _, df, success, elapsed_time, error_message, progress_updates = future.result()

                        # 更新该平台的最终状态
                        if success:
                            platform_status[platform_name]['status'].success(f"✅ {platform_name} 完成")
                            platform_status[platform_name]['details'].success(progress_updates[-1]['message'] if progress_updates else "完成")
                            platform_status[platform_name]['time'].success(f"⏱️ 用时: {elapsed_time:.2f} 秒")
                            platform_status[platform_name]['progress'].progress(1.0)

                            if df is not None:
                                all_dfs.append(df)
                        else:
                            platform_status[platform_name]['status'].error(f"❌ {platform_name} 失败")
                            platform_status[platform_name]['details'].error(error_message)
                            platform_status[platform_name]['time'].error(f"⏱️ 用时: {elapsed_time:.2f} 秒")

                    except Exception as e:
                        platform_status[platform_name]['status'].error(f"❌ {platform_name} 异常")
                        platform_status[platform_name]['details'].error(f"执行异常: {e}")

                    # 更新总体进度
                    overall_progress.info(f"🎯 总体进度：{completed_count}/{total_count} 个平台完成")

            # 短暂休眠避免过度占用CPU
            time.sleep(0.5)

    total_elapsed_time = time.time() - total_start_time
    overall_progress.success(f"🎯 并行抓取完成！总用时：{total_elapsed_time:.2f} 秒")

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
        "🌲 Model Tree 统计",
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

            if use_parallel:
                # 并行执行模式
                all_dfs, total_elapsed_time = run_platforms_parallel(
                    platforms, fetchers_to_use, save_to_database
                )
            else:
                # 串行执行模式（原有逻辑）
                total_start_time = time.time()
                st.markdown("### ⏳ 串行更新进度")
                progress_placeholder = st.empty()

                for idx, platform in enumerate(platforms, start=1):
                    progress_placeholder.info(f"正在更新：**{platform}** ({idx}/{len(platforms)})")

                    # 调用平台抓取函数
                    fetch_func = fetchers_to_use.get(platform)
                    if fetch_func:
                        df = run_platform_fetcher(platform, fetch_func, save_to_database)
                        if df is not None:
                            all_dfs.append(df)

                        elapsed = time.time() - total_start_time
                        status_msg = "数据已保存" if save_to_database else "仅预览"
                        st.success(f"✅ {platform} 完成，用时 {elapsed:.2f} 秒，{status_msg}")

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
                tables = format_report_tables(report_data)

                st.success(f"✅ 周报生成成功！对比时间段：{previous_date} → {current_date}")

                # 检查并显示负增长警告
                warnings_df = tables.get('negative_growth_warnings')
                if warnings_df is not None and not warnings_df.empty:
                    st.markdown("### ⚠️ 负增长警告")
                    st.error(f"检测到 {len(warnings_df)} 个模型出现负增长！这可能表示数据采集问题或模型被下架。")
                    st.dataframe(warnings_df, use_container_width=True)
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

                # 检查并显示负增长警告
                warnings_df = tables.get('negative_growth_warnings')
                if warnings_df is not None and not warnings_df.empty:
                    st.markdown("### ⚠️ 负增长警告")
                    st.error(f"检测到 {len(warnings_df)} 个模型出现负增长！这可能表示数据采集问题或模型被下架。")
                    st.dataframe(warnings_df, use_container_width=True)
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

# ================= Model Tree 统计 =================
elif page == "🌲 Model Tree 统计":
    st.markdown("## 🌲 Model Tree 统计")
    from ernie_tracker.analysis import get_available_dates

    available_dates = get_available_dates()
    if not available_dates:
        st.warning("⚠️ 数据库中暂无数据，请先更新或导入数据。")
    else:
        date_options = ["全部"] + available_dates
        selected_date = st.selectbox("📅 选择日期", options=date_options, index=1 if len(date_options) > 1 else 0)
        date_filter = None if selected_date == "全部" else selected_date

        df = load_data_from_db(date_filter=date_filter)

        if df.empty:
            st.warning(f"⚠️ {selected_date} 没有数据")
            st.stop()

        # 清洗 base_model 中的空值字符串
        if 'base_model' in df.columns:
            df['base_model'] = df['base_model'].apply(
                lambda v: None if str(v).strip().lower() in ['', 'none', 'nan'] else v
            )

        # 仅保留 ERNIE 相关，排除 PaddleOCR-VL
        df = df[df['model_category'] != 'paddleocr-vl']

        total = len(df)
        original_count = len(df[df['model_type'] == 'original']) if 'model_type' in df.columns else 0
        derivative_count = total - original_count

        col_total1, col_total2, col_total3 = st.columns(3)
        with col_total1:
            st.metric("总模型数", total)

        st.markdown("### 📊 数据来源分布")
        source_counts = df['data_source'].fillna('unknown').value_counts().reset_index()
        source_counts.columns = ['data_source', 'count']
        st.dataframe(source_counts, use_container_width=True)

        st.markdown("### 🧭 分类统计")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            cat_counts = df['model_category'].fillna('unknown').value_counts().reset_index()
            cat_counts.columns = ['model_category', 'count']
            st.dataframe(cat_counts, use_container_width=True)
        with col_stats2:
            type_counts = df['model_type'].fillna('unknown').value_counts().reset_index()
            type_counts.columns = ['model_type', 'count']
            st.dataframe(type_counts, use_container_width=True)

        class_total = cat_counts['count'].sum() if not cat_counts.empty else 0
        with col_total2:
            st.metric("分类合计", class_total)
        with col_total3:
            st.metric("衍生模型数", derivative_count)
        if class_total != total:
            st.warning(f"分类计数({class_total})与总模型数({total})不一致，请刷新或检查数据。")

        derivative_df = df[df['base_model'].notna() & (df['base_model'] != '') & (df['model_type'] != 'original')]

        if not derivative_df.empty:
            st.markdown("### 🌳 按基座汇总")
            base_summary = (
                derivative_df.groupby('base_model')
                .agg(
                    derivative_count=('model_name', 'count'),
                    downloads=('download_count', lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()),
                )
                .reset_index()
                .sort_values('derivative_count', ascending=False)
            )
            st.dataframe(base_summary, use_container_width=True)

            st.markdown("### 🏆 下载量 Top 衍生模型")
            top_derivatives = derivative_df.copy()
            top_derivatives['download_count'] = pd.to_numeric(top_derivatives['download_count'], errors='coerce').fillna(0)
            top_derivatives = top_derivatives.sort_values('download_count', ascending=False).head(30)
            display_cols = [
                'model_name',
                'publisher',
                'base_model',
                'download_count',
                'model_type',
                'model_category',
                'data_source',
            ]
            top_derivatives = top_derivatives[[c for c in display_cols if c in top_derivatives.columns]]
            st.dataframe(top_derivatives, use_container_width=True)

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
    from ernie_tracker.analysis import get_available_dates
    from ernie_tracker.model_analysis import analyze_derivative_ecosystem, OFFICIAL_MODEL_GROUPS
    import plotly.express as px
    import plotly.graph_objects as go
    from io import BytesIO

    st.markdown("## 🌳 衍生模型生态分析")
    st.info("📊 分析 ERNIE-4.5 和 PaddleOCR-VL 的衍生模型生态，包括 Finetune、Adapter、量化模型等。支持按模型系列筛选，可单独分析 ERNIE-4.5 或 PaddleOCR-VL。")

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
            # 数据源筛选
            data_source_filter = st.radio(
                "📂 数据来源筛选",
                options=["全部模型", "仅 Model Tree"],
                index=0,
                horizontal=True,
                help="选择要分析的模型范围"
            )

        # 模型系列筛选
        st.markdown("#### 🎯 模型系列筛选")
        selected_series = st.multiselect(
            "选择要分析的模型系列",
            options=["ERNIE-4.5", "PaddleOCR-VL", "其他ERNIE"],
            default=["ERNIE-4.5", "PaddleOCR-VL"],
            help="可以选择一个或多个模型系列进行分析"
        )

        if not selected_series:
            st.warning("⚠️ 请至少选择一个模型系列")
            st.stop()

        # 显示筛选说明
        series_info = "、".join(selected_series)
        if data_source_filter == "仅 Model Tree":
            st.info(f"🌳 **Model Tree 模式** | 📊 **分析系列**: {series_info} | 仅分析通过 Model Tree 找到的衍生模型（有明确的 base_model 关系）")
        else:
            st.info(f"🔍 **全部模型模式** | 📊 **分析系列**: {series_info} | 分析所有相关模型（包括通过搜索和 Model Tree 发现的）")

        if st.button("🔍 开始分析", type="primary"):
            with st.spinner("正在分析衍生模型生态..."):
                # 加载数据
                df = load_data_from_db(date_filter=selected_date)

                if df.empty:
                    st.error(f"❌ {selected_date} 没有数据")
                else:
                    # 筛选 HuggingFace 平台的 ERNIE 和 PaddleOCR 相关数据
                    hf_df = df[df['repo'] == 'Hugging Face'].copy()

                    if hf_df.empty:
                        st.warning("⚠️ 该日期没有 Hugging Face 平台的数据")
                    else:
                        # 根据数据源筛选选项过滤数据
                        if data_source_filter == "仅 Model Tree":
                            # 只保留通过 Model Tree 找到的模型（data_source = 'model_tree' 或 'both'）
                            # 或者至少要有 base_model 的记录
                            if 'data_source' in hf_df.columns:
                                hf_df = hf_df[
                                    (hf_df['data_source'].isin(['model_tree', 'both'])) |
                                    (hf_df['base_model'].notna() & (hf_df['base_model'] != '') & (hf_df['base_model'] != 'None'))
                                ].copy()
                            else:
                                # 如果没有 data_source 列，使用 base_model 判断
                                hf_df = hf_df[
                                    hf_df['base_model'].notna() &
                                    (hf_df['base_model'] != '') &
                                    (hf_df['base_model'] != 'None')
                                ].copy()

                            if hf_df.empty:
                                st.warning("⚠️ 该日期没有 Model Tree 衍生模型数据")
                                st.stop()

                            st.success(f"✅ 筛选后共 {len(hf_df)} 个 Model Tree 衍生模型")
                        else:
                            st.success(f"✅ 共 {len(hf_df)} 个 HuggingFace 模型")

                        # 根据模型系列筛选
                        if 'model_category' in hf_df.columns:
                            # 映射用户选择到 model_category 值
                            series_mapping = {
                                "ERNIE-4.5": "ernie-4.5",
                                "PaddleOCR-VL": "paddleocr-vl",
                                "其他ERNIE": "other-ernie"
                            }
                            selected_categories = [series_mapping[s] for s in selected_series if s in series_mapping]

                            if selected_categories:
                                hf_df = hf_df[hf_df['model_category'].isin(selected_categories)].copy()

                                if hf_df.empty:
                                    st.warning(f"⚠️ 没有找到符合选择的模型系列（{series_info}）的数据")
                                    st.stop()

                                st.info(f"🎯 模型系列筛选后: {len(hf_df)} 个模型")

                        # 进行衍生生态分析
                        analysis_result = analyze_derivative_ecosystem(hf_df, infer_missing=True)

                        st.success(f"✅ 分析完成！分析日期：{selected_date}")

                        # ========== 1. 总体概览 ==========
                        st.markdown("### 📊 总体概览")

                        col1, col2, col3, col4 = st.columns(4)

                        total_models = len(hf_df)
                        derivative_models = analysis_result['total_derivatives']
                        inferred_models = analysis_result['total_inferred']
                        official_models = len(hf_df[hf_df.get('model_type') == 'original']) if 'model_type' in hf_df.columns else 0

                        with col1:
                            st.metric("总模型数", f"{total_models:,}")

                        with col2:
                            st.metric("衍生模型数", f"{derivative_models:,}")

                        with col3:
                            derivative_rate = (derivative_models / total_models * 100) if total_models > 0 else 0
                            st.metric("衍生率", f"{derivative_rate:.1f}%")

                        with col4:
                            st.metric("推断的 base_model", f"{inferred_models:,}")

                        st.markdown("---")

                        # ========== 2. 按 model_category 统计 ==========
                        st.markdown("### 📈 按模型系列分类")

                        if 'model_category' in hf_df.columns:
                            category_counts = hf_df[hf_df['model_category'].notna()]['model_category'].value_counts()

                            if not category_counts.empty:
                                col_chart1, col_chart2 = st.columns([1, 1])

                                with col_chart1:
                                    # 饼图
                                    fig_pie = px.pie(
                                        values=category_counts.values,
                                        names=category_counts.index,
                                        title="模型系列分布",
                                        hole=0.3
                                    )
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig_pie, use_container_width=True)

                                with col_chart2:
                                    # 表格
                                    category_df = pd.DataFrame({
                                        '模型系列': category_counts.index,
                                        '数量': category_counts.values,
                                        '占比': [f"{v/category_counts.sum()*100:.1f}%" for v in category_counts.values]
                                    })
                                    st.dataframe(category_df, use_container_width=True, height=250)

                        st.markdown("---")

                        # ========== 3. 按 model_type 统计 ==========
                        st.markdown("### 🔧 按模型类型分类")
                        st.info("📌 统计衍生模型类型，不包括官方原始模型（original）")

                        if 'model_type' in hf_df.columns:
                            # 过滤掉 'original' 类型（官方原始模型）
                            type_df_filtered = hf_df[
                                hf_df['model_type'].notna() &
                                (hf_df['model_type'] != 'original')
                            ]
                            type_counts = type_df_filtered['model_type'].value_counts()

                            if not type_counts.empty:
                                col_chart3, col_chart4 = st.columns([1, 1])

                                with col_chart3:
                                    # 柱状图
                                    fig_bar = px.bar(
                                        x=type_counts.index,
                                        y=type_counts.values,
                                        title="模型类型分布",
                                        labels={'x': '模型类型', 'y': '数量'},
                                        text=type_counts.values
                                    )
                                    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                                    fig_bar.update_layout(showlegend=False)
                                    st.plotly_chart(fig_bar, use_container_width=True)

                                with col_chart4:
                                    # 表格
                                    type_df = pd.DataFrame({
                                        '模型类型': type_counts.index,
                                        '数量': type_counts.values,
                                        '占比': [f"{v/type_counts.sum()*100:.1f}%" for v in type_counts.values]
                                    })

                                    # 添加类型说明
                                    type_labels = {
                                        'quantized': '量化模型',
                                        'finetune': '微调模型',
                                        'adapter': 'Adapter模型',
                                        'lora': 'LoRA模型',
                                        'merge': '合并模型',
                                        'original': '官方原始',
                                        'other': '其他'
                                    }
                                    type_df['说明'] = type_df['模型类型'].map(type_labels).fillna('未知')

                                    st.dataframe(type_df, use_container_width=True, height=250)

                        st.markdown("---")

                        # ========== 4. 按官方模型分组统计 ==========
                        st.markdown("### 🏷️ 按官方模型分组统计")
                        st.info("📌 统计每个官方模型的衍生生态情况")

                        # 显示汇总表格
                        group_summary_data = []
                        for group_name, group_data in analysis_result['by_group'].items():
                            if group_data['total'] > 0:
                                group_summary_data.append({
                                    '模型分组': group_name,
                                    '衍生模型总数': group_data['total'],
                                    'Quantized': group_data['by_type'].get('quantized', 0),
                                    'Finetune': group_data['by_type'].get('finetune', 0),
                                    'Adapter': group_data['by_type'].get('adapter', 0),
                                    'Merge': group_data['by_type'].get('merge', 0),
                                    'Other': group_data['by_type'].get('other', 0)
                                })

                        if group_summary_data:
                            summary_df = pd.DataFrame(group_summary_data)
                            st.dataframe(summary_df, use_container_width=True)

                            # 可视化：各分组的衍生模型数量对比
                            fig_group = px.bar(
                                summary_df,
                                x='模型分组',
                                y='衍生模型总数',
                                title="各官方模型分组的衍生模型数量",
                                text='衍生模型总数'
                            )
                            fig_group.update_traces(texttemplate='%{text}', textposition='outside')
                            fig_group.update_layout(showlegend=False)
                            st.plotly_chart(fig_group, use_container_width=True)

                            # 详细展开
                            st.markdown("#### 📋 各分组详细信息")

                            for group_name, group_data in analysis_result['by_group'].items():
                                if group_data['total'] > 0:
                                    with st.expander(f"🔍 {group_name} ({group_data['total']} 个衍生模型)", expanded=False):
                                        st.markdown(f"**包含的官方模型：**")
                                        for base_model in group_data['base_models']:
                                            st.markdown(f"- `{base_model}`")

                                        st.markdown(f"\n**类型分布：**")
                                        type_dist_data = []
                                        for model_type, count in sorted(group_data['by_type'].items(), key=lambda x: x[1], reverse=True):
                                            percentage = (count / group_data['total']) * 100
                                            type_dist_data.append({
                                                '类型': model_type,
                                                '数量': count,
                                                '占比': f"{percentage:.1f}%"
                                            })

                                        st.dataframe(pd.DataFrame(type_dist_data), use_container_width=True)

                                        if group_data['by_data_source']:
                                            st.markdown(f"\n**数据来源：**")
                                            source_labels = {
                                                'search': '搜索发现',
                                                'model_tree': 'Model Tree',
                                                'both': '搜索+Model Tree'
                                            }
                                            for source, count in group_data['by_data_source'].items():
                                                label = source_labels.get(source, source)
                                                st.markdown(f"- {label}: {count} 个")

                                        st.markdown(f"\n**样本模型（前10个）：**")
                                        if group_data['models']:
                                            samples = group_data['models'][:10]
                                            sample_df = pd.DataFrame(samples)
                                            sample_df['download_count'] = sample_df['download_count'].apply(lambda x: int(x) if pd.notna(x) else 0)
                                            st.dataframe(sample_df, use_container_width=True)
                        else:
                            st.info("暂无衍生模型数据")

                        st.markdown("---")

                        # ========== 5. 衍生模型详细列表 ==========
                        st.markdown("### 📋 衍生模型详细列表")

                        # 获取所有衍生模型
                        derivatives = hf_df[
                            hf_df['base_model'].notna() &
                            (hf_df['base_model'] != '') &
                            (hf_df['base_model'] != 'None')
                        ].copy()

                        if not derivatives.empty:
                            # 筛选器
                            col_filter1, col_filter2, col_filter3 = st.columns(3)

                            with col_filter1:
                                category_options = ['全部'] + sorted(derivatives['model_category'].dropna().unique().tolist())
                                selected_category = st.selectbox("筛选模型系列", category_options, key="filter_category")

                            with col_filter2:
                                type_options = ['全部'] + sorted(derivatives['model_type'].dropna().unique().tolist())
                                selected_type = st.selectbox("筛选模型类型", type_options, key="filter_type")

                            with col_filter3:
                                base_options = ['全部'] + sorted(derivatives['base_model'].dropna().unique().tolist())
                                selected_base = st.selectbox("筛选基础模型", base_options, key="filter_base")

                            # 应用筛选
                            filtered_derivatives = derivatives.copy()

                            if selected_category != '全部':
                                filtered_derivatives = filtered_derivatives[filtered_derivatives['model_category'] == selected_category]

                            if selected_type != '全部':
                                filtered_derivatives = filtered_derivatives[filtered_derivatives['model_type'] == selected_type]

                            if selected_base != '全部':
                                filtered_derivatives = filtered_derivatives[filtered_derivatives['base_model'] == selected_base]

                            st.info(f"📊 共 {len(filtered_derivatives)} 个衍生模型符合筛选条件")

                            # 选择要显示的列
                            display_cols = ['model_name', 'publisher', 'download_count', 'model_type',
                                          'model_category', 'base_model', 'data_source']
                            available_cols = [col for col in display_cols if col in filtered_derivatives.columns]

                            display_df = filtered_derivatives[available_cols].copy()
                            display_df['download_count'] = display_df['download_count'].apply(lambda x: int(x) if pd.notna(x) else 0)
                            display_df = display_df.sort_values('download_count', ascending=False)

                            # 显示列名中文化
                            display_df = display_df.rename(columns={
                                'model_name': '模型名称',
                                'publisher': '发布者',
                                'download_count': '下载量',
                                'model_type': '模型类型',
                                'model_category': '模型系列',
                                'base_model': '基础模型',
                                'data_source': '数据来源'
                            })

                            st.dataframe(display_df, use_container_width=True, height=400)

                            # ========== 6. 导出功能 ==========
                            st.markdown("### 💾 导出分析结果")

                            col_export1, col_export2 = st.columns([3, 1])

                            with col_export2:
                                # 导出到 Excel
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    # Sheet 1: 总体概览
                                    overview_data = {
                                        '指标': ['总模型数', '衍生模型数', '衍生率', '推断的base_model数'],
                                        '数值': [total_models, derivative_models, f"{derivative_rate:.1f}%", inferred_models]
                                    }
                                    pd.DataFrame(overview_data).to_excel(writer, sheet_name='总体概览', index=False)

                                    # Sheet 2: 模型系列分布
                                    if 'model_category' in hf_df.columns:
                                        category_df.to_excel(writer, sheet_name='模型系列分布', index=False)

                                    # Sheet 3: 模型类型分布
                                    if 'model_type' in hf_df.columns:
                                        type_df.to_excel(writer, sheet_name='模型类型分布', index=False)

                                    # Sheet 4: 分组汇总
                                    if group_summary_data:
                                        summary_df.to_excel(writer, sheet_name='分组汇总', index=False)

                                    # Sheet 5: 衍生模型详细列表
                                    display_df.to_excel(writer, sheet_name='衍生模型列表', index=False)

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
