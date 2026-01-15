"""
Streamlit UI测试：验证进度条实时更新
"""
import streamlit as st
import time
import threading
from concurrent.futures import ThreadPoolExecutor


st.set_page_config(page_title="进度测试", layout="wide")
st.title("🧪 并行进度条测试")


def test_real_time_progress():
    """测试Streamlit中的实时进度更新"""

    if st.button("开始测试"):
        # 共享状态
        progress_state = {
            'task_a': {'latest': None, 'lock': threading.Lock()},
            'task_b': {'latest': None, 'lock': threading.Lock()}
        }

        logs = []
        log_lock = threading.Lock()

        def worker(task_name, duration):
            """模拟工作线程"""
            for i in range(1, 11):
                time.sleep(duration / 10)

                # 更新共享状态
                with progress_state[task_name]['lock']:
                    progress = i / 10
                    progress_state[task_name]['latest'] = {
                        'progress': progress,
                        'message': f'{task_name}: 已处理 {i}/10'
                    }

                # 添加日志
                with log_lock:
                    logs.append(f"[{time.strftime('%H:%M:%S')}] {task_name}: 步骤 {i}")

            return task_name, True

        # 创建UI容器
        st.markdown("### ⏳ 任务进度")

        # 创建状态容器
        status_container = st.container()

        with status_container:
            # 任务A的状态
            with st.expander("🔄 任务 A", expanded=True):
                status_a = st.empty()
                progress_a = st.progress(0)
                details_a = st.empty()
                status_a.info("🔄 任务 A 等待中...")

            # 任务B的状态
            with st.expander("🔄 任务 B", expanded=True):
                status_b = st.empty()
                progress_b = st.progress(0)
                details_b = st.empty()
                status_b.info("🔄 任务 B 等待中...")

            # 日志区域
            st.markdown("---")
            st.markdown("#### 📝 实时日志")
            log_placeholder = st.empty()

        # 总体进度
        overall_placeholder = st.empty()

        # 执行并行任务
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(worker, 'task_a', 3): 'task_a',
                executor.submit(worker, 'task_b', 5): 'task_b'
            }

            completed = 0

            # 监控循环
            while completed < 2:
                # 更新所有任务的进度
                with progress_state['task_a']['lock']:
                    latest_a = progress_state['task_a']['latest']
                    if latest_a and latest_a['progress']:
                        progress_a.progress(latest_a['progress'])
                        details_a.info(latest_a['message'])

                with progress_state['task_b']['lock']:
                    latest_b = progress_state['task_b']['latest']
                    if latest_b and latest_b['progress']:
                        progress_b.progress(latest_b['progress'])
                        details_b.info(latest_b['message'])

                # 更新日志
                with log_lock:
                    if logs:
                        recent_logs = logs[-15:]
                        log_text = "\n".join(recent_logs)
                        log_placeholder.text(log_text)

                # 检查完成的任务
                for future in list(futures.keys()):
                    if future.done():
                        task_name = futures.pop(future)
                        completed += 1

                        if task_name == 'task_a':
                            status_a.success("✅ 任务 A 完成")
                            progress_a.progress(1.0)
                        else:
                            status_b.success("✅ 任务 B 完成")
                            progress_b.progress(1.0)

                        overall_placeholder.info(f"🎯 总体进度：{completed}/2 个任务完成")

                time.sleep(0.2)

        overall_placeholder.success("🎉 所有任务完成！")


if __name__ == "__main__":
    test_real_time_progress()
