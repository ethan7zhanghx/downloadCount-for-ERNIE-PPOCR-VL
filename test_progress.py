"""
简单测试：验证并行执行时的进度回调是否工作
"""
import time
import threading
from collections import deque


def test_parallel_progress():
    """测试并行执行时的进度更新"""

    # 共享状态
    progress_state = {
        'platform_a': {'latest': None, 'lock': threading.Lock()},
        'platform_b': {'latest': None, 'lock': threading.Lock()}
    }

    logs = deque(maxlen=20)
    log_lock = threading.Lock()

    def worker(platform_name, duration):
        """模拟工作线程"""
        for i in range(1, 6):
            time.sleep(duration / 5)  # 模拟工作

            # 更新共享状态
            with progress_state[platform_name]['lock']:
                progress = i / 5
                progress_state[platform_name]['latest'] = {
                    'progress': progress,
                    'message': f'{platform_name}: 已处理 {i}/5'
                }

            # 添加日志
            with log_lock:
                logs.append(f"[{time.strftime('%H:%M:%S')}] {platform_name}: 进度 {i*20}%")

        return platform_name, True

    # 启动并行任务
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(worker, 'platform_a', 2): 'platform_a',
            executor.submit(worker, 'platform_b', 3): 'platform_b'
        }

        completed = 0
        last_log_len = 0

        print("=== 开始并行测试 ===\n")

        while completed < 2:
            # 显示所有平台的进度
            for platform in ['platform_a', 'platform_b']:
                with progress_state[platform]['lock']:
                    latest = progress_state[platform]['latest']
                    if latest:
                        print(f"🔄 {platform}: {latest['message']} - 进度条: {int(latest['progress']*100)}%")

            # 显示日志
            with log_lock:
                if len(logs) > last_log_len:
                    print("\n📝 日志:")
                    for log in list(logs)[last_log_len:]:
                        print(f"  {log}")
                    last_log_len = len(logs)

            # 检查完成的任务
            for future in list(futures.keys()):
                if future.done():
                    platform_name = futures.pop(future)
                    completed += 1
                    print(f"\n✅ {platform_name} 完成!")

            time.sleep(0.3)
            print("-" * 50)

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_parallel_progress()
