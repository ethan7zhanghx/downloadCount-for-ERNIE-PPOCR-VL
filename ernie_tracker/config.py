"""
配置文件 - 存储所有常量和配置项
"""

# 数据库配置
DB_PATH = "ernie_downloads.db"
DATA_TABLE = "model_downloads"
STATS_TABLE = "platform_stats"

# 搜索关键词 - 统一搜索所有PaddlePaddle模型
SEARCH_QUERY = "PaddlePaddle"

# Selenium 配置
SELENIUM_TIMEOUT = 40
SELENIUM_WINDOW_SIZE = "1920,1080"
# 控制 Selenium 是否使用无头模式（统一入口，避免多版本代码）
SELENIUM_HEADLESS = False

# GitCode 模型链接列表
GITCODE_MODEL_LINKS = [
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

# CAICT (鲸智) 模型链接列表
CAICT_MODEL_LINKS = [
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
    "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-21B-A3B-Paddle",
    "https://aihub.caict.ac.cn/models/yiyan/ERNIE-4.5-21B-A3B-PT",
    "https://aihub.caict.ac.cn/models/PaddlePaddle/ERNIE-4.5-0.3B-Base-Paddle"
]

# 平台名称映射
PLATFORM_NAMES = {
    "huggingface": "Hugging Face",
    "modelscope": "ModelScope",
    "aistudio": "AI Studio",
    "gitcode": "GitCode",
    "caict": "鲸智",
    "modelers": "魔乐 Modelers",
    "gitee": "Gitee"
}
