#!/bin/bash

echo "======================================"
echo "ERNIE 模型下载量统计系统"
echo "======================================"
echo ""

# 检查是否安装了依赖
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  检测到未安装依赖，正在安装..."
    pip3 install -r requirements.txt
fi

echo "🚀 启动应用..."
echo ""

# 启动 Streamlit 应用
python3 -m streamlit run app.py
