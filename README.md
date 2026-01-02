# ERNIE 模型下载量统计系统

本项目是一个使用 Streamlit 构建的 Web 应用，用于统计和展示 ERNIE 模型的下载量数据。

## 项目结构

- `ernie_tracker/`：核心代码包（配置、数据库访问、分析、统一抓取器与 Selenium 抓取器）。
- `scripts/`：运维/数据修复脚本（重分类、修复标签、导入导出等）。
- `tests/`：测试脚本。
- `app.py`：Streamlit 入口。
- `ernie_downloads.db`：默认本地数据库（未纳入 Git）。
- `start.sh`：一键启动脚本。

## 如何启动

1.  **安装依赖:**

    在首次运行或依赖缺失时，系统会自动检查并安装所需的 Python 包。您也可以手动执行以下命令来安装依赖：

    ```bash
    pip3 install -r requirements.txt
    ```

2.  **启动应用:**

    执行以下脚本来启动 Streamlit 应用：

    ```bash
    ./start.sh
    ```

    或者，您可以直接使用 `python3 -m streamlit` 命令：

    ```bash
    python3 -m streamlit run app.py
    ```

    启动后，应用将在您的默认浏览器中打开。
