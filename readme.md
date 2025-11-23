# Gemini Pro Web UI

这是一个为 Gemini Pro 模型设计的 Web 用户界面，专注于图像生成。它提供了一个简单易用的界面来与 Gemini API 交互，并管理您的生成历史和设置。

## ✨ 主要功能

-   **直观的用户界面**: 简洁明了的界面，用于输入文本提示、调整参数并生成图像。
-   **参数可调**: 支持调整多种生成参数，如分辨率、宽高比、温度（Temperature）和 Top P。
-   **高级模式**: 提供绕过安全限制的选项，并允许设置自定义系统提示词。
-   **历史记录**: 自动保存所有成功的生成记录，包括图像、提示和元数据。
-   **图片管理**: 内置的图库可以方便地查看、搜索和下载历史生成作品。
-   **配置持久化**: 所有设置都会自动保存在服务器上，方便下次使用。
-   **预设管理**: 可以将常用的提示（Prompt）保存为预设，方便快速调用。

## 🚀 如何运行

本项目包含一个 Python 后端和一个纯 HTML/CSS/JS 的前端。

### 1. 安装依赖

项目依赖 Python。首先，需要安装 `requirements.txt` 文件中列出的所有依赖项。

打开终端（或命令提示符），进入项目根目录，然后运行：

```bash
pip install -r requirements.txt
```

### 2. 运行服务器

依赖安装完成后，运行后端服务器。

```bash
python server.py
```

或者，在 Windows 系统上，您可以直接双击运行 `run_server.bat` 脚本，它会自动完成依赖安装和服务器启动。

### 3. 访问 Web 界面

服务器启动后，您会在终端看到类似以下的输出：

```
Starting server at http://127.0.0.1:8040
```

打开您的浏览器，访问 `http://127.0.0.1:8040` 即可开始使用。

## 📁 文件结构

```
.
├── data/                  # 存储生成的图片和缩略图
├── static/                # 存放所有前端文件 (HTML, JS, CSS)
│   ├── index.html         # 应用主页面
│   ├── app.js             # 前端交互逻辑
│   └── styles.css         # 界面样式
├── config.json            # 存储用户配置
├── history.db             # SQLite 数据库，用于存储生成历史
├── requirements.txt       # Python 依赖列表
├── run_server.bat         # (Windows) 一键启动脚本
└── server.py              # FastAPI 后端服务器主程序
