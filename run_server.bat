@echo off
chcp 65001
echo 正在检查并安装依赖...
pip install -r requirements.txt
echo.
echo 依赖安装完成。正在启动服务器...
echo 请在浏览器访问: http://127.0.0.1:8000
echo.
python server.py
pause