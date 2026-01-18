#!/usr/bin/env python3
"""
图片预览工具 - 启动HTTP服务器预览指定文件夹下的所有图片
用法: python3 preview_images.py [文件夹路径] [端口号]
示例: python3 preview_images.py ./images 8080
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
import threading
from pathlib import Path
from urllib.parse import urlparse

def generate_index_html(directory):
    """在指定目录生成index.html文件"""
    directory = Path(directory).resolve()
    
    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.ico'}
    
    # 扫描图片文件
    images = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            images.append(file_path.name)
    
    # 生成HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片查看器 - {directory.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }}

        .info {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}

        .stats {{
            margin-bottom: 20px;
            color: #666;
            font-size: 14px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 6px;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .image-item {{
            position: relative;
            background: #f5f5f5;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }}

        .image-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
        }}

        .image-item img {{
            width: 100%;
            height: 250px;
            object-fit: cover;
            display: block;
        }}

        .image-name {{
            padding: 10px;
            background: white;
            font-size: 12px;
            color: #333;
            word-break: break-all;
            text-align: center;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            cursor: pointer;
        }}

        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }}

        .modal-content img {{
            max-width: 100%;
            max-height: 90vh;
            object-fit: contain;
        }}

        .close {{
            position: absolute;
            top: 20px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}

        .close:hover {{
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ 图片查看器</h1>
        <div class="info">目录: {directory.name}</div>
        <div class="stats">找到 {len(images)} 张图片</div>
        
        <div class="gallery">
"""
    
    # 添加图片项
    for image_name in images:
        html_content += f"""            <div class="image-item" onclick="openModal('{image_name}')">
                <img src="{image_name}" alt="{image_name}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'250\\' height=\\'250\\'%3E%3Crect fill=\\'%23ddd\\' width=\\'250\\' height=\\'250\\'/%3E%3Ctext fill=\\'%23999\\' font-family=\\'sans-serif\\' font-size=\\'14\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dy=\\'.3em\\'%3E无法加载图片%3C/text%3E%3C/svg%3E'">
                <div class="image-name">{image_name}</div>
            </div>
"""
    
    html_content += """        </div>
    </div>

    <div id="modal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <div class="modal-content">
            <img id="modalImage" src="" alt="">
        </div>
    </div>

    <script>
        function openModal(imagePath) {
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modalImage');
            modalImg.src = imagePath;
            modal.style.display = 'block';
        }

        function closeModal() {
            const modal = document.getElementById('modal');
            modal.style.display = 'none';
        }

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""
    
    # 保存HTML文件
    output_file = directory / 'index.html'
    output_file.write_text(html_content, encoding='utf-8')
    return len(images), output_file


def open_browser(url, delay=1.5):
    """延迟打开浏览器，等待服务器启动"""
    def _open():
        import time
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description='启动HTTP服务器预览指定文件夹下的所有图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 preview_images.py                          # 预览当前目录，使用默认端口8000
  python3 preview_images.py ./images                 # 预览./images目录，使用默认端口8000
  python3 preview_images.py ./images 8080            # 预览./images目录，使用端口8080
  python3 preview_images.py /path/to/images 2345     # 预览指定目录，使用端口2345
        """
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要预览的文件夹路径（默认为当前目录）'
    )
    parser.add_argument(
        'port',
        nargs='?',
        type=int,
        default=8000,
        help='HTTP服务器端口号（默认为8000）'
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    directory = Path(args.directory).resolve()
    if not directory.exists():
        print(f"❌ 错误: 目录不存在: {directory}")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"❌ 错误: 不是目录: {directory}")
        sys.exit(1)
    
    # 生成index.html
    print(f"📁 扫描目录: {directory}")
    try:
        image_count, html_file = generate_index_html(directory)
        print(f"✅ 找到 {image_count} 张图片")
        print(f"✅ 已生成: {html_file}")
    except Exception as e:
        print(f"❌ 生成HTML文件时出错: {e}")
        sys.exit(1)
    
    # 切换到目标目录
    os.chdir(directory)
    
    # 启动HTTP服务器
    port = args.port
    url = f"http://localhost:{port}"
    
    print(f"\n🚀 启动HTTP服务器...")
    print(f"📍 访问地址: {url}")
    print(f"📂 服务目录: {directory}")
    print(f"\n按 Ctrl+C 停止服务器\n")
    
    # 延迟打开浏览器
    open_browser(url)
    
    try:
        with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 错误: 端口 {port} 已被占用，请使用其他端口")
        else:
            print(f"❌ 错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")


if __name__ == '__main__':
    main()
