from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import base64
import io

app = Flask(__name__)

# 加载模型和类别标签
model = tf.keras.models.load_model('model/lego_brick_model.keras')
with open('model/class_labels.json', 'r') as f:
    class_labels = json.load(f)
    # 将键转换为整数
    class_labels = {int(k): v for k, v in class_labels.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return redirect(request.url)
        
        results = []
        for file in files:
            if file:
                # 读取图片
                img = Image.open(file)
                
                # 转换为 RGB 模式（处理 RGBA、WebP、HEIC 等格式）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 预处理图片
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # 模型预测
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                predicted_label = class_labels[predicted_class]
                confidence = predictions[0][predicted_class]
                
                # 将图片转换为 base64 编码，以便在网页中显示
                img_io = io.BytesIO()
                img.save(img_io, format='PNG')
                img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
                
                # 检查是否为乐高积木（置信度阈值）
                is_lego = confidence > 0.5
                
                results.append({
                    'image_url': f'data:image/png;base64,{img_base64}',
                    'prediction': predicted_label,
                    'confidence': f"{confidence * 100:.2f}",
                    'is_lego': is_lego
                })
        
        return render_template('result.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    # 创建 templates 目录
    os.makedirs('templates', exist_ok=True)
    
    # 创建 index.html 模板
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>乐高积木识别</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        .form-container {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .drop-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background-color: #f9f9f9;
            transition: all 0.3s ease;
        }
        .drop-area:hover {
            border-color: #4CAF50;
            background-color: #f0f9f0;
        }
        .drop-area.dragover {
            border-color: #4CAF50;
            background-color: #e8f5e8;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        #preview {
            margin: 20px 0;
            text-align: center;
        }
        #preview img {
            max-width: 200px;
            max-height: 200px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px;
        }
        .preview-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }
        .file-item {
            margin: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>乐高积木识别</h1>
    <div class="form-container">
        <form method="POST" enctype="multipart/form-data" id="upload-form">
            <div class="drop-area" id="drop-area">
                <p>拖放图片到这里，或</p>
                <input type="file" name="files" id="files" accept="image/*" multiple style="display: none;">
                <label for="files" style="cursor: pointer; color: #4CAF50; text-decoration: underline;">点击选择文件（支持批量选择）</label>
            </div>
            <div id="preview" class="preview-container"></div>
            <div style="text-align: center; margin-top: 20px;">
                <button type="button" id="submit-btn" disabled>识别</button>
            </div>
        </form>
        <div id="loading" style="display: none; text-align: center; margin-top: 20px;">
            <p>正在识别...</p>
        </div>
    </div>
    <div style="margin-top: 20px;">
        <h3>项目信息</h3>
        <p>这是一个使用深度学习识别乐高积木的应用</p>
        <p>支持识别多种基础砖块规格：1×1、1×2、1×3、1×4、1×6、2×2、2×3、2×4</p>
    </div>
    <script>
        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('files');
        const preview = document.getElementById('preview');
        const submitBtn = document.getElementById('submit-btn');
        let selectedFiles = [];
        
        // 拖放事件处理
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // 拖放样式
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('dragover');
        }
        
        function unhighlight() {
            dropArea.classList.remove('dragover');
        }
        
        // 处理文件拖放
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        // 处理文件选择
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
        
        // 转换图片为 PNG 格式（处理微信 HEIC 等特殊格式）
        async function convertToPNG(file) {
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = new Image();
                    img.onload = function() {
                        const canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        canvas.toBlob((blob) => {
                            const newFile = new File([blob], file.name.replace(/\.[^.]+$/, '.png'), { type: 'image/png' });
                            resolve(newFile);
                        }, 'image/png');
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            });
        }
        
        async function handleFiles(files) {
            if (files.length > 0) {
                // 转换所有图片为 PNG 格式
                const convertedFiles = [];
                for (const file of files) {
                    const newFile = await convertToPNG(file);
                    convertedFiles.push(newFile);
                }
                selectedFiles = convertedFiles;
                
                // 预览图片
                preview.innerHTML = '';
                selectedFiles.forEach((file, index) => {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        fileItem.innerHTML = `<img src="${e.target.result}" alt="预览图片 ${index+1}">`;
                        preview.appendChild(fileItem);
                    };
                    reader.readAsDataURL(file);
                });
                // 启用提交按钮
                submitBtn.disabled = false;
            }
        }
        
        // 提交表单
        submitBtn.addEventListener('click', async function() {
            if (selectedFiles.length === 0) return;
            
            const loading = document.getElementById('loading');
            loading.style.display = 'block';
            submitBtn.disabled = true;
            
            const formData = new FormData();
            selectedFiles.forEach((file) => {
                formData.append('files', file);
            });
            
            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });
                const html = await response.text();
                document.open();
                document.write(html);
                document.close();
            } catch (error) {
                alert('上传失败: ' + error);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
''')
    
    # 创建 result.html 模板
    with open('templates/result.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>识别结果</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        .result-container {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .result-item {
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        img {
            max-width: 200px;
            max-height: 200px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .confidence {
            font-size: 14px;
            color: #666;
        }
        .not-lego {
            color: #f44336;
            font-weight: bold;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        a:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>识别结果</h1>
    <div class="result-container">
        <div class="result-grid">
            {% for result in results %}
            <div class="result-item">
                <img src="{{ result.image_url }}" alt="上传的图片">
                {% if result.is_lego %}
                <h3>识别结果: {{ result.prediction }}</h3>
                <p class="confidence">置信度: {{ result.confidence }}%</p>
                {% else %}
                <h3 class="not-lego">识别结果: 不是乐高积木</h3>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    <div style="text-align: center; margin-top: 20px;">
        <a href="/">返回上传页面</a>
    </div>
</body>
</html>
''')
    
    # 安装 Flask
    import subprocess
    subprocess.run(['pip', 'install', 'flask'])
    
    # 运行应用
    app.run(debug=True, host='0.0.0.0', port=5000)
