import streamlit as st

# 页面标题
st.markdown("# ℹ️ 关于项目")
st.markdown("---")

# 项目介绍卡片
st.markdown("""
<div class="card">
    <h3>🎯 项目简介</h3>
    <p class="description">
        这是一个基于深度学习技术的乐高积木智能识别系统，能够准确识别多种基础砖块规格。
        系统采用卷积神经网络（CNN）架构，通过大量乐高积木图片训练而成，
        为乐高爱好者、收藏家和教育工作者提供便捷的积木识别服务。
    </p>
</div>
""", unsafe_allow_html=True)

# 支持的积木类型
st.markdown("### 🧱 支持的积木类型")
st.markdown("""
<div class="card">
    <p class="description">
        系统支持识别以下 8 种基础砖块规格：
    </p>
    <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
        <li><strong>1×1</strong> - 基础单位积木</li>
        <li><strong>1×2</strong> - 常用长条积木</li>
        <li><strong>1×3</strong> - 中等长度积木</li>
        <li><strong>1×4</strong> - 标准长条积木</li>
        <li><strong>1×6</strong> - 加长条形积木</li>
        <li><strong>2×2</strong> - 基础方形积木</li>
        <li><strong>2×3</strong> - 矩形积木</li>
        <li><strong>2×4</strong> - 标准矩形积木</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 功能特点
st.markdown("### ✨ 功能特点")
# 使用st.columns创建两栏布局，确保对齐整齐
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <h4>🚀 核心功能</h4>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>✅ 支持批量上传图片识别</li>
            <li>✅ 显示识别概率分布</li>
            <li>✅ 智能识别非乐高图片</li>
            <li>✅ 实时识别结果反馈</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4>💡 用户体验</h4>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>🎨 简洁直观的界面设计</li>
            <li>📊 可视化概率分布图表</li>
            <li>🔍 详细的识别结果展示</li>
            <li>⚡ 快速响应的识别速度</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 技术栈（使用折叠面板）
st.markdown("### 🔧 技术栈")

with st.expander("📱 前端技术", expanded=False):
    st.markdown("""
    <div class="card">
        <h4>Streamlit</h4>
        <p class="description">
            Streamlit 是一个开源的 Python 框架，专为机器学习和数据科学应用设计。
            它能够快速创建交互式 Web 应用，无需编写前端代码。
        </p>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>快速原型开发</li>
            <li>实时数据可视化</li>
            <li>简洁的 API 设计</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🧠 深度学习框架", expanded=False):
    st.markdown("""
    <div class="card">
        <h4>PyTorch</h4>
        <p class="description">
            PyTorch 是一个开源的机器学习库，基于 Torch 库，用于计算机视觉和自然语言处理等应用。
            它提供了灵活的张量计算和动态计算图，是深度学习领域的主流框架之一。
        </p>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>动态计算图</li>
            <li>丰富的神经网络模块</li>
            <li>强大的 GPU 加速支持</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🏗️ 模型架构", expanded=False):
    st.markdown("""
    <div class="card">
        <h4>卷积神经网络 (CNN)</h4>
        <p class="description">
            卷积神经网络是一种专门处理具有类似网格结构数据（如图像）的深度学习模型。
            通过卷积层、池化层和全连接层的组合，能够自动提取图像特征并进行分类。
        </p>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>自动特征提取</li>
            <li>平移不变性</li>
            <li>参数共享机制</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📊 数据可视化", expanded=False):
    st.markdown("""
    <div class="card">
        <h4>Plotly</h4>
        <p class="description">
            Plotly 是一个交互式数据可视化库，支持创建高质量的图表和仪表板。
            它提供了丰富的图表类型和交互功能，使数据展示更加生动直观。
        </p>
        <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
            <li>交互式图表</li>
            <li>多种图表类型</li>
            <li>响应式设计</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 项目结构
st.markdown("### 📁 项目结构")
st.markdown("""
<div class="card">
<pre style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.25rem; overflow-x: auto; margin-bottom: 0;">
lego/
├── streamlit_app.py              # 主应用文件
├── pages/                        # 页面模块
│   ├── image_recognition.py      # 图片识别页面
│   └── about.py                  # 关于项目页面
├── model/                        # 模型相关
│   ├── model_def.py              # 模型定义
│   └── best_model.pth            # 最佳模型权重
└── utils/                        # 工具函数
    └── inference_new.py           # 推理函数
</pre>
</div>
""", unsafe_allow_html=True)

# 使用说明
st.markdown("### 📖 使用说明")
st.markdown("""
<div class="card">
    <h4>🚀 快速开始</h4>
    <ol style="color: #666; line-height: 1.8; margin-bottom: 0;">
        <li>在侧边栏选择"📷 图片识别"功能</li>
        <li>点击上传区域或拖拽乐高积木图片</li>
        <li>支持批量上传多张图片</li>
        <li>点击"🚀 开始识别"按钮</li>
        <li>查看识别结果和概率分布</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# 注意事项
st.markdown("### ⚠️ 注意事项")
st.markdown("""
<div class="card">
    <ul style="color: #666; line-height: 1.8; margin-bottom: 0;">
        <li>📌 支持的图片格式：JPG、JPEG、PNG</li>
        <li>📌 建议图片大小：不超过 10MB</li>
        <li>📌 识别准确率：取决于图片质量和清晰度</li>
        <li>📌 置信度低于 0.5 时会提示"不是乐高积木"</li>
        <li>📌 建议使用正面拍摄的积木图片以获得最佳识别效果</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 联系方式
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>📧 联系我们</h4>
    <p style="margin-bottom: 1rem;">如有任何问题或建议，欢迎通过以下方式联系我们</p>
    <p style="margin-top: 1rem; line-height: 1.6;">
        <strong>项目地址：</strong>乐高积木智能识别系统<br>
        <strong>技术支持：</strong>深度学习实验室
    </p>
</div>
""", unsafe_allow_html=True)