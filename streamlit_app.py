import streamlit as st
import os

# 删除不需要的页面文件，避免自动生成导航
if os.path.exists("pages/app.py"):
    os.remove("pages/app.py")
if os.path.exists("pages/example_demo.py"):
    os.remove("pages/example_demo.py")

# 设置页面配置
st.set_page_config(
    page_title="乐高积木智能识别系统",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主容器样式 */
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding-top: 1rem;
    }
    
    /* 卡片样式 */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #E3000B;
    }
    
    /* 标题样式 */
    h1 {
        color: #E3000B;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #FFD500;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #0055BF;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    h4 {
        color: #333;
        font-size: 1.1rem;
        font-weight: bold;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #E3000B;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #C0000A;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(227, 0, 11, 0.3);
    }
    
    /* 上传区域样式 */
    .upload-container {
        background-color: #f8f9fa;
        padding: 3rem;
        border-radius: 0.5rem;
        border: 2px dashed #E3000B;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 结果区域样式 */
    .result-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    /* 成功消息样式 */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    /* 警告消息样式 */
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    /* 错误消息样式 */
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    /* 信息消息样式 */
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    /* 说明文字样式 */
    .description {
        color: #666;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* 拖拽上传区域样式 */
    .stFileUploader {
        margin-bottom: 1.5rem;
    }
    
    .stFileUploader label {
        font-size: 1rem;
        font-weight: 500;
        color: #333;
        display: none !important;
    }
    
    .stFileUploader div {
        border: 2px dashed #E3000B !important;
        border-radius: 0.5rem !important;
        background-color: #f8f9fa !important;
        padding: 3rem !important;
    }
    
    .stFileUploader div:hover {
        border-color: #C0000A !important;
        background-color: #f0f2f5 !important;
    }
    
    /* 自定义上传区域提示文字 */
    .custom-upload-text {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* 双栏布局样式 */
    .two-column-layout {
        display: flex;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .left-column {
        flex: 1;
    }
    
    .right-column {
        flex: 1.5;
    }
    
    @media (max-width: 768px) {
        .two-column-layout {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.markdown("---")

# 导航菜单，将图片识别放在最上方
page = st.sidebar.radio(
    "导航",
    ["📷 图片识别", "ℹ️ 关于项目"],
    label_visibility="collapsed"
)

# 导入页面模块
if page == "📷 图片识别":
    import pages.image_recognition
elif page == "ℹ️ 关于项目":
    import pages.about
