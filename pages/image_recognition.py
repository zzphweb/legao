import streamlit as st
import plotly.express as px
import pandas as pd
from utils.inference_new import predict_from_bytes
import time

# 页面标题
st.markdown("# 📷 乐高积木智能识别")
st.markdown("---")

# 上传区域卡片
st.markdown("""
<div class="card">
    <h3>📤 上传图片</h3>
    <p class="description">支持 JPG、JPEG、PNG 格式，单张图片大小不超过 10MB，可批量上传多张图片</p>
</div>
""", unsafe_allow_html=True)

# 文件上传组件 - 自定义样式已在主CSS中设置
uploaded_files = st.file_uploader(
    "拖拽图片到此处或点击上传",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="支持JPG、JPEG和PNG格式的图片，单张不超过10MB",
    max_upload_size=10  # 设置最大上传大小为10MB
)

# 空状态提示
if not uploaded_files:
    st.markdown("""
    <div class="info-message">
        <strong>📌 提示：</strong>请上传一张或多张乐高积木图片开始识别
    </div>
    """, unsafe_allow_html=True)

if uploaded_files:
    # 显示上传的图片数量
    st.markdown(f"""
    <div class="success-message">
        <strong>✅ 上传成功：</strong>已上传 {len(uploaded_files)} 张图片
    </div>
    """, unsafe_allow_html=True)
    
    # 识别按钮
    st.markdown("---")
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if st.button("🚀 开始识别", use_container_width=True, type="primary"):
            try:
                # 显示加载动画
                with st.spinner("🔄 正在识别中，请稍候..."):
                    time.sleep(0.5)
                    
                    # 对每张图片进行识别
                    results = []
                    for i, file in enumerate(uploaded_files):
                        # 检查文件格式
                        if not file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            results.append((file, None, f"文件 {file.name} 不是有效的图片格式"))
                            continue
                        
                        # 检查文件大小
                        if file.size > 10 * 1024 * 1024:  # 10MB
                            results.append((file, None, f"文件 {file.name} 大小超过 10MB 限制"))
                            continue
                        
                        # 进行推理
                        result = predict_from_bytes(file)
                        results.append((file, result, None))
                    
                    # 显示识别结果
                    st.markdown("---")
                    st.markdown("### 📊 识别结果")
                    
                    for i, (file, result, error) in enumerate(results):
                        st.markdown(f"#### 📷 图片 {i+1}")
                        
                        # 双栏布局：左栏显示原图，右栏显示识别结果
                        col_left, col_right = st.columns([1, 1.5])
                        
                        with col_left:
                            st.markdown("""
                            <div class="card">
                                <h4>📸 原始图片</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(file, width="stretch")
                        
                        with col_right:
                            st.markdown("""
                            <div class="card">
                                <h4>🎯 识别结果</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if error:
                                st.markdown(f"""
                                <div class="error-message">
                                    <strong>❌ 错误：</strong>{error}
                                </div>
                                """, unsafe_allow_html=True)
                            elif result:
                                # 检查是否为乐高积木（置信度阈值）
                                if result['confidence'] < 0.5:
                                    st.markdown(f"""
                                    <div class="warning-message">
                                        <strong>⚠️ 识别结果：</strong>不是乐高积木<br>
                                        <strong>置信度：</strong>{result['confidence']:.2f}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="success-message">
                                        <strong>✅ 识别结果：</strong>{result['predicted_class']}<br>
                                        <strong>置信度：</strong>{result['confidence']:.2f}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 显示概率分布条形图
                                st.markdown("#### 📈 概率分布")
                                
                                # 准备数据
                                data = pd.DataFrame({
                                    '类别': result['class_names'],
                                    '概率': result['probabilities']
                                })
                                
                                # 按概率降序排列
                                data = data.sort_values('概率', ascending=False)
                                
                                # 获取最高概率的类别
                                max_prob_index = data['概率'].idxmax()
                                max_class = data.loc[max_prob_index, '类别']
                                
                                # 创建条形图
                                fig = px.bar(
                                    data, 
                                    x='类别', 
                                    y='概率',
                                    text='概率',
                                    title='各类别识别概率分布',
                                    height=400
                                )
                                
                                # 设置文本格式
                                fig.update_traces(
                                    texttemplate='%{text:.2f}', 
                                    textposition='outside',
                                    marker_color=['#E3000B' if cat == max_class else '#0055BF' for cat in data['类别']]
                                )
                                
                                # 更新布局
                                fig.update_layout(
                                    xaxis_title="积木类型",
                                    yaxis_title="概率",
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(size=12)
                                )
                                
                                # 显示图表
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.markdown("""
                                <div class="error-message">
                                    <strong>❌ 识别失败：</strong>图片可能损坏或格式不正确，请重试
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                    
                    st.markdown("""
                    <div class="success-message" style="text-align: center;">
                        <strong>🎉 识别完成！</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    <strong>❌ 处理过程中出现错误：</strong>{str(e)}<br>
                    <strong>💡 建议：</strong>请检查上传的图片是否正确，或稍后重试
                </div>
                """, unsafe_allow_html=True)