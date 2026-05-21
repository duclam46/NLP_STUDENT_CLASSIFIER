import streamlit as st
from ui.classify_page import render_classify_page
from ui.manage_data_page import render_manage_data_page

st.set_page_config(
    page_title="Hệ thống NLP Sinh viên", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Custom Card */
    div.st-emotion-cache-1r6slb0 {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }
    
    /* Title Gradient */
    h1 {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #edf2f7;
    }
    
    /* Button Hover Effect */
    .stButton > button {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 10px;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    # Avatar area
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 50px;">🎓</div>
            <h2 style="margin-bottom: 0;">NLP Classifier</h2>
            <p style="color: #6B7280; font-size: 0.9em;">Student Query System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.selectbox(
        "📍 Điều hướng",
        ["🔍 Phân loại câu hỏi", "⚙️ Quản lý dữ liệu"],
        index=0
    )
    
    st.markdown("---")
    st.info(f"""
        **Thông tin SV:**
        - 👤: Nhóm 4
        - 🏫: Đại học Vinh
        - 📚: Xử lý Ngôn ngữ Tự nhiên
    """)

if "🔍" in page:
    render_classify_page()
else:
    render_manage_data_page()
