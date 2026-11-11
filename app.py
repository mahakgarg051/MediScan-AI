"""
MediScan AI — Modular Healthcare Intelligence Platform
========================================================
A multi-task AI healthcare research platform featuring:
1. Chest X-ray Analysis — MobileNetV3 fine-tuned pneumonia classifier & Grad-CAM (Active)
2. Brain MRI Analysis — Neuro-oncology multi-class tumor classification (Scaffolding / Awaiting Training)
3. Medicine Identification — Computer vision pill & packaging identification (Scaffolding / Awaiting Catalog)
4. Model Evaluation & Metrics — Validated test cohort evaluation & error reports

Hardware & VRAM Optimization: Designed for 4GB GPU VRAM environments.
"""

from pathlib import Path
import sys

import streamlit as st
import torch

# Ensure repository root is in Python sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import (
    render_chest_xray_module,
    render_chest_xray_evaluation,
    render_brain_mri_module,
)


# ---------------- Streamlit Page Configuration ----------------
st.set_page_config(
    page_title="MediScan AI — Healthcare Research Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Global CSS & Design System ----------------
st.markdown("""
<style>
/* Clean Font Stack */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Futuristic Dark Medical Palette */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: radial-gradient(circle at 50% 0%, #0d1e38 0%, #070e1b 45%, #040812 100%) !important;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f1e 0%, #0a172c 55%, #050b14 100%) !important;
    border-right: 1px solid rgba(0, 210, 255, 0.18) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
}

/* Hide Default Streamlit Chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent; visibility: hidden; height: 0px;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #060d17;
}
::-webkit-scrollbar-thumb {
    background: #1e3a5f;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00d2ff;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
p, span, label {
    color: #cbd5e1;
}

/* Medical Glassmorphism Card System */
.med-card {
    background: linear-gradient(145deg, rgba(13, 27, 48, 0.82) 0%, rgba(8, 17, 32, 0.92) 100%);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    transition: all 0.25s ease;
    margin-bottom: 18px;
}
.med-card:hover {
    border-color: rgba(0, 210, 255, 0.35);
    box-shadow: 0 12px 35px -8px rgba(0, 210, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

/* Top Header Banner */
.main-header-banner {
    background: linear-gradient(135deg, rgba(15, 32, 59, 0.95) 0%, rgba(8, 17, 33, 0.98) 100%);
    border: 1px solid rgba(0, 210, 255, 0.25);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 16px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}
.main-header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d2ff, #3b82f6, #06b6d4, #00d2ff);
    background-size: 200% 100%;
}

.brand-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.brand-title span.glow {
    background: linear-gradient(90deg, #38bdf8 0%, #00d2ff 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-subtitle {
    font-size: 0.93rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Dynamic Three-Step Workflow Indicators */
.workflow-pills {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.wf-step {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(14, 31, 56, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 9999px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
}
.wf-step.active {
    background: rgba(0, 210, 255, 0.14);
    border-color: #00d2ff;
    color: #38bdf8;
    box-shadow: 0 0 14px rgba(0, 210, 255, 0.25);
}
.wf-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #00d2ff;
    color: #050b14;
    font-size: 0.72rem;
    font-weight: 800;
}
.wf-arrow {
    color: #475569;
    font-size: 0.85rem;
}

/* Safety Disclaimer Bar */
.disclaimer-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 10px 18px;
    color: #fde68a;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

/* Streamlit Container (Border=True) Override as Polish Glass Cards */
div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="column"] > div {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    height: 100% !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(13, 27, 48, 0.92) 0%, rgba(7, 15, 29, 0.96) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.25s ease !important;
    flex: 1 1 auto !important;
    height: 100% !important;
    margin-bottom: 0px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0, 210, 255, 0.38) !important;
    box-shadow: 0 14px 40px rgba(0, 210, 255, 0.12) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 20px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    gap: 12px !important;
}

/* Card Header Bar */
.card-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 36px;
    padding-bottom: 10px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.card-header-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge-tag {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    letter-spacing: 0.04em;
}

/* Prediction Result Badges */
.badge-result-normal {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
    border: 1.5px solid #10b981;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 0 24px rgba(16, 185, 129, 0.2);
    margin-bottom: 12px;
}
.badge-result-pneumonia {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.28) 100%);
    border: 1.5px solid #ef4444;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 0 24px rgba(239, 68, 68, 0.25);
    margin-bottom: 12px;
}
.result-headline {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin: 4px 0 2px 0;
}
.result-score-highlight {
    font-size: 0.98rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* Probabilities Progress Container */
.prob-card {
    background: rgba(10, 20, 37, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 5px;
}
.prob-num {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

/* Threshold Info Box */
.threshold-pill {
    background: rgba(14, 30, 56, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* Buttons */
div.stButton > button[kind="primary"], div.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #00b4d8 100%) !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 18px rgba(0, 180, 216, 0.35) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #0096c7 100%) !important;
    border-color: #7dd3fc !important;
    box-shadow: 0 6px 24px rgba(0, 210, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Download Buttons */
div[data-testid="stDownloadButton"] > button {
    background: rgba(14, 31, 56, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0, 210, 255, 0.16) !important;
    border-color: #00d2ff !important;
    color: #ffffff !important;
    box-shadow: 0 0 18px rgba(0, 210, 255, 0.35) !important;
}

/* Sidebar Branding & HUD Card (Bug-Free Pure CSS Illustration) */
.sidebar-brand-box {
    padding: 10px 6px 14px 6px;
    border-bottom: 1px solid rgba(0, 210, 255, 0.15);
    margin-bottom: 14px;
}
.sidebar-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
    letter-spacing: 0.03em;
    margin-top: 6px;
}
.status-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

/* CSS Medical Scanner HUD Card */
.sidebar-hud-card {
    background: linear-gradient(145deg, #09172c 0%, #050d1a 100%);
    border: 1px solid rgba(0, 210, 255, 0.25);
    border-radius: 14px;
    padding: 14px 12px;
    text-align: center;
    margin: 8px 0 16px 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}
.sidebar-hud-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d2ff, transparent);
    animation: scanAnim 3s infinite linear;
}
@keyframes scanAnim {
    0% { top: 0%; opacity: 0.3; }
    50% { opacity: 1; }
    100% { top: 98%; opacity: 0.3; }
}
.hud-icon-badge {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    margin: 0 auto 8px auto;
    background: radial-gradient(circle, rgba(0, 210, 255, 0.2) 0%, rgba(14, 31, 56, 0.6) 100%);
    border: 1px solid rgba(0, 210, 255, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 16px rgba(0, 210, 255, 0.3);
}
.hud-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hud-tag {
    font-size: 0.68rem;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 3px;
}

/* Sidebar Hardware Spec Card */
.sidebar-spec-card {
    background: rgba(10, 22, 40, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 12px;
}
.spec-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.77rem;
    padding: 3px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.spec-row:last-child {
    border-bottom: none;
}
.spec-key {
    color: #94a3b8;
}
.spec-val {
    color: #e2e8f0;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric Display Overrides */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(13, 27, 48, 0.8) 0%, rgba(8, 17, 32, 0.9) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    padding: 14px 16px !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}

/* File Uploader Container Styling */
div[data-testid="stFileUploader"] {
    background: rgba(10, 22, 40, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Image Containers - Consistent Uniform Height & Centering Inside Cards */
div[data-testid="stImage"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    background: #040912 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    padding: 8px !important;
    overflow: hidden !important;
}
div[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: none !important;
    box-shadow: none !important;
    height: 230px !important;
    max-height: 230px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    display: block !important;
}
div[data-testid="stImage"] [data-testid="stImageCaption"] {
    text-align: center !important;
    color: #94a3b8 !important;
    font-size: 0.77rem !important;
    margin-top: 6px !important;
}

/* Tabs styling inside cards */
div[data-testid="stTabs"] {
    width: 100% !important;
}
div[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 6px 12px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d2ff !important;
    border-bottom-color: #00d2ff !important;
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d2ff, #38bdf8) !important;
    border-radius: 9999px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Session State Initialization ----------------
if "active_module" not in st.session_state:
    st.session_state.active_module = "🫁 Chest X-ray"
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---------------- Sidebar Navigation & Platform Status ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #0284c7, #00d2ff); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 16px rgba(0, 210, 255, 0.4);">
                🫁
            </div>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                    MediScan <span style="color: #00d2ff;">AI</span>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Healthcare AI Suite
                </div>
            </div>
        </div>
        <div class="sidebar-chip">
            <span class="status-dot"></span>
            Modular Research v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Medical Scanner HUD Card
    st.markdown("""
    <div class="sidebar-hud-card">
        <div class="hud-icon-badge">🏥</div>
        <div class="hud-title">Multi-Task Clinical AI Console</div>
        <div class="hud-tag">RESEARCH SUITE // MODULAR</div>
    </div>
    """, unsafe_allow_html=True)

    # Modular Navigation Selector
    # Modality Navigation Selector
    st.markdown("<p style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 6px;'>SELECT IMAGING MODALITY</p>", unsafe_allow_html=True)
    
    selected_module = st.radio(
        "Modality Selection",
        options=[
            "🫁 Chest X-ray",
            "🧠 Brain MRI",
            "📊 Model Evaluation & Metrics",
        ],
        label_visibility="collapsed",
    )

    # GPU VRAM Conservation Trigger: Clear PyTorch cache when switching tasks
    if selected_module != st.session_state.active_module:
        st.session_state.active_module = selected_module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Module Status Summary Card in Sidebar
    if "Chest X-ray" in selected_module:
        status_chip_text = "🟢 ACTIVE MODEL (MobileNetV3)"
        status_chip_color = "#34d399"
    elif "Brain MRI" in selected_module:
        status_chip_text = "🟢 ACTIVE MODEL (MobileNetV3 4-Class)"
        status_chip_color = "#34d399"
    else:
        status_chip_text = "📊 TEST COHORT BENCHMARK"
        status_chip_color = "#38bdf8"

    st.markdown(f"""
    <div style="background: rgba(12, 25, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
        <div style="font-size: 0.76rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em;">
            Current Modality State
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; color: {status_chip_color};">
            {status_chip_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hardware & 4GB VRAM Protection Spec Card
    device_label = "CUDA (NVIDIA GPU)" if torch.cuda.is_available() else "CPU Mode"
    vram_status = "Protected (On-Demand Load)" if torch.cuda.is_available() else "N/A (CPU)"
    
    st.markdown(f"""
    <div class="sidebar-spec-card">
        <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            HARDWARE & VRAM ALLOCATION
        </div>
        <div class="spec-row">
            <span class="spec-key">Inference Device:</span>
            <span class="spec-val" style="color: {'#34d399' if torch.cuda.is_available() else '#94a3b8'};">{device_label}</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">4GB VRAM Status:</span>
            <span class="spec-val" style="color: #38bdf8;">{vram_status}</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Active Modality:</span>
            <span class="spec-val" style="font-size: 0.72rem;">{selected_module.split(' ')[1]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #64748b; font-size: 0.72rem; line-height: 1.4; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 10px;">
        Educational research prototype.<br>Not a medical diagnostic tool.
    </div>
    """, unsafe_allow_html=True)


# ---------------- Dynamic Header Banner & Workflow Pills ----------------
if "Chest X-ray" in selected_module:
    banner_title = "🫁 <span class='glow'>MediScan AI — Chest X-ray Analysis</span>"
    banner_desc = "Deep learning radiography evaluation for pneumonia screening with explainable Grad-CAM feature localization."
    w1, w2, w3 = "1. Upload Chest X-ray", "2. MobileNetV3 Inference", "3. Grad-CAM Visualization"
elif "Brain MRI" in selected_module:
    banner_title = "🧠 <span class='glow'>MediScan AI — Brain MRI Analysis</span>"
    banner_desc = "Deep learning neuro-oncology MRI classification across 4 categories with Grad-CAM feature localization."
    w1, w2, w3 = "1. Upload MRI Slice", "2. MobileNetV3 Inference", "3. 4-Class Confidence & CAM"
else:
    banner_title = "📊 <span class='glow'>MediScan AI — Model Evaluation & Metrics</span>"
    banner_desc = "Quantitative performance metrics, confusion matrices, and error reports on reserved test cohorts."
    w1, w2, w3 = "1. Test Cohorts", "2. Confusion Matrices", "3. Error Analysis"

st.markdown(f"""
<div class="main-header-banner">
    <div class="brand-title">
        {banner_title}
    </div>
    <div class="brand-subtitle">
        {banner_desc}
    </div>
    <div class="workflow-pills">
        <div class="wf-step active">
            <span class="wf-num">1</span> {w1}
        </div>
        <span class="wf-arrow">➔</span>
        <div class="wf-step">
            <span class="wf-num">2</span> {w2}
        </div>
        <span class="wf-arrow">➔</span>
        <div class="wf-step">
            <span class="wf-num">3</span> {w3}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Universal Safety Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <span style="font-size: 1.1rem;">⚠️</span>
    <div>
        <strong>Safety Notice:</strong> Educational research prototype only. Model outputs represent uncalibrated statistical confidence scores. This tool does not provide medical diagnosis, clinical decisions, or treatment recommendations.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- Module Dispatcher ----------------
if "Chest X-ray" in selected_module:
    render_chest_xray_module()

elif "Brain MRI" in selected_module:
    render_brain_mri_module()

elif "Evaluation" in selected_module:
    render_chest_xray_evaluation()


# ---------------- Footer ----------------
st.markdown("""
<div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 18px; margin-top: 30px; text-align: center; color: #64748b; font-size: 0.8rem;">
    <strong>MediScan AI</strong> • Modular Healthcare Intelligence Research Platform • Deep Learning Medical Vision<br>
    Educational research prototype only. Model outputs represent statistical associations and require physician verification.
</div>
""", unsafe_allow_html=True)
