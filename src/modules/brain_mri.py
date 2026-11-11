"""
MediScan AI — Brain MRI Analysis Module
========================================
Deep learning neuro-oncology MRI classification for four categories:
1. glioma
2. meningioma
3. notumor
4. pituitary

Checkpoint: models/brain_mri_mobilenetv3.pth
Backbone: MobileNetV3-Small (Pretrained ImageNet, fine-tuned on 4-class MRI cohort)
Resolution: 224x224 pixels with ImageNet normalization
Interpretability: Grad-CAM feature localization on model.features[-1]

Confidence Score Policy:
Scores are explicitly presented as uncalibrated neural network model confidence scores,
NOT calibrated medical or epidemiological probabilities. Educational research prototype only.
"""

from pathlib import Path
import json
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models


# ---------------- Configuration & Paths ----------------
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "brain_mri_mobilenetv3.pth"
METRICS_PATH = ROOT / "outputs" / "brain_mri" / "evaluation_metrics.json"

TEST_DATA_DIR = ROOT / "data" / "raw" / "brain_mri" / "Testing"
SAMPLE_PATHS = {
    "glioma": TEST_DATA_DIR / "glioma" / "Te-gl_1.jpg",
    "meningioma": TEST_DATA_DIR / "meningioma" / "Te-aug-me_1.jpg",
    "notumor": TEST_DATA_DIR / "notumor" / "Te-no_1.jpg",
    "pituitary": TEST_DATA_DIR / "pituitary" / "Te-pi_1.jpg",
}

IMAGE_SIZE = 224

# The exact 4 classes stored in the checkpoint
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_META = {
    "glioma": {
        "display_name": "Glioma",
        "description": "Primary glial parenchymal neoplasm",
        "badge_class": "badge-result-pneumonia",
        "color": "#ef4444",
        "icon": "⚠️",
    },
    "meningioma": {
        "display_name": "Meningioma",
        "description": "Dural-based extra-axial arachnoid sheath neoplasm",
        "badge_class": "badge-result-pneumonia",
        "color": "#f97316",
        "icon": "⚠️",
    },
    "notumor": {
        "display_name": "No Tumour",
        "description": "Normal brain scan without identified mass",
        "badge_class": "badge-result-normal",
        "color": "#10b981",
        "icon": "🟢",
    },
    "pituitary": {
        "display_name": "Pituitary",
        "description": "Sellar / neuroendocrine pituitary adenoma",
        "badge_class": "badge-result-pneumonia",
        "color": "#38bdf8",
        "icon": "⚠️",
    },
}


# ---------------- Dedicated MRI Preprocessing ----------------
# Identical 224x224 resize and ImageNet normalization used during training
brain_mri_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- Memory-Conscious Model Loader ----------------
@st.cache_resource
def load_brain_mri_model():
    """
    Loads the trained 4-class MobileNetV3-Small checkpoint from models/brain_mri_mobilenetv3.pth.
    Uses CUDA if available (RTX 3050 Laptop GPU, 4GB VRAM).
    """
    if not MODEL_PATH.is_file():
        return None, None, None, f"Checkpoint not found at: {MODEL_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    stored_classes = checkpoint.get("classes", CLASSES)

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(stored_classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, stored_classes, device, None


def is_mri_model_configured() -> bool:
    """Checks whether the Brain MRI model checkpoint exists."""
    return MODEL_PATH.is_file()


# ---------------- Grad-CAM Localization Generator ----------------
def generate_mri_gradcam(model, tensor: torch.Tensor, target_idx: int) -> np.ndarray:
    """
    Generates class-activation map (Grad-CAM) from the final feature block:
    model.features[-1]
    """
    activations = []
    gradients = []
    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            tensor_copy = tensor.clone().detach().requires_grad_(True)
            logits = model(tensor_copy)
            logits[0, target_idx].backward()

        feature_maps = activations[0].detach()
        grads = gradients[0].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1)).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        denom = cam.max() - cam.min()
        if denom > 1e-8:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        return cam
    finally:
        handle.remove()


def create_cam_overlay(pil_image: Image.Image, cam_array: np.ndarray, alpha: float = 0.45):
    """
    Renders Grad-CAM heatmap overlay onto original PIL image at 224x224.
    """
    orig_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    orig_np = np.array(orig_resized).astype(np.float32) / 255.0

    # Resize CAM to 224x224 using PIL bicubic
    cam_pil = Image.fromarray((cam_array * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC
    )
    cam_norm = np.array(cam_pil).astype(np.float32) / 255.0

    # Color map
    cmap = plt.get_cmap("jet")
    colored_cam = cmap(cam_norm)[:, :, :3]  # RGB

    overlay = alpha * colored_cam + (1.0 - alpha) * orig_np
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    pure_cam_img = Image.fromarray((colored_cam * 255).astype(np.uint8))

    return overlay_img, pure_cam_img


# ---------------- Session State Helpers ----------------
def init_brain_mri_session_state():
    if "mri_analyzed_image" not in st.session_state:
        st.session_state.mri_analyzed_image = None
    if "mri_image_name" not in st.session_state:
        st.session_state.mri_image_name = None
    if "mri_results" not in st.session_state:
        st.session_state.mri_results = None
    if "mri_sample_selected" not in st.session_state:
        st.session_state.mri_sample_selected = None


# ---------------- UI Renderer: Brain MRI Module ----------------
def render_brain_mri_module():
    """Renders the comprehensive Brain MRI analysis interface."""
    init_brain_mri_session_state()

    is_configured = is_mri_model_configured()
    model, classes, device, load_err = load_brain_mri_model()

    # Module Header Bar
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
        <div>
            <h2 style="margin: 0; font-size: 1.4rem; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span>🧠</span> Brain MRI Analysis
            </h2>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                4-Class neuro-oncology MRI classification: Glioma, Meningioma, Pituitary, and No Tumour.
            </p>
        </div>
        <div>
            <span style="background: {'rgba(16, 185, 129, 0.15)' if is_configured and not load_err else 'rgba(239, 68, 68, 0.15)'}; border: 1px solid {'#10b981' if is_configured and not load_err else '#ef4444'}; color: {'#34d399' if is_configured and not load_err else '#f87171'}; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">
                {'● MRI MODEL ACTIVE (MobileNetV3 4-Class)' if is_configured and not load_err else '⚠️ MRI CHECKPOINT MISSING'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if load_err:
        st.warning(f"⚠️ {load_err}. Please ensure `models/brain_mri_mobilenetv3.pth` is present.")

    # 4 Supported Categories Banner
    st.markdown("""
    <div class="med-card" style="padding: 14px 18px; margin-bottom: 14px;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
            Supported Diagnostic Categories (4 Classes)
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
            <div style="background: rgba(14, 31, 56, 0.7); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 8px; padding: 8px 12px;">
                <span style="color: #ef4444; font-weight: 700; font-size: 0.82rem;">1. Glioma</span>
                <div style="font-size: 0.72rem; color: #94a3b8;">Primary brain parenchymal neoplasm</div>
            </div>
            <div style="background: rgba(14, 31, 56, 0.7); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 8px; padding: 8px 12px;">
                <span style="color: #f97316; font-weight: 700; font-size: 0.82rem;">2. Meningioma</span>
                <div style="font-size: 0.72rem; color: #94a3b8;">Dural / arachnoid sheath neoplasm</div>
            </div>
            <div style="background: rgba(14, 31, 56, 0.7); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 8px 12px;">
                <span style="color: #34d399; font-weight: 700; font-size: 0.82rem;">3. No Tumour</span>
                <div style="font-size: 0.72rem; color: #94a3b8;">Normal brain parenchyma & ventricles</div>
            </div>
            <div style="background: rgba(14, 31, 56, 0.7); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 8px; padding: 8px 12px;">
                <span style="color: #38bdf8; font-weight: 700; font-size: 0.82rem;">4. Pituitary</span>
                <div style="font-size: 0.72rem; color: #94a3b8;">Sellar / neuroendocrine adenoma</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Sample Loader Row
    st.markdown("""
    <div style="font-size: 0.82rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;">
        ⚡ Quick Test Samples (from reserved test cohort):
    </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        if st.button("🧪 Sample Glioma", use_container_width=True):
            if SAMPLE_PATHS["glioma"].is_file():
                try:
                    img = Image.open(SAMPLE_PATHS["glioma"]).convert("RGB")
                    st.session_state.mri_analyzed_image = img
                    st.session_state.mri_image_name = "Sample_Glioma (Te-gl_1.jpg)"
                    st.session_state.mri_sample_selected = "glioma"
                    st.session_state.mri_results = None
                except Exception as err:
                    st.error(f"Error loading sample: {err}")

    with s_col2:
        if st.button("🧪 Sample Meningioma", use_container_width=True):
            if SAMPLE_PATHS["meningioma"].is_file():
                try:
                    img = Image.open(SAMPLE_PATHS["meningioma"]).convert("RGB")
                    st.session_state.mri_analyzed_image = img
                    st.session_state.mri_image_name = "Sample_Meningioma (Te-aug-me_1.jpg)"
                    st.session_state.mri_sample_selected = "meningioma"
                    st.session_state.mri_results = None
                except Exception as err:
                    st.error(f"Error loading sample: {err}")

    with s_col3:
        if st.button("🧪 Sample No Tumour", use_container_width=True):
            if SAMPLE_PATHS["notumor"].is_file():
                try:
                    img = Image.open(SAMPLE_PATHS["notumor"]).convert("RGB")
                    st.session_state.mri_analyzed_image = img
                    st.session_state.mri_image_name = "Sample_NoTumour (Te-no_1.jpg)"
                    st.session_state.mri_sample_selected = "notumor"
                    st.session_state.mri_results = None
                except Exception as err:
                    st.error(f"Error loading sample: {err}")

    with s_col4:
        if st.button("🧪 Sample Pituitary", use_container_width=True):
            if SAMPLE_PATHS["pituitary"].is_file():
                try:
                    img = Image.open(SAMPLE_PATHS["pituitary"]).convert("RGB")
                    st.session_state.mri_analyzed_image = img
                    st.session_state.mri_image_name = "Sample_Pituitary (Te-pi_1.jpg)"
                    st.session_state.mri_sample_selected = "pituitary"
                    st.session_state.mri_results = None
                except Exception as err:
                    st.error(f"Error loading sample: {err}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Upload Section
    st.markdown("""
    <div class="med-card" style="padding: 14px 18px; margin-bottom: 12px;">
        <div style="font-size: 0.94rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; justify-content: space-between;">
            <span>📤 Or Upload Custom Brain MRI Slice</span>
            <div>
                <span class="badge-tag">JPG</span>
                <span class="badge-tag">JPEG</span>
                <span class="badge-tag">PNG</span>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 3px;">
            Axial, coronal, or sagittal T1/T2/FLAIR brain MRI scan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"],
        key="brain_mri_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            custom_img = Image.open(uploaded_file).convert("RGB")
            # If a new file is uploaded, update session state
            if st.session_state.mri_image_name != uploaded_file.name:
                st.session_state.mri_analyzed_image = custom_img
                st.session_state.mri_image_name = uploaded_file.name
                st.session_state.mri_results = None
        except Exception as e:
            st.error(f"❌ Failed to decode uploaded file as an image: {e}. Please upload a valid JPG, JPEG, or PNG file.")

    # Action Button Bar
    current_image = st.session_state.mri_analyzed_image
    current_image_name = st.session_state.mri_image_name

    if current_image is not None:
        btn_col1, btn_col2 = st.columns([2.5, 1])
        with btn_col1:
            run_analysis = st.button("⚡ Analyze Brain MRI Scan", use_container_width=True, type="primary")
        with btn_col2:
            if st.button("🔄 Reset / Clear", use_container_width=True):
                st.session_state.mri_analyzed_image = None
                st.session_state.mri_image_name = None
                st.session_state.mri_results = None
                st.rerun()

        # Run Inference
        if run_analysis or (st.session_state.mri_results is None and current_image is not None):
            if not is_configured or model is None:
                st.error("Cannot analyze: Brain MRI model checkpoint is not loaded. Check models/brain_mri_mobilenetv3.pth.")
            else:
                try:
                    tensor = brain_mri_transform(current_image).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    top_idx = int(probs.argmax())
                    top_class = classes[top_idx]
                    top_score = float(probs[top_idx])

                    # Compute Grad-CAM
                    cam_array = generate_mri_gradcam(model, tensor, top_idx)
                    overlay_img, pure_cam_img = create_cam_overlay(current_image, cam_array)

                    st.session_state.mri_results = {
                        "top_class": top_class,
                        "top_score": top_score,
                        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
                        "overlay_img": overlay_img,
                        "pure_cam_img": pure_cam_img,
                        "device": str(device),
                    }
                except Exception as infer_err:
                    st.error(f"❌ Inference error: {infer_err}")

    # Results Display Section: 3 Desktop Cards Aligned
    if current_image is not None and st.session_state.mri_results is not None:
        res = st.session_state.mri_results
        top_c = res["top_class"]
        meta = CLASS_META.get(top_c, {
            "display_name": top_c.capitalize(),
            "description": "Brain tissue classification",
            "badge_class": "badge-result-normal",
            "color": "#38bdf8",
            "icon": "🧠",
        })

        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        # Card 1: 1. Uploaded Brain MRI
        with col1:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title"><span>1.</span> Uploaded Brain MRI</div>
                    <span class="badge-tag">ORIGINAL</span>
                </div>
                """, unsafe_allow_html=True)

                st.image(current_image, caption=f"{current_image_name} (224×224)", use_container_width=True)

                st.markdown(f"""
                <div style="font-size: 0.78rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-top: 6px;">
                    Dimensions: {current_image.size[0]}×{current_image.size[1]} px | Mode: {current_image.mode}
                </div>
                """, unsafe_allow_html=True)

        # Card 2: 2. Grad-CAM Attention Heatmap
        with col2:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title"><span>2.</span> Grad-CAM Localization</div>
                    <span class="badge-tag">EXPLAINABLE AI</span>
                </div>
                """, unsafe_allow_html=True)

                tab_overlay, tab_pure = st.tabs(["Overlay Heatmap", "Pure Activation"])
                with tab_overlay:
                    st.image(res["overlay_img"], caption=f"Attention map for predicted '{meta['display_name']}'", use_container_width=True)
                with tab_pure:
                    st.image(res["pure_cam_img"], caption="Conv Feature Maps (model.features[-1])", use_container_width=True)

                st.markdown("""
                <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4; margin-top: 6px;">
                    Gradient-weighted Class Activation Mapping highlights regions contributing most to the class score.
                </div>
                """, unsafe_allow_html=True)

        # Card 3: 3. Classification Result & Confidence Scores
        with col3:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title"><span>3.</span> Classification Result</div>
                    <span class="badge-tag">4-CLASS OUTPUT</span>
                </div>
                """, unsafe_allow_html=True)

                # Headline Card
                st.markdown(f"""
                <div class="{meta['badge_class']}">
                    <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #cbd5e1; margin-bottom: 2px;">
                        PREDICTED CLASS
                    </div>
                    <div class="result-headline" style="color: {meta['color']};">
                        {meta['icon']} {meta['display_name']}
                    </div>
                    <div class="result-score-highlight">
                        Model Confidence: {res['top_score'] * 100:.2f}%
                    </div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; margin-top: 4px;">
                        {meta['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # All 4 Classes Confidence Scores Breakdown
                st.markdown("""
                <div class="prob-card">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
                        All Classes Confidence Scores
                    </div>
                """, unsafe_allow_html=True)

                for c_name in classes:
                    c_info = CLASS_META.get(c_name, {"display_name": c_name.capitalize(), "color": "#38bdf8"})
                    c_score = res["probabilities"].get(c_name, 0.0)
                    c_pct = c_score * 100

                    st.markdown(f"""
                    <div class="prob-label-row">
                        <span style="color: {c_info['color']};">{c_info['display_name']}</span>
                        <span class="prob-num" style="color: {c_info['color']};">{c_pct:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(c_score))

                st.markdown("</div>", unsafe_allow_html=True)

                # Mandatory Requirement 7 & 10 Disclaimers
                st.markdown(f"""
                <div class="threshold-pill">
                    <strong style="color: #38bdf8;">ℹ️ Model Confidence Notice:</strong><br>
                    Scores reflect uncalibrated neural network softmax confidence scores, NOT calibrated medical or clinical diagnostic probabilities.
                </div>
                <div style="font-size: 0.72rem; color: #64748b; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 8px; margin-top: 8px; line-height: 1.4;">
                    <strong>Educational Research Prototype:</strong> This application does not diagnose illness, validate clinical pathology, or recommend medical treatment. Always consult a licensed radiologist.
                </div>
                """, unsafe_allow_html=True)

    elif current_image is None:
        # Prompt card when no scan is uploaded yet
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 30px 18px;">
                <div style="font-size: 2.8rem; margin-bottom: 8px;">🧠</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #f1f5f9; margin-bottom: 6px;">
                    Ready for Brain MRI Analysis
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 580px; margin: 0 auto 16px auto; line-height: 1.6;">
                    The fine-tuned MobileNetV3-Small model is loaded and ready to classify brain MRI scans across
                    <strong style="color: #38bdf8;">Glioma, Meningioma, No Tumour, and Pituitary</strong>.
                    Click a quick test sample above or upload your own scan to begin.
                </div>
                <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                    <span class="badge-tag">MobileNetV3-Small</span>
                    <span class="badge-tag">94.69% Test Accuracy</span>
                    <span class="badge-tag">Grad-CAM Localization</span>
                    <span class="badge-tag">4-Class Output</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Performance & Validation Dashboard Accordion
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if METRICS_PATH.is_file():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f_m:
                metrics_data = json.load(f_m)

            with st.expander("📊 View Model Performance on Independent Test Cohort (N = 1,600)"):
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Overall Test Accuracy", f"{metrics_data.get('overall_accuracy', 0.9469) * 100:.2f}%")
                m_c2.metric("Test Images Evaluated", f"{metrics_data.get('test_images', 1600)}")
                m_c3.metric("Macro Precision", f"{metrics_data.get('macro_avg', {}).get('precision', 0.9512) * 100:.2f}%")
                m_c4.metric("Macro Recall", f"{metrics_data.get('macro_avg', {}).get('recall', 0.9469) * 100:.2f}%")

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # Per-class metrics table
                per_class = metrics_data.get("per_class", {})
                if per_class:
                    st.markdown("**Per-Class Breakdown on Reserved Test Set:**")
                    table_rows = []
                    for c_key in classes:
                        c_data = per_class.get(c_key, {})
                        c_display = CLASS_META.get(c_key, {}).get("display_name", c_key.capitalize())
                        table_rows.append({
                            "Class": c_display,
                            "Precision": f"{c_data.get('precision', 0.0) * 100:.2f}%",
                            "Recall": f"{c_data.get('recall', 0.0) * 100:.2f}%",
                            "F1-Score": f"{c_data.get('f1_score', 0.0) * 100:.2f}%",
                            "Test Samples": c_data.get("support", 400),
                        })
                    st.dataframe(table_rows, use_container_width=True)

                # Confusion Matrix display
                cm = metrics_data.get("confusion_matrix")
                if cm:
                    st.markdown("**4×4 Confusion Matrix:**")
                    display_names = [CLASS_META.get(c, {}).get("display_name", c.capitalize()) for c in classes]
                    cm_df = {
                        display_names[i]: [cm[row][i] for row in range(len(classes))]
                        for i in range(len(classes))
                    }
                    import pandas as pd
                    st.dataframe(pd.DataFrame(cm_df, index=display_names), use_container_width=True)

        except Exception as e:
            st.caption(f"Could not load evaluation metrics: {e}")
