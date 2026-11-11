"""
MediScan AI — Chest X-ray Analysis Module
==========================================
Deep learning radiography analysis for pneumonia screening and Grad-CAM interpretability.
Architecture: MobileNetV3-Small (Fine-Tuned)
Decision Threshold: 0.70
"""

from pathlib import Path
import io
import json

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms


# ---------------- Configuration & Paths ----------------
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "mediscan_mobilenetv3_finetuned.pth"
METRICS_PATH = ROOT / "outputs" / "evaluation_metrics.json"
ERROR_REPORT_PATH = ROOT / "outputs" / "error_report.csv"
SAMPLE_NORMAL_PATH = ROOT / "data" / "raw" / "chest_xray" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
SAMPLE_PNEUMONIA_PATH = ROOT / "data" / "raw" / "chest_xray" / "chest_xray" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"

IMAGE_SIZE = 224
PNEUMONIA_THRESHOLD = 0.70
MODULE_STATUS = "IMPLEMENTED_ACTIVE"


# ---------------- Preprocessing Pipeline ----------------
chest_xray_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- Model Loading (Cached & Memory-Protected) ----------------
@st.cache_resource
def load_chest_xray_model():
    """Loads MobileNetV3-Small fine-tuned weights onto GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


# ---------------- Grad-CAM Explainable AI ----------------
def generate_gradcam(model, image_tensor, target_idx):
    """Computes Grad-CAM activations on target layer model.features[-1]."""
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
            logits = model(image_tensor)
            logits[0, target_idx].backward()

        feature_maps = activations[0].detach()
        grads = gradients[0].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        minimum, maximum = cam.min(), cam.max()
        if maximum - minimum > 1e-8:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:
        handle.remove()


def make_visuals(image: Image.Image, cam: np.ndarray):
    """Produces 224x224 RGB image, Jet heatmap, and 40% blend overlay."""
    original = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")

    heat_array = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heat_array)
    overlay = Image.blend(original, heatmap, alpha=0.40)

    return original, heatmap, overlay


# ---------------- UI Renderer: Chest X-ray Module ----------------
def render_chest_xray_module():
    """Renders the main Chest X-ray analysis interface with upload and Grad-CAM breakdown."""

    # Module Header Bar with Status
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
        <div>
            <h2 style="margin: 0; font-size: 1.4rem; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span>🫁</span> Chest X-ray Analysis
            </h2>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                MobileNetV3-Small classifier for pneumonia detection with Grad-CAM localization.
            </p>
        </div>
        <div>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">
                ● MODEL IMPLEMENTED & ACTIVE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Upload & Demo Samples
    upload_c1, sample_c2 = st.columns([2.5, 1.2])

    with upload_c1:
        st.markdown("""
        <div class="med-card" style="padding: 16px 20px; margin-bottom: 12px;">
            <div style="font-size: 0.98rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; justify-content: space-between;">
                <span>📤 Upload Chest Radiograph (PA/AP Frontal View)</span>
                <div>
                    <span class="badge-tag">JPG</span>
                    <span class="badge-tag">JPEG</span>
                    <span class="badge-tag">PNG</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                Standard frontal chest projection recommended for pneumonia screening.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Chest X-ray",
            type=["jpg", "jpeg", "png"],
            key=f"chest_xray_uploader_{st.session_state.get('uploader_key', 0)}",
            help="Supported formats: JPG, JPEG, PNG. Image resized to 224x224 RGB.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                loaded_img = Image.open(uploaded_file).convert("RGB")
                if st.session_state.get("image_filename") != uploaded_file.name:
                    st.session_state["current_image"] = loaded_img
                    st.session_state["image_filename"] = uploaded_file.name
                    st.session_state["prediction_result"] = None
            except Exception as err:
                st.error(f"Failed to read image file: {err}")

    with sample_c2:
        st.markdown("""
        <div class="med-card" style="padding: 16px 18px; margin-bottom: 12px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">
                ⚡ Quick Demo Samples
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px;">
                Load verified test set radiographs:
            </div>
        </div>
        """, unsafe_allow_html=True)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            if st.button("Normal Sample", use_container_width=True, help="Load verified Normal test X-ray"):
                if SAMPLE_NORMAL_PATH.is_file():
                    st.session_state["current_image"] = Image.open(SAMPLE_NORMAL_PATH).convert("RGB")
                    st.session_state["image_filename"] = "IM-0001-0001.jpeg (Normal Test Sample)"
                    st.session_state["prediction_result"] = None
                    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                    st.rerun()
                else:
                    st.warning("Normal sample file not found in test directory.")
        with s_col2:
            if st.button("Pneumonia Sample", use_container_width=True, help="Load verified Pneumonia test X-ray"):
                if SAMPLE_PNEUMONIA_PATH.is_file():
                    st.session_state["current_image"] = Image.open(SAMPLE_PNEUMONIA_PATH).convert("RGB")
                    st.session_state["image_filename"] = "person100_bacteria_475.jpeg (Pneumonia Test Sample)"
                    st.session_state["prediction_result"] = None
                    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                    st.rerun()
                else:
                    st.warning("Pneumonia sample file not found in test directory.")

    # State 1: Image Loaded, Waiting for Analyze Click
    if st.session_state.get("current_image") is not None and st.session_state.get("prediction_result") is None:
        curr_img = st.session_state["current_image"]
        curr_name = st.session_state.get("image_filename", "Loaded Chest X-ray")

        pre_c1, pre_c2 = st.columns([1, 1.4])
        with pre_c1:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title"><span>🩻</span> Uploaded X-ray Preview</div>
                    <span class="badge-tag">READY</span>
                </div>
                """, unsafe_allow_html=True)
                st.image(curr_img, caption=curr_name, width="stretch")
                st.markdown(f"""
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                    Dimensions: {curr_img.size[0]}×{curr_img.size[1]} px | Color: {curr_img.mode}
                </div>
                """, unsafe_allow_html=True)

        with pre_c2:
            with st.container(border=True):
                st.markdown("""
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;">
                    ⚡ Execute Deep Radiography Inference
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 22px;">
                    The radiograph is loaded in memory. Click the button below to run the fine-tuned MobileNetV3-Small neural network and compute Grad-CAM activation gradients on the final convolutional layer.
                </div>
                """, unsafe_allow_html=True)

                run_analysis = st.button("🔍 Analyze X-ray", type="primary", use_container_width=True)

                st.markdown("""
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 14px; text-align: center;">
                    Inference executes on GPU/CPU without modifying model weights.
                </div>
                """, unsafe_allow_html=True)

                if run_analysis:
                    if not MODEL_PATH.is_file():
                        st.error(f"Model file missing: {MODEL_PATH}")
                        st.stop()

                    with st.spinner("Analyzing radiograph with MobileNetV3 and extracting Grad-CAM activations..."):
                        try:
                            model, classes, device = load_chest_xray_model()

                            if "NORMAL" not in classes or "PNEUMONIA" not in classes:
                                st.error("Model checkpoint has unexpected class names.")
                                st.stop()

                            img_tensor = chest_xray_transform(curr_img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                outputs = model(img_tensor)
                                probabilities = torch.softmax(outputs, dim=1)[0]

                            pneumonia_idx = classes.index("PNEUMONIA")
                            normal_idx = classes.index("NORMAL")

                            pneumonia_score = float(probabilities[pneumonia_idx].item())
                            normal_score = float(probabilities[normal_idx].item())

                            pred_class = "PNEUMONIA" if pneumonia_score >= PNEUMONIA_THRESHOLD else "NORMAL"
                            target_idx = classes.index(pred_class)
                            model_score = float(probabilities[target_idx].item())

                            # Grad-CAM computation
                            cam = generate_gradcam(model, img_tensor, target_idx)
                            orig_vis, heat_vis, ovl_vis = make_visuals(curr_img, cam)

                            # Save to session state so state is preserved across reruns
                            st.session_state["prediction_result"] = {
                                "prediction": pred_class,
                                "model_score": model_score,
                                "normal_score": normal_score,
                                "pneumonia_score": pneumonia_score,
                                "target_idx": target_idx,
                                "cam": cam,
                                "original": orig_vis,
                                "heatmap": heat_vis,
                                "overlay": ovl_vis,
                                "device": str(device),
                            }
                            st.rerun()

                        except Exception as exc:
                            st.error(f"Inference error: {exc}")

    # State 2: No Image Loaded Yet
    elif st.session_state.get("current_image") is None:
        st.markdown("""
        <div style="background: linear-gradient(145deg, rgba(13, 27, 48, 0.6) 0%, rgba(8, 17, 32, 0.8) 100%); border: 2px dashed rgba(56, 189, 248, 0.3); border-radius: 18px; padding: 36px 24px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.8rem; margin-bottom: 8px;">🩻</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">Ready to Analyze Chest X-ray</div>
            <div style="font-size: 0.88rem; color: #94a3b8; max-width: 540px; margin: 0 auto 14px auto;">
                Please upload a radiograph above or choose a quick demo test sample. The “Analyze X-ray” button and Grad-CAM results will appear immediately.
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                <span class="badge-tag">Frontal PA/AP View</span>
                <span class="badge-tag">RGB 224×224 Input</span>
                <span class="badge-tag">MobileNetV3 Backbone</span>
                <span class="badge-tag">Decision Threshold: 0.70</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # State 3: Prediction Completed -> Three Visually Balanced Aligned Cards
    if st.session_state.get("prediction_result") is not None:
        res = st.session_state["prediction_result"]
        img_name = st.session_state.get("image_filename", "Chest Radiograph")

        # Action bar to re-run or test another
        bar_c1, bar_c2 = st.columns([3, 1])
        with bar_c1:
            st.markdown("""
            <div style="font-size: 1.18rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px; margin: 6px 0 14px 0;">
                <span>📊</span> Inference Results & Explainable AI Breakdown
            </div>
            """, unsafe_allow_html=True)
        with bar_c2:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.session_state["current_image"] = None
                st.session_state["image_filename"] = None
                st.session_state["prediction_result"] = None
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.rerun()

        # Three aligned cards in one row on desktop using native container wrapping
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])

        # ----------------- Card 1: Uploaded X-ray -----------------
        with col_c1:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title">
                        <span>1.</span> Uploaded X-ray
                    </div>
                    <span class="badge-tag">INPUT TENSOR</span>
                </div>
                """, unsafe_allow_html=True)

                st.image(res["original"], caption="Input Tensor (224×224 px)", width="stretch")

                st.markdown(f"""
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 10px; line-height: 1.5;">
                    <strong style="color: #e2e8f0;">Source:</strong> <span style="font-family: 'JetBrains Mono', monospace;">{img_name}</span><br>
                    <strong style="color: #e2e8f0;">Preprocessing:</strong> ImageNet Mean & Std<br>
                    <strong style="color: #e2e8f0;">Format:</strong> RGB 3-Channel Normalized
                </div>
                <div style="font-size: 0.74rem; color: #64748b; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 8px; margin-top: 10px;">
                    Frontal chest radiograph projection
                </div>
                """, unsafe_allow_html=True)

        # ----------------- Card 2: Grad-CAM Heatmap -----------------
        with col_c2:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title">
                        <span>2.</span> Grad-CAM Heatmap
                    </div>
                    <span class="badge-tag">EXPLAINABLE AI</span>
                </div>
                """, unsafe_allow_html=True)

                tab_ovl, tab_heat = st.tabs(["Heatmap Overlay", "Pure Heatmap"])
                with tab_ovl:
                    st.image(res["overlay"], caption="40% Jet Colormap Blend", width="stretch")
                with tab_heat:
                    st.image(res["heatmap"], caption="Pure Activation Heatmap", width="stretch")

                buffer = io.BytesIO()
                res["overlay"].save(buffer, format="PNG")
                st.download_button(
                    "⬇️ Download Heatmap Overlay",
                    data=buffer.getvalue(),
                    file_name=f"mediscan_gradcam_{res['prediction'].lower()}.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.markdown("""
                <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.4; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 8px; margin-top: 10px;">
                    <strong style="color: #38bdf8;">Notice:</strong> Grad-CAM highlights spatial features influencing the score. It is an experimental visualization and does not prove medical correctness.
                </div>
                """, unsafe_allow_html=True)

        # ----------------- Card 3: Prediction Result -----------------
        with col_c3:
            with st.container(border=True):
                st.markdown("""
                <div class="card-header-bar">
                    <div class="card-header-title">
                        <span>3.</span> Prediction Result
                    </div>
                    <span class="badge-tag">MODEL OUTPUT</span>
                </div>
                """, unsafe_allow_html=True)

                is_pneumonia = (res["prediction"] == "PNEUMONIA")
                if is_pneumonia:
                    badge_style = "badge-result-pneumonia"
                    status_icon = "⚠️"
                    color_code = "#f87171"
                else:
                    badge_style = "badge-result-normal"
                    status_icon = "✅"
                    color_code = "#34d399"

                st.markdown(f"""
                <div class="{badge_style}">
                    <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #cbd5e1; margin-bottom: 2px;">
                        MODEL PREDICTION
                    </div>
                    <div class="result-headline" style="color: {color_code};">
                        {status_icon} {res['prediction']}
                    </div>
                    <div class="result-score-highlight">
                        Model Score: {res['model_score'] * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities Breakdown
                st.markdown("""
                <div class="prob-card">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
                        Class Probabilities
                    </div>
                """, unsafe_allow_html=True)

                norm_pct = res["normal_score"] * 100
                st.markdown(f"""
                <div class="prob-label-row">
                    <span style="color: #34d399;">NORMAL</span>
                    <span class="prob-num" style="color: #34d399;">{norm_pct:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(res["normal_score"])

                pneu_pct = res["pneumonia_score"] * 100
                st.markdown(f"""
                <div class="prob-label-row" style="margin-top: 8px;">
                    <span style="color: #f87171;">PNEUMONIA</span>
                    <span class="prob-num" style="color: #f87171;">{pneu_pct:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(res["pneumonia_score"])

                st.markdown("</div>", unsafe_allow_html=True)

                # Configured Threshold Box
                st.markdown(f"""
                <div class="threshold-pill">
                    <strong style="color: #38bdf8;">Configured Pneumonia Threshold:</strong> {PNEUMONIA_THRESHOLD:.2f}<br>
                    Requires Pneumonia probability &ge; 70.0% to classify as positive.
                </div>
                <div style="font-size: 0.73rem; color: #64748b; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 8px; margin-top: 10px; display: flex; justify-content: space-between;">
                    <span>Inference Device: <code>{res['device']}</code></span>
                    <span>MobileNetV3</span>
                </div>
                """, unsafe_allow_html=True)


# ---------------- UI Renderer: Evaluation Dashboard ----------------
def render_chest_xray_evaluation():
    """Renders the comprehensive performance evaluation dashboard and error reports."""
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
        <div>
            <h2 style="margin: 0; font-size: 1.45rem; color: #ffffff;">📊 Model Evaluation & Performance Metrics</h2>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #94a3b8;">
                Validation results on 624 independent chest X-ray test images with decision threshold 0.70.
            </p>
        </div>
        <div>
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">
                TEST COHORT N=624
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics_data = None
    if METRICS_PATH.is_file():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation metrics JSON: {e}")

    # Top KPI Metrics Row
    if metrics_data:
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

        with m_col1:
            st.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0.8942) * 100:.2f}%")
        with m_col2:
            st.metric("Pneumonia Recall", f"{metrics_data.get('recall', [0, 0.9795])[1] * 100:.2f}%", help="Sensitivity / Detection Rate")
        with m_col3:
            st.metric("Normal Precision", f"{metrics_data.get('precision', [0.9565, 0])[0] * 100:.2f}%")
        with m_col4:
            st.metric("Pneumonia F1-Score", f"{metrics_data.get('f1_score', [0, 0.9205])[1] * 100:.2f}%")
        with m_col5:
            st.metric("Test Set Size", f"{metrics_data.get('test_images', 624)} imgs")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Performance Breakdown & Confusion Matrix
    eval_c1, eval_c2 = st.columns([1.1, 1.4])

    with eval_c1:
        st.markdown("""
        <div class="med-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 14px;">
                🎯 Class-wise Performance Metrics
            </div>
        """, unsafe_allow_html=True)

        if metrics_data:
            perf_df = pd.DataFrame({
                "Class": metrics_data.get("classes", ["NORMAL", "PNEUMONIA"]),
                "Precision": [f"{p * 100:.2f}%" for p in metrics_data.get("precision", [0, 0])],
                "Recall (Sensitivity)": [f"{r * 100:.2f}%" for r in metrics_data.get("recall", [0, 0])],
                "F1-Score": [f"{f * 100:.2f}%" for f in metrics_data.get("f1_score", [0, 0])],
                "Support": metrics_data.get("support", [0, 0]),
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin-top: 12px;">
                <strong>Clinical Takeaway:</strong> The model achieves a critical <strong style="color: #34d399;">97.95% Recall on Pneumonia</strong>, missing only 8 cases out of 390 in the test cohort. This minimizes false negatives during screening.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Metrics JSON not available.")

        st.markdown("</div>", unsafe_allow_html=True)

    with eval_c2:
        st.markdown("""
        <div class="med-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between;">
                <span>🔲 Visual Confusion Matrix</span>
                <span class="badge-tag">THRESHOLD = 0.70</span>
            </div>
        """, unsafe_allow_html=True)

        if metrics_data and "confusion_matrix" in metrics_data:
            cm = metrics_data["confusion_matrix"]
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 14px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">True Normal (TN)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #34d399; font-family: 'JetBrains Mono', monospace;">{tn}</div>
                    <div style="font-size: 0.74rem; color: #cbd5e1;">Actual Normal → Predicted Normal</div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 12px; padding: 14px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">False Positive (FP)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24; font-family: 'JetBrains Mono', monospace;">{fp}</div>
                    <div style="font-size: 0.74rem; color: #cbd5e1;">Actual Normal → Predicted Pneumonia</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 14px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">False Negative (FN)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #f87171; font-family: 'JetBrains Mono', monospace;">{fn}</div>
                    <div style="font-size: 0.74rem; color: #cbd5e1;">Actual Pneumonia → Predicted Normal</div>
                </div>
                <div style="background: rgba(14, 165, 233, 0.12); border: 1px solid rgba(14, 165, 233, 0.35); border-radius: 12px; padding: 14px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">True Positive (TP)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">{tp}</div>
                    <div style="font-size: 0.74rem; color: #cbd5e1;">Actual Pneumonia → Predicted Pneumonia</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Misclassification Error Report Section
    st.markdown("""
    <div class="med-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9;">
                📋 Error Report Analysis (Misclassified Test Samples)
            </div>
        </div>
    """, unsafe_allow_html=True)

    if ERROR_REPORT_PATH.is_file():
        try:
            error_df = pd.read_csv(ERROR_REPORT_PATH)

            err_col1, err_col2 = st.columns([1.5, 1])
            with err_col1:
                filter_choice = st.selectbox(
                    "Filter by Error Category:",
                    options=["All Errors", "False Positive (NORMAL as PNEUMONIA)", "False Negative (PNEUMONIA as NORMAL)"],
                )
            with err_col2:
                with open(ERROR_REPORT_PATH, "rb") as f_err:
                    st.download_button(
                        "⬇️ Download Error Report (CSV)",
                        data=f_err,
                        file_name="mediscan_error_report.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            if filter_choice == "False Positive (NORMAL as PNEUMONIA)":
                filtered_df = error_df[error_df["error_type"].str.contains("Positive", case=False, na=False)]
            elif filter_choice == "False Negative (PNEUMONIA as NORMAL)":
                filtered_df = error_df[error_df["error_type"].str.contains("Negative", case=False, na=False)]
            else:
                filtered_df = error_df

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=320,
            )
            st.caption(f"Showing {len(filtered_df)} of {len(error_df)} total misclassifications.")

        except Exception as e:
            st.error(f"Error reading error report CSV: {e}")
    else:
        st.info("Error report CSV not found in outputs directory.")

    st.markdown("</div>", unsafe_allow_html=True)
