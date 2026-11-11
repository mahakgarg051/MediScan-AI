"""
MediScan AI — Modular Healthcare Research Suite
================================================
Modules:
- chest_xray: Chest radiograph pneumonia screening & Grad-CAM (Active Model)
- brain_mri: Brain MRI 4-class tumor classification & Grad-CAM (Active Model)
"""

from .chest_xray import (
    render_chest_xray_module,
    render_chest_xray_evaluation,
    load_chest_xray_model,
)
from .brain_mri import (
    render_brain_mri_module,
    load_brain_mri_model,
    is_mri_model_configured,
)

__all__ = [
    "render_chest_xray_module",
    "render_chest_xray_evaluation",
    "load_chest_xray_model",
    "render_brain_mri_module",
    "load_brain_mri_model",
    "is_mri_model_configured",
]
