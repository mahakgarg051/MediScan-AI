"""
MediScan AI — Brain MRI Subsystem
==================================
Modular training, evaluation, and dataset tools for 4-class Brain MRI classification:
1. glioma
2. meningioma
3. pituitary tumour
4. no tumour
"""

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumour",
    "pituitary_tumor": "Pituitary Tumour",
}
IMAGE_SIZE = 224
