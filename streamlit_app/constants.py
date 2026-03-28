"""
constants.py — Project-wide constants for the ScanAid Streamlit application.

All path constants are built as absolute paths anchored to PROJECT_ROOT so the
app works regardless of the working directory from which it is launched.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at:  <project_root>/streamlit_app/constants.py
# Two levels up → <project_root>
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Use .keras format (Keras 3 native) — version-agnostic, preferred over .h5
# Fall back to .h5 if .keras file doesn't exist yet
def _model_path(name):
    keras_path = os.path.join(PROJECT_ROOT, "models", f"{name}.keras")
    h5_path    = os.path.join(PROJECT_ROOT, "models", f"{name}.h5")
    return keras_path if os.path.exists(keras_path) else h5_path

DOWN_SYNDROME_MODEL_PATH = _model_path("down_syndrome_detector")
ANGELMAN_MODEL_PATH      = _model_path("angelman_siamese_model")

# Primary reference directory (curated reference images for Angelman screening)
ANGELMAN_REFERENCE_DIR   = os.path.join(PROJECT_ROOT, "data", "references", "angelman")

# Fallback: use training data directory if the reference dir is absent / empty
ANGELMAN_FALLBACK_DIR    = os.path.join(PROJECT_ROOT, "syndrome_dataset", "angelman")

# ---------------------------------------------------------------------------
# Image / model hyper-parameters
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)

# Down Syndrome binary classifier — sigmoid output
# prediction < HIGH_RISK_THRESHOLD  →  "High Risk"
HIGH_RISK_THRESHOLD = 0.5

# Angelman Siamese model — euclidean distance output
# avg_distance < SIAMESE_DISTANCE_THRESHOLD  →  "High Risk"
SIAMESE_DISTANCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_VERSION = "v0.1.0 (Research Prototype)"

MAX_UPLOAD_MB = 10

# ---------------------------------------------------------------------------
# Disclaimers
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "IMPORTANT MEDICAL DISCLAIMER: ScanAid is an experimental research prototype "
    "intended solely for academic and investigational purposes. It is NOT a "
    "certified medical device and must NOT be used as a substitute for professional "
    "medical diagnosis, advice, or treatment. The AI-generated risk assessments are "
    "based on limited training data and may be inaccurate. All results must be "
    "interpreted by a qualified healthcare professional in the context of a full "
    "clinical evaluation. The developers accept no liability for any clinical "
    "decisions made on the basis of this tool. If you have concerns about a child's "
    "development, please consult a licensed medical specialist."
)

DISCLAIMER_SHORT = (
    "ScanAid is a research prototype — results are not a medical diagnosis and "
    "must be reviewed by a qualified healthcare professional."
)
