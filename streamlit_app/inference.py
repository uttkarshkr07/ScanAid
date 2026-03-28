"""
inference.py — Model loading and inference functions for the ScanAid app.

All model-loading functions are decorated with @st.cache_resource so that
heavy TensorFlow models are loaded only once per Streamlit server session.
"""

import os
import time
import sys

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on the path so sibling packages resolve correctly.
# (app.py also does this, but inference.py may be imported independently.)
# ---------------------------------------------------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from streamlit_app.constants import (
    DOWN_SYNDROME_MODEL_PATH,
    ANGELMAN_MODEL_PATH,
    ANGELMAN_REFERENCE_DIR,
    ANGELMAN_FALLBACK_DIR,
    IMG_SIZE,
    HIGH_RISK_THRESHOLD,
    SIAMESE_DISTANCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def detect_face_from_array(img_rgb_uint8: np.ndarray):
    """Detect and crop the largest face from an RGB uint8 numpy array.

    Replicates the exact same Haar Cascade logic used during training in
    src/preprocessing.py so that inference inputs are consistent with the
    training distribution.

    Parameters
    ----------
    img_rgb_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8, RGB colour order.

    Returns
    -------
    tuple[np.ndarray | None, bool, str]
        (face_uint8_rgb, success, message)
        face_uint8_rgb is resized to IMG_SIZE and dtype uint8, or None if
        no face was detected.
    """
    if img_rgb_uint8 is None or img_rgb_uint8.ndim != 3:
        return None, False, "Invalid image array supplied."

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    if len(faces) == 0:
        return None, False, (
            "No face detected in the uploaded image. "
            "Please upload a clear, front-facing photograph."
        )

    # Select the largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # 10 % margin (clamped to image boundaries)
    margin_w = int(w * 0.10)
    margin_h = int(h * 0.10)
    img_h, img_w = img_rgb_uint8.shape[:2]

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)

    face_crop = img_rgb_uint8[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)

    return face_resized.astype(np.uint8), True, "Face detected successfully."


# ---------------------------------------------------------------------------
# Model loading  (cached per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_down_syndrome_model():
    """Load the Down Syndrome binary classifier from disk.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model ready for inference.

    Raises
    ------
    FileNotFoundError
        Re-raised with a user-friendly message if the model file is absent.
    """
    import tensorflow as tf

    if not os.path.exists(DOWN_SYNDROME_MODEL_PATH):
        raise FileNotFoundError(
            f"Down Syndrome model not found at:\n{DOWN_SYNDROME_MODEL_PATH}\n\n"
            "Please ensure the model file has been placed in the models/ directory."
        )
    model = tf.keras.models.load_model(DOWN_SYNDROME_MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def load_angelman_model():
    """Load the Angelman Siamese model with its custom objects.

    The EuclideanDistance layer is defined inline here to keep inference.py
    self-contained and avoid import-order issues.

    Returns
    -------
    tf.keras.Model
        The loaded Siamese model.

    Raises
    ------
    FileNotFoundError
        Re-raised with a user-friendly message if the model file is absent.
    """
    import tensorflow as tf

    if not os.path.exists(ANGELMAN_MODEL_PATH):
        raise FileNotFoundError(
            f"Angelman model not found at:\n{ANGELMAN_MODEL_PATH}\n\n"
            "Please ensure the model file has been placed in the models/ directory."
        )

    class EuclideanDistance(tf.keras.layers.Layer):
        """Custom layer that computes the Euclidean distance between two vectors."""

        def call(self, inputs):
            x, y = inputs
            sum_square = tf.math.reduce_sum(
                tf.math.square(x - y), axis=1, keepdims=True
            )
            return tf.math.sqrt(tf.math.maximum(sum_square, 1e-8))

    def contrastive_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model = tf.keras.models.load_model(
        ANGELMAN_MODEL_PATH,
        custom_objects={
            "EuclideanDistance": EuclideanDistance,
            "contrastive_loss":  contrastive_loss,
        },
    )
    return model


# ---------------------------------------------------------------------------
# Reference embeddings for Angelman (cached per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_reference_embeddings(base_network):
    """Compute and cache per-class prototype embeddings for Angelman screening.

    Scans ANGELMAN_REFERENCE_DIR for sub-folders (one per class).  Falls back
    to ANGELMAN_FALLBACK_DIR if the primary directory is absent or empty.
    Each class prototype is the mean of all per-image embeddings in that folder.

    Parameters
    ----------
    base_network : tf.keras.Model
        The feature-extractor sub-model extracted from the Siamese model.

    Returns
    -------
    dict[str, np.ndarray] | None
        Mapping from class name to its prototype embedding vector, or None if
        no usable reference images were found.
    """
    import tensorflow as tf
    from PIL import Image

    # Decide which directory to use
    ref_dir = None
    for candidate in [ANGELMAN_REFERENCE_DIR, ANGELMAN_FALLBACK_DIR]:
        if os.path.isdir(candidate):
            sub_dirs = [
                d for d in os.listdir(candidate)
                if os.path.isdir(os.path.join(candidate, d))
            ]
            if sub_dirs:
                ref_dir = candidate
                break

    if ref_dir is None:
        return None

    prototypes = {}
    valid_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for class_name in sorted(os.listdir(ref_dir)):
        class_path = os.path.join(ref_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        embeddings = []
        for fname in os.listdir(class_path):
            if os.path.splitext(fname)[1].lower() not in valid_exts:
                continue
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)
                emb = base_network.predict(arr, verbose=0) # (1, embedding_dim)
                embeddings.append(emb[0])
            except Exception:
                continue  # skip unreadable files silently

        if embeddings:
            prototypes[class_name] = np.mean(embeddings, axis=0)

    return prototypes if prototypes else None


# ---------------------------------------------------------------------------
# Per-call inference helpers
# ---------------------------------------------------------------------------

def _extract_base_network(siamese_model):
    """Return the 'feature_extractor' sub-model from the Siamese model.

    Mirrors the logic in src/evaluate.py so that embedding extraction is
    identical between training-time evaluation and app inference.

    Parameters
    ----------
    siamese_model : tf.keras.Model

    Returns
    -------
    tf.keras.Model | None
    """
    # Prefer the named sub-model
    for layer in siamese_model.layers:
        if layer.name == "feature_extractor":
            return layer
    # Fallback: first layer that is itself a Model
    import tensorflow as tf
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def predict_down_syndrome(face_uint8_rgb: np.ndarray):
    """Run Down Syndrome binary classifier on a pre-detected face crop.

    Parameters
    ----------
    face_uint8_rgb : np.ndarray
        Shape (224, 224, 3), dtype uint8, RGB.

    Returns
    -------
    tuple[str, float, float, np.ndarray]
        (risk_label, confidence, inference_time_seconds, face_float32)
        risk_label  : "High Risk" | "Low Risk"
        confidence  : float in [0, 1]
        face_float32: normalised face array ready for Grad-CAM
    """
    model = load_down_syndrome_model()

    face_float32 = face_uint8_rgb.astype(np.float32) / 255.0
    img_array    = np.expand_dims(face_float32, axis=0)  # (1, 224, 224, 3)

    t0   = time.perf_counter()
    pred = float(model.predict(img_array, verbose=0)[0][0])
    elapsed = time.perf_counter() - t0

    # class_indices: {'down_syndrome': 0, 'typical': 1}
    # sigmoid output: low value → down_syndrome (high risk)
    if pred < HIGH_RISK_THRESHOLD:
        risk_label = "High Risk"
        confidence = float(1.0 - pred)
    else:
        risk_label = "Low Risk"
        confidence = float(pred)

    return risk_label, confidence, elapsed, face_float32


def predict_angelman(face_uint8_rgb: np.ndarray, base_network, ref_embeddings: dict):
    """Run Angelman Siamese screening on a pre-detected face crop.

    Parameters
    ----------
    face_uint8_rgb : np.ndarray
        Shape (224, 224, 3), dtype uint8, RGB.
    base_network : tf.keras.Model
        Feature-extractor sub-model from the Siamese model.
    ref_embeddings : dict[str, np.ndarray]
        Per-class prototype embeddings returned by load_reference_embeddings().

    Returns
    -------
    tuple[str, float, float, np.ndarray]
        (risk_label, confidence, inference_time_seconds, face_float32)
    """
    face_float32 = face_uint8_rgb.astype(np.float32) / 255.0
    img_array    = np.expand_dims(face_float32, axis=0)  # (1, 224, 224, 3)

    t0  = time.perf_counter()
    emb = base_network.predict(img_array, verbose=0)[0]   # (embedding_dim,)
    elapsed = time.perf_counter() - t0

    # Euclidean distance to each class prototype, then average
    distances = []
    for prototype in ref_embeddings.values():
        dist = float(np.linalg.norm(emb - prototype))
        distances.append(dist)

    avg_distance = float(np.mean(distances))

    if avg_distance < SIAMESE_DISTANCE_THRESHOLD:
        risk_label = "High Risk"
        confidence = float(max(0.0, min(1.0, 1.0 - avg_distance)))
    else:
        risk_label = "Low Risk"
        confidence = float(max(0.0, min(1.0, avg_distance)))

    return risk_label, confidence, elapsed, face_float32
