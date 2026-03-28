"""
gradcam.py — Grad-CAM overlay generation for the ScanAid Streamlit app.

Wraps the lower-level functions from src/gradcam.py, adapting them for the
array-based inputs used in the app (as opposed to the file-path-based API
used during training).
"""

import sys
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — ensure src/ is importable
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.gradcam import find_last_conv_layer, make_gradcam_heatmap  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam_overlay(
    face_float32: np.ndarray,
    model,
    model_type: str = "standard",
) -> np.ndarray | None:
    """Generate a Grad-CAM saliency overlay for a face crop.

    Parameters
    ----------
    face_float32 : np.ndarray
        Shape (224, 224, 3), dtype float32, values in [0, 1].  This is the
        normalised face array returned by the predict_* inference functions.
    model : tf.keras.Model
        The model to explain.  For 'siamese' this should be the full Siamese
        model; the base network is extracted internally.
    model_type : str
        'standard' — use *model* directly (Down Syndrome classifier).
        'siamese'  — extract the feature_extractor sub-model and use the L2
                     norm of its output as the gradient target (Angelman).

    Returns
    -------
    np.ndarray | None
        RGB uint8 array of shape (224, 224, 3) with the heatmap blended onto
        the original face, or None if Grad-CAM generation fails.
    """
    try:
        import tensorflow as tf

        img_array = np.expand_dims(face_float32, axis=0)  # (1, 224, 224, 3)

        if model_type == "siamese":
            # ----------------------------------------------------------------
            # For Siamese models we explain the base (feature_extractor) net.
            # The gradient target is the L2 norm of the embedding vector —
            # the same approach used in src/gradcam.py run_gradcam_siamese.
            # ----------------------------------------------------------------
            base_net = _extract_base_network(model)
            if base_net is None:
                return None

            last_conv_name = find_last_conv_layer(base_net)
            if last_conv_name is None:
                return None

            # Build a sub-model that outputs the last conv activations and the
            # embedding, then use tf.GradientTape for the L2-norm score.
            grad_model = tf.keras.models.Model(
                inputs=base_net.inputs,
                outputs=[
                    base_net.get_layer(last_conv_name).output,
                    base_net.output,
                ],
            )

            with tf.GradientTape() as tape:
                conv_outputs, embeddings = grad_model(img_array, training=False)
                # L2 norm of the embedding as a scalar score
                score = tf.norm(embeddings, axis=-1)

            grads      = tape.gradient(score, conv_outputs)            # (1, h, w, c)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))       # (c,)
            conv_outputs = conv_outputs[0]                             # (h, w, c)

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]     # (h, w, 1)
            heatmap = tf.squeeze(heatmap)                              # (h, w)
            heatmap = tf.nn.relu(heatmap)
            heatmap_np = heatmap.numpy()

            # Normalise to [0, 1]
            h_max = heatmap_np.max()
            if h_max > 0:
                heatmap_np = heatmap_np / h_max
            heatmap_np = heatmap_np.astype(np.float32)

        else:
            # ----------------------------------------------------------------
            # Standard binary classifier  (Down Syndrome)
            # ----------------------------------------------------------------
            last_conv_name = find_last_conv_layer(model)
            if last_conv_name is None:
                return None

            heatmap_np = make_gradcam_heatmap(model, img_array, last_conv_name)

        # --------------------------------------------------------------------
        # Blend heatmap onto the original face image
        # --------------------------------------------------------------------
        # Resize heatmap to 224×224
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))

        # Convert to uint8 and apply JET colormap  → BGR
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # BGR

        # Convert face back to uint8 RGB for blending
        face_uint8_rgb = np.uint8(face_float32 * 255)
        face_uint8_bgr = cv2.cvtColor(face_uint8_rgb, cv2.COLOR_RGB2BGR)

        # Alpha blend: 60 % original, 40 % heatmap
        overlay_bgr = cv2.addWeighted(face_uint8_bgr, 0.6, heatmap_color, 0.4, 0)

        # Return as RGB
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return overlay_rgb.astype(np.uint8)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_base_network(siamese_model):
    """Return the 'feature_extractor' sub-model from the Siamese model.

    Mirrors src/evaluate.py _extract_base_network so the correct sub-model
    is targeted for both Grad-CAM and inference.
    """
    import tensorflow as tf

    for layer in siamese_model.layers:
        if layer.name == "feature_extractor":
            return layer

    # Fallback: first layer that is itself a Model
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    return None
