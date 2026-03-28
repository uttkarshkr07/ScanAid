"""
app.py — Main entry point for the ScanAid Streamlit web application.

Run with:
    streamlit run app.py
"""

# ---------------------------------------------------------------------------
# Path bootstrap — must be the very first executable lines so every subsequent
# import can resolve project-relative packages (streamlit_app, src, etc.).
# ---------------------------------------------------------------------------
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Standard library / third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from streamlit_app.constants import (
    APP_VERSION,
    DISCLAIMER,
    DISCLAIMER_SHORT,
    MAX_UPLOAD_MB,
)
from streamlit_app.inference import (
    detect_face_from_array,
    load_down_syndrome_model,
    load_angelman_model,
    load_reference_embeddings,
    predict_down_syndrome,
    predict_angelman,
    _extract_base_network,
)
from streamlit_app.gradcam import generate_gradcam_overlay
from streamlit_app.ui_components import (
    get_custom_css,
    render_disclaimer_banner,
    render_sidebar,
    render_module_cards,
    render_risk_result,
    render_image_row,
)

# ---------------------------------------------------------------------------
# Page configuration  (must be called before any other st.* command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScanAid",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_STATE_DEFAULTS = {
    "page":          "home",       # 'home' | 'down_syndrome' | 'angelman'
    "uploaded_file": None,
    "analysis_done": False,
    "results":       None,         # dict with result payload when done
}

for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar  (always visible)
# ---------------------------------------------------------------------------
render_sidebar()


# ===========================================================================
# Page: Home
# ===========================================================================
def render_home_page() -> None:
    """Render the ScanAid landing / module-selection page."""
    render_disclaimer_banner()

    st.markdown('<h1 class="scanaid-title">🔬 ScanAid</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="scanaid-subtitle">'
        "AI-assisted facial phenotype screening · Research Prototype"
        "</p>",
        unsafe_allow_html=True,
    )

    clicked = render_module_cards()

    if clicked is not None:
        # Reset analysis state when switching modules
        st.session_state.page          = clicked
        st.session_state.uploaded_file = None
        st.session_state.analysis_done = False
        st.session_state.results       = None
        st.rerun()


# ===========================================================================
# Page: Module (shared for Down Syndrome and Angelman)
# ===========================================================================

_MODULE_META = {
    "down_syndrome": {
        "title":       "Down Syndrome Screening",
        "icon":        "🧬",
        "description": (
            "This module uses a fine-tuned MobileNetV2 binary classifier to assess "
            "whether a facial photograph exhibits visual markers associated with "
            "Down Syndrome (Trisomy 21).  The model was trained on labelled facial "
            "images and produces a probabilistic risk score."
        ),
    },
    "angelman": {
        "title":       "Angelman Syndrome Screening",
        "icon":        "🔬",
        "description": (
            "This module uses a Siamese MobileNetV2 network to compare the facial "
            "embedding of the uploaded photograph against a set of reference images. "
            "A low Euclidean distance to Angelman reference embeddings is flagged as "
            "higher risk."
        ),
    },
}


def render_module_page(module: str) -> None:
    """Render the analysis page for *module* ('down_syndrome' or 'angelman')."""
    meta = _MODULE_META[module]

    # --- Back button --------------------------------------------------------
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.page          = "home"
        st.session_state.uploaded_file = None
        st.session_state.analysis_done = False
        st.session_state.results       = None
        st.rerun()

    st.markdown(f"## {meta['icon']} {meta['title']}")
    st.markdown(f"_{meta['description']}_")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- Model loading (with user-friendly errors) --------------------------
    model        = None
    base_network = None
    ref_embeddings = None
    model_ok     = False

    if module == "down_syndrome":
        try:
            with st.spinner("Loading AI model…"):
                model = load_down_syndrome_model()
            model_ok = True
        except FileNotFoundError as exc:
            st.error(f"Model file not found. {exc}")
        except Exception as exc:
            st.error(f"Failed to load the Down Syndrome model: {exc}")

    else:  # angelman
        try:
            with st.spinner("Loading Angelman Siamese model…"):
                angelman_model = load_angelman_model()
                base_network   = _extract_base_network(angelman_model)
                if base_network is None:
                    st.error(
                        "Could not locate the 'feature_extractor' sub-model inside "
                        "the Angelman Siamese model.  Please check the model file."
                    )
                else:
                    model = angelman_model
                    model_ok = True
        except FileNotFoundError as exc:
            st.error(f"Model file not found. {exc}")
        except Exception as exc:
            st.error(f"Failed to load the Angelman model: {exc}")

        if model_ok:
            try:
                with st.spinner("Loading reference embeddings…"):
                    ref_embeddings = load_reference_embeddings()
                if ref_embeddings is None:
                    st.warning(
                        "No reference images were found. "
                        "Please add reference images to "
                        "`data/references/angelman/<class>/` "
                        "before running Angelman screening."
                    )
                    model_ok = False
                else:
                    n_classes = len(ref_embeddings)
                    st.success(
                        f"Reference embeddings loaded for {n_classes} class(es): "
                        + ", ".join(ref_embeddings.keys())
                    )
            except Exception as exc:
                st.error(f"Failed to load reference embeddings: {exc}")
                model_ok = False

    # --- File uploader ------------------------------------------------------
    st.markdown("### Upload a Photograph")
    uploaded = st.file_uploader(
        "Choose a front-facing facial photograph (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{module}",
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB",
    )

    if uploaded is None:
        st.info("Please upload an image to begin.")
        return

    # Check file size
    uploaded.seek(0, 2)
    file_size_mb = uploaded.tell() / (1024 * 1024)
    uploaded.seek(0)
    if file_size_mb > MAX_UPLOAD_MB:
        st.error(
            f"The uploaded file is {file_size_mb:.1f} MB, which exceeds the "
            f"{MAX_UPLOAD_MB} MB limit.  Please upload a smaller image."
        )
        return

    # Load PIL image
    try:
        pil_image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not open the uploaded file as an image.  Please try a different file.")
        return

    img_array = np.array(pil_image, dtype=np.uint8)

    # --- Face detection preview ---------------------------------------------
    st.markdown("### Face Detection")
    with st.spinner("Detecting face…"):
        face_uint8, face_ok, face_msg = detect_face_from_array(img_array)

    render_image_row(pil_image, face_uint8 if face_ok else None)

    if face_ok:
        st.success(face_msg)
    else:
        st.warning(face_msg)

    # --- Analysis button ----------------------------------------------------
    run_disabled = (not face_ok) or (not model_ok)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    run_clicked = st.button(
        "Run Screening Analysis",
        key=f"run_{module}",
        disabled=run_disabled,
        type="primary",
        use_container_width=True,
    )

    # Show a hint when the button is disabled
    if run_disabled and not face_ok:
        st.caption("Analysis is disabled — face detection must succeed first.")
    elif run_disabled and not model_ok:
        st.caption("Analysis is disabled — model could not be loaded.")

    if not run_clicked:
        return

    # --- Run analysis -------------------------------------------------------
    risk_label   = None
    confidence   = None
    elapsed      = None
    face_float32 = None
    overlay      = None

    try:
        with st.spinner("Running AI analysis…"):
            if module == "down_syndrome":
                risk_label, confidence, elapsed, face_float32 = predict_down_syndrome(face_uint8)
            else:
                risk_label, confidence, elapsed, face_float32 = predict_angelman(
                    face_uint8, base_network, ref_embeddings
                )
    except Exception as exc:
        st.error(f"An error occurred during AI analysis: {exc}")
        return

    try:
        with st.spinner("Generating explainability heatmap…"):
            overlay = generate_gradcam_overlay(
                face_float32,
                model=model if module == "down_syndrome" else model,
                model_type="standard" if module == "down_syndrome" else "siamese",
            )
    except Exception:
        overlay = None  # Grad-CAM failure is non-fatal

    # --- Display results ----------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## Results")

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        render_risk_result(risk_label, confidence)
        st.markdown("</div>", unsafe_allow_html=True)

    with res_col2:
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM Explainability Heatmap", width="stretch")
            st.caption(
                "The heatmap highlights facial regions that most influenced the prediction. "
                "Warmer colours (red/yellow) indicate higher importance."
            )
        else:
            st.image(face_uint8, caption="Detected Face Crop", width="stretch")
            st.caption("Grad-CAM overlay could not be generated for this image.")

    # --- Technical details expander -----------------------------------------
    with st.expander("Technical Details"):
        st.markdown(
            f"""
            | Parameter | Value |
            |---|---|
            | Module | {meta['title']} |
            | Risk label | {risk_label} |
            | Confidence | {confidence * 100:.1f}% |
            | Inference time | {elapsed * 1000:.1f} ms |
            | Input size | 224 × 224 px |
            | Normalisation | ÷ 255.0 |
            """
        )
        if module == "angelman" and ref_embeddings is not None:
            st.markdown(
                f"**Reference classes used:** {', '.join(ref_embeddings.keys())}"
            )

    # --- Disclaimer footer --------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="disclaimer-banner">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Router
# ===========================================================================

def main() -> None:
    """Main router — dispatches to the correct page based on session state."""
    page = st.session_state.get("page", "home")

    if page == "home":
        render_home_page()
    elif page in ("down_syndrome", "angelman"):
        render_module_page(page)
    else:
        # Unknown state — reset to home
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()
else:
    # Streamlit executes app.py as a module, so main() must be called at
    # module scope as well.
    main()
