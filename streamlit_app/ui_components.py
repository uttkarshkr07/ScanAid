"""
ui_components.py — Reusable Streamlit UI components for the ScanAid app.

Each function either returns a string (for CSS / HTML) or renders directly
to the Streamlit page via st.* calls.
"""

import numpy as np
import streamlit as st
from PIL import Image

from streamlit_app.constants import APP_VERSION, DISCLAIMER, DISCLAIMER_SHORT


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def get_custom_css() -> str:
    """Return a <style> block with ScanAid custom CSS classes."""
    return """
    <style>
        /* Disclaimer banner */
        .disclaimer-banner {
            background-color: #fff8e1;
            color: #5d4037;
            border-left: 5px solid #ff8f00;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 18px;
            font-size: 0.88rem;
            line-height: 1.6;
        }

        /* Risk result cards */
        .risk-high {
            border: 2px solid #d32f2f;
            background-color: #ffebee;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .risk-low {
            border: 2px solid #2e7d32;
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }

        /* Module selection cards */
        .module-card {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            padding: 28px 22px;
            text-align: center;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
            cursor: pointer;
        }
        .module-card:hover {
            box-shadow: 0 6px 20px rgba(26,115,232,0.18);
            transform: translateY(-3px);
        }

        /* Result section wrapper */
        .result-section {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            padding: 22px 24px;
            margin-top: 14px;
        }

        /* Metric label */
        .metric-label {
            color: #757575;
            font-size: 0.80rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        /* General utility */
        .scanaid-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1a73e8;
            text-align: center;
            margin-bottom: 0;
        }
        .scanaid-subtitle {
            font-size: 1.05rem;
            color: #5f6368;
            text-align: center;
            margin-top: 6px;
            margin-bottom: 28px;
        }
        .section-divider {
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 24px 0;
        }
    </style>
    """


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def render_disclaimer_banner() -> None:
    """Render the full medical disclaimer as a styled banner."""
    st.markdown(
        f'<div class="disclaimer-banner"><strong>Medical Disclaimer:</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the persistent sidebar with About, How It Works, and version info."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/microscope.png",
            width=64,
        )
        st.markdown("## ScanAid")
        st.caption(APP_VERSION)
        st.markdown("---")

        with st.expander("About ScanAid", expanded=True):
            st.markdown(
                """
                **ScanAid** is an AI-assisted screening research prototype that uses
                deep learning to analyse facial photographs for visual markers
                associated with certain genetic syndromes.

                It is intended for **research and educational use only** and is
                not a certified diagnostic tool.
                """
            )

        with st.expander("How It Works"):
            st.markdown(
                """
                1. **Select a module** — choose the syndrome you wish to screen for.
                2. **Upload a photograph** — a clear, front-facing image works best.
                3. **Face detection** — the app automatically locates and crops the face.
                4. **AI analysis** — a deep learning model produces a risk assessment.
                5. **Explainability** — a Grad-CAM heatmap highlights the regions that
                   most influenced the prediction.

                Results should always be reviewed by a qualified clinician.
                """
            )

        st.markdown("---")
        st.caption(DISCLAIMER_SHORT)
        st.markdown("---")
        st.caption("ScanAid · Research Prototype · " + APP_VERSION)


# ---------------------------------------------------------------------------
# Module selection cards
# ---------------------------------------------------------------------------

def render_module_cards() -> str | None:
    """Render two side-by-side module selection cards.

    Returns
    -------
    str | None
        'down_syndrome', 'angelman', or None if no card was clicked.
    """
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Select a Screening Module")

    col_a, col_b = st.columns(2, gap="large")

    clicked = None

    with col_a:
        st.markdown(
            """
            <div class="module-card">
                <div style="font-size:2.4rem;">🧬</div>
                <h3 style="color:#1a73e8; margin:10px 0 6px;">Down Syndrome</h3>
                <p style="color:#5f6368; font-size:0.9rem;">
                    Binary risk screening using a fine-tuned MobileNetV2 classifier
                    trained on facial photographs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Open Down Syndrome Module", key="btn_down", use_container_width=True):
            clicked = "down_syndrome"

    with col_b:
        st.markdown(
            """
            <div class="module-card">
                <div style="font-size:2.4rem;">🔬</div>
                <h3 style="color:#1a73e8; margin:10px 0 6px;">Angelman Syndrome</h3>
                <p style="color:#5f6368; font-size:0.9rem;">
                    Similarity-based screening using a Siamese MobileNetV2 network
                    that compares facial embeddings to reference images.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Open Angelman Module", key="btn_angelman", use_container_width=True):
            clicked = "angelman"

    return clicked


# ---------------------------------------------------------------------------
# Risk result rendering
# ---------------------------------------------------------------------------

def render_risk_result(risk_label: str, confidence: float) -> None:
    """Render a coloured risk card with a confidence progress bar.

    Parameters
    ----------
    risk_label : str
        'High Risk' or 'Low Risk'.
    confidence : float
        Value in [0, 1] representing model confidence in the given label.
    """
    css_class  = "risk-high" if risk_label == "High Risk" else "risk-low"
    emoji      = "⚠️" if risk_label == "High Risk" else "✅"
    color      = "#d32f2f" if risk_label == "High Risk" else "#2e7d32"
    pct        = int(round(confidence * 100))

    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="font-size:2rem;">{emoji}</div>
            <h2 style="color:{color}; margin:8px 0 4px;">{risk_label}</h2>
            <p class="metric-label">Model confidence</p>
            <h3 style="color:{color}; margin:0;">{pct}%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.progress(confidence, text=f"Confidence: {pct}%")


# ---------------------------------------------------------------------------
# Image preview row
# ---------------------------------------------------------------------------

def render_image_row(original_pil: Image.Image, face_uint8: np.ndarray | None) -> None:
    """Render the original uploaded image and the detected face crop side by side.

    Parameters
    ----------
    original_pil : PIL.Image.Image
        The full uploaded image.
    face_uint8 : np.ndarray | None
        The detected face crop as a (224,224,3) uint8 RGB array, or None if
        detection failed.
    """
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.image(original_pil, caption="Uploaded image", use_container_width=True)

    with col2:
        if face_uint8 is not None:
            st.image(face_uint8, caption="Detected face crop (224×224)", use_container_width=True)
        else:
            st.warning("No face detected — the analysis cannot proceed.")
