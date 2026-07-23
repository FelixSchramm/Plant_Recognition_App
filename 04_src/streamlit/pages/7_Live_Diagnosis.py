"""Interactive live diagnosis page.

Upload a leaf image (or pick one of the bundled examples) and get an on-the-fly
prediction of the plant, the disease (or a healthy verdict), the model
confidence and the top-3 alternatives.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Make sure the parent ``streamlit`` folder is importable when the page is run
# directly, so ``import diagnosis`` resolves in every launch mode.
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

import diagnosis  # noqa: E402 - path setup has to run first

st.header("Live Diagnosis")
st.write(
    "Upload a leaf image and the trained model will predict the plant, tell "
    "you whether it looks healthy or diseased and show how confident it is."
)

# --- Disclaimer ------------------------------------------------------------
st.warning(
    "This is a model estimate for demonstration, not a definitive diagnosis. "
    "The model only reliably recognises the 38 plant/disease classes it was "
    "trained on and does not replace professional agronomic advice."
)

class_names = diagnosis.load_class_names()
model_bundle = diagnosis.load_diagnosis_model()

# --- Demo-mode banner ------------------------------------------------------
if diagnosis.is_mock_backend(model_bundle):
    st.info(
        "Demo mode: no trained model is connected yet, so predictions are "
        "simulated. Once the model is published on the Hugging Face Hub and "
        "configured in the app secrets, real predictions appear here "
        "automatically."
    )

# --- Supported classes -----------------------------------------------------
with st.expander(f"Show the {len(class_names)} supported classes"):
    supported = sorted(diagnosis.format_class_name(name) for name in class_names)
    st.write(", ".join(supported))

# --- Image input -----------------------------------------------------------
st.subheader("1. Choose an image")

uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

examples = diagnosis.list_example_images()
example_choice = None
if examples:
    example_labels = ["-"] + [name for name, _ in examples]
    selected = st.selectbox(
        "...or try one of the bundled example images", example_labels
    )
    if selected != "-":
        example_choice = dict(examples)[selected]

# The upload takes precedence over the example selection.
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif example_choice is not None:
    image = Image.open(example_choice)

if image is None:
    st.info("Upload an image or pick an example to start the diagnosis.")
    st.stop()

st.image(image, caption="Selected image", width=320)

# --- Diagnosis -------------------------------------------------------------
st.subheader("2. Run the diagnosis")

if not st.button("Run diagnosis", type="primary"):
    st.stop()

with st.spinner("Analysing the image..."):
    predictions, _ = diagnosis.predict(model_bundle, image, class_names, top_k=3)

top = predictions[0]

# --- Result panel ----------------------------------------------------------
st.subheader("3. Result")

if top.is_healthy:
    st.success(f"Healthy leaf detected: **{top.plant}**")
else:
    st.error(f"Disease detected: **{top.disease}** on **{top.plant}**")

col_plant, col_status, col_confidence = st.columns(3)
col_plant.metric("Plant", top.plant)
col_status.metric("Condition", "Healthy" if top.is_healthy else top.disease)
col_confidence.metric("Confidence", f"{top.confidence * 100:.1f} %")

# --- Top-3 chart -----------------------------------------------------------
st.markdown("**Top 3 predictions**")
chart_data = pd.DataFrame(
    {
        "Class": [diagnosis.format_class_name(p.label) for p in predictions],
        "Confidence": [p.confidence * 100 for p in predictions],
    }
)
figure = px.bar(
    chart_data,
    x="Confidence",
    y="Class",
    orientation="h",
    range_x=[0, 100],
    text=chart_data["Confidence"].map(lambda value: f"{value:.1f} %"),
)
figure.update_layout(yaxis=dict(autorange="reversed"), height=260)
st.plotly_chart(figure, use_container_width=True)
