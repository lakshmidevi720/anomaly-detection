
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_DIR = Path.home() / "Downloads" / "brain_mri_models"
ENCODER_PATH = MODEL_DIR / "mri_encoder.keras"
DECODER_PATH = MODEL_DIR / "mri_decoder.keras"
SETTINGS_PATH = MODEL_DIR / "model_settings.json"

st.set_page_config(
    page_title="Brain MRI Anomaly Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("Brain MRI Anomaly Detection")
st.caption("VAE-based reconstruction error research prototype")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
def sample_latent(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon


@st.cache_resource
def load_models():
    with open(SETTINGS_PATH, "r") as file:
        settings = json.load(file)

    encoder = tf.keras.models.load_model(
        ENCODER_PATH, compile=False, safe_mode=False,
        custom_objects={"sample_latent": sample_latent}
    )
    decoder = tf.keras.models.load_model(
        DECODER_PATH, compile=False, safe_mode=False
    )
    return encoder, decoder, settings


if not all(p.exists() for p in
           [ENCODER_PATH, DECODER_PATH, SETTINGS_PATH]):
    st.error("Model files are missing. Check ~/Downloads/brain_mri_models/")
    st.stop()

try:
    encoder, decoder, settings = load_models()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

img_size = int(settings["image_size"])
threshold = float(settings["threshold"])

st.success("Trained VAE model loaded successfully.")

# --------------------------------------------------
# Upload MRI
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is None:
    st.info("Upload an MRI image to begin analysis.")
    st.stop()

try:
    pil_image = Image.open(uploaded_file).convert("L")
    original = np.asarray(pil_image, dtype=np.float32) / 255.0

    resized_image = pil_image.resize((img_size, img_size))
    model_image = np.asarray(
        resized_image, dtype=np.float32
    ) / 255.0
    model_input = model_image[np.newaxis, ..., np.newaxis]

    # Encoder returns mean, log variance and sampled latent vector.
    # Use the mean for deterministic reconstruction.
    z_mean, _, _ = encoder.predict(model_input, verbose=0)
    reconstruction_batch = decoder.predict(z_mean, verbose=0)
    reconstruction = np.clip(reconstruction_batch[0, ..., 0], 0, 1)

    # Absolute pixel-wise reconstruction error.
    error_map = np.abs(model_image - reconstruction)
    score = float(np.mean(np.square(model_image - reconstruction)))

    # Normalize the heatmap for visualization only.
    max_error = float(error_map.max())
    if max_error > 0:
        heatmap = error_map / max_error
    else:
        heatmap = error_map

except Exception as exc:
    st.error(f"Could not process this image: {exc}")
    st.stop()

# --------------------------------------------------
# Results
# --------------------------------------------------
st.header("Analysis Results")

m1, m2, m3 = st.columns(3)
m1.metric("Reconstruction MSE", f"{score:.6f}")
m2.metric("Provisional threshold", f"{threshold:.6f}")
m3.metric(
    "Threshold comparison",
    "Above threshold" if score > threshold else "At/below threshold"
)

st.caption(
    "The threshold was calibrated using normal test images in this "
    "initial experiment. It is provisional and requires independent validation."
)

# --------------------------------------------------
# Images
# --------------------------------------------------
st.subheader("Image Comparison")
c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Original MRI**")
    st.image(original, clamp=True, width="stretch")

with c2:
    st.write("**VAE Reconstruction**")
    st.image(reconstruction, clamp=True, width="stretch")

with c3:
    st.write("**Reconstruction Error Heatmap**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Methodology")
st.write(
    "MRI preprocessing → VAE encoder → latent mean → VAE decoder → "
    "reconstruction error → provisional threshold comparison."
)

st.warning(
    "Research prototype only. This model is not a validated medical "
    "diagnostic system. An above-threshold result does not establish "
    "the presence of a tumor or other disease."
)

