import streamlit as st
import requests
from PIL import Image
import io

st.title("🔬 Cancer Detection Demo")

# 1) Upload
uploaded = st.file_uploader("Upload a lesion image (PNG/JPG)", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

# 2) Read bytes once
raw_bytes = uploaded.read()

# 3) Show it
img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
st.image(img, caption="Input image", use_container_width=True)

# 4) When user clicks, POST those same bytes
if st.button("Run prediction"):
    files = {
        "file": (uploaded.name, raw_bytes, uploaded.type)
    }
    try:
        resp = requests.post("http://api:8000/predict", files=files, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prob = data.get("probability_malignant")
        uncertainty = data.get("uncertainty_entropy")
        if prob is None:
            st.error("API returned no probability.")
        else:
            st.metric("Predictive uncertainty", f"{uncertainty:.2f}")
            st.metric("Probability malignant", f"{prob:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
