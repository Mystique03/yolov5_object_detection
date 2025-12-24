import streamlit as st
import torch
import numpy as np
from PIL import Image

# Page title
st.title("YOLOv5 Object Detection")

# Load YOLOv5 model (cached so it loads only once)
@st.cache_resource
def load_model():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )
    model.conf = 0.15  # confidence threshold
    return model

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# Run detection when an image is uploaded
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.write("Running object detection...")

    # Load model & run inference
    model = load_model()
    results = model(img_array)

    # Render results
    rendered_img = results.render()[0]

    # Display output
    st.image(
        rendered_img,
        caption="Detected objects",
        use_container_width=True
    )
