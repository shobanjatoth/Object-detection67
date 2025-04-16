import streamlit as st
import os
import cv2
import base64
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
import tempfile
import numpy as np

# --- BASE64 BACKGROUND IMAGE ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_path):
    base64_img = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Set background
set_background("background.jpg")  # Replace with your image filename

# Set Streamlit config
st.set_page_config(page_title="License Plate Recognition", layout="centered")

# Custom CSS for UI
st.markdown(
    """
    <style>
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        text-align: center;
        color: #0a3d62;
    }
    .stButton > button {
        background-color: #0984e3;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    .stButton > button:hover {
        background-color: #0652DD;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main">🚘 License Plate Detection & OCR</h1>', unsafe_allow_html=True)

# Load models
yolo_model = YOLO("best.pt")  # Replace with your YOLOv8 model path
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')  # English OCR

# File upload
uploaded_file = st.file_uploader("Upload an image or a video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    suffix = os.path.splitext(uploaded_file.name)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # For Images
    if "image" in file_type:
        image = Image.open(temp_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        results = yolo_model.predict(temp_path)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                image_np = np.array(image)
                plate_crop = image_np[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                st.image(plate_crop, caption="Detected License Plate")
                ocr_result = ocr_model.ocr(plate_crop, cls=True)
                for line in ocr_result:
                    for word_info in line:
                        detected_text = word_info[1][0]
                        st.success(f"📌 Detected Text: `{detected_text}`")
                        break

    # For Videos
    elif "video" in file_type:
        st.video(temp_path)
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 == 0:
                results = yolo_model.predict(frame)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size == 0:
                            continue
                        ocr_result = ocr_model.ocr(plate_crop, cls=True)
                        for line in ocr_result:
                            for word_info in line:
                                detected_text = word_info[1][0]
                                st.info(f"🎥 Frame {frame_count}: `{detected_text}`")
                                break
            if frame_count > 300:
                break
        cap.release()

    os.remove(temp_path)

# --- CONTACT INFO ---
st.markdown('<div class="contact">Contact: <a href="mailto:shobanbabujatoth@gmail.com">shobanbabujatoth@gmail.com</a></div>', unsafe_allow_html=True)

