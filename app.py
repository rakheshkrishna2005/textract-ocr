import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import cv2
import numpy as np

# Configure the page: wide layout and page title
st.set_page_config(layout="wide", page_title="OCR Application")

# Center the title using Markdown with HTML
st.markdown("<h1 style='text-align: center;'>📄 Textract OCR</h1>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Sidebar: Predefined Advanced Settings (available before upload)
# ----------------------------------------------------------------
st.sidebar.subheader("Predefined Settings")

grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True)
denoising = st.sidebar.checkbox("Apply Denoising", value=False)
denoise_strength = st.sidebar.slider("Denoising Strength", min_value=1, max_value=40, value=10, step=1)
thresholding = st.sidebar.checkbox("Apply Thresholding", value=False)
threshold_level = st.sidebar.slider("Threshold Level", min_value=0, max_value=255, value=128, step=1)
rotate90 = st.sidebar.checkbox("Rotate (90° steps)", value=False)
angle_free = st.sidebar.slider("Rotation Angle (0, 90, 180, 270)", min_value=0, max_value=270, value=0, step=90)
align_text_option = st.sidebar.checkbox("Align Text", value=False)

# ----------------------------------------------------------------
# Upload Multiple Images
# ----------------------------------------------------------------
uploaded_files = st.file_uploader("Upload Images (png, jpg, jpeg) or PDF", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Adjust the predefined settings from the sidebar and upload images to begin.")
    st.stop()

# ----------------------------------------------------------------
# Preprocessing Functions
# ----------------------------------------------------------------
def preprocess_image(image, grayscale_flag, denoising_flag, denoise_strength,
                     threshold_flag, threshold_level, rotate_flag, angle_value):
    image_cv = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    if grayscale_flag:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    if denoising_flag:
        image_cv = cv2.fastNlMeansDenoising(image_cv, None, h=denoise_strength, templateWindowSize=7, searchWindowSize=21)
    
    if threshold_flag:
        _, image_cv = cv2.threshold(image_cv, threshold_level, 255, cv2.THRESH_BINARY)
    
    if rotate_flag:
        k = angle_value // 90
        image_cv = np.rot90(image_cv, k=k)
    
    return image_cv

def align_text(image_cv, thresh_val=128):
    if len(image_cv.shape) == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv.copy()
    _, img_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(img_thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_text(image_cv):
    return pytesseract.image_to_string(image_cv)

# ----------------------------------------------------------------
# Loop Over Each Uploaded Image
# ----------------------------------------------------------------
for uploaded_file in uploaded_files:    
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not open image {uploaded_file.name}: {e}")
        continue

    # Show Original and Grayscale Images Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        gray_preview = ImageOps.grayscale(image)
        st.markdown("### Grayscale Preview")
        st.image(gray_preview, caption="Grayscale", use_container_width=True)

    # Preprocessing
    with st.spinner("Processing image..."):
        processed_image = preprocess_image(
            image,
            grayscale,
            denoising,
            denoise_strength,
            thresholding,
            threshold_level,
            rotate90,
            angle_free
        )
        if align_text_option:
            processed_image = align_text(processed_image)

    # Show Processed Image
    if len(processed_image.shape) == 2:
        display_processed = Image.fromarray(processed_image)
    else:
        display_processed = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    st.markdown("### Processed Image")
    st.image(display_processed, use_container_width=True)

    # OCR Extraction
    with st.spinner("Extracting text..."):
        ocr_text = extract_text(processed_image)

    st.markdown("### Extracted Text")
    st.text_area("", ocr_text, height=300, key=uploaded_file.name)
    st.markdown("---")
