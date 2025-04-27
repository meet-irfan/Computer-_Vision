import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import pytesseract
import os

# Set page config
st.set_page_config(
    page_title="Pakil OCR App",
    page_icon="📄",
    layout="wide"
)

# Load the model (or create new instance if not found)
@st.cache_resource
def load_model():
    try:
        with open('ocr_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        # If model file doesn't exist, create new instance
        from model import OCRModel
        return OCRModel()

model = load_model()

# App title and description
st.title("📄 Pakil Document OCR")
st.markdown("""
Upload an image of a document, and this app will extract the text using OCR.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    show_processed = st.checkbox("Show processed image", True)
    language = st.selectbox("OCR Language", ["eng", "eng+fil"])

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of the document"
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)
    
    # Process and show results
    with col2:
        try:
            # Extract text
            text = model.extract_text("temp_image.png")
            
            # Show processed image if enabled
            if show_processed:
                processed = model.preprocess_image(cv2.imread("temp_image.png"))
                st.subheader("Processed Image")
                st.image(processed, use_column_width=True, clamp=True)
            
            # Show extracted text
            st.subheader("Extracted Text")
            st.text_area("OCR Results", text, height=300)
            
            # Add download button
            st.download_button(
                label="Download Text",
                data=text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Clean up
    os.remove("temp_image.png")

# Add some instructions
st.markdown("""
### Tips for better results:
- Use clear, well-lit images
- Ensure text is horizontal
- Higher resolution images work better
- For Filipino text, select 'eng+fil' in settings
""")