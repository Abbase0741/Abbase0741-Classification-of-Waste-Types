import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import gdown
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.applications import DenseNet169

# Page Configuration & UI Settings
st.set_page_config(page_title="Waste Classification Engine - DenseNet169", layout="centered")

# Custom CSS for Modern Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stWidgetLabel"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(45deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Waste Classification Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload a waste sample image to classify its type via DenseNet169 model</p>", unsafe_allow_html=True)

# 1. Class Mapping & Threshold Configuration
CLASS_MAPPING = {
    0: 'Recyclable',
    1: 'Electronic',
    2: 'Organic'
}
TARGET_SIZE = (150, 150)
CONFIDENCE_THRESHOLD = 0.50  # 50% Threshold

# 2. Load Keras Model with Cache
# File ID Google Drive milikmu
GDRIVE_FILE_ID = '16I8s00QRFE1u2uW_ose3QyEioigqnl5B' 
MODEL_FILENAME = 'modelDenseNet169v3.keras'

def build_densenet_model():
    # 1. Inisialisasi Base Model DenseNet169 (tanpa top classification head)
    base_model = DenseNet169(weights=None, include_top=False, input_shape=(150, 150, 3))
    
    # 2. Reconstruct Sequential Architecture sesuai config kamu
    model = Sequential([
        InputLayer(input_shape=(150, 150, 3)),
        base_model,
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    return model

@st.cache_resource
def load_densenet_model():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}&confirm=t'
            gdown.download(url, MODEL_FILENAME, quiet=False)
            
    try:
        # Bangun arsitektur terlebih dahulu
        model = build_densenet_model()
        # Muat bobot (weights) saja dari file .keras
        model.load_weights(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None

model = load_densenet_model()

# 3. Image Preprocessing & Inference Function
def predict_waste_sample(img_input, model):
    # Resize image to (150, 150) matching the training pipeline
    img_resized = img_input.resize(TARGET_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Rescale pixel values (1./255)
    img_array = img_array / 255.0
    
    # Expand dims for batch format (1, 150, 150, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict logits/probabilities
    preds = model.predict(img_batch, verbose=0)
    
    # Apply Softmax if model outputs raw logits
    if np.sum(preds[0]) > 1.01 or np.sum(preds[0]) < 0.99:
        probabilities = tf.nn.softmax(preds[0]).numpy()
    else:
        probabilities = preds[0]
        
    predicted_class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class_idx])
    predicted_class_name = CLASS_MAPPING.get(predicted_class_idx, "Unknown")
    
    return predicted_class_name, confidence

# 4. Streamlit Upload & Predict Interface
uploaded_file = st.file_uploader("Upload waste image sample", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    container_action = st.empty()
    btn_predict = container_action.button("Start Computer Vision Inference", type="primary", use_container_width=True)
    
    st.image(img, caption="Uploaded Waste Image Sample", use_container_width=True)
    
    if btn_predict:
        if model is not None:
            progress_bar = container_action.progress(0, text="Preprocessing image tensor to (150, 150, 3)...")
            
            for percent in range(1, 101, 10):
                time.sleep(0.03)
                if percent == 30:
                    progress_bar.progress(percent, text="Normalizing pixel scale (1./255)...")
                elif percent == 60:
                    progress_bar.progress(percent, text="Extracting feature representations via DenseNet169...")
                elif percent == 90:
                    progress_bar.progress(percent, text="Computing class probability distribution...")
                else:
                    progress_bar.progress(percent)
                    
            pred_class, confidence = predict_waste_sample(img, model)
            container_action.empty()
            
            # --- RESULTS PANEL ---
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Predicted Class", value=pred_class)
                with col2:
                    st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
                with col3:
                    st.metric(label="Applied Threshold", value=f"{CONFIDENCE_THRESHOLD * 100:.0f}%")
                    
            st.write("")
            st.button("🔄 Reset and Upload New Image", use_container_width=True)
        else:
            st.error("The classification system is unready because the model failed to initialize.")
