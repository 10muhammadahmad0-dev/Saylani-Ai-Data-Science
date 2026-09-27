import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\Lapzone.pk\OneDrive\Desktop\Mnist deployment\mnist_model.keras"   # apne trained model ka naam/path yahan set karo

# ---------- LOAD MODEL (cached so it loads only once) ----------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# ---------- PAGE ----------
st.title("MNIST Digit Predictor")
st.write("Upload a handwritten digit")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Predict Digit"):
        # ---- Preprocess ----
        img = image.convert("L")                # grayscale
        img = ImageOps.invert(img) if np.array(img).mean() > 127 else img  # ensure white digit on black bg
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 784)   # flattened input, matches your model's expected shape (None, 784)

        # ---- Predict ----
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.success("Prediction Complete!")
        st.write("### Predicted Digit")
        st.write(f"# {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")