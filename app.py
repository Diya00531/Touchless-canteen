import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model safely
model = tf.keras.models.load_model("food_model.h5", compile=False)

# Load classes
with open("classes.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

# Price list
prices = {
    "Chaat": 30,
    "Maggie": 30,
    "Noodles": 70,
    "Paratha": 50,
    "Samosa": 20
}

# Prediction function
def predict(image):
    if image is None:
        return "No image uploaded"

    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = image.reshape(1, 128, 128, 3)

    pred = model.predict(image)
    idx = np.argmax(pred)

    food = classes.get(idx, "Unknown")
    price = prices.get(food, 0)

    return f"{food} → ₹{price}"

# UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Touchless Canteen AI",
    description="Upload food image to detect item and price"
)

demo.launch()