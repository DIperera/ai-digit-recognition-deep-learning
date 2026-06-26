# =========================
# 1. IMPORT LIBRARIES
# =========================
import gradio as gr
import numpy as np
import tensorflow as tf
from google.colab import drive

drive.mount('/content/drive')

model = tf.keras.models.load_model('/content/drive/MyDrive/mnist_model.h5')

# =========================
# 3. PREDICTION FUNCTION
# =========================
def predict(image):
    # convert to grayscale
    image = image.convert("L")

    # resize to MNIST format
    image = image.resize((28, 28))

    # convert to array
    img_array = np.array(image).astype("float32")

    # normalize
    img_array = img_array / 255.0

    # FIX: handle background inversion safely
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    # reshape for model
    img_array = img_array.reshape(1, 28, 28)

    # prediction
    pred = model.predict(img_array, verbose=0)
    result = np.argmax(pred)

    return f"Predicted Digit: {result}"

# =========================
# 4. GRADIO UI
# =========================
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="MNIST Digit Recognition (Stable Version)",
    description="Upload a handwritten digit image (0-9) and get prediction"
)

# =========================
# 5. LAUNCH APP
# =========================
app.launch(share=True, debug=True)