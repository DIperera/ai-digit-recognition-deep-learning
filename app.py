from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("mnist_model.h5")


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(img):
    img = img.convert("L")

    # convert to numpy
    img = np.array(img)

    # normalize
    img = img / 255.0

    # FIX polarity (auto-detect instead of always invert)
    if np.mean(img) > 0.5:
        img = 1 - img

    # remove noise
    img = np.where(img > 0.2, img, 0)

    # resize AFTER cleaning
    img = tf.image.resize(img[..., np.newaxis], (28, 28)).numpy()

    return img.reshape(1, 28, 28)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    image = Image.open(io.BytesIO(file.read()))

    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)
    result = int(np.argmax(prediction))

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)