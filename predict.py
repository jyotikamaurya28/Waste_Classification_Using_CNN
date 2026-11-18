"""Usage: python predict.py path/to/image.jpg"""
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import config


def predict_image(model, img_path):
    img = load_img(img_path, target_size=config.IMG_SIZE)
    arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    p = float(model.predict(arr, verbose=0)[0][0])
    label = "Recyclable" if p > 0.5 else "Organic"
    conf = p if p > 0.5 else 1 - p
    return label, conf, p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py path/to/image.jpg")
    model = load_model(config.MODEL_PATH)
    label, conf, raw = predict_image(model, sys.argv[1])
    print(f"Raw score: {raw:.4f}")
    print(f"Prediction: {label} waste ({conf:.1%} confidence)")