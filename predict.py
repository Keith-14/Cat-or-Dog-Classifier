import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = "cat_dog_classifier.h5"
model = load_model(MODEL_PATH)

INPUT_H = model.input_shape[1]
INPUT_W = model.input_shape[2]


def get_actual_label(img_path):
    folder = os.path.basename(os.path.dirname(img_path)).lower()
    filename = os.path.basename(img_path).lower()

    # Folder name detection
    if "cat" in folder:
        return "Cat"
    if "dog" in folder:
        return "Dog"

    # Filename detection
    if "cat" in filename:
        return "Cat"
    if "dog" in filename:
        return "Dog"

    return "Unknown"


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(INPUT_H, INPUT_W))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    predicted_label = "Dog" if prediction >= 0.5 else "Cat"
    actual_label = get_actual_label(img_path)

    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label}   |   Actual: {actual_label}", fontsize=16)
    plt.show()

    print(f"\nPredicted: {predicted_label}")
    print(f"Actual: {actual_label}")


if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)
