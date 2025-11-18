import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "cat_dog_classifier.h5"
model = load_model(MODEL_PATH)

# Automatically detect correct input size
INPUT_H = model.input_shape[1]
INPUT_W = model.input_shape[2]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(INPUT_H, INPUT_W))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        print("Prediction: DOG")
    else:
        print("Prediction: CAT")


if __name__ == "__main__":
    path = input("Enter image path: ")
    predict_image(path)