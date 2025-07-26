import pickle
import streamlit as st
from PIL import Image
import numpy as np


# model path
MODEL_PATH = 'model/brainTumorMRIClassificationModel-v1.pkl'
# class label path
LABEL_PATH = "model_dataset_lalel.txt"

# load model
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# load class labels
def load_labels():
    model_classification_labels = []
    with open(LABEL_PATH, "r") as f:
        label = f.read()
        for class_label in label.split(","):
            model_classification_labels.append(class_label.strip())
    return model_classification_labels

model = load_model()
class_labels = load_labels()

st.title("Brain Tumor MRI Classification Demo")
st.write("Upload an MRI image to classify the type of brain tumor.")

# file (image) uploader
upload_file = st.file_uploader("choose an MRI image...", type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
    # display image
    image = Image.open(upload_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess image
    image = image.resize((255, 255))
    image_array = np.array(image) /255.0
    image_array = np.expand_dims(image_array, axis=0)

    # predict
    prediction = model.predict(image_array)
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]

    st.markdown(f"### Prediction: `{pred_label}`")
    st.bar_chart(prediction[0])

st.markdown("---")
st.write("Example test images are available")
st.write("You can find them in the `test_images` directory.")
st.write("Feel free to upload your own images for classification.")