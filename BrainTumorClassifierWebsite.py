import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# Load the pre-trained model
model = keras.models.load_model('BTMC_CNN_Model.h5')

# Define the preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image

# Define the prediction function
def predict(image):
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=-1)
    return prediction

def get_class_explanation(class_index):
    if class_index == 0:
        return 'Meningioma: Meningioma is a type of brain tumor that arises from the meninges, the protective membranes that surround the brain and spinal cord. Meningiomas are usually benign (non-cancerous) tumors, but can sometimes become malignant (cancerous). They are the most common type of brain tumor in adults and typically grow slowly over time. They can cause symptoms such as headaches, seizures, and vision problems, but often do not cause any symptoms and are found incidentally on imaging studies performed for other reasons. Treatment options for meningiomas depend on the size, location, and symptoms of the tumor, but may include surgical removal, radiation therapy, or observation with imaging follow-up.'
    elif class_index == 1:
        return 'Notumor: Notumor is a term used to describe a situation where an imaging study shows an area that is suspected to be a brain tumor, but further testing and evaluation reveals that it is not actually a tumor. This can be due to various factors such as a normal variant of anatomy, an artifact on the imaging, or inflammation. In some cases, a biopsy or other diagnostic test may be needed to confirm the absence of a tumor. If a notumor is identified, no treatment is needed and regular monitoring is recommended to ensure that the lesion does not change over time.'
    elif class_index == 2:
        return 'Pituitary: Pituitary tumors are growths that occur in the pituitary gland, a small organ located at the base of the brain that plays a key role in regulating hormone levels. Pituitary tumors can be either benign (non-cancerous) or malignant (cancerous) and may produce hormones, leading to changes in hormone levels in the body. Symptoms of pituitary tumors can include changes in vision, headaches, changes in menstrual cycle, decreased sex drive, and changes in hormone levels leading to conditions such as hyperthyroidism or hypothyroidism. Treatment options for pituitary tumors depend on the type and size of the tumor, as well as the presence of any symptoms, and may include surgical removal, radiation therapy, medications to control hormone levels, or observation with imaging follow-up.'
    else:
        return 'Glioma: Glioma is a type of brain tumor that arises from the supportive tissues of the brain, known as glial cells. Gliomas can be benign or malignant and can range from low-grade tumors that grow slowly to high-grade tumors that grow quickly and aggressively. Symptoms of gliomas can include headaches, seizures, changes in speech, vision, or coordination, and memory problems. Treatment options for gliomas depend on the location and grade of the tumor, and may include surgical removal, radiation therapy, chemotherapy, or a combination of these treatments. In some cases, it may also be appropriate to closely monitor the tumor with imaging follow-up instead of treating it immediately. The prognosis and survival rate for patients with gliomas vary widely depending on the grade and location of the tumor, as well as the overall health of the patient.'


# Write the main Streamlit app
st.set_page_config(page_title="Brain Tumor Classifier", page_icon=":brain:", layout="wide")

st.title("Brain Tumor Classifier using CNN")

# Add a header and subheader
st.header("Upload MRI Image")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png"])

# Display the image and prediction result
if uploaded_file is not None:
    col1, col2 = st.columns([400,750])
    image = Image.open(uploaded_file)
    col1.image(image, caption="MRI Image",width=350)
    image = preprocess_image(image)
    prediction = predict(image)
    classes = ["Meningioma", "Notumor", "Pituitary", "Glioma"]
    col2.markdown("This MRI image belongs to the following brain tumor class:")
    col2.subheader(classes[prediction[0]])
    col2.markdown("About")
    # col1.st.success("Prediction Completed!")
    explanation = get_class_explanation(prediction[0])
    col2.markdown(explanation)
