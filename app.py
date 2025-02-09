import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MTCNN and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed feature list and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Create upload directory if not exists
os.makedirs('uploads', exist_ok=True)

# Save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

# Extract features using VGGFace
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        st.error("No face detected in the image. Please upload a clearer image.")
        return None
    elif len(results) > 1:
        st.error("Multiple faces detected. Please avoid uploading group images.")
        return None

    # If a single face is detected, process it
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # Resize and preprocess
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    # Preprocess and predict
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

# Recommend closest match
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app
st.title('Which Bollywood Celebrity are You?')

# Warning for users
st.info("Note: Please avoid uploading group images. Ensure the uploaded image is clear.")

uploaded_image = st.file_uploader('Choose an Image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        # Load and display uploaded image
        display_image = Image.open(uploaded_image)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Extract features and recommend
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = "".join (filenames[index_pos]).split('\\')[1].split('_')

            # Display results
            st.header("Your Match")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(display_image, use_column_width=True)
            with col2:
                st.subheader("Seems like "+ predicted_actor[0])
                st.image(filenames[index_pos], use_column_width=True)
 # venv\Scripts\Activate.ps1