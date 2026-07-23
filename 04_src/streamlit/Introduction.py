import streamlit as st

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Plant Disease Recognition App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main Page Content ---

# Add a title to the app
st.title("Plant Disease Recognition App")

# --- Executive Summary Section ---
st.markdown("---")
st.header("Executive Summary")

st.markdown("""
This application is a powerful tool for **plant disease recognition using deep learning**. It allows users to explore the image dataset, visualize data distributions, and interact with the trained model to understand its predictions.
""")

st.markdown("""
**Project Goal:** To develop a robust deep learning model that accurately identifies various plant types and diseases from images.

**Methods:** The process covers the complete machine learning workflow:
* Data Exploration and Visualization
* Model Training and Evaluation
* Interpreting Predictions with Grad-CAM and SHAP

**Application:** This interactive platform demonstrates an end-to-end ML workflow—from data to deployment—designed to address real-world challenges in agriculture.
""")

# --- About the Data Section ---
st.markdown("---")
st.header("About the Data")

st.markdown("""
#### New Plant Diseases Dataset
- **Source**: [Kaggle](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- **Content**: 38 classes of healthy/diseased plant leaves.
- **Use Case**: Disease classifier for common crops (e.g., tomato, potato).
- **Pros**: Large, well-organized, high-quality images.
- **Cons**: Limited to specific crops.
""")

# --- Page Descriptions Section ---
st.markdown("---")
st.header("Content Overview")

st.markdown("""
Here is a guide to the different sections of this application, which you can navigate to using the sidebar on the left:

**1. Exploration - Data Exploration & Analysis:**
* Provides a deep dive into the dataset used for training the recognition models.
* Explore the dataset's composition, including the total number of images, the different plant species, and the distribution of images across various diseases.
* Use interactive charts to visualize the number of images per plant and disease type.
* Analyze the perceptual brightness of the images to ensure the quality and consistency of the dataset.

**2. Data Visualization - Visualizing the Plant Dataset:**
* Get a firsthand look at the images that the models learn from.
* View a gallery of sample images from the training set, showcasing the diversity of plant species and diseases.
* Use dropdown menus to select a specific plant and disease to view a corresponding image.
* Helps in understanding the visual characteristics the model uses to differentiate between healthy and diseased plants.

**3. Modelling - Initial Model: A Simple CNN:**
* Details the first attempt at building a plant disease classifier using a foundational Convolutional Neural Network (CNN).
* Inspect the model's architecture, layer by layer, and view its training history through interactive plots of accuracy and loss.
* See the model's performance evaluation on the validation set, with a detailed classification report and an interactive confusion matrix.
* Uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image the model focuses on when making a prediction.

**4. Modelling Advanced - Advanced Model: Transfer Learning with MobileNetV2:**
* Introduces a more sophisticated model that utilizes **transfer learning** with the pre-trained MobileNetV2 architecture.
* Explore its structure, training history, and detailed performance evaluation.
* Employs techniques like Grad-CAM and SHAP (SHapley Additive exPlanations) to provide deeper insights into how this complex model arrives at its predictions.

**5. Live Diagnosis - Try the Model Yourself:**
* Upload your own leaf image (or pick a bundled example) and get an instant prediction.
* See the recognised plant, whether it looks healthy or diseased, the model's confidence and the top-3 alternatives.
* Turns the project from a static results showcase into a hands-on, end-to-end demonstration of the trained model.

**6. About Page:**
* Learn more about the project's background and its creation as part of a Data Science program.
* Meet the development team and the project supervisor.
* Find links to the team's professional profiles.
""")

# --- Sidebar Configuration ---
st.sidebar.title("Table of Contents")
st.sidebar.info(
    "Select a page above to explore different aspects of the project."
)
