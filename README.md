# Plant Disease Recognition using Deep Learning

### Problem Statement
This project addresses the challenge of accurately identifying various plant diseases and healthy leaves from images using deep learning. The goal is to develop a robust image classification model that can distinguish between 38 different classes of plant diseases and healthy leaves, with a secondary objective of building a user-friendly application to showcase the results.

### Data Source
The project uses the **New Plant Diseases Dataset**, which is available on Kaggle. It contains images of 38 classes of healthy and diseased plant leaves. The dataset is well-organized, with separate sub-folders for each class, making it suitable for image classification tasks.

### Tech Stack
* **Python**: Core programming language.
* **TensorFlow/Keras**: For building and training deep learning models (Convolutional Neural Networks and transfer learning with MobileNetV2).
* **Streamlit**: For creating the interactive web application and dashboard.
* **Pandas**: For data handling and manipulation.
* **Matplotlib & Seaborn**: For creating visualizations like bar charts, loss curves, and confusion matrices.
* **SHAP & Grad-CAM**: For model interpretability and explaining model predictions.

### Key Findings
The project successfully developed a highly accurate deep learning model using **transfer learning** with the **MobileNetV2** architecture. The model was trained in two phases: an initial training phase with frozen weights, followed by a fine-tuning phase where the top layers were made trainable.

This approach resulted in strong generalization capabilities, as evidenced by the convergence of training and validation accuracy curves near 100%. The model's reasoning was validated using interpretability techniques like Grad-CAM and SHAP, which showed the model correctly focused on relevant features like leaf texture and vein patterns to make predictions.

### Live Diagnosis Feature
The Streamlit app includes an interactive **Live Diagnosis** page: upload a leaf image (or pick a bundled example) and the app predicts the plant, whether it is healthy or diseased, the model confidence and the top-3 alternatives. The trained MobileNetV2 model is loaded at runtime from the Hugging Face Hub (configured via the app secrets `hf_model_repo` / `hf_model_filename`); until it is connected, the page runs in a clearly labelled demo mode with simulated predictions.

### Deployed App
https://plant-recognition-app.streamlit.app/ 

### How to Run
1.  Clone this repository.
2.  Install the required libraries listed in the `requirements.txt` file.
3.  Ensure the dataset is structured correctly in the `original_data` directory as described in the `1_Exploration.py` file.
4.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run Introduction.py
    ```
5.  The app will open in your browser, allowing you to interact with the dashboard and explore the project.
