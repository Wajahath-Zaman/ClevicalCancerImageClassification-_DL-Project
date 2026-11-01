# Cervical Cancer Image Classification - ML

**Project Type:** Deep Learning, Computer Vision and MLOps
**Dataset:** [https://www.kaggle.com/datasets/robiulhasan0/cervical-cancer-image-dataset]
**Model Framework:** PyTorch
**Development:** VSCode + Python
**Training:** Google Colab
**Experiment Tracking:** MLflow
**Orchestration:** Airflow
**API**: FastAPI
**Containerization:** Docker
**CI/CD:** Github Actions
**Monitoring Prospects:** Data Quality, prediction Behavior, drift and model performance

# Cervical Cytology Image Classification

### An End-to-End Deep Learning and MLOps Project

**A deep learning application for classifying cervical cytology images into five diagnostic categories using ResNet50, PyTorch, FastAPI, Docker, Streamlit and GitHub Actions.**

---

## 1. Introduction

This project started with a simple objective: develop a deep learning model capable of classifying cervical cytology images into five different categories.

However, I wanted to take it beyond simply training a neural network and reporting its accuracy.

The goal was to build an end-to-end machine learning application that covers the entire development lifecycle, from dataset preparation and model experimentation to experiment tracking, API development, containerization, automated testing and cloud deployment.

The project involved several experiments, a few unsuccessful approaches, and some important architectural decisions along the way.

One of the most significant decisions was moving away from a custom CNN architecture towards transfer learning with ResNet50 after evaluating the performance of our initial experiments.

The final model was integrated into a FastAPI application, connected to a Streamlit frontend, containerized using Docker and deployed as two independent services on Render.

The application is now publicly accessible.

**Important:** This is an experimental research and educational project. Its predictions must not be interpreted as medical diagnoses or used to make clinical decisions.

---

## 2. Project Objectives

The primary objectives were to:

1. Prepare and organize a cervical cytology image dataset.
2. Perform dataset inspection and understand the class distribution.
3. Develop and experiment with custom CNN architectures.
4. Evaluate different approaches to improve classification performance.
5. Implement transfer learning using a pretrained ResNet50 architecture.
6. Track experiments, parameters and evaluation metrics using MLflow.
7. Export and manage the trained model artifact.
8. Develop an inference API using FastAPI.
9. Implement automated API testing using Pytest.
10. Containerize the backend using Docker.
11. Build a user-facing application using Streamlit.
12. Integrate the frontend and backend through HTTP requests.
13. Implement continuous integration using GitHub Actions.
14. Deploy the complete application to the cloud.

---

## 3. Dataset Overview

The project uses a cervical cytology image dataset obtained from Kaggle.

The dataset contains images belonging to five different classes.

### Classification Categories

| Class | Description                                          |
| ----- | ---------------------------------------------------- |
| ASCUS | Atypical Squamous Cells of Undetermined Significance |
| HSIL  | High-Grade Squamous Intraepithelial Lesion           |
| LSIL  | Low-Grade Squamous Intraepithelial Lesion            |
| SCC   | Squamous Cell Carcinoma                              |
| NILM  | Negative for Intraepithelial Lesion or Malignancy    |

### Original Class Distribution

The original dataset inspection reported the following image counts:

| Class | Original Image Count |
| ----- | -------------------: |
| ASCUS |                  782 |
| HSIL  |                1,583 |
| LSIL  |                1,339 |
| NILM  |                2,101 |
| SCC   |                  462 |

These are the original class counts observed during dataset inspection, before the final cleaned working dataset was prepared.

The dataset was not perfectly balanced. In particular, SCC and ASCUS had fewer images compared with HSIL and NILM.

This class imbalance was one of the considerations during model development.

### Dataset Preparation

The dataset preparation process involved:

* Inspecting the directory structure.
* Identifying the available image classes.
* Checking image dimensions and formats.
* Examining image distribution across classes.
* Preparing the image data for model development.
* Organizing the dataset into training, validation and testing subsets.

The final cleaned working dataset contained **5,566 images**.

### Dataset Split

A stratified split was used to preserve the approximate class distribution across the subsets.

| Dataset    |    Images | Percentage |
| ---------- | --------: | ---------: |
| Training   |     3,896 |        70% |
| Validation |       835 |        15% |
| Testing    |       835 |        15% |
| **Total**  | **5,566** |   **100%** |

The training dataset was used for model learning, the validation dataset was used for model selection and evaluation during development, and the test dataset was reserved for final performance evaluation.

**Dataset limitation:** Patient or specimen identifiers were not available to establish a patient-level split. Therefore, patient-level separation cannot be verified, and potential correlations between images from the same source cannot be ruled out.

---

## 4. Image Preprocessing

Image preprocessing was an important part of the development process.

The dataset contained images with different dimensions and aspect ratios. Directly resizing every image to a square could distort the original image.

To address this, a custom `ResizeWithPadding` transformation was implemented.

### Preprocessing Pipeline

The preprocessing workflow consisted of:

1. Loading the image.
2. Converting it into RGB format.
3. Resizing the image while preserving its original aspect ratio.
4. Padding the resized image to obtain a square image.
5. Converting the image into a PyTorch tensor.
6. Applying ImageNet normalization.

### Image Configuration

| Parameter         | Value                                |
| ----------------- | ------------------------------------ |
| Input image size  | 512 × 512                            |
| Image format      | RGB                                  |
| Tensor conversion | ToTensor                             |
| Normalization     | ImageNet mean and standard deviation |

Normalization parameters:

**Mean**

* 0.485
* 0.456
* 0.406

**Standard Deviation**

* 0.229
* 0.224
* 0.225

These normalization values were used because the final model was initialized with ImageNet pretrained weights.

---

## 5. Model Development: From Custom CNN to Transfer Learning

This was one of the most important parts of the project.

Rather than immediately selecting a pretrained architecture, I initially experimented with custom CNN models to understand how the dataset behaved and how well a network trained from scratch could learn the classification task.

The initial experiments did not produce satisfactory results.

Instead of continuing to increase model complexity without a clear improvement, I evaluated the results and decided to explore transfer learning.

### 5.1 Experiment 1 — Baseline Custom CNN

The first experiment involved developing a baseline custom CNN architecture.

The objective was to establish a starting point and evaluate whether a CNN trained from scratch could learn useful features from the cervical cytology images.

**Results:**

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy | 40.12% |
| Macro F1-score      | 25.74% |

The results indicated that the model struggled to distinguish between the five classes effectively.

Although the network was learning, its classification performance was not satisfactory.

### 5.2 Experiment 2 — Custom CNN with Class Weighting

The second experiment focused on addressing the class imbalance.

Class weighting was introduced to give greater importance to underrepresented classes during training.

The intention was to reduce the model's tendency to favor classes with more training samples.

However, the results were worse than the baseline.

**Results:**

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy | 25.99% |
| Macro F1-score      |  8.25% |

The model exhibited a prediction collapse, predominantly predicting HSIL.

This experiment demonstrated that introducing class weights does not automatically solve class imbalance. The resulting model behavior must also be examined.

### 5.3 Experiment 3 — Custom CNN with Higher Image Resolution

For the third experiment, the baseline CNN approach was revisited with an increased input resolution.

The input image size was increased from 512 × 512 to 1024 × 1024.

The motivation was to investigate whether preserving more visual information would help the network learn more discriminative features.

**Results:**

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy | 55.69% |
| Macro F1-score      | 45.59% |

The higher-resolution experiment improved the results considerably compared with the initial baseline.

However, the model still did not achieve the classification performance required for the project.

### 5.4 Why I Moved to Transfer Learning

After three custom CNN experiments, it became clear that training a CNN from scratch was not producing sufficiently reliable classification performance.

The experiments revealed several limitations:

* The baseline CNN struggled to learn discriminative features.
* Class weighting resulted in undesirable prediction behavior.
* Increasing image resolution improved performance but did not fully address the classification problem.
* Training a useful feature extractor from scratch with the available dataset was challenging.

This led to the decision to explore transfer learning.

Instead of learning all visual representations from scratch, I decided to use a network pretrained on ImageNet and adapt its learned feature representations to the cervical cytology classification task.

**ResNet50 was selected for this purpose.**

This was not simply a change of architecture. It was an engineering decision based on the results of the previous experiments.

---

## 6. ResNet50 Transfer Learning

The final model was developed using ResNet50 with ImageNet pretrained weights.

ResNet50 is a deep convolutional neural network that uses residual connections to facilitate the training of deeper networks.

The original classification layer was replaced with a new fully connected layer containing five output neurons, corresponding to the five cervical cytology classes.

The model was trained using a two-stage transfer learning strategy.

### 6.1 Stage 1 — Frozen Backbone

During the first stage, the pretrained ResNet50 backbone was frozen.

Only the newly introduced classification head was trained.

The purpose was to adapt the final classification layer to the target dataset without modifying the pretrained feature extractor.

**Configuration:**

| Parameter           | Value               |
| ------------------- | ------------------- |
| Architecture        | ResNet50            |
| Pretrained weights  | ImageNet            |
| Input size          | 512 × 512           |
| Batch size          | 16                  |
| Epochs              | 5                   |
| Optimizer           | Adam                |
| Learning rate       | 0.001               |
| Backbone            | Frozen              |
| Trainable component | Classification head |

**Results:**

| Metric                   |        Result |
| ------------------------ | ------------: |
| Best Validation Loss     |        0.8852 |
| Best Validation Accuracy |        65.27% |
| Best Epoch               |             4 |
| Training Duration        | 24.37 minutes |

The first stage established a stronger starting point compared with the custom CNN experiments.

### 6.2 Stage 2 — Fine-Tuning

After training the classification head, the second stage focused on fine-tuning the pretrained network.

The final residual block (`layer4`) was unfrozen along with the classification head.

Earlier backbone layers remained frozen.

Discriminative learning rates were used to allow the pretrained feature extractor and the classification head to adapt at different learning rates.

**Configuration:**

| Parameter                         | Value     |
| --------------------------------- | --------- |
| Architecture                      | ResNet50  |
| Input size                        | 512 × 512 |
| Batch size                        | 16        |
| Maximum epochs                    | 5         |
| Trainable backbone component      | layer4    |
| Classification head               | Trainable |
| Layer4 learning rate              | 0.0001    |
| Classification head learning rate | 0.001     |

### Fine-Tuning Results

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| ----- | ------------: | ----------------: | --------------: | ------------------: |
| 1     |        0.6223 |            75.92% |          0.6899 |              73.77% |
| 2     |        0.5289 |            80.08% |          0.5854 |              78.56% |
| 3     |        0.4564 |            82.78% |          0.5229 |              80.72% |
| 4     |        0.3953 |            84.52% |          0.5194 |              81.44% |
| 5     |        0.3646 |            85.81% |          0.5178 |              82.40% |

The validation performance improved throughout fine-tuning.

The best validation loss was achieved at epoch 5.

| Metric               |        Result |
| -------------------- | ------------: |
| Best Validation Loss |        0.5178 |
| Validation Accuracy  |        82.40% |
| Training Duration    | 24.77 minutes |

The improvement over the initial custom CNN experiments supported the decision to use transfer learning.

---

## 7. Final Model Evaluation

After completing the two-stage training process, the selected ResNet50 model was evaluated on the held-out test dataset.

The test dataset contained 835 images.

The following metrics were used to assess classification performance:

* Accuracy
* Precision
* Recall
* F1-score
* Macro averages
* Weighted averages

Macro metrics are particularly important in this project because the dataset contains classes with different sample counts.

### 7.1 Overall Test Performance

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 79.52% |
| Macro Precision   | 77.23% |
| Macro Recall      | 75.20% |
| Macro F1-score    | 75.83% |
| Weighted F1-score | 78.97% |

The final model achieved an overall test accuracy of **79.52%**.

### 7.2 Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| ASCUS |    43.04% | 35.42% |   38.86% |      96 |
| HSIL  |    81.42% | 85.19% |   83.26% |     216 |
| LSIL  |    91.22% | 72.97% |   81.08% |     185 |
| SCC   |    91.18% | 89.86% |   90.51% |      69 |
| NILM  |    79.30% | 92.57% |   85.42% |     269 |

### 7.3 Understanding the Results

The model performed differently across the five classes.

SCC achieved an F1-score of 90.51%, while NILM achieved an F1-score of 85.42%.

HSIL and LSIL also showed reasonably strong classification performance.

However, ASCUS remained the weakest-performing class.

Its recall was only 35.42%, meaning that the model failed to correctly identify a substantial proportion of the actual ASCUS samples.

This is an important limitation and should not be overlooked simply because the overall accuracy is close to 80%.

The results demonstrate that overall accuracy alone does not adequately describe the model's behavior on an imbalanced multi-class classification problem.

The model requires further investigation and improvement before it could be considered for any clinical research or practical screening application.

**The held-out test results are reported for final evaluation and were not used as a basis for further model tuning.**

---

## 8. Experiment Tracking with MLflow and DagsHub

Experiment tracking was incorporated to maintain a record of the model development process.

MLflow was used to log experiments, parameters, metrics and model artifacts.

DagsHub was used as the remote MLflow tracking server.

The ResNet50 development process was tracked across two separate runs:

1. ResNet50 Stage 1 — Frozen Backbone.
2. ResNet50 Stage 2 — Fine-Tuning.

The following information was tracked:

* Model configuration.
* Training hyperparameters.
* Learning rates.
* Epoch-wise training loss.
* Epoch-wise training accuracy.
* Validation loss.
* Validation accuracy.
* Best validation metrics.
* Model checkpoint.
* Training history.

The final test evaluation metrics were also incorporated into the experiment tracking workflow.

This made it possible to compare experiments and maintain a record of the model development process rather than relying only on notebook outputs.

**Experiment tracking platform:** MLflow + DagsHub.

---

## 9. Model Artifact Export and Management

After completing model development and evaluation, the selected ResNet50 checkpoint was exported from Google Colab.

The exported artifact contains:

* Model architecture information.
* Model state dictionary.
* Number of classes.
* Class names.
* Input image size.
* Normalization parameters.
* Best training epoch.
* Best validation loss.

The artifact was downloaded and integrated into the local application repository.

**Model artifact:**

`artifacts/resnet50_cervical_cytology_best.pth`

The model is approximately 91 MB.

The backend reconstructs the ResNet50 architecture, loads the saved state dictionary and switches the model into evaluation mode.

The model is loaded onto the available inference device. In the current deployment, CPU inference is used.

---

## 10. Backend Development — FastAPI

Once model development and evaluation were complete, the next step was to make the model accessible through an API.

FastAPI was selected to build the inference backend.

The API is responsible for receiving uploaded images, validating the requests, performing inference and returning the classification results.

### API Endpoints

| Method | Endpoint   | Purpose                       |
| ------ | ---------- | ----------------------------- |
| GET    | `/`        | Root endpoint                 |
| GET    | `/health`  | Health-check endpoint         |
| POST   | `/predict` | Image classification endpoint |

### Prediction Workflow

The prediction workflow follows these steps:

1. The client uploads a cervical cytology image.
2. The API checks the file extension.
3. The API checks whether the uploaded file is empty.
4. Pillow attempts to decode the image.
5. The image is converted into RGB format.
6. The preprocessing pipeline resizes and normalizes the image.
7. The image is converted into a PyTorch tensor.
8. The tensor is passed to the loaded ResNet50 model.
9. The model generates class logits.
10. Softmax converts the logits into class probabilities.
11. The highest-probability class is selected.
12. The predicted class and confidence score are returned as JSON.

### Example Prediction Response

```json
{
    "filename": "sample_image.png",
    "predicted_class": "HSIL",
    "confidence": 94.27,
    "disclaimer": "Experimental classification output, not a medical diagnosis."
}
```

The confidence value represents the model's highest softmax probability. It should not be interpreted as a calibrated probability of disease or diagnostic certainty.

### Error Handling

The API includes handling for:

* Unsupported file formats.
* Empty uploads.
* Invalid or corrupted image files.
* Failed image decoding.

The API returns appropriate HTTP error responses for invalid requests.

---

## 11. Automated Testing — Pytest

Automated testing was introduced to verify the behavior of the FastAPI application.

The tests were implemented using Pytest and FastAPI's TestClient.

The current test suite contains four tests.

| Test                       | Purpose                                      |
| -------------------------- | -------------------------------------------- |
| `test_home_endpoint`       | Checks the root endpoint                     |
| `test_health_endpoint`     | Checks the API health endpoint               |
| `test_predict_endpoint`    | Tests image upload and prediction            |
| `test_invalid_file_format` | Checks rejection of unsupported file formats |

### Test Results

```text
collected 4 items

tests/test_api.py::test_home_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_endpoint PASSED
tests/test_api.py::test_invalid_file_format PASSED

4 passed
```

All four tests passed successfully during local testing.

Automated testing helped verify that the API endpoints and the prediction workflow behaved as expected before containerization and deployment.

---

## 12. Containerization with Docker

Docker was used to package the application and its dependencies into reproducible containers.

The backend and frontend were containerized independently.

This separation keeps the two application components independent and allows them to be built, run and deployed separately.

### Backend Docker Image

The backend image contains:

* Python runtime.
* FastAPI.
* Uvicorn.
* PyTorch.
* Torchvision.
* Pillow.
* Python multipart support.
* Application source code.
* Trained ResNet50 model artifact.

The backend uses CPU-only PyTorch wheels because GPU inference is not required for the current deployment.

**Backend image:**

`cervical-cytology-api:v1`

### Frontend Docker Image

The frontend image contains:

* Python runtime.
* Streamlit.
* Requests.
* Pillow.
* Streamlit application code.

The trained model is not included in the frontend image. All model inference takes place in the backend.

**Frontend image:**

`cervical-cytology-frontend:v1`

### Docker Network

A custom Docker bridge network was created to allow communication between the two containers.

Network name:

`cervical-cytology-network`

The frontend communicates with the backend through its Docker container name.

Backend address inside the Docker network:

`http://cervical-cytology-container:8000`

The frontend container receives the backend address through the `API_URL` environment variable.

### Local Docker Architecture

```text
                  Browser
                     |
                     |
                     v
            Streamlit Container
                 Port 8501
                     |
                     |
             Docker Network
                     |
                     |
                     v
              FastAPI Container
                  Port 8000
                     |
                     |
                     v
                ResNet50
```

This architecture was tested locally, and predictions were successfully generated through the containerized frontend and backend.

---

## 13. Frontend Development — Streamlit

Streamlit was selected to build a simple user-facing interface.

The frontend communicates with the FastAPI backend through HTTP requests.

The interface allows users to upload a cervical cytology image and obtain the model's classification output.

### Frontend Features

* Image upload.
* Support for JPG, JPEG and PNG files.
* Uploaded image preview.
* Classification button.
* Predicted class display.
* Confidence score display.
* Loading indicator during inference.
* Backend connection error handling.
* Request timeout handling.
* API error handling.
* Medical and research disclaimer.

### Frontend-Backend Communication

The frontend sends an HTTP POST request to the backend's `/predict` endpoint.

The uploaded image is transferred as multipart form data.

The backend processes the image and returns the prediction result as JSON.

Streamlit then displays the predicted class and confidence score.

The backend URL is configured through an environment variable:

`API_URL`

This allows the same frontend code to work with the local Docker backend and the publicly deployed backend without modifying the application source code.

---

## 14. Continuous Integration — GitHub Actions

GitHub Actions was integrated to automate the testing process.

The CI workflow is located at:

`.github/workflows/ci.yml`

The workflow is triggered when:

* Code is pushed to the `main` branch.
* A pull request targets the `main` branch.

### CI Workflow

The workflow performs the following operations:

1. Checks out the GitHub repository.
2. Sets up Python 3.13.
3. Installs CPU-only PyTorch and Torchvision.
4. Installs the project dependencies.
5. Executes the Pytest test suite.

### CI Pipeline

```text
             Git Push
                 |
                 v
           GitHub Actions
                 |
                 v
          Checkout Repository
                 |
                 v
            Setup Python
                 |
                 v
       Install CPU-only PyTorch
                 |
                 v
       Install Dependencies
                 |
                 v
            Run Pytest
                 |
          +------+------+
          |             |
          v             v
        PASS          FAIL
          |             |
          v             v
      Successful     Investigate
       CI Run          Errors
```

The workflow executed successfully on GitHub's hosted runner.

All four automated tests passed.

**Note:** The current GitHub Actions workflow implements continuous integration and automated testing. Cloud deployment is managed separately through Render's GitHub integration.

---

## 15. Cloud Deployment — Render

The application was deployed using Render.

Rather than combining the frontend and backend into a single service, both components were deployed independently.

This preserves the separation established during local Docker development.

### Backend Deployment

The FastAPI backend was deployed as a Docker-based Web Service.

During deployment, Render:

1. Cloned the GitHub repository.
2. Checked out the `main` branch.
3. Detected the backend Dockerfile.
4. Built the Docker image.
5. Installed CPU-only PyTorch and the remaining dependencies.
6. Copied the source code and model artifact.
7. Started the Uvicorn server.
8. Loaded the ResNet50 model successfully.

The deployment logs confirmed:

```text
ResNet50 loaded successfully.
Device: cpu
Classes: ['ASCUS', 'HSIL', 'LSIL', 'SCC', 'NILM']
Image size: 512
```

The backend was successfully deployed and its health endpoint was verified.

### Frontend Deployment

The Streamlit frontend was deployed as a separate Docker-based Web Service.

The frontend was configured with the following environment variable:

```text
API_URL=https://clevicalcancerimageclassification-dl.onrender.com
```

This allows the Streamlit application to communicate with the publicly deployed FastAPI backend.

The frontend deployment completed successfully.

### Production Architecture

```text
                     USER
                       |
                       v
             PUBLIC STREAMLIT URL
                       |
                       v
              STREAMLIT SERVICE
                   RENDER
                       |
                       |
                 HTTPS REQUEST
                       |
                       v
              PUBLIC FASTAPI URL
                       |
                       v
                FASTAPI SERVICE
                    RENDER
                       |
                       v
                 IMAGE INPUT
                       |
                       v
                PREPROCESSING
                       |
                       v
                   RESNET50
                       |
                       v
                  PREDICTION
                       |
                       v
                JSON RESPONSE
                       |
                       v
                STREAMLIT UI
                       |
                       v
                     USER
```

### Deployment Links

Both services are publicly accessible.

* Frontend: `https://clevicalcancerimageclassification-1dq9.onrender.com`
* Backend: `https://clevicalcancerimageclassification-dl.onrender.com`

The frontend has been tested using the publicly deployed backend, and image predictions have been generated successfully.

---

## 16. Technology Stack

The following technologies and tools were used throughout the project.

| Category                   | Technology                |
| -------------------------- | ------------------------- |
| Programming Language       | Python 3.13               |
| Deep Learning Framework    | PyTorch                   |
| Computer Vision            | Torchvision, Pillow       |
| Model Architecture         | ResNet50                  |
| Data Processing            | NumPy, Pandas             |
| Model Evaluation           | Scikit-learn              |
| Experiment Tracking        | MLflow                    |
| Remote Experiment Tracking | DagsHub                   |
| Backend Framework          | FastAPI                   |
| ASGI Server                | Uvicorn                   |
| Frontend Framework         | Streamlit                 |
| HTTP Communication         | Requests                  |
| Automated Testing          | Pytest                    |
| Containerization           | Docker                    |
| Continuous Integration     | GitHub Actions            |
| Cloud Deployment           | Render                    |
| Development Environment    | VS Code, Jupyter Notebook |
| Model Training Environment | Google Colab              |

---

## 17. Project Structure

The repository is organized to keep model development, backend implementation, frontend implementation, testing and deployment configuration separate.

```text
CervialCancerImageClassification - ML/
│
├── artifacts/
│   └── resnet50_cervical_cytology_best.pth
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_deep_learning.ipynb
│   └── 03_testing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   ├── predict.py
│   └── preprocessing.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The raw and processed datasets are excluded from Git tracking to avoid committing large collections of image files.

The trained model artifact is maintained in the `artifacts/` directory.

---

## 18. Running the Project Locally

The application can be executed locally using either the Python environment or Docker.

### Prerequisites

* Python 3.13.
* Git.
* Docker Desktop.
* A working internet connection for dependency installation.

### Clone the Repository

```bash
git clone https://github.com/Wajahath-Zaman/ClevicalCancerImageClassification-_DL-Project.git

cd "CervialCancerImageClassification - ML"
```

### Create a Virtual Environment

```bash
python -m venv .cervical-cancer-ml
```

Activate it on Windows:

```bash
source .cervical-cancer-ml/Scripts/activate
```

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Start the FastAPI Backend

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be accessible locally at:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

### Install Frontend Dependencies

Open another terminal from the project root.

```bash
pip install -r frontend/requirements.txt
```

### Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

The frontend will be accessible at:

`http://localhost:8501`

By default, the frontend communicates with the backend at:

`http://localhost:8000`

### Running with Docker

Build the backend image:

```bash
docker build -t cervical-cytology-api:v1 .
```

Create a Docker network:

```bash
docker network create cervical-cytology-network
```

Start the backend:

```bash
docker run -d \
  --name cervical-cytology-container \
  --network cervical-cytology-network \
  -p 8000:8000 \
  cervical-cytology-api:v1
```

Build the frontend image:

```bash
docker build -t cervical-cytology-frontend:v1 ./frontend
```

Start the frontend:

```bash
docker run -d \
  --name cervical-cytology-frontend \
  --network cervical-cytology-network \
  -p 8501:8501 \
  -e API_URL=http://cervical-cytology-container:8000 \
  cervical-cytology-frontend:v1
```

Open:

`http://localhost:8501`

Both services should now be running independently.

---

## 19. Challenges and Lessons Learned

This project involved more than implementing a neural network. Several challenges influenced the final architecture and development decisions.

### 19.1 Custom CNN Performance

The initial custom CNN experiments did not deliver satisfactory classification results.

Three experiments were conducted before moving to transfer learning.

The experiments highlighted the importance of evaluating class-wise performance rather than relying exclusively on accuracy.

### 19.2 Class Imbalance

The dataset contained different numbers of images across the five categories.

An experiment using class weighting resulted in undesirable prediction behavior, demonstrating that imbalance handling needs to be evaluated carefully.

### 19.3 Choosing the Right Architecture

Increasing the input resolution improved the baseline CNN performance, but it did not fully solve the problem.

The move to ResNet50 was based on the results of the earlier experiments and the need for a stronger feature extractor.

### 19.4 Training Environment

Model development and training were performed using Google Colab to access GPU resources.

The final model artifact was then downloaded and integrated into the local application.

This separated computationally intensive model training from local application development.

### 19.5 Separating Frontend and Backend

Initially, the API and frontend were tested locally.

Later, both components were containerized independently and connected through a custom Docker network.

The same separation was maintained during cloud deployment.

### 19.6 Model Deployment

The final application needed to load a relatively large PyTorch model artifact and perform CPU-based inference.

The Docker image was configured to install CPU-only PyTorch dependencies rather than unnecessary GPU packages.

### 19.7 Continuous Integration

GitHub Actions was introduced to automate the API testing process.

This helped establish a repeatable verification step whenever changes were pushed to the main branch.

---

## 20. Current Limitations

Although the application is successfully deployed, there are several limitations that should be acknowledged.

1. **ASCUS performance:** The final model achieved only 35.42% recall for ASCUS on the test dataset.

2. **Dataset limitations:** The dataset does not provide verified patient or specimen identifiers, so patient-level separation cannot be established.

3. **Dataset size and diversity:** The model's generalization to images from different laboratories, equipment and acquisition conditions has not been established.

4. **Confidence interpretation:** Softmax confidence is not equivalent to calibrated diagnostic certainty.

5. **CPU inference:** The deployed backend currently performs inference on CPU. Processing time may increase under concurrent requests.

6. **Clinical validation:** The model has not undergone clinical validation or external validation by medical professionals.

7. **Production monitoring:** Automated monitoring of data drift, prediction distributions and model performance has not yet been implemented.

8. **Retraining workflow:** A controlled automated retraining and model promotion pipeline has not yet been implemented.

For these reasons, this application should be treated strictly as an educational and experimental machine learning project.

It must not be used for patient diagnosis, screening decisions or treatment recommendations.

---

## 21. Future Improvements

There are several opportunities to extend this project.

Potential improvements include:

* Investigating the low ASCUS recall.
* Evaluating additional pretrained architectures.
* Improving generalization across different image acquisition conditions.
* Calibrating model confidence scores.
* Adding prediction history.
* Implementing structured application logging.
* Introducing request-size limits and additional input validation.
* Implementing API performance monitoring.
* Monitoring prediction distributions and potential data drift.
* Establishing a controlled model retraining pipeline.
* Improving deployment observability.
* Exploring model explainability techniques such as Grad-CAM.

These improvements would help make the application more robust and improve the reproducibility and maintainability of the overall system.

---

## 22. Final Thoughts

This project was a valuable learning experience because it required working through the complete lifecycle of a deep learning application.

The development did not go exactly as initially planned.

The first three custom CNN experiments produced disappointing results. Instead of continuing with the same approach, I evaluated the outcomes and moved towards transfer learning with ResNet50.

After obtaining better results, I integrated the model into a FastAPI backend, implemented automated testing, containerized the application and built a separate Streamlit frontend.

The final application was deployed on Render, with both services communicating through a public API.

More importantly, the project helped connect several areas of machine learning engineering that are often studied independently:

* Deep learning and computer vision.
* Model experimentation and evaluation.
* Experiment tracking.
* Model artifact management.
* Backend development.
* Automated testing.
* Docker and containerization.
* Continuous integration.
* Cloud deployment.

The objective was not just to produce a model with a good accuracy score, but to understand how a trained model can be transformed into an application that can be accessed and tested.

That is what this project ultimately demonstrates.

---

**Project Type:** Deep Learning | Computer Vision | MLOps

**Model:** ResNet50

**Framework:** PyTorch

**Backend:** FastAPI

**Frontend:** Streamlit

**Containerization:** Docker

**Experiment Tracking:** MLflow + DagsHub

**CI:** GitHub Actions

**Cloud Platform:** Render

**Status:** Deployed — Experimental Research Application
