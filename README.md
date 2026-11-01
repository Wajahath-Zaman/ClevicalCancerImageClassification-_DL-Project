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

**An End-to-End Deep Learning and MLOps Project**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)

## 1. Project Overview

This project focuses on developing an image classification system capable of classifying cervical cytology images into five predefined categories using deep learning.

The project follows an end-to-end machine learning engineering workflow, covering dataset preparation, model experimentation, transfer learning, experiment tracking, model evaluation, API development, containerization and automated testing.

The ultimate objective is to develop a reproducible and deployable deep learning application that demonstrates both computer vision and MLOps engineering capabilities.

**Important:** This is an experimental research and educational project. Its predictions are not intended for clinical diagnosis or medical decision-making.

## 2. Project Objectives

* Develop a reproducible image preprocessing pipeline.
* Experiment with custom CNN architectures.
* Implement transfer learning using pretrained deep learning architectures.
* Evaluate models using appropriate classification metrics.
* Track experiments using MLflow and DagsHub.
* Export and load trained model artifacts.
* Develop a REST API using FastAPI.
* Containerize the inference application using Docker.
* Implement automated testing and CI/CD.
* Develop an interactive frontend using Streamlit.
* Deploy the application to cloud infrastructure.

## 3. Dataset Information

**Dataset:** Cervical Cancer Image Dataset
**Source:** Kaggle
**Framework:** PyTorch

The dataset contains five classification categories:

| Class | Description                                          |
| ----- | ---------------------------------------------------- |
| ASCUS | Atypical Squamous Cells of Undetermined Significance |
| HSIL  | High-Grade Squamous Intraepithelial Lesion           |
| LSIL  | Low-Grade Squamous Intraepithelial Lesion            |
| NILM  | Negative for Intraepithelial Lesion or Malignancy    |
| SCC   | Squamous Cell Carcinoma                              |

The cleaned working dataset contains **5,566 images**.

### Dataset Distribution

| Class | Original Image Count |
| ----- | -------------------: |
| ASCUS |                  782 |
| HSIL  |                1,583 |
| LSIL  |                1,339 |
| NILM  |                2,101 |
| SCC   |                  462 |

### Dataset Splitting

The dataset was divided using stratified sampling:

| Dataset    | Percentage |    Images |
| ---------- | ---------: | --------: |
| Training   |        70% |     3,896 |
| Validation |        15% |       835 |
| Testing    |        15% |       835 |
| **Total**  |   **100%** | **5,566** |

## 4. Deep Learning Implementation

The model development phase involved experimenting with custom convolutional neural networks before transitioning to transfer learning.

### Custom CNN Experiments

Three custom CNN configurations were evaluated:

* Baseline CNN with 512 × 512 input resolution.
* Class-weighted CNN with 512 × 512 input resolution.
* Baseline CNN with 1024 × 1024 input resolution.

These experiments helped identify the limitations of training a model from scratch with the available dataset and computational resources.

### Transfer Learning: ResNet50

The final model uses a pretrained ResNet50 architecture with ImageNet weights.

The training process was divided into two stages:

**Stage 1 — Feature Extraction**

* Backbone layers frozen.
* Classification head trained independently.
* Five training epochs.
* Learning rate: 0.001.

**Stage 2 — Fine-Tuning**

* Final ResNet layer (`layer4`) unfrozen.
* Classification head remained trainable.
* Discriminative learning rates:

  * Layer 4: 0.0001
  * Classification head: 0.001
* Five additional training epochs.

The best-performing checkpoint was selected based on validation loss.

### Model Configuration

| Parameter          | Value                           |
| ------------------ | ------------------------------- |
| Architecture       | ResNet50                        |
| Framework          | PyTorch                         |
| Pretrained weights | ImageNet                        |
| Input resolution   | 512 × 512                       |
| Number of classes  | 5                               |
| Training strategy  | Transfer Learning + Fine-Tuning |
| Inference device   | CPU                             |

## 5. Model Evaluation

The final ResNet50 model was evaluated using the held-out test dataset.

### Overall Performance

| Metric            | Test Result |
| ----------------- | ----------: |
| Accuracy          |      79.52% |
| Macro Precision   |      77.23% |
| Macro Recall      |      75.20% |
| Macro F1-Score    |      75.83% |
| Weighted F1-Score |      78.97% |

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| ASCUS |    43.04% | 35.42% |   38.86% |
| HSIL  |    81.42% | 85.19% |   83.26% |
| LSIL  |    91.22% | 72.97% |   81.08% |
| SCC   |    91.18% | 89.86% |   90.51% |
| NILM  |    79.30% | 92.57% |   85.42% |

**Evaluation observations:**

* The model achieved an overall test accuracy of 79.52%.
* SCC achieved the highest class-wise F1-score.
* ASCUS remains the weakest-performing category.
* The class-wise performance indicates that overall accuracy alone is insufficient to assess this classification task.

These results are experimental and require further validation before any consideration of clinical applications.

## 6. Experiment Tracking — MLflow and DagsHub

MLflow was integrated to maintain experiment reproducibility and track model development.

Experiment tracking includes:

* Model configuration and hyperparameters.
* Training and validation losses.
* Training and validation accuracy.
* Best validation metrics.
* Model checkpoints and experiment artifacts.
* Final test evaluation metrics.

DagsHub was used as the remote MLflow tracking destination.

## 7. Model Artifact Management

The selected ResNet50 model was exported from Google Colab and transferred to the local development environment.

The exported checkpoint contains:

* Model architecture information.
* Model state dictionary.
* Number of classes.
* Class names.
* Input image resolution.
* Image normalization parameters.
* Best validation epoch.
* Best validation loss.

The model is reconstructed and loaded locally for CPU-based inference.

## 8. Backend Development — FastAPI

A REST API was developed using FastAPI to expose the trained model for image classification.

### Available Endpoints

| Method | Endpoint   | Description              |
| ------ | ---------- | ------------------------ |
| GET    | `/`        | API welcome endpoint     |
| GET    | `/health`  | Application health check |
| POST   | `/predict` | Image classification     |

### Prediction Workflow

1. Client uploads a cervical cytology image.
2. API validates the file format.
3. Image is loaded and converted to RGB.
4. Preprocessing is applied.
5. Image is passed to the ResNet50 model.
6. Model generates class probabilities.
7. Predicted class and confidence are returned.

### Example Prediction Response

```json
{
    "filename": "sample.jpg",
    "predicted_class": "HSIL",
    "confidence": 92.45,
    "disclaimer": "Experimental classification output, not a medical diagnosis."
}
```

The values above are illustrative.

## 9. Automated Testing

Pytest was used to implement automated API tests.

The current test suite covers:

* Home endpoint functionality.
* Health endpoint functionality.
* Image prediction endpoint.
* Invalid file format handling.

**Initial test results:**

```text
4 passed, 1 warning
```

## 10. Docker Containerization

The FastAPI inference application was successfully containerized using Docker.

The container uses:

* Python 3.13 slim base image.
* CPU-only PyTorch.
* Torchvision.
* FastAPI.
* Uvicorn.
* Pillow.
* Exported ResNet50 model artifact.

### Build Docker Image

```bash
docker build -t cervical-cytology-api:v1 .
```

### Run Docker Container

```bash
docker run -d \
    --name cervical-cytology-container \
    -p 8000:8000 \
    cervical-cytology-api:v1
```

### Access API

Homepage:

```text
http://localhost:8000
```

Health endpoint:

```text
http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## 11. Technology Stack

| Category             | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Deep Learning        | PyTorch             |
| Computer Vision      | Torchvision, Pillow |
| Model Architecture   | ResNet50            |
| Experiment Tracking  | MLflow              |
| Remote Tracking      | DagsHub             |
| API Development      | FastAPI             |
| API Server           | Uvicorn             |
| Automated Testing    | Pytest              |
| Containerization     | Docker              |
| Version Control      | Git, GitHub         |
| Training Environment | Google Colab        |
| Local Development    | VS Code             |

## 12. Project Structure

```text
CervialCancerImageClassification/
│
├── artifacts/
│   └── resnet50_cervical_cytology_best.pth
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
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
│   ├── conftest.py
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The dataset is excluded from version control because of its size.

## 13. Planned Development

The following components are planned for subsequent implementation:

* [ ] GitHub Actions CI/CD pipeline.
* [ ] Streamlit interactive frontend.
* [ ] Streamlit and FastAPI integration.
* [ ] Cloud deployment of the inference API.
* [ ] Cloud deployment of the Streamlit frontend.
* [ ] Application monitoring and logging.
* [ ] Model prediction monitoring.
* [ ] Data and model drift monitoring.
* [ ] Controlled model retraining and promotion.

## 14. Limitations

* Model performance varies across the five classification categories.
* ASCUS currently has comparatively weak classification performance.
* The dataset does not provide verified patient-level identifiers for patient-independent splitting.
* The model has not undergone clinical validation.
* Prediction confidence should not be interpreted as clinical certainty.

## 15. Disclaimer

This project is developed exclusively for educational, experimental and research purposes.

It is not a clinically validated medical diagnostic system and must not be used as a substitute for professional medical examination, laboratory testing or clinical judgment.
