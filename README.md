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


## Objective:
- The objective is to build a reproducible image classification system that is capable of:
1. Ingesting and Validating image data 
2. Train and Evaluate Deep Learning Models
3. Track Experiments and Register Models
4. Automatically Orchestrate the ML workflow
5. Serve Predictions through an API
6. Package the Application into a Docker Container.
7. Automate Testing and delivery.
8. Monitor The deployed model
9. Support controlled retraining and model promotion.
- Each of these steps will be implemented incrementally and documented.

## Implementation:
### PHASE I: Initial Setup:

- Created Project root directory and initialize git. 
- Created a virtual environment and create git ignore.
- Built the starting point architecture like:
- Install the initially required dependencies.

### PHASE II: Dataset Acquisition And Exploration:
- Downloaded the dataset from the source website and organized it in the `/data/raw` directory.
- Now the raw data is in the directory and the data architecture is:
```text
CervialCancerImageClassification - ML/
│
└── data/
    └── raw/
        └── image/
            │
            ├── ASCUS/
            │
            ├── HSIL/
            │
            ├── LSIL/
            │
            ├── NILM/
            │
            └── SCC/
```
- The meaning of each term here is:
1. NILM: Negative for intraepithelial Lesion or Malignancy.
2. ASCUS: Atypical Squamous cells of Undetermined Significance. 
3. LSIL: Low-Grade Squamous Intraepithelial Lesion.
4. HSIL: High-Grade Squamous Intraepithelial Lesion.
5. SCC: Squamous Cell Carcinoma.

- Lets understand the situation where this is helpful:
```text
We are developing a Deep Learning model that analyzes microscopic cervical cytology images and predicts which of five predefined categories—ASCUS, HSIL, LSIL, NILM, or SCC—the image most likely belongs to, with the potential to support research into AI-assisted cervical cytology classification.
```