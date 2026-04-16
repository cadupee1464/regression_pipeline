# Azure ML Regression Pipeline: Student Performance Prediction

## Overview

This project implements a fully reproducible, cloud-based machine
learning pipeline designed to predict **ELA summative exam scores**
using student testing data, demographics, and school-level features.

Built with the **Azure Machine Learning SDK**, the pipeline emphasizes
**modular design, experiment tracking, and multi-model
evaluation**—demonstrating production-oriented machine learning
practices.

## Highlights

- Azure ML SDK pipeline (component-based)
- GridSearchCV multi-model training
- MLflow experiment tracking
- MAE / RMSE / R² evaluation
- YAML-based deployment (Azure CLI)

------------------------------------------------------------------------

## Business Problem

Educational institutions need to **identify at-risk students and schools
early** in order to allocate intervention resources effectively.

This project enables:

- Prediction of student performance on ELA summative exams
- Early identification of students likely to underperform
- Data-informed targeting of interventions at both student and school
  levels

------------------------------------------------------------------------

## Solution Summary

- Constructs an **end-to-end regression pipeline** in Azure ML
- Supports **multiple model types** within a single experiment framework
- Uses **GridSearchCV** to determine best-performing models
- Tracks and compares results across runs using MLflow

------------------------------------------------------------------------

## Data

- Source: **Anonymized student testing dataset**

- Features include:

  - Prior assessment scores
  - Demographic indicators
  - School and location attributes

- Target:

  - **ELA Summative Scale Score**

------------------------------------------------------------------------

## Pipeline Architecture

The pipeline is orchestrated using the Azure ML SDK and composed of
modular components:

### Pipeline Steps

1.  **Data Ingestion & Cleaning**

2.  **Train / Validation / Test Split**

3.  **Baseline Model (Median Dummy Regressor)**

4.  **Model Training with GridSearchCV**

    - Linear Regression
    - Ridge Regression
    - Random Forest

5.  **Evaluation & Metric Logging**

Each step is implemented as a **separate script component**, enabling:

- Clear separation of concerns
- Reusability and extensibility
- Scalable orchestration in a cloud environment

------------------------------------------------------------------------

## Modeling Approach

### Models Evaluated

- Dummy Regressor (median baseline)
- Linear Regression
- Ridge Regression (hyperparameter tuned)
- Random Forest (hyperparameter tuned)

### Model Selection

- **GridSearchCV** used to identify optimal hyperparameters
- Final model selected based on validation performance

------------------------------------------------------------------------

## Evaluation Metrics

Models are evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

Metrics are:

- Logged via MLflow
- Stored as artifacts (including `.json` outputs)
- Used for comparison across experiment runs

------------------------------------------------------------------------

##️ Azure ML Integration

### Key Features

- Pipeline orchestration via **Azure ML SDK (v2)**
- Experiment tracking with **MLflow autologging**
- Component-based pipeline architecture
- Reproducible environments via `conda.yml`

### Deployment

- Includes **endpoint deployment configuration via YAML**
- Deployment performed using **Azure CLI**

> ⚠️ Note: The model endpoint is **not actively deployed** due to
> subscription limitations. However, all deployment scripts and
> configurations are included.

------------------------------------------------------------------------

## Outputs & Artifacts

The repository includes:

- Full pipeline graph (Azure ML Studio)
- Model comparison across multiple runs
- Best estimator metrics (Random Forest)
- JSON metric output files
- Endpoint deployment configuration (YAML)
- Screenshots of pipeline execution and results

------------------------------------------------------------------------

## Project Structure

    ```
    regression_pipeline/
    │
    ├── data/
    │   ├── raw/
    │   │   └── portfolio_test_data.csv
    │   ├── interim/
    │   ├── split/
    │
    ├── env/
    │   └── conda.yml              # reproducible environment
    │
    ├── src/                       # all executable pipeline logic
    │   ├── data_prep/
    │   │   └── 01_import_clean.py
    │   │
    │   ├── split/
    │   │   └──02_split_data.py
    │   │
    │   ├── baseline/
    │   │   └── 03_dummy_baseline.py
    │   │
    │   ├── training/
    │   │   └── 04_train.py
    │   │
    │   └── evaluation/
    │       └── 05_evaluate_model.py
    │
    ├── components/                # Azure ML component definitions
    │   ├── 01_import_clean.yml
    │   ├── 02_split_data.yml
    │   ├── 03_dummy_baseline.yml
    │   ├── 04_train.yml
    │   └── 05_evaluate_model.yml
    │
    ├── pipeline/
    │   └── pipeline.py            # pipeline orchestration (SDK)
    │
    ├── deployment/
    │   ├── endpoint.yml           # endpoint config
    │   └── rf_deployment.yml      # model-specific deployment
    │
    ├── config.json
    │
    ├── images/                    
    │   ├── pipeline.png
    │   ├── models_created.png
    │   ├── model_metric_comparisons.png
    │   ├── random_forest_metrics_and_parameters.png
    │   ├── json_model_metrics.png
    │   ├── prediction_output_csv.png
    │   └── endpoint.png
    │
    ├── .gitignore
    └── README.md

    ```

------------------------------------------------------------------------
