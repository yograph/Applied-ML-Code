# Applied Machine Learning Project

## Breast Cancer Detection

This project focuses on breast cancer detection using machine learning techniques. It involves processing medical images, training various models, and evaluating their performance to classify the presence of cancer. The repository contains multiple scripts and modules that handle different aspects of the project, from data preprocessing to model deployment.

## Project Structure

- **Cancer_Detection**: Main directory for model development and evaluation.
  - **main_model**: Contains scripts for model building, training, and evaluation.
    - `base_model.py`: Implements cancer detection using DenseNet201.
    - `convnet_one.py`: Example of using ConvNet architecture.
    - `new_model_type.py`: Specialized model with LSTM features.
    - `training_utils_normal.py`: Training pipeline with various utilities.
    - `training_utils_uncertainty.py`: Advanced training pipeline with uncertainty handling.
    - `test_models.py`: Script to evaluate trained models.
    - `api.py`: FastAPI service for model deployment.
    - **data**: Scripts for data handling and processing.
      - `taking_dicoms.py`: Download and process DICOM files.
      - `processing_to_png.py`: Convert DICOMs to PNGs.
      - `using_glob`: Utilities for file management.
      - `select_images_training.py`: Image selection for training.
  - **features**: Scripts for feature extraction and evaluation.
    - `evaluating.py`: Extended toolkit for evaluation metrics.
- **API**: Contains deployment scripts for model serving.
  - `api.py`: FastAPI script for model deployment and inference.

## Key Features

- **Data Handling**: Comprehensive scripts for downloading, processing, and converting medical images.
- **Model Training**: Supports multiple architectures with customizable training utilities, including class weighting and augmentation.
- **Evaluation**: Detailed evaluation metrics, including confusion matrix, ROC curves, and classification reports.
- **Deployment**: FastAPI-based deployment scripts for serving models with robust error handling and validation.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/applied-ml-code.git
   cd applied-ml-code
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Data Processing**:
   - Use `taking_dicoms.py` to download and process DICOM images.

2. **Model Training**:
   - Run `training_utils_normal.py` or `training_utils_uncertainty.py` to train models.

3. **Evaluation**:
   - Use `test_models.py` to evaluate trained models on test data.

4. **Deployment**:
   - install the requirements
   - this code is just so that you can train the code and test it
   - Deploy models using `api.py` with FastAPI.
   - Start the server with:
     ```sh
     uvicorn api:app --reload --port 8000
     ```
   - To deploy the model, use the second script

5. **Trainig/Testing**:
   - To run the training and testing, run the main.py andpass the arguments --mode and --dataset.

**To run the code**: 
   Run these 2 commands to run the docker
   - docker-compose build --no-cache
   - docker-compose up

   Then run this command to run the streamlit app.
   - http://localhost:8501
   