# Titanic Survival Prediction

## Objective
This project aims to predict passenger survival on the Titanic using machine learning. Key objectives include:
- Exploratory Data Analysis (EDA) to uncover patterns in survival.
- Handling missing data and preprocessing features (scaling, encoding, imputation).
- Training and evaluating multiple models (Random Forest, Gradient Boosting, Logistic Regression, SVM).
- Hyperparameter tuning using `GridSearchCV` for optimal model performance.
- Evaluating models with metrics like accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset
The dataset (`tested.csv`) contains information about Titanic passengers, including features like class, age, fare, and survival status. Download it from [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file) and place it in the project directory.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/GrowthLink-assignment-.git
   cd GrowthLink-assignment-
   
2. **Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter

##Usage
1. **Launch Jupyter Notebook:

   ```bash
   jupyter notebook

2.Open GrowthLink(assignment).ipynb.

3.Ensure tested.csv is in the same directory.

4.Run all cells sequentially to execute the analysis.

## Code Structure
Data Loading & Inspection: Load the dataset and display basic statistics.

EDA: Visualize survival rates by gender, class, and age distribution.

Preprocessing: Handle missing values, encode categorical features, and split data into train/test sets.

Model Training: Train and compare multiple classifiers.

Hyperparameter Tuning: Optimize models using GridSearchCV.

Evaluation: Report performance metrics and visualize results.

## Key Features
Robust Preprocessing: Uses ColumnTransformer and pipelines for scalable data handling.

Comprehensive EDA: Interactive visualizations with matplotlib and seaborn.

Model Comparison: Evaluates 4 algorithms with cross-validation.

Innovation: Automated hyperparameter tuning and detailed metric reporting.

## Results
In the provided tested.csv, every female passenger has Survived=1, and every male has Survived=0.
Models detected the trivial relationship: Sex=female → Survived=1, Sex=male → Survived=0.
This is why Logistic Regression, Random Forest, and Gradient Boosting achieve 100% accuracy.
SVM’s slightly lower accuracy (97.6%) might stem from hyperparameter settings (e.g., kernel, regularization) or minor noise in the data.
According to me the current results are correct but reflect an artificial scenario.

## Evaluation Criteria
Functionality: Full end-to-end workflow from EDA to model deployment.

Code Quality: Modular, well-commented, and PEP8-compliant.

Innovation: Integration of pipelines and GridSearchCV for optimization.

Documentation: Clear README and in-code explanations.

## Contact: Dharmanshu Singh – dharmanshus1012@gmail.com
