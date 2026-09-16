# ML_case_study2
# Credit Card Fraud Detection 💳🕵️‍♂️

This project focuses on detecting fraudulent credit card transactions using Machine Learning. Since fraud detection datasets are typically highly imbalanced, this project employs **SMOTE (Synthetic Minority Over-sampling Technique)** for data balancing and **XGBoost** for robust classification. Furthermore, we tune the decision thresholds to optimize the trade-off between Precision and Recall.

## 📌 Project Overview
Fraudulent transactions make up a very small fraction of all credit card transactions. Standard machine learning algorithms often struggle with such heavy class imbalance, biasing towards the majority class. This project tackles this by:
1. **Preprocessing & Scaling:** Standardizing transaction `Amount` and `Time`.
2. **Oversampling:** Generating synthetic samples for the minority (fraud) class using SMOTE.
3. **Modeling:** Training an optimized XGBoost Classifier.
4. **Threshold Tuning:** Evaluating custom decision thresholds to maximize fraud detection (Recall) while maintaining reasonable Precision.
5. **Interpretability:** Extracting and visualizing Feature Importance scores.

## 📊 Dataset
The dataset used contains credit card transactions featuring numerical input variables (resulting from a PCA transformation, typically `V1` to `V28`), along with `Time`, `Amount`, and `Class` (0 for normal, 1 for fraud). 
*(e.g., The Kaggle Credit Card Fraud Detection dataset).*

## 🛠️ Tech Stack & Libraries
- **Python 3**
- **Pandas & NumPy** (Data manipulation)
- **Scikit-Learn** (Preprocessing, Data Splitting, Evaluation Metrics)
- **Imbalanced-Learn** (SMOTE)
- **XGBoost** (Gradient Boosting Classifier)
- **Matplotlib** (Data Visualization)

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Removed null/missing values.
- Scaled the `Amount` and `Time` columns using `StandardScaler` to ensure all features are on a similar scale.
- Dropped the original `Time` and `Amount` columns to rely on the standardized versions.

### 2. Train-Test Split & SMOTE
- Split the dataset into training (80%) and testing (20%) sets using stratified sampling to maintain the original class distribution in the test set.
- Applied **SMOTE** (`k_neighbors=5`) *only on the training data* to balance the classes.

### 3. Model Training (XGBoost)
Trained an XGBoost Classifier with the following key parameters:
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `objective`: binary:logistic
- `eval_metric`: logloss

### 4. Decision Threshold Tuning
Since standard accuracy is misleading for imbalanced data, the model predicts probabilities, and we evaluate multiple decision thresholds (e.g., 0.1, 0.3, 0.5, 0.7, 0.9). 
* Lower thresholds increase **Recall** (catching more frauds but with more false positives).
* Higher thresholds increase **Precision** (fewer false positives but missing some frauds).

### 5. Feature Importance
Extracted the top 10 features driving the model's predictions based on the 'gain' metric to interpret which variables contribute most to identifying fraudulent transactions.

## 🚀 How to Run
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
