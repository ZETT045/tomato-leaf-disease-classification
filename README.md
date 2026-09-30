# Tomato Leaf Disease Classification: Random Forest vs XGBoost Comparison

This project implements a comprehensive machine learning pipeline to classify tomato leaf diseases based on hand-crafted image features. The primary goal is to compare the performance of **Random Forest** and **XGBoost** algorithms in terms of accuracy, efficiency, and statistical significance.

## 📌 Project Overview
The system classifies tomato leaves into 8 different categories (including healthy and various diseases) by extracting a wide array of features from images, rather than using a deep learning approach (CNN), to maintain model interpretability and low computational cost.

## 🛠 Technical Implementation

### 1. Feature Extraction (217 Features)
To capture the unique characteristics of each disease, the following features were extracted:
- **Color Features:** RGB and HSV Color Histograms (capturing intensity distribution).
- **Texture Features:** 
  - **GLCM (Gray-Level Co-occurrence Matrix):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation.
  - **LBP (Local Binary Pattern):** Capturing local texture patterns.
  - **Gabor Filters:** Capturing multi-scale and multi-orientation textures.
- **Shape Features:** Area, Perimeter, Compactness, and Aspect Ratio.

### 2. Machine Learning Pipeline
- **Preprocessing:** Standard Scaling and Label Encoding.
- **Hyperparameter Tuning:** Used `RandomizedSearchCV` to optimize both models.
- **Validation:** 5-Fold Stratified Cross-Validation to ensure model robustness.
- **Statistical Testing:** Conducted a **Paired t-test** to verify if the difference in performance between the two models was statistically significant.

## 📊 Results & Comparison

| Metric | Random Forest | XGBoost | Winner |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 88.58% | **91.17%** | **XGBoost** |
| **Training Time** | 20.48s | **18.78s** | **XGBoost** |
| **Inference Time/Sample** | **0.076ms** | 0.126ms | **Random Forest** |
| **Model Size** | 117.12 MB | **6.58 MB** | **XGBoost** |
| **p-value (T-Test)** | - | **0.0001** | **Significant** |

**Conclusion:** XGBoost is the superior model for this dataset, providing higher accuracy and a significantly smaller memory footprint, with a statistically significant performance lead over Random Forest.

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost opencv-python scikit-image matplotlib seaborn
   ```
3. Run the provided Jupyter Notebooks:
   - `EkstraksiFiturSkripsi.ipynb`: For feature extraction.
   - `PerbandinganAlgoritma.ipynb`: For model training and comparison.

## 📁 Project Structure
- `EkstraksiFiturSkripsi.ipynb`: Image processing and feature engineering.
- `PerbandinganAlgoritma.ipynb`: Model tuning, evaluation, and statistical testing.
- `comparison_results.json`: Final metrics exported from the experiment.
- `confusion_matrix_comparison.png`: Visual comparison of the two models.
