# Predicting Thyroid Cancer Risk using Machine Learning

## 📖 Description

This project explores the use of machine learning for non-invasive, preoperative risk stratification of thyroid malignancy using basic demographic and clinical information. Using a comprehensive dataset of over 210,000 patient records, this study aims to build a predictive model to assist clinicians in making faster, more accurate diagnostic decisions. The project involves extensive exploratory data analysis (EDA), various feature selection techniques, and the implementation and evaluation of several machine learning models in both Python and R.

## ✨ Features

* **Comprehensive EDA**: In-depth analysis of feature distributions, correlations, and associations with the diagnosis.
* **Feature Engineering & Selection**: Applied one-hot encoding, binary mapping, and standardization. Used Lasso regularization, statistical tests (Mann-Whitney U, Chi-squared), and dimensionality reduction (FAMD) to identify the most predictive features.
* **Machine Learning Models**: Trained and evaluated several models, including:
    * Logistic Regression
    * Random Forest
    * XGBoost
    * Naive Bayes
**Clustering Analysis**: Performed K-Means and Hierarchical Clustering (using FAMD) to identify patient profiles and latent patterns in risk factors.
* **Handling Class Imbalance**: Addressed the imbalanced dataset using both SMOTE (in Python) and ROSE (in R) techniques.
* **Web Application**: Includes a Flask app for demonstrating the model in action.

## 📊 Dataset

The dataset used in this project is the **Thyroid Cancer Risk Data**, which contains 15 features from over 210,000 patient records. The features include demographic information (age, gender, country, ethnicity), clinical history (family history, radiation exposure), and key health metrics (smoking, obesity, diabetes, hormone levels). The target variable for prediction is `Diagnosis` (Benign or Malignant).

## 💻 Technologies Used

**Programming Languages:**
* Python
* R

**Python Libraries:**
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `imblearn`
* `xgboost`
* `flask`

**R Libraries:**
* `dplyr`
* `ggplot2`
* `caret`
* `glmnet`
* `pROC`
* `ROSE`
* `FactoMineR`
* `factoextra`
* `cluster`

## 📂 Project Structure

```
thyroid-cancer-prediction/
│
├── app/                  # Contains the Flask web application files
├── data/                 # Contains the raw dataset (thyroid_cancer_risk_data.csv)
├── notebooks/            # Contains Jupyter notebooks for EDA and modeling
├── report/               # Contains the final project report
├── scripts/              # Contains R scripts for analysis and modeling
└── README.md             # This file!
```

## ⚙️ Installation

To run this project, you'll need both Python and R installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/luxan-shanmugalingam/thyroid-cancer-prediction.git](https://github.com/luxan-shanmugalingam/thyroid-cancer-prediction.git)
    cd thyroid-cancer-prediction
    ```

2.  **Set up the Python environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up the R environment:**
    Open R or RStudio and run the following command to install the required packages:
    ```R
    install.packages(c("dplyr", "ggplot2", "caret", "glmnet", "pROC", "ROSE", "FactoMineR", "factoextra", "cluster", "fastDummies", "MLmetrics"))
    ```

## ▶️ Usage

* **Jupyter Notebooks**: Open the notebooks in the `notebooks/` folder to see the Python-based EDA and model training process.
* **R Scripts**: Run the scripts in the `scripts/` folder using R or RStudio to see the R-based analysis.
* **Flask App**: Navigate to the `app/` directory and run the main application file to start the web server.

## 📈 Results

The project found that **Random Forest** and **XGBoost** were the best-performing models. The most predictive features for thyroid cancer malignancy were identified as **Country, Ethnicity, Family History, Radiation Exposure, and Iodine Deficiency**. The final models show promise as a supportive tool for clinicians in early risk assessment.

## 📜 References

1.  **Dataset**: [Thyroid Cancer Risk Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/proytosh/thyroid-cancer-risk-prediction-dataset)
2.  Clark, E., Price, S., Lucena, T., Haberlein, B., Wahbeh, A., & Seetan, R. (2024). Predictive Analytics for Thyroid Cancer Recurrence: A Machine Learning Approach. *Knowledge, 4*, 557-570.
3.  Salman, K., & Sonuç, E. (2021). Comparative Analysis of Machine Learning Models for Thyroid Cancer Recurrence Prediction. *J. Phys.: Conf. Ser., 1963*, 012140.