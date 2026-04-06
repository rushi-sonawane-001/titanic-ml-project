# 🚢 Titanic Survival Prediction (Machine Learning)

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques. The dataset is taken from the Kaggle competition *Titanic - Machine Learning from Disaster*.

---

## 🎯 Objective

Build a classification model to predict survival (`0 = Not Survived`, `1 = Survived`) based on passenger features.

---

## 📊 Dataset

* Source: Kaggle Titanic Dataset
* Files used:

  * `train.csv` → Training data
  * `test.csv` → Test data

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing

* Handled missing values (`Age`, `Embarked`, `Fare`)
* Selected important features:

  * `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`
* Encoded categorical variables using Label Encoding

### 2. Exploratory Data Analysis (EDA)

* Survival distribution analysis
* Feature correlation analysis
* Visualization using Seaborn & Matplotlib

### 3. Model Building

* Model Used: **LightGBM (LGBMClassifier)**
* Reason: High performance on tabular data and faster training

### 4. Model Evaluation

* Metric: Accuracy Score
* Achieved Accuracy: **~80–85%**

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Joblib

---

## 📁 Project Structure

```
titanic-ml-project/
│── data/
│   ├── train.csv
│   ├── test.csv
│
│── notebooks/
│   └── titanic-analysis.ipynb
│
│── src/
│   ├── train.py
│   ├── predict.py
│
│── models/
│   ├── model.pkl
│   ├── le_sex.pkl
│   ├── le_embarked.pkl
│
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/your-username/titanic-ml-project.git
cd titanic-ml-project
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Train Model

```
python src/train.py
```

### 5. Make Predictions

```
python src/predict.py
```

---

## 📈 Output

* Generates `submission.csv` file
* Contains predicted survival values for test dataset

---

## 💡 Key Learnings

* Data preprocessing and handling missing values
* Feature encoding techniques
* Model training and evaluation
* Working with LightGBM
* Structuring ML projects for real-world use

---

## 🚀 Future Improvements

* Feature engineering (Family size, Title extraction)
* Hyperparameter tuning
* Try advanced models (XGBoost, stacking)
* Deploy as a web app (Streamlit)

---

## 🔗 Author

**Your Name**

* GitHub: https://github.com/your-username

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
