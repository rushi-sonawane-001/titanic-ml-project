import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

#load the data
train = pd.read_csv("data/train.csv")

#features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

#handels missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

#Encode categorical data
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

train['Sex'] = le_sex.fit_transform(train['Sex'])
train['Embarked'] = le_embarked.fit_transform(train['Embarked'])

# define x and y
x = train[features]
y = train['Survived']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Model training (LightGBM)
model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train)

# Evaluation
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save Model + Encoders
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sex, "models/le_sex.pkl")
joblib.dump(le_embarked, "models/le_embarked.pkl")

print("Model and encoders saved!")