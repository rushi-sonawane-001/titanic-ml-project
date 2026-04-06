import pandas as pd
import joblib

#Load model and encoders
model = joblib.load("models/model.pkl")
le_sex = joblib.load("models/le_sex.pkl")
le_embarked = joblib.load("models/le_embarked.pkl")

#load test data

test = pd.read_csv("data/test.csv")

#features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# handle missing values
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

#Encode
test['Sex'] = le_sex.transform(test['Sex'])
test['Embarked'] = le_embarked.transform(test['Embarked'])

#prepare data
x_test = test[features]

#predict
predictions = model.predict(x_test)

# save submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv("data/submission.csv", index=False)

print("Submission File Created")