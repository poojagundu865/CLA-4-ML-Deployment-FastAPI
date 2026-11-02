import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Load the dataset
df = pd.read_csv("train.csv")

# Select one unseen passenger record
passenger = df.loc[[700], [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

# Make prediction
prediction = model.predict(passenger)[0]

print("Passenger Details:")
print(passenger)

print("\nPrediction:")

if prediction == 1:
    print("Survived")
else:
    print("Did Not Survive")