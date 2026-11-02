import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Load the dataset
df = pd.read_csv("train.csv")

# Select 20 unseen passenger records
batch_data = df.loc[701:720, [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]].copy()

# Generate predictions
predictions = model.predict(batch_data)

# Add predictions to the data
batch_data["Prediction"] = predictions

# Convert 0 and 1 into readable results
batch_data["Prediction"] = batch_data["Prediction"].map({
    0: "Did Not Survive",
    1: "Survived"
})

# Save results
batch_data.to_csv("batch_predictions.csv", index=False)

print("Batch inference completed.")
print("\nPredictions:")
print(batch_data)

print("\nSaved results as batch_predictions.csv")