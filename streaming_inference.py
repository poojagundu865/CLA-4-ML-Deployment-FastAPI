import pandas as pd
import joblib
import time

# Load the saved model
model = joblib.load("model.pkl")

# Load the dataset
df = pd.read_csv("train.csv")

# Select 10 unseen passenger records
stream_data = df.loc[721:730, [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

print("Starting Streaming Inference...\n")

# Process one passenger at a time
for index, row in stream_data.iterrows():

    passenger = pd.DataFrame([row])

    prediction = model.predict(passenger)[0]

    if prediction == 1:
        result = "Survived"
    else:
        result = "Did Not Survive"

    print("Passenger", index, "->", result)

    time.sleep(1)

print("\nStreaming inference completed.")