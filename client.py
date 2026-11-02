import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 18,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 227.525,
    "Embarked": "C"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())