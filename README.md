# ML Model Deployment Strategies and Model Serving using FastAPI

## Project Description

This project demonstrates machine learning model deployment and different inference strategies using the Titanic – Machine Learning from Disaster dataset.

A classification model is trained using the Titanic `train.csv` dataset. The trained pipeline is saved using Joblib as `model.pkl` and reused for:

- Online Inference
- Batch Inference
- Streaming Inference
- FastAPI Model Serving

The project also includes API validation using Pydantic and testing through Swagger UI and a Python client.

---

## Dataset

**Dataset:** Titanic – Machine Learning from Disaster

**Source:** Kaggle

**File used:** `train.csv`

**Dataset Link:** Kaggle Titanic – Machine Learning from Disaster

The dataset is used according to the assignment requirements.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Requests
- Pydantic

---

## Project Structure

```text
CLA-4/
│
├── train.csv
├── train.py
├── model.pkl
│
├── online_inference.py
├── batch_inference.py
├── batch_predictions.csv
├── streaming_inference.py
│
├── app.py
├── client.py
│
├── requirements.txt
├── README.md
├── report.pdf
│
└── screenshots/
    ├── training.png
    ├── online.png
    ├── batch.png
    ├── streaming.png
    ├── health.png
    ├── predict_valid.png
    ├── predict_invalid.png
    └── client.png
```

---

## Tasks Implemented

### Task 1 – Model Training

The Titanic dataset was loaded and preprocessed. A classification model was trained using the labelled dataset.

The dataset was divided into training and testing portions and the model was evaluated using accuracy.

The complete trained pipeline was saved as:

```text
model.pkl
```

### Task 2 – Online Inference

One unseen passenger record was selected and passed to the saved model.

The model generated an individual prediction for the passenger.

### Task 3 – Batch Inference

20 unseen passenger records were selected and processed together.

The predictions were saved in:

```text
batch_predictions.csv
```

### Task 4 – Streaming Inference

10 unseen passenger records were processed one at a time.

A 1-second delay was used between records to simulate streaming inference.

### Task 5 – FastAPI Model Serving

A FastAPI application was created in:

```text
app.py
```

The API provides the following endpoints:

```text
GET  /health
POST /predict
```

Input validation is performed using Pydantic.

The API was tested using Swagger UI at:

```text
/docs
```

Both valid and invalid requests were tested.

A Python client was also created in:

```text
client.py
```

---

## How to Run the Project

### 1. Install the required packages

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This creates:

```text
model.pkl
```

### 3. Run Online Inference

```bash
python online_inference.py
```

### 4. Run Batch Inference

```bash
python batch_inference.py
```

This generates:

```text
batch_predictions.csv
```

### 5. Run Streaming Inference

```bash
python streaming_inference.py
```

### 6. Start the FastAPI Server

```bash
python -m uvicorn app:app --reload
```

Open the Swagger documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

### 7. Test the Python Client

Keep the FastAPI server running and open another terminal:

```bash
python client.py
```

---

## Results

### Model Training

The model was trained successfully.

**Test Accuracy: 81.56%**

The trained pipeline was successfully saved as:

```text
model.pkl
```

### Online Inference

One unseen passenger record was processed successfully and a survival prediction was generated.

### Batch Inference

20 unseen passenger records were processed together.

The results were successfully saved to:

```text
batch_predictions.csv
```

### Streaming Inference

10 unseen passenger records were processed one at a time with a 1-second delay.

### FastAPI

The FastAPI application was successfully tested.

| Test | Result |
|---|---|
| `/health` | HTTP 200 – API is running |
| Valid `/predict` | HTTP 200 – Prediction returned |
| Invalid `/predict` | HTTP 422 – Validation error |
| Python client | HTTP 200 – Prediction returned |

---

## Inference Strategy Comparison

### Online Inference

Online inference processes one record or request at a time and returns the prediction immediately.

### Batch Inference

Batch inference processes multiple records together in a single operation. It is useful when a group of records needs to be processed at once.

### Streaming Inference

Streaming inference processes records continuously as they arrive, one record at a time. In this project, streaming was simulated using 10 records with a 1-second delay.

---

## FastAPI Model Serving

FastAPI is used to expose the trained machine learning model through REST API endpoints.

It provides:

- Fast request handling
- Pydantic-based input validation
- Automatic Swagger API documentation
- Easy integration with Python machine learning models

The API loads the previously saved `model.pkl` instead of training the model again.

---

## Reusing the Same Trained Model

The same saved model is reused for Online, Batch, Streaming and FastAPI inference.

This ensures:

- Consistent preprocessing
- Consistent model behaviour
- Consistent predictions
- No unnecessary retraining
- The deployed model remains the same as the evaluated model

---

## Screenshots

The `screenshots` folder contains evidence of the completed tasks:

- Model training output
- Online inference output
- Batch inference output
- Streaming inference output
- FastAPI `/health` test
- Valid `/predict` test
- Invalid `/predict` validation test
- Python client test

---

## Report

A brief assignment report containing the required observations and conclusion is included as:

```text
report.pdf
```

---

## Conclusion

This project successfully demonstrates the complete machine learning deployment workflow using the Titanic dataset.

The trained model was saved and reused for Online, Batch and Streaming inference. The same model was also deployed through FastAPI and tested using valid and invalid requests through Swagger UI and a Python client.

The project demonstrates how a trained machine learning model can be reused across different inference strategies and exposed as a REST API for practical model serving.