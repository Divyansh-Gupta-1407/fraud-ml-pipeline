# Fraud Detection ML Pipeline

An end-to-end Machine Learning pipeline for fraud detection built with MLOps best practices.

The project demonstrates a complete ML workflow including data versioning, experiment tracking, model training, inference serving, and containerized deployment.

---

## Features

- End-to-end ML training pipeline
- Data Version Control (DVC)
- Experiment Tracking with MLflow
- FastAPI inference service
- Dockerized deployment
- Modular and scalable project structure
- Reproducible workflows

---

## Tech Stack

### Machine Learning
- Python
- Scikit-Learn
- Pandas
- NumPy

### MLOps
- DVC
- MLflow

### Deployment
- FastAPI
- Docker

---

## System Architecture

<img width="1218" height="333" alt="image" src="https://github.com/user-attachments/assets/8e283f03-52e6-45a5-86cd-b84a87144a93" />


---

## Repository Structure

```text
fraud-ml-pipeline/
│
├── .dvc/
├── artifacts/
├── data/
│   └── processed/
│
├── src/
│
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── mlflow.db
│
├── requirements.txt
├── requirements-inference.txt
│
└── README.md
```
<img width="910" height="268" alt="Screenshot 2026-06-01 205202" src="https://github.com/user-attachments/assets/aca0cfef-ff28-4f2d-99da-f8d9e4210a9d" />

---

## In Action 
### Model Registry

<img width="1875" height="995" alt="Screenshot 2026-06-01 205803" src="https://github.com/user-attachments/assets/9b745085-012b-4a3d-8f89-41a99847f832" />


Key observations:

* Multiple model versions were created and evaluated during experimentation.
* Version 1 and Version 4 emerged as the strongest candidates after validation.
* Both versions were based on the Random Forest algorithm.
* Version 4 was ultimately promoted to the **Production** stage after final evaluation.


<img width="1879" height="888" alt="Screenshot 2026-06-01 210100" src="https://github.com/user-attachments/assets/9c710771-d3e2-4907-9cad-2c9ce83599df" />

### Experiment Tracking
<img width="1871" height="1005" alt="Screenshot 2026-06-01 210203" src="https://github.com/user-attachments/assets/e239bba5-f33c-43d7-b9ea-9207ed1e52cc" />


Key observations:

* MLflow was used to track experiment metrics, parameters, and model artifacts.
* The best-performing run achieved an F1 score of **99.94%**.
* Such a high score should be interpreted carefully due to the severe class imbalance present in the dataset.
* Fraudulent transactions represent only a small fraction of the overall data, making F1 score a more meaningful metric than raw accuracy.


---

## Workflow

### 1. Data Versioning

DVC is used to track datasets and pipeline stages.

```bash
dvc repro
```

---

### 2. Experiment Tracking

MLflow is used to track:

- Parameters
- Metrics
- Models

Launch MLflow UI:

```bash
mlflow ui
```

---

### 3. Model Training

Run the training pipeline:

```bash
python src/main.py
```

---

### 4. Inference API

Start FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://localhost:8000/docs
```

to access Swagger UI.

---

### 5. Docker Deployment

Build image:

```bash
docker build -t fraud-detection .
```

Run container:

```bash
docker run -p 8000:8000 fraud-detection
```

---

## Example Prediction

Request:

```json
{
  "feature_1": 12.3,
  "feature_2": 5.6
}
```

Response:

```json
{
  "prediction": "Fraud"
}
```

---

## Future Improvements

- CI/CD using GitHub Actions
- Automated retraining pipeline
- Model monitoring
- Data drift detection
- Cloud deployment (AWS/GCP/Azure)

---

## Key Learnings

This project demonstrates:

- Building reproducible ML pipelines
- Experiment tracking with MLflow
- Data versioning using DVC
- Containerized ML deployment
- Serving ML models through REST APIs

---

## Author

Divyansh Gupta

LinkedIn: linkedin.com/in/gdivyansh/

GitHub: https://github.com/Divyansh-Gupta-1407
