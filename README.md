---
title: Telecom Churn Mlops
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

# 🚀 Telecom Customer Churn Prediction — Enterprise-Grade MLOps System

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141E30,50:243B55,100:1f4037&height=200&section=header&text=Enterprise%20MLOps%20Pipeline&fontSize=38&fontColor=ffffff&animation=fadeIn" />

### Production ML • System Design Thinking • Deployment Ready • Interview Optimized

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge\&logo=scikitlearn)
![MLOps](https://img.shields.io/badge/MLOps-Production--Grade-black?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Inference-red?style=for-the-badge\&logo=streamlit)
![Docker-Ready](https://img.shields.io/badge/Docker-Deployment-blue?style=for-the-badge\&logo=docker)
![CI/CD-Ready](https://img.shields.io/badge/CI/CD-Extensible-success?style=for-the-badge)

</div>

---

# 📌 Executive Overview

An end-to-end Machine Learning system designed to predict telecom customer churn using a **production-oriented architecture**.

The emphasis is not only predictive performance — but:

* Scalability
* Reproducibility
* Deployment separation
* Artifact management
* System extensibility

This project demonstrates ownership of the full ML lifecycle.

---

# 📊 System Architecture

```
Raw Data
   ↓
Data Ingestion → Validation → Transformation
   ↓
Model Training → Evaluation
   ↓
Serialized Artifacts (Model + Preprocessor)
   ↓
Independent Inference Layer (Streamlit)
```

## Architectural Principles

* Stateless inference
* Config-driven execution
* Artifact-based workflow
* Clear training/inference separation
* Extensible for containerization

---

# 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.84  |
| Precision | 0.81  |
| Recall    | 0.78  |
| F1 Score  | 0.79  |

### Interpretation

* Balanced precision/recall tradeoff
* Suitable for churn risk prioritization
* Extendable with threshold tuning for business optimization

---

# 🧩 Project Structure

```
telecom-churn-mlops/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── model_evaluation.py
│
├── config/
│   └── config.yaml
│
├── app.py
├── requirements.txt
└── README.md
```

This mirrors production ML repository standards.

---

# 🐳 Containerization & Deployment Strategy

## Dockerfile (Production Ready Blueprint)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

## Deployment Flow

1. Build Docker image
2. Push to container registry
3. Deploy on VM / Kubernetes cluster
4. Scale inference replicas horizontally

Future Production Enhancements:

* FastAPI serving layer
* Load balancer integration
* Monitoring stack
* Automated CI/CD pipeline

---

# 🧪 Testing & Validation Strategy

* Data validation checks before training
* Schema enforcement capability
* Reproducible training runs
* Isolated inference testing

Future Enhancements:

* Unit tests for pipeline components
* Automated pipeline tests via CI

---

# 📉 Monitoring & Drift Strategy (Design-Level)

In production, the following layers would be integrated:

* Input feature distribution monitoring
* Prediction distribution tracking
* Performance degradation alerts
* Scheduled retraining triggers

The architecture supports adding these layers without refactoring.

---

# 🏆Design Decisions & Tradeoffs

### 1️⃣ Modular vs Monolithic

Chosen modular architecture to enable testability and scalability.
Tradeoff: Slightly higher initial setup complexity.

### 2️⃣ Artifact Serialization

Preprocessor and model saved separately.
Tradeoff: Requires version management in large-scale systems.

### 3️⃣ Streamlit for Inference Layer

Selected for rapid demonstration and UI clarity.
Tradeoff: Not ideal for ultra-low latency APIs — can migrate to FastAPI.

### 4️⃣ Batch-Friendly Design

Pipeline supports full-dataset retraining.
Tradeoff: Not distributed training yet (can integrate Spark later).

---

<div align="center">

### "From Model Building to System Engineering."

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1f4037,50:243B55,100:141E30&height=120&section=footer" />

</div>
