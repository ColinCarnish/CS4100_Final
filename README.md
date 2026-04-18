# MBTA Delay Prediction Tool

## Overview
This project predicts MBTA subway delays using multiple machine learning approaches for two tasks:

1. **Delay Probability Prediction**  
Predict whether a train will be delayed.

2. **Delay Duration Prediction**  
Predict how long a delay will last.

The project compares four models:
- Long Short-Term Memory (LSTM) - prediction
- Hidden Markov Model (HMM) - prediction
- Random Forest Regressor - duration
- Gradient Boosting Regressor - duration

The system includes data preprocessing pipelines, model training, saved model artifacts, and a Streamlit-based frontend for generating predictions.

---

**Goal:**  
Evaluate how different machine learning models perform in predicting both the likelihood and duration of MBTA delays using a limited but well-engineered set of features.

# Project Structure

```text
src/models/lstm.py
src/models/hmm.py
src/models/random_forest.py
src/models/gradient_boosting.py

src/preprocessing/
src/visualizations/

Datasets/
src/models/model_storage

app.py
requirements.txt
README.md
```

---

# Installation

## Prerequisites
Python 3.10+ (we used Python 3.14)

Required libraries:
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- streamlit
- pyyaml

---

## Install Dependencies

Clone the repository:

```bash
git clone https://github.com/ColinCarnish/CS4100_Final.git
cd CS4100_Final
```

Install packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train Models
```bash
python CS4100_Final/src/models/lstm.py  
python CS4100_Final/src/models/hmm_model.py   
python CS4100_Final/src/models/GBM.py  
python CS4100_Final/src/models/forest/forest.py                 
```

---

## Launch Frontend
```bash
streamlit run app/main.py
```

---

# Libraries Used

Core libraries:
```bash
pip install -r requirements.txt
```
---