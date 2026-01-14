# Machine Learning Practical Work (M2-GI)

**Author:** Donfack Pascal Arthur

## Project Overview

This project is part of the **Machine Learning practical assignments** for the **M2-GI program**.
The main objective is to design, train, and evaluate machine learning models while ensuring proper experiment tracking using **MLflow**.

The work includes:

* Preparing the environment and dependencies.
* Implementing a training pipeline with configurable hyperparameters.
* Tracking experiments, parameters, metrics, and models with **MLflow**.
* Saving trained models for later usage and reproducibility.

---

## Installation & Setup

### 1. Clone the repository (if applicable)

```bash
git clone https://github.com/Tiger-Foxx/DeepLearning1
cd DeepLearning1
```

### 2. Create and activate a virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Training

To launch training with MLflow tracking:

```bash
python train.py
```

The training function supports configurable hyperparameters such as:

* **max\_iter** → Number of iterations for optimization.
* **random\_state** → Seed for reproducibility.
* **C** → Regularization strength.
* **solver** → Optimization algorithm used.

Each run is automatically logged by **MLflow** with:

* Parameters used.
* Evaluation metrics (e.g., accuracy).
* The trained model as an artifact.

---

## Experiment Tracking with MLflow

To visualize experiments in a browser:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore:

* Metrics evolution.
* Comparison between runs.
* Stored models and artifacts.


## Learning Outcomes

Through this assignment, I practiced:

* Setting up reproducible ML experiments.
* Using **MLflow** for experiment tracking and model versioning.
* Understanding hyperparameter tuning and evaluation.
* Structuring machine learning projects in a professional way.

---

## Author

**Donfack Pascal Arthur**
M2-GI – Machine Learning Practical Work
