# 🩺 Disease Prediction ML Web App

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> A machine learning web application that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Liver Disease** based on patient clinical data — built with scikit-learn and deployed via Streamlit.

**🔗 Live Demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## 📸 Demo

![App Screenshot](assets/demo.gif)

---

## 🎯 Features

- Predicts 3 diseases: Diabetes, Heart Disease, Liver Disease
- Trained on real-world clinical datasets (UCI / Kaggle)
- Displays prediction confidence score with a probability bar
- Clean, mobile-friendly Streamlit UI
- Model comparison dashboard (Random Forest vs XGBoost vs Logistic Regression)
- Explainability section showing top contributing features (SHAP values)

---

## 🏗️ Project Structure

```
disease-prediction-app/
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── predict.py               # Prediction logic
│   └── utils.py                 # Helper functions
│
├── models/
│   ├── diabetes_model.pkl       # Trained Random Forest model
│   ├── heart_model.pkl          # Trained XGBoost model
│   └── liver_model.pkl          # Trained Logistic Regression model
│
├── data/
│   ├── raw/                     # Original CSV datasets
│   └── processed/               # Cleaned & encoded datasets
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb   # Feature engineering & encoding
│   ├── 03_ModelTraining.ipynb   # Training, evaluation, comparison
│   └── 04_SHAP_Explainability.ipynb  # Feature importance
│
├── assets/
│   └── demo.gif                 # App demo screenshot
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Datasets Used

| Disease | Dataset | Source | Samples | Features |
|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes | [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) | 768 | 8 |
| Heart Disease | Cleveland Heart Disease | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) | 303 | 13 |
| Liver Disease | ILPD (Indian Liver) | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29) | 583 | 10 |

---

## 🤖 Model Performance

### Diabetes Prediction (Random Forest)
| Metric | Score |
|---|---|
| Accuracy | 78.3% |
| Precision | 76.1% |
| Recall | 72.4% |
| F1 Score | 74.2% |
| ROC-AUC | 0.843 |

### Heart Disease Prediction (XGBoost)
| Metric | Score |
|---|---|
| Accuracy | 85.2% |
| Precision | 84.0% |
| Recall | 86.1% |
| F1 Score | 85.0% |
| ROC-AUC | 0.921 |

### Liver Disease Prediction (Logistic Regression)
| Metric | Score |
|---|---|
| Accuracy | 72.1% |
| Precision | 73.4% |
| Recall | 70.8% |
| F1 Score | 72.1% |
| ROC-AUC | 0.787 |

> **Note:** Results based on 80/20 train-test split with 5-fold cross-validation.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Web Framework | Streamlit |
| Data Processing | pandas, NumPy |
| Visualisation | Plotly, matplotlib |
| Deployment | Streamlit Cloud / Render |
| Version Control | Git + GitHub |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/disease-prediction-app.git
cd disease-prediction-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app locally
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

---

## 📦 requirements.txt

```
streamlit==1.32.0
scikit-learn==1.4.0
xgboost==2.0.3
pandas==2.2.0
numpy==1.26.4
shap==0.44.1
plotly==5.19.0
matplotlib==3.8.3
joblib==1.3.2
```

---

## 🧠 How It Works

```
User Input (clinical features)
        │
        ▼
  Data Preprocessing
  (scaling + encoding)
        │
        ▼
  Trained ML Model
  (RF / XGBoost / LR)
        │
        ▼
  Prediction + Confidence Score
        │
        ▼
  SHAP Feature Explanation
  (why this prediction was made)
```

1. The user enters clinical values (e.g. glucose level, BMI, blood pressure) via the Streamlit form
2. Input is scaled using the same `StandardScaler` fitted during training
3. The trained model returns a prediction (Positive / Negative) and probability score
4. SHAP values are computed to explain which features contributed most to this individual prediction

---

## 🖥️ App Walkthrough

### Step 1 — Select a disease
Choose Diabetes, Heart Disease, or Liver Disease from the sidebar.

### Step 2 — Enter patient data
Fill in the clinical input fields. Tooltips explain what each field means.

### Step 3 — Get your prediction
The app displays the prediction result, confidence probability (0–100%), and a feature importance chart.

---

## ⚠️ Disclaimer

This application is built for **educational and portfolio purposes only**. It is **not a medical tool** and should not be used to make any real health decisions. Always consult a qualified healthcare professional for medical advice.

---

## 📁 Notebooks Guide

| Notebook | What it covers |
|---|---|
| `01_EDA.ipynb` | Distribution plots, correlation heatmaps, missing value analysis |
| `02_Preprocessing.ipynb` | Imputation, scaling, encoding, train-test split |
| `03_ModelTraining.ipynb` | Training 3+ models, cross-validation, ROC curves, confusion matrices |
| `04_SHAP_Explainability.ipynb` | SHAP summary plots, force plots, beeswarm plots |

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app/main.py`
5. Click **Deploy**

### Deploy to Render (Alternative)

```bash
# Add a Procfile
echo "web: streamlit run app/main.py --server.port=$PORT" > Procfile
```

Then connect to [render.com](https://render.com) and deploy as a Web Service.

---

## 🔮 Future Improvements

- [ ] Add more diseases (Kidney Disease, Parkinson's)
- [ ] Replace classical ML with a neural network (PyTorch)
- [ ] Add patient history tracking (SQLite backend)
- [ ] REST API endpoint using FastAPI
- [ ] Docker containerisation
- [ ] CI/CD pipeline with GitHub Actions

---

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [github.com/yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the datasets
- [Kaggle](https://www.kaggle.com/) for the Pima Indians Diabetes dataset
- [SHAP library](https://shap.readthedocs.io/) for model explainability
- DeepLearning.AI & Stanford ML Course for the foundational knowledge
