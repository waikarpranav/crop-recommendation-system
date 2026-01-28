# 🌾 Crop Recommendation System Using Machine Learning

An AI-powered **crop recommendation system** that suggests the most suitable crop to grow based on soil nutrients and climate conditions.  
It uses machine learning models trained on real-world agricultural data (N, P, K, temperature, humidity, pH, rainfall) to maximize crop yield.

---

## 🔍 Problem Statement

Farmers often decide which crop to grow based on experience or guesswork, which can lead to:
- Low yield
- Poor soil utilization
- Increased risk under changing climate

This project builds a machine learning–based decision support system that recommends the **best crop** for given soil and weather conditions.

---

## 📊 Dataset

The project uses two CSV files:

- `Datasets/crop_data1.csv`  
- `Datasets/crop_data2.csv`  

**Features:**
- `N` – Nitrogen content in soil (kg/ha)  
- `P` – Phosphorus content in soil (kg/ha)  
- `K` – Potassium content in soil (kg/ha)  
- `temperature` – Average temperature (°C)  
- `humidity` – Relative humidity (%)  
- `ph` – Soil pH (0–14)  
- `rainfall` – Annual rainfall (mm)  

**Target:**
- `label` – Name of the crop (e.g., rice, wheat, maize, etc.)

The two datasets are combined into a single DataFrame before training.

---

## 🧠 Machine Learning Approach

Steps followed in the notebook:

1. **Data Loading & Cleaning**
   - Read both CSV files
   - Concatenate them into one dataset
   - Basic checks for missing values and data types

2. **Exploratory Data Analysis (EDA)**
   - Histograms for N, P, K, temperature, humidity, pH, rainfall
   - Crop distribution and number of unique crops

3. **Preprocessing**
   - Features: `['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']`
   - Target: `label`
   - Standardization using `StandardScaler`

4. **Train–Test Split**
   - 80% training, 20% testing

5. **Models Trained**
   - Gaussian Naive Bayes  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - Support Vector Machine (SVM)  
   - Gradient Boosting Classifier  

6. **Evaluation**
   - Accuracy on test data  
   - Classification report for the best model  
   - Feature importance (Random Forest)

7. **Best Model**
   - Random Forest (used in the Streamlit app)

---

## 📂 Project Structure

```text
Crop-Recommendation-System-Using-Machine-Learning/
├── Datasets/
│   ├── crop_data1.csv
│   └── crop_data2.csv
├── Notebook/
│   ├── Crop-recommendation-final.ipynb
│   └── Crop_Recommendation.ipynb
├── app.py                # Streamlit web app
├── requirements.txt
├── README.md
└── .venv/ (virtual environment - not committed)
