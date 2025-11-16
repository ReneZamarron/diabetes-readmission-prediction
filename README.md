

# Diabetes 130-US Hospitals Readmission Prediction  
**Author:** Rene Zamarron  

---

## 📘 Project Overview  
This project applies machine learning classification to the Diabetes 130-US Hospitals dataset to predict whether a patient will be readmitted to the hospital.  
We reproduce the steps discussed in the research paper **"Explainable Multi-Class Classification of Medical Data"**.

The following models were implemented and compared:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  

---

## 🧩 Dataset  
Dataset Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains over **100,000 clinical records**, and the target variable is **readmitted** (`NO`, `<30`, `>30`).

---

## ⚙️ Requirements  
Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib
 