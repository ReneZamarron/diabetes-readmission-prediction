
# Step 2: Load and Explore the Diabetes Dataset

# ✅ 1. Import pandas
import pandas as pd

# ✅ 2. Load the dataset (adjust the path if needed)
df = pd.read_csv(r"C:\Users\admin\Desktop\DATA ANALISIS\301 Machine Learging\Midterm Project\diabetes_130_us_hospitals_for_years_1999_2008\diabetic_data.csv")

# ✅ 3. Show basic info
print("✅ Dataset loaded successfully!")
print("Shape of the data:", df.shape)
print("\nColumn names:\n", df.columns.tolist()[:10])
print("\nFirst 5 rows:")
print(df.head())


#---------------Handle_Missing_Values_and_Drop_Unnecessary_Columns
#---STEP_3:_DATA_PREPROCESSING_AND_FEATURE_ENGINEERING ---

import numpy as np

# Replace '?' with NaN for easier handling
df.replace('?', np.nan, inplace=True)

# Drop columns that are identifiers or not useful for prediction
cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("\n✅ Columns dropped and missing values handled.")
print("Remaining columns:", len(df.columns))

#----------#Convert_Target_Column_'readmitted'_to_Binary_(0/1)
#Create_a_binary_target_variable
df['readmitted_binary'] = df['readmitted'].replace({'>30': 1, '<30': 1, 'NO': 0})

print("\nTarget distribution:")
print(df['readmitted_binary'].value_counts())

#-----#Encode_Categorical_Columns
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Exclude the original target column (we already encoded it)
cat_cols.remove('readmitted')

# Apply Label Encoding to each categorical column
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

print("\n✅ Categorical columns encoded successfully.")


#--------------#Split Dataset for Training and Testing
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = df.drop(columns=['readmitted', 'readmitted_binary'])
y = df['readmitted_binary']

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n✅ Data split into training and test sets.")
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])


#--------------#_Scale Numerical Features (for KNN and Logistic Regression)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Feature scaling completed.")


#---------------#Confirm Final Shape

print("\nFinal shapes:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)

#------------------------------------------
#------------------------------------------
# ------------------ STEP 4: MODEL TRAINING & EVALUATION ------------------

print("\n🔹 STEP 4: Training and evaluating models...")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\n----- {name} -----")
    # Use scaled data for models that need it
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"Accuracy": acc, "AUC": auc}

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------ ROC-AUC CURVE COMPARISON ------------------

print("\n🔹 Plotting ROC-AUC curves...")

plt.figure(figsize=(8, 6))

for name, model in models.items():
    if name in ["Logistic Regression", "KNN"]:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUC']:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison – Diabetes Readmission")
plt.legend()
plt.grid(True)
plt.show()

# ------------------ SUMMARY ------------------

print("\n🔹 Model Summary (Accuracy & AUC):")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['Accuracy']:.4f}, AUC = {metrics['AUC']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['AUC'])
print(f"\n✅ Best model based on AUC: {best_model[0]}")

#-----------------------------
#-----------------------------
# ------------------ STEP 5: MODEL TRAINING ------------------

print("\n🔹 STEP 5: Training the three classification models...")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Initialize models
logreg = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(criterion='gini', random_state=42)

# 1️⃣ Logistic Regression
logreg.fit(X_train_scaled, y_train)
y_pred_log = logreg.predict(X_test_scaled)
y_prob_log = logreg.predict_proba(X_test_scaled)[:, 1]
print("\n🔸 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("AUC:", roc_auc_score(y_test, y_prob_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# 2️⃣ KNN
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
print("\n🔸 KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("AUC:", roc_auc_score(y_test, y_prob_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# 3️⃣ Decision Tree
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
print("\n🔸 Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("AUC:", roc_auc_score(y_test, y_prob_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

print("\n✅ Step 5 completed: All three models trained and evaluated successfully.")

#--------------------------------
#--------------------------------
# ------------------ STEP 6: MODEL EVALUATION & COMPARISON ------------------

print("\n🔹 STEP 6: Evaluating and comparing model performance...")

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Store results in a dictionary for plotting
results = {
    "Logistic Regression": {"y_prob": y_prob_log},
    "KNN": {"y_prob": y_prob_knn},
    "Decision Tree": {"y_prob": y_prob_dt}
}

# Compute AUC values and plot ROC curves
plt.figure(figsize=(8, 6))

for model_name, values in results.items():
    fpr, tpr, _ = roc_curve(y_test, values["y_prob"])
    auc_score = roc_auc_score(y_test, values["y_prob"])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")
    results[model_name]["AUC"] = auc_score

# Plot reference line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison – Diabetes Readmission")
plt.legend()
plt.grid(True)
plt.show()

# Print a summary table
print("\n🔹 Model Performance Summary:")
for model_name, metrics in results.items():
    print(f"{model_name}: AUC = {metrics['AUC']:.4f}")

# Determine best model
best_model = max(results.items(), key=lambda x: x[1]["AUC"])
print(f"\n✅ Best model based on AUC: {best_model[0]} (AUC = {best_model[1]['AUC']:.4f})")

plt.show(block=True)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Step 7: Interpretation & Suggestions

## Model Interpretation

After training and evaluating all three classification algorithms (Logistic Regression, KNN, and Decision Tree) on the Diabetes 130-US Hospitals dataset, the following results were obtained:

| Model | Accuracy | AUC Score | Observation |
|--------|-----------|------------|--------------|
| Logistic Regression | ~0.62 | ~0.65 | Best overall performance |
| KNN | ~0.57 | ~0.59 | Moderate performance, slightly lower generalization |
| Decision Tree | ~0.56 | ~0.56 | Lower performance, potential overfitting |

From the results, **Logistic Regression** achieved the **highest AUC score (~0.65)** and the **best accuracy (~0.62)** among the three models.
This means that Logistic Regression was slightly better at distinguishing between patients who were readmitted and those who were not.

Although the overall accuracy values appear modest, this is **expected in medical datasets**, which are typically complex and imbalanced.
Factors like patient demographics, admission type, and medical history contribute non-linear relationships that are hard to capture with simple models.

---

## Explanation of the Results

- **Logistic Regression:**
  Performed best because it handles high-dimensional, numeric, and categorical data efficiently after scaling.
  It generalizes better and avoids overfitting compared to Decision Trees.

- **KNN:**
  Performance was slightly worse, as it is sensitive to data scaling and struggles when the dataset is large (more than 100,000 rows).
  The curse of dimensionality likely affected its distance-based predictions.

- **Decision Tree:**
  Showed the lowest AUC, possibly due to overfitting on training data.
  Trees can capture complex patterns but tend to lose generalization when not pruned or regularized.

---

## Suggestions for Model Improvement

1. **Feature Engineering:**
   Derive new meaningful features such as combining admission type and time in hospital, or patient age group patterns.

2. **Data Balancing:**
   The dataset is somewhat imbalanced. Applying techniques like **SMOTE (Synthetic Minority Oversampling Technique)** or undersampling could help balance the readmission classes.

3. **Hyperparameter Tuning:**
   Use **GridSearchCV** or **RandomizedSearchCV** to find optimal parameters for each model (e.g., max_depth for Decision Tree, C for Logistic Regression, K for KNN).

4. **Ensemble Models:**
   Try advanced models such as **Random Forest** or **Gradient Boosting (XGBoost)** to improve predictive power.

5. **Cross-Validation:**
   Apply k-fold cross-validation to get more reliable and stable performance estimates.

---

## Final Remark

Among the three models tested, **Logistic Regression** is the most reliable and interpretable model for this medical classification problem.
It provides a good trade-off between performance and explainability, which is critical in healthcare decision-making.



