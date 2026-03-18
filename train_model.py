# ============================
# TRAIN MODEL FOR NUTRISCAN
# ============================

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Convert to numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================
# LOAD DATASET
# ============================

df = pd.read_excel("final_nutriscan_dataset.xlsx")

print("\nDataset preview:")
print(df.head())

y = df["label"].values


flip_count = int(0.07 * len(y))

indices = np.random.choice(len(y), flip_count, replace=False)

for i in indices:
    y[i] = 1 - y[i]





# ============================
# FEATURES AND LABEL
# ============================

X = df[[
    "allergy_conflict",
    "condition_conflict",
    "diet_conflict"
]]

y = df["label"]


# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y

)


# ============================
# TRAIN MODEL
# ============================

model = RandomForestClassifier(
    n_estimators=50,     # reduce trees
    max_depth=3,         # limit learning
    random_state=42
)
model.fit(X_train, y_train)


# ============================
# PREDICTIONS
# ============================

y_pred = model.predict(X_test)


# ============================
# ACCURACY
# ============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"\nAccuracy = {accuracy*100:.2f} %")


# ============================
# CONFUSION MATRIX
# ============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")

print("            Predicted")
print("Actual   Safe  Unsafe")

print(f"Safe      {cm[0][0]}     {cm[0][1]}")
print(f"Unsafe    {cm[1][0]}     {cm[1][1]}")


# ============================
# CLASSIFICATION REPORT
# ============================

print("\nClassification Report:")

print(classification_report(y_test, y_pred))



    # ============================
# GRAPH 1: Confusion Matrix (2 colors)
# ============================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Unsafe"]
)

disp.plot(cmap="Blues")   # 2-color graph

plt.title("NutriScan Confusion Matrix")

plt.show()



# ============================
# GRAPH 2: Feature Importance (2 colors)
# ============================

importance = model.feature_importances_

features = [
    "Allergy Conflict",
    "Condition Conflict",
    "Diet Conflict"
]

plt.bar(features, importance, color=["#AEC6CF", "#6FA8DC", "#3D85C6"])

plt.title("NutriScan Feature Importance")

plt.ylabel("Importance")

plt.show()

# ============================
# GRAPH 3: Training vs Testing Accuracy
# ============================

# Calculate training accuracy
y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_pred)


# Plot graph

labels = ["Training Accuracy", "Testing Accuracy"]

values = [train_accuracy*100, test_accuracy*100]


plt.figure()

bars = plt.bar(labels, values, color=["#93C47D", "#6FA8DC"])


# Add accuracy text on bars

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height,

        f"{height:.2f}%",

        ha="center",

        va="bottom"

    )


plt.title("NutriScan Training vs Testing Accuracy")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.show()

# ============================
# SAVE MODEL
# ============================

joblib.dump(model, "model.pkl")

print("\n✅ Model saved as model.pkl")

print("\nTraining Complete")