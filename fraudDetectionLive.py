##MAIN MODEL


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load and preprocess the dataset
creditcard_df = pd.read_csv("creditcard.csv")
scaler = StandardScaler()
creditcard_df["NormalizedAmount"] = scaler.fit_transform(creditcard_df["Amount"].values.reshape(-1, 1))
creditcard_df["NormalizedTime"] = scaler.fit_transform(creditcard_df["Time"].values.reshape(-1, 1))
creditcard_df.drop(["Amount", "Time"], axis=1, inplace=True)

X = creditcard_df.drop("Class", axis=1)
y = creditcard_df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance the dataset
smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Save the trained model
model_filename = "fraud_detection_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
