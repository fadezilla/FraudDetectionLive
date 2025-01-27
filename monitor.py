import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

log_file = "predictions.log"
live_data_file = "live_data.csv"

def monitor_performance():
    try:
        live_data = pd.read_csv(live_data_file)
    except FileNotFoundError:
        print(f"Error: {live_data_file} not found.")
        return

    if "Prediction" not in live_data.columns or "Actual" not in live_data.columns:
        print("Error: Live data must contain 'Prediction' and 'Actual' columns.")
        return

    predictions = live_data['Prediction']
    actuals = live_data['Actual']

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    if recall < 0.85:
        print("Consider lowering the threshold for higher recall.")
    elif precision < 0.70:
        print("Consider raising the threshold for higher precision.")

if __name__ == "__main__":
    monitor_performance()
