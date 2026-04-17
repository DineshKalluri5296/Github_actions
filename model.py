import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score 
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# MLflow setup
mlflow.set_tracking_uri("http://34.234.65.117:5000/")
mlflow.set_experiment("Seattle_weather_prediction_Final")

# Load data
df = pd.read_csv("seattle-weather.csv")
df = df.dropna()

X = df.drop(["date", "weather"], axis=1)
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=1)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Create classification report
    report = classification_report(y_test, pred, zero_division=1)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    # ✅ FIXED artifact logging
    mlflow.log_artifact("classification_report.txt", artifact_path="reports")

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="SeattleWeatherModel26"
    )

    joblib.dump(model, "model.pkl")
    print("Model saved locally as model.pkl")

# Add description
client = MlflowClient()

latest_version = client.get_latest_versions("SeattleWeatherModel26")[0].version

client.update_model_version(
    name="SeattleWeatherModel26",
    version=latest_version,
    description="LogisticRegression model trained on Seattle weather dataset"
)

print(f"Model Version {latest_version} updated successfully!")
