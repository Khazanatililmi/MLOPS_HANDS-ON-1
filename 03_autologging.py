import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-clf")

# Activate autologging
mlflow.sklearn.autolog()

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run(run_name="logreg-autolog"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    acc = model.score(X_val, y_val)
    print(f"[03_autologging] Autologging completed. Validation accuracy: {acc:.4f}")
