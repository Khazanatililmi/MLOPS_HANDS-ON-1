import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-clf")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}

with mlflow.start_run(run_name="rf-baseline"):
    mlflow.set_tags({"stage": "baseline", "author": "luky"})
    mlflow.log_params(params)
    
    model = RandomForestClassifier(**params).fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]
    
    val_f1 = f1_score(y_val, proba > 0.5)
    val_auc = roc_auc_score(y_val, proba)
    
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("val_auc", val_auc)
    
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_val[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print(f"[01_basic_logging] Run finished successfully.")
    print(f"   val_f1  : {val_f1:.4f}")
    print(f"   val_auc : {val_auc:.4f}")
