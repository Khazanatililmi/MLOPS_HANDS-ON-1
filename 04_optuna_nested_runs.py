import mlflow
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast-cancer-clf")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_eval(params):
    clf = RandomForestClassifier(**params)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    return f1_score(y_val, preds)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "random_state": 42
    }
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        val_f1 = train_and_eval(params)
        mlflow.log_metric("val_f1", val_f1)
        return val_f1

optuna.logging.set_verbosity(optuna.logging.WARNING)

with mlflow.start_run(run_name="optuna-sweep-01"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_f1", study.best_value)
    print(f"[04_optuna_nested_runs] Optuna Sweep finished successfully!")
    print(f"   Best params : {study.best_params}")
    print(f"   Best val_f1 : {study.best_value:.4f}")
