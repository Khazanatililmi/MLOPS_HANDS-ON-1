import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
name = "breast-cancer-clf"

# Search specifically for runs that logged a model (e.g. rf-baseline or logreg-autolog)
runs = mlflow.search_runs(
    experiment_names=["breast-cancer-clf"],
    filter_string="tags.mlflow.runName = 'rf-baseline'",
    order_by=["metrics.val_f1 DESC"],
    max_results=1,
)

if not runs.empty:
    best_run_id = runs.iloc[0]["run_id"]
    print(f"[06_model_registry] Registering model from best run_id ({runs.iloc[0]['tags.mlflow.runName']}): {best_run_id}")
    
    # 1. Registrasi model dari run terbaik
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", name)
    
    # 2. Anotasi & tandai sebagai challenger
    client.update_model_version(name, mv.version, description="RandomForest model for Breast Cancer Classification")
    client.set_model_version_tag(name, mv.version, "validation_status", "passed")
    client.set_registered_model_alias(name, "challenger", mv.version)
    
    # 3. Promosikan ke champion
    client.set_registered_model_alias(name, "champion", mv.version)
    
    # 4. Memuat via alias @champion dan melakukan prediksi
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    
    X, y = load_breast_cancer(return_X_y=True)
    preds = model.predict(X[:5])
    print(f"[06_model_registry] Successfully registered & loaded model via @champion!")
    print(f"   Registered Version : v{mv.version}")
    print(f"   Sample predictions : {preds}")
else:
    print("No matching run with model found to register.")
