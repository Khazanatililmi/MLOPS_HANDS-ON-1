import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

runs = mlflow.search_runs(
    experiment_names=["breast-cancer-clf"],
    filter_string="metrics.val_f1 > 0.80",
    order_by=["metrics.val_f1 DESC"],
    max_results=10,
)

print("[05_search_and_compare] Programmatic search finished successfully.")
if not runs.empty:
    cols = [c for c in ["run_id", "tags.mlflow.runName", "metrics.val_f1", "metrics.val_auc"] if c in runs.columns]
    print(runs[cols].to_string(index=False))
    best_run_id = runs.iloc[0]["run_id"]
    print(f"\nBest run_id selected: {best_run_id}")
else:
    print("No matching runs found.")
