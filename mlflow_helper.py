import mlflow
from train_model_helper import evaluate_model

def log_model_with_grid(model, model_name,X_test, y_test, le):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.best_params_)
        model_winner= model.best_estimator_
        metrics, conf_path = evaluate_model(model_winner, X_test, y_test, le.classes_, model_name=model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model_winner, model_name)
        if conf_path:
            mlflow.log_artifact(conf_path)

            
def log_model(model, model_name,X_test, y_test, le):
    with mlflow.start_run(run_name=model_name):
        metrics, conf_path = evaluate_model(model, X_test, y_test, le.classes_, model_name=model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)
        if conf_path:
            mlflow.log_artifact(conf_path)           