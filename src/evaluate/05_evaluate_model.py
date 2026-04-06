import os
import joblib
import json
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_features_and_target(data, target_column):
    data = pd.read_csv(data)
    data = data.dropna(subset=[target_column])
    
    X= data.drop(columns = [target_column])
    y = data[target_column]
    return X, y

def load_model(model_dir):
    model = mlflow.sklearn.load_model(model_dir)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "test_mae": mean_absolute_error(y_test, preds),
        "test_rmse": mean_squared_error(y_test, preds) ** 0.5,
        "test_r2": r2_score(y_test, preds)
    }

    return preds, metrics

def export_predictions(preds, y_test, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(output_directory, "test_predictions.csv")

    pred_df = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "prediction": preds
    })

    pred_df.to_csv(output_path, index=False)
    return output_path

def export_summary(metrics, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    summary_path = os.path.join(output_directory, "test_metrics.json")

    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent = 2)

    return summary_path

def main(args):
    #mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name = args.run_name):
        mlflow.log_param("evaluation_split", "test")
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("model_path", args.model_path)

        X_test, y_test = load_features_and_target(args.test_data, args.target_column)
        model = load_model(args.model_path)

        preds, metrics = evaluate_model(model, X_test, y_test)

        predictions_path = export_predictions(preds, y_test, args.output_directory)
        summary_path = export_summary(metrics, args.output_directory)

        mlflow.log_metrics(metrics)
        
        print("Test evaluation complete")
        print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default = "portfolio_regression_pipeline")
    parser.add_argument("--run_name", type=str, default = "ridge_test_evaluation")

    args = parser.parse_args()

    main(args)
