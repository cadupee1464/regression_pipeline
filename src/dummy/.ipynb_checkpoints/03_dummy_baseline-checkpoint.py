import os
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_split_cols (data, target_column):
    data = pd.read_csv(data)
    data = data.dropna(subset=[target_column])

    X = data.drop(columns = [target_column])
    y = data[target_column]

    return X, y

def median_dummy_make_and_predict (train_set_x, train_set_y, val_set_x):
    model = DummyRegressor(strategy = 'median')
    model.fit(train_set_x, train_set_y)

    preds = model.predict(val_set_x)

    return model, preds

def get_and_log_metrics(preds, val_set_y):
    metrics = {
            "val_mae": mean_absolute_error(val_set_y, preds),
            "val_rmse": mean_squared_error(val_set_y, preds) ** 0.5,
            "val_r2": r2_score(val_set_y, preds),
        }

    mlflow.log_metrics(metrics)

    return metrics

def export_and_log_predictions(preds, val_set_y, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, "dummy_predictions.csv")

    pred_df = pd.DataFrame({
        "actual": val_set_y.reset_index(drop=True),
        "prediction": preds
    })

    pred_df.to_csv(output_path, index=False)
    return output_path

def main (args):
    #mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name = args.run_name):
        mlflow.sklearn.autolog()

        mlflow.log_param("model_type", "DummyRegressor")
        mlflow.log_param("dummy_strategy", "median")
        mlflow.log_param("evaluation_split", "validation")
        mlflow.log_param("target_column", args.target_column)

        train_set_X, train_set_y = load_and_split_cols(args.train_data, args.target_column)
        val_set_X, val_set_y = load_and_split_cols(args.val_data, args.target_column)

        model, preds = median_dummy_make_and_predict(train_set_X, train_set_y, val_set_X)

        metrics = get_and_log_metrics(preds, val_set_y)
        output_path = export_and_log_predictions(preds, val_set_y, args.output_directory)

        print("Run complete")
        print(f"File save to {output_path}")
        print(metrics)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default = "portfolio_regression_pipeline")
    parser.add_argument("--run_name", type=str, default = "dummy_baseline")

    args = parser.parse_args()
    main(args)