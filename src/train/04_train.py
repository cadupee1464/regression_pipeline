import os
import argparse
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# define variables from known columns

TEST_COLS_MASTER = ['CAA ELA Grade 7',
 'CAA ELA Grade 8',
 'CAA Math Grade 7',
 'CAA Math Grade 8',
 'CAA Science Grade 8',
 'CAST Summative Grade 8',
 'CAST Summative Grade HS',
 'CSA Summative Grade HS',
 'Grade 10 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 10 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 11 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 11 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 11-12 ELPAC IA-Listening I',
 'Grade 11-12 ELPAC IA-Reading I',
 'Grade 11-12 ELPAC IA-Speaking I',
 'Grade 4 ELA - Listen/Interpret (FIAB)',
 'Grade 4 ELA - Research: Use Evidence (FIAB)',
 'Grade 4 MATH - Number and Operations in Base Ten (IAB)',
 'Grade 5 ELA - Language and Vocabulary Use (FIAB)',
 'Grade 5 ELA - Listen/Interpret (FIAB)',
 'Grade 5 ELA - Read Literary Texts (IAB)',
 'Grade 5 MATH - Convert Measurements (FIAB)',
 'Grade 5 MATH - Geometry (FIAB)',
 'Grade 5 MATH - Number and Operations - Fractions (IAB)',
 'Grade 5 MATH - Number and Operations in Base Ten (IAB)',
 'Grade 5 MATH - Numerical Expressions (FIAB)',
 'Grade 5 MATH - Operations with Whole Numbers and Decimals (FIAB)',
 'Grade 6 ELA - Editing (FIAB)',
 'Grade 6 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 6 ELA - Language and Vocabulary Use (FIAB)',
 'Grade 6 ELA - Listen/Interpret (FIAB)',
 'Grade 6 ELA - Performance Task - Multivitamins (IAB)',
 'Grade 6 ELA - Read Informational Texts (IAB)',
 'Grade 6 ELA - Read Literary Texts (IAB)',
 'Grade 6 ELA - Research (IAB)',
 'Grade 6 ELA - Research: Analyze and Integrate Information (FIAB)',
 'Grade 6 ELA - Research: Evaluate Information and Sources (FIAB)',
 'Grade 6 ELA - Research: Use Evidence (FIAB)',
 'Grade 6 ELA - Revision (IAB)',
 'Grade 6 ELA - Write and Revise Argumentative Texts (FIAB)',
 'Grade 6 ELA - Write and Revise Narratives (FIAB)',
 'Grade 6 ELPAC IA-Listening I',
 'Grade 6 ELPAC IA-Reading I',
 'Grade 6 ELPAC IA-Writing I',
 'Grade 6 MATH - Algebraic Expressions (FIAB)',
 'Grade 6 MATH - Dependent and Independent Variables (FIAB)',
 'Grade 6 MATH - Divide Fractions by Fractions (FIAB)',
 'Grade 6 MATH - Expressions and Equations (IAB)',
 'Grade 6 MATH - Geometry (FIAB)',
 'Grade 6 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 6 MATH - Multidigit Numbers, Factors, and Multiples (FIAB)',
 'Grade 6 MATH - One-Variable Expressions and Equations (FIAB)',
 'Grade 6 MATH - Performance Task - Cell Phone Plan (IAB)',
 'Grade 6 MATH - Performance Task - Feeding the Giraffe (IAB)',
 'Grade 6 MATH - Rational Number System II (FIAB)',
 'Grade 6 MATH - Ratios and Proportional Relationships (FIAB)',
 'Grade 6 MATH - Statistics and Probability (FIAB)',
 'Grade 6 MATH - The Number System (IAB)',
 'Grade 7 ELA - Brief Writes (IAB)',
 'Grade 7 ELA - Editing (FIAB)',
 'Grade 7 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 7 ELA - Language and Vocabulary Use (FIAB)',
 'Grade 7 ELA - Listen/Interpret (FIAB)',
 'Grade 7 ELA - Performance Task - Mobile Ed Technology (IAB)',
 'Grade 7 ELA - Read Informational Texts (IAB)',
 'Grade 7 ELA - Read Literary Texts (IAB)',
 'Grade 7 ELA - Research (IAB)',
 'Grade 7 ELA - Research: Analyze and Integrate Information (FIAB)',
 'Grade 7 ELA - Research: Evaluate Information and Sources (FIAB)',
 'Grade 7 ELA - Research: Use Evidence (FIAB)',
 'Grade 7 ELA - Revision (IAB)',
 'Grade 7 ELA - Write and Revise Argumentative Texts (FIAB)',
 'Grade 7 ELA - Write and Revise Explanatory Texts (FIAB)',
 'Grade 7 ELA - Write and Revise Narratives (FIAB)',
 'Grade 7 ELPAC IA-Listening I',
 'Grade 7 ELPAC IA-Reading I',
 'Grade 7 ELPAC IA-Speaking I',
 'Grade 7 MATH - Algebraic Expressions and Equations (FIAB)',
 'Grade 7 MATH - Angles, Areas, and Volume (FIAB)',
 'Grade 7 MATH - Equivalent Expressions (FIAB)',
 'Grade 7 MATH - Expressions and Equations (IAB)',
 'Grade 7 MATH - Geometric Figures (FIAB)',
 'Grade 7 MATH - Geometry (IAB)',
 'Grade 7 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 7 MATH - Performance Task - Camping Tasks (IAB)',
 'Grade 7 MATH - Ratios and Proportional Relationships (FIAB)',
 'Grade 7 MATH - Statistics and Probability (FIAB)',
 'Grade 7 MATH - The Number System (FIAB)',
 'Grade 8 ELA - Brief Writes (IAB)',
 'Grade 8 ELA - Edit/Revise (IAB)',
 'Grade 8 ELA - Editing (FIAB)',
 'Grade 8 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 8 ELA - Language and Vocabulary Use (FIAB)',
 'Grade 8 ELA - Listen/Interpret (FIAB)',
 'Grade 8 ELA - Performance Task - Women In Space (IAB)',
 'Grade 8 ELA - Read Informational Texts (IAB)',
 'Grade 8 ELA - Read Literary Texts (IAB)',
 'Grade 8 ELA - Research (IAB)',
 'Grade 8 ELA - Research: Analyze and Integrate Information (FIAB)',
 'Grade 8 ELA - Research: Evaluate Information and Sources (FIAB)',
 'Grade 8 ELA - Research: Use Evidence (FIAB)',
 'Grade 8 ELA - Write and Revise Argumentative Texts (FIAB)',
 'Grade 8 ELA - Write and Revise Explanatory Texts (FIAB)',
 'Grade 8 ELPAC IA-Listening I',
 'Grade 8 ELPAC IA-Listening I Braille',
 'Grade 8 ELPAC IA-Reading I',
 'Grade 8 ELPAC IA-Speaking I',
 'Grade 8 ELPAC IA-Writing I',
 'Grade 8 MATH - Analyze and Solve Linear Equations (FIAB)',
 'Grade 8 MATH - Congruence and Similarity (FIAB)',
 'Grade 8 MATH - Expressions and Equations I (IAB)',
 'Grade 8 MATH - Expressions and Equations II (FIAB)',
 'Grade 8 MATH - Functions (FIAB)',
 'Grade 8 MATH - Geometry (IAB)',
 'Grade 8 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 8 MATH - Performance Task - Baseball Tickets (IAB)',
 'Grade 8 MATH - Proportional Relationships, Lines, and Linear Equations (FIAB)',
 'Grade 8 MATH - The Number System (FIAB)',
 'Grade 8 MATH - Volume of Cylinders, Cones, and Spheres (FIAB)',
 'Grade 9 ELA - Interim Comprehensive Assessment (ICA)',
 'Grade 9 MATH - Interim Comprehensive Assessment (ICA)',
 'Grade 9-10 ELPAC IA-Listening I',
 'Grade 9-10 ELPAC IA-Reading I',
 'Grade 9-10 ELPAC IA-Speaking I',
 'Grade 9-10 ELPAC IA-Writing I',
 'High School CAST IA-Earth and Space Sciences I',
 'High School CAST IA-Life Sciences I',
 'High School CAST IA-Physical Sciences I',
 'High School ELA - Brief Writes (IAB)',
 'High School ELA - Editing (FIAB)',
 'High School ELA - Language and Vocabulary Use (FIAB)',
 'High School ELA - Listen/Interpret (FIAB)',
 'High School ELA - Performance Task - How We Learn (IAB)',
 'High School ELA - Read Informational Texts (IAB)',
 'High School ELA - Read Literary Texts (IAB)',
 'High School ELA - Research (IAB)',
 'High School ELA - Research: Evaluate Information and Sources (FIAB)',
 'High School ELA - Research: Use Evidence (FIAB)',
 'High School ELA - Revision (IAB)',
 'High School ELA - Write and Revise Narratives (FIAB)',
 'High School MATH - Algebra and Functions I (IAB)',
 'High School MATH - Algebra and Functions II (IAB)',
 'High School MATH - Create Equations: Linear and Exponential (FIAB)',
 'High School MATH - Create Equations: Quadratic (FIAB)',
 'High School MATH - Equations and Reasoning (FIAB)',
 'High School MATH - Geometry Congruence (IAB)',
 'High School MATH - Geometry Measurement and Modeling (IAB)',
 'High School MATH - Geometry and Right Triangle Trigonometry (FIAB)',
 'High School MATH - Interpreting Functions (FIAB)',
 'High School MATH - Number and Quantity (FIAB)',
 'High School MATH - Performance Task - Teen Driving Restrictions (IAB)',
 'High School MATH - Seeing Structure in Expressions/Polynomial Expressions (FIAB)',
 'High School MATH - Solve Equations and Inequalities: Linear and Exponential (FIAB)',
 'High School MATH - Solve Equations and Inequalities: Quadratic (FIAB)',
 'Math Summative Grade 11',
 'Math Summative Grade 6',
 'Math Summative Grade 7',
 'Math Summative Grade 8',
 'Middle School CAST IA-Earth and Space Sciences I',
 'Middle School CAST IA-Life Sciences I',
 'Middle School CAST IA-Physical Sciences I',
 'Summative ELPAC Grade 10',
 'Summative ELPAC Grade 11',
 'Summative ELPAC Grade 12',
 'Summative ELPAC Grade 6',
 'Summative ELPAC Grade 7',
 'Summative ELPAC Grade 8',
 'Summative ELPAC Grade 9']

ONE_HOT_COLS_MASTER = [
    'SchoolName',
    'GradeLevelWhenAssessed',
    'LanguageCode',
    'LanguageAltCode',
    'EnglishLanguageAcquisitionStatus'
]

BIN_COLS_MASTER = [
    'MigrantStatus',
    'HispanicOrLatinoEthnicity',
    'AmericanIndianOrAlaskaNative',
    'Asian',
    'BlackOrAfricanAmerican',
    'White',
    'NativeHawaiianOrOtherPacificIslander',
    'TwoOrMoreRaces',
    'Filipino'
]

# load data
def load_and_extract_features(data, target_column):
    df = pd.read_csv(data)
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def build_preprocessor(test_cols, one_hot_cols, bin_cols):
    test_transform = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
    ])

    binary_transform = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant', fill_value='no')),
        ('binarizer', OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
    ])

    one_hot_transform = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('test', test_transform, test_cols),
            ('bin', binary_transform, bin_cols),
            ('one', one_hot_transform, one_hot_cols)
        ],
        remainder='drop'
    )

def get_model_and_param_grid(model_name):
    if model_name == "linear":
        model = LinearRegression()
        param_grid = {}

    elif model_name == "ridge":
        model = Ridge()
        param_grid = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }

    elif model_name == "rf":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_split": [2, 5]
        }

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, param_grid

def build_pipeline(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def build_param_grid():
    return {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

def run_grid_search(X_train, y_train, test_cols, one_hot_cols, bin_cols, model_name, scoring, cv, n_jobs):
    preprocessor = build_preprocessor(test_cols, one_hot_cols, bin_cols)
    model, param_grid = get_model_and_param_grid(model_name)
    pipeline = build_pipeline(preprocessor, model)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)
    return grid

def evaluate_and_log(best_model, X_val, y_val, output_dir):
    preds = best_model.predict(X_val)

    metrics = {
        "val_mae": mean_absolute_error(y_val, preds),
        "val_rmse": mean_squared_error(y_val, preds) ** 0.5,
        "val_r2": r2_score(y_val, preds)
    }

    mlflow.log_metrics(metrics)

    os.makedirs(output_dir, exist_ok=True)
    preds_path = os.path.join(output_dir, "val_predictions.csv")

    pd.DataFrame({
        "actual": y_val.reset_index(drop=True),
        "prediction": preds
    }).to_csv(preds_path, index=False)

    
    return metrics, preds_path

def log_grid_results(grid, output_dir):
    mlflow.log_metric("best_cv_score", grid.best_score_)
    mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})

    os.makedirs(output_dir, exist_ok=True)
    cv_path = os.path.join(output_dir, "cv_results.csv")

    pd.DataFrame(grid.cv_results_).to_csv(cv_path, index=False)
    

    return cv_path

def main(args):
    #with mlflow.start_run(run_name=args.run_name):
    
        mlflow.sklearn.autolog(log_models=False)

        mlflow.set_tag("mlflow.runName", args.run_name)
        mlflow.set_tag("model_type", args.model_name)
        
     
        mlflow.log_param("model_type", args.model_name)
        mlflow.log_param("search_type", "GridSearchCV")
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("evaluation_split", "validation")

        X_train, y_train = load_and_extract_features(args.train_data, args.target_column)
        X_val, y_val = load_and_extract_features(args.val_data, args.target_column)

        test_cols = [col for col in TEST_COLS_MASTER if col in X_train.columns]
        one_hot_cols = [col for col in ONE_HOT_COLS_MASTER if col in X_train.columns]
        bin_cols = [col for col in BIN_COLS_MASTER if col in X_train.columns]

        grid = run_grid_search(
            X_train,
            y_train,
            test_cols,
            one_hot_cols,
            bin_cols,
            args.model_name,
            args.scoring,
            args.cv,
            args.n_jobs
        )

        best_model = grid.best_estimator_

        metrics, preds_path = evaluate_and_log(best_model, X_val, y_val, args.output_directory)
        
        cv_path = log_grid_results(grid, args.output_directory)


        registered_model_name = f"portfolio_regression_{args.model_name}_model"
        mlflow.set_tag("registered_model_name", registered_model_name)


        #mlflow.sklearn.log_model(
        #sk_model = best_model,
        #artifact_path = "model",
        #registered_model_name = registered_model_name)

        os.makedirs(args.model_output, exist_ok=True)

        mlflow.sklearn.save_model(
        sk_model=best_model,
        path=args.model_output
        )

        summary = {
            "best_params": grid.best_params_,
            "best_cv_score": grid.best_score_,
            "validation_metrics": metrics
        }

        summary_path = os.path.join(
            args.output_directory, "training_summary.json"
        )

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        

        print("Run complete")
        print(summary)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)

    parser.add_argument("--experiment_name", type=str, default = "portfolio_regression_pipeline")
    parser.add_argument("--run_name", type=str, default="ridge_gridsearch_and_train")

    parser.add_argument("--scoring", type=str, default="neg_root_mean_squared_error")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--model_output", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        required=True, 
        choices=["linear", "ridge", "rf"]
    )

    args = parser.parse_args()
    main(args)    
