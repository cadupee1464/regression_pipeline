import argparse
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

print("1. Building ML client")
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print("2. Loading components")
import_clean_component = load_component(source="components/01_import_clean.yml")
split_component = load_component(source="components/02_split_data.yml")
dummy_component = load_component(source="components/03_dummy_baseline.yml")
train_component = load_component(source="components/04_train.yml")
evaluate_component = load_component(source="components/05_evaluate_model.yml")


@pipeline(
    name="portfolio_regression_pipeline",
    description="Import, clean, split, baseline, train, and evaluate on test"
)
def portfolio_pipeline(
    input_data,
    target_column,
    experiment_name,
    dummy_run_name,
    train_run_name,
    eval_run_name,
    scoring,
    cv,
    n_jobs,
    model_name
):
    import_clean_step = import_clean_component(
        input_filename=input_data
    )

    split_step = split_component(
        input_filename=import_clean_step.outputs.output_filename
    )

    dummy_step = dummy_component(
        train_data=split_step.outputs.train_output,
        val_data=split_step.outputs.val_output,
        target_column=target_column,
        experiment_name=experiment_name,
        run_name=dummy_run_name,
    )

    train_step = train_component(
        train_data=split_step.outputs.train_output,
        val_data=split_step.outputs.val_output,
        target_column=target_column,
        model_name=model_name,
        experiment_name=experiment_name,
        run_name=train_run_name,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )

    evaluate_step = evaluate_component(
        test_data=split_step.outputs.test_output,
        target_column=target_column,
        model_path=train_step.outputs.model_output,
        experiment_name=experiment_name,
        run_name=eval_run_name,
    )

    return {
        "cleaned_data": import_clean_step.outputs.output_filename,
        "train_data": split_step.outputs.train_output,
        "val_data": split_step.outputs.val_output,
        "test_data": split_step.outputs.test_output,
        "dummy_outputs": dummy_step.outputs.output_directory,
        "train_outputs": train_step.outputs.output_directory,
        "model_output": train_step.outputs.model_output,
        "evaluation_outputs": evaluate_step.outputs.output_directory,
    }

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="ELA Summative")
    parser.add_argument("--model_name", type=str, required=True, choices=["linear", "ridge", "rf"])

    parser.add_argument("--experiment_name", type=str, default="portfolio_regression_pipeline")
    parser.add_argument("--dummy_run_name", type=str, default="dummy_baseline")
    parser.add_argument("--train_run_name", type=str, default=None)
    parser.add_argument("--eval_run_name", type=str, default="test_evaluation")

    parser.add_argument("--scoring", type=str, default="neg_root_mean_squared_error")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--default_compute", type=str, default="regression-cluster")

    return parser.parse_args()


def main():
    args = parse_args()
    train_run_name = f"{args.model_name}_training"
    
    print("3. Building pipeline job")
    pipeline_job = portfolio_pipeline(
        input_data=Input(
            type="uri_file",
            path=args.input_data,
        ),
        target_column=args.target_column,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        dummy_run_name=args.dummy_run_name,
        train_run_name=train_run_name,
        eval_run_name=args.eval_run_name,
        scoring=args.scoring,
        cv=args.cv,
        n_jobs=args.n_jobs,
    )

    pipeline_job.settings.default_compute = args.default_compute

    print("4. Setting compute / experiment")
    pipeline_job.experiment_name = args.experiment_name
    pipeline_job.display_name = f"{args.model_name}_pipeline"
    
    print("5. Submitting job")
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Submitted job: {returned_job.name}")


if __name__ == "__main__":
    main()