import argparse
import json
import os
import shutil
from argparse import Namespace
from typing import Any

import mlflow
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential
from config import get_config_dict, load_azure_service_principal_environment_vars


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-download-path", type=str)
    parser.add_argument("--preprocessor-download-path", type=str)
    args = parser.parse_args()
    return args


def setup_download_dirs(model_download_path: str, preprocessor_download_path: str):
    if os.path.isdir(model_download_path):
        shutil.rmtree(model_download_path)
    if os.path.isdir(preprocessor_download_path):
        shutil.rmtree(preprocessor_download_path)
    os.makedirs(model_download_path, exist_ok=True)
    os.makedirs(preprocessor_download_path, exist_ok=True)


def get_model_metrics(
    ml_client: MLClient, model_name: str, model_version: str
) -> dict[str, dict[str, Any]]:
    model = ml_client.models.get(model_name, model_version)
    job_name = model.job_name
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    metrics = ["train_mae", "test_mae", "train_rmse", "test_rmse", "r_precision_at_5"]
    metrics_dict: dict[str, dict[str, Any]] = {}
    for metric_name in metrics:
        retrieved_metric = client.get_metric_history(job_name, metric_name)
        retrieved_metric_as_dict = dict(retrieved_metric[0])
        metrics_dict[retrieved_metric_as_dict["key"]] = {
            key: val for key, val in retrieved_metric_as_dict.items() if key != "key"
        }
    return metrics_dict


def main():

    args = parse_args()
    model_download_path = args.model_download_path
    preprocessor_download_path = args.preprocessor_download_path
    setup_download_dirs(model_download_path, preprocessor_download_path)

    load_azure_service_principal_environment_vars("config.yaml", "config_secret.yaml")
    config_dict = get_config_dict("config.yaml")

    model_name = config_dict["model"]["name"]
    model_version = config_dict["model"]["version"]
    preprocessor_name = config_dict["preprocessor"]["name"]
    preprocessor_version = config_dict["preprocessor"]["version"]

    cred = EnvironmentCredential()

    ml_client = MLClient(
        subscription_id="94f3bfe4-d65b-4af2-959a-f4cc3f4fef6a",
        resource_group_name="movie-recommender",
        workspace_name="movie-recommender-dev",
        credential=cred,
    )
    ml_client.models.download(
        name=model_name,
        version=model_version,
        download_path=model_download_path,
    )

    ml_client.models.download(
        name=preprocessor_name,
        version=preprocessor_version,
        download_path=preprocessor_download_path,
    )

    model_dir = os.path.join("model_api", "model")
    preprocessor_dir = os.path.join("model_api", "preprocessor")

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    if os.path.isdir(preprocessor_dir):
        shutil.rmtree(preprocessor_dir)

    shutil.move(
        os.path.join(model_download_path, model_name, "model"),
        "model_api",
    )
    shutil.rmtree(model_download_path)

    shutil.move(
        os.path.join(preprocessor_download_path, preprocessor_name, "preprocessor"),
        "model_api",
    )
    shutil.rmtree(preprocessor_download_path)

    metrics_dict = get_model_metrics(ml_client, model_name, model_version)
    print(metrics_dict)

    metrics_file = os.path.join("model_api", "metrics.json")

    if os.path.isfile(metrics_file):
        os.remove(path=metrics_file)

    with open(metrics_file, "w") as fp:
        json.dump(metrics_dict, fp, indent=4)


if __name__ == "__main__":
    main()
