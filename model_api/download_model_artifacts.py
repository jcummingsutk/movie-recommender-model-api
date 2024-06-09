import argparse
import os
import shutil

from azure.ai.ml import Input, MLClient
from azure.identity import EnvironmentCredential
from config import get_config_dict, load_azure_service_principal_environment_vars
from mlflow import MlflowClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-path", type=str)
    args = parser.parse_args()

    os.makedirs(args.download_path, exist_ok=True)

    load_azure_service_principal_environment_vars("config.yaml", "config_secret.yaml")
    config_dict = get_config_dict("config.yaml")
    model_name = config_dict["model"]["name"]
    model_version = config_dict["model"]["version"]

    cred = EnvironmentCredential()

    ml_client = MLClient(
        subscription_id="94f3bfe4-d65b-4af2-959a-f4cc3f4fef6a",
        resource_group_name="movie-recommender",
        workspace_name="movie-recommender-dev",
        credential=cred,
    )

    ml_client.models.download(model_name, model_version, args.download_path)

    shutil.copytree(
        os.path.join(args.download_path, "movie-recommender", "model"),
        os.path.join("model_api", "model"),
    )
