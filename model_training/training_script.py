import argparse

import mlflow
import torch
import yaml

from movie_recommender_data.movie_recommender_data.config import get_dev_db_params

from .code_training.data_utils import (
    create_dataloaders,
    create_train_test_split,
    load_dataframe,
)
from .code_training.model_utils import RecSysModel, train


def main(parameters_file: str, config_file: str, config_secrets_file: str):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    with open(parameters_file) as fp:
        parameters = yaml.safe_load(fp)
    mlflow.log_params(parameters)
    params = get_dev_db_params(
        config_file,
        config_secrets_file,
    )

    df = load_dataframe(
        params["host"],
        params["database"],
        params["user"],
        params["password"],
        params["port"],
    )

    model = RecSysModel(df).to(device)

    df = model.preprocess_data(df)

    df_train, df_test = create_train_test_split(df, parameters["data"]["test_size"])

    train_loader, test_loader = create_dataloaders(
        df_train, df_test, device, parameters["data"]["batch_size"]
    )

    train(
        model,
        train_loader,
        test_loader,
        model_params=parameters["model"],
        metric_params=parameters["metrics"],
    )


if __name__ == "__main__":
    mlflow.set_experiment("movie-recommender-model-training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters-file", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--config-secrets-file", type=str)
    args = parser.parse_args()

    parameters_file = args.parameters_file

    print(args.config_file)
    print(args.config_secrets_file)

    with mlflow.start_run():
        main(args.parameters_file, args.config_file, args.config_secrets_file)
