import argparse
import os

from mlflow import MlflowClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--download-path", type=str)
    args = parser.parse_args()

    os.makedirs(args.download_path, exist_ok=True)

    client = MlflowClient()
    client.download_artifacts(
        run_id=args.run_id, path="model", dst_path=args.download_path
    )
