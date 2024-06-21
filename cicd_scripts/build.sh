#! /bin/bash
python model_api/download_model_artifacts.py --model-download-path ./downloaded_model --preprocessor-download-path ./downloaded_preprocessor

docker build -t movie-recommender-api model_api && docker run -p 8001:8001 movie-recommender-api:latest