#! /bin/bash
docker tag movie-recommender-api registry.digitalocean.com/streamlit-apps/movie-recommender-api:latest

docker push registry.digitalocean.com/streamlit-apps/movie-recommender-api:latest