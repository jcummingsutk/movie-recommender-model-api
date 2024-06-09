FROM python:3.11-slim-buster

WORKDIR /app

COPY ./model/requirements.txt ./model/

COPY modify_requirements.py ./

RUN python modify_requirements.py --requirements-file model/requirements.txt --torch-requirements-out-file torch_requirements.txt --non-torch-requirements-out-file non_torch_requirements.txt

RUN pip install --upgrade pip

RUN pip install -r ./non_torch_requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cpu -r ./torch_requirements.txt 

RUN pip install fastapi fastapi-cli uvicorn

COPY . .

EXPOSE 8001

CMD ["fastapi", "run", "app/main.py", "--port", "8001"]