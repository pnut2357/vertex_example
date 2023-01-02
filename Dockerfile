FROM python:3.7.10

COPY ./utils.py ./app/utils.py
COPY ./diagnosis_utils.py ./app/diagnosis_utils.py
COPY ./requirements.txt ./app/requirements.txt

WORKDIR ./app
RUN apt-get update && apt-get install gcc libffi-dev -y

RUN pip install -r requirements.txt
