FROM python:3.7
#FROM python:3.7-slim

RUN pwd
RUN ls

COPY ./utils.py ./app/utils.py
COPY ./diagnosis_utils.py ./app/diagnosis_utils.py
COPY ./requirements.txt ./app/requirements.txt
# Copy and Rund run_test.py via bash script
# COPY ./run_test.py ./app/run_test.py
#
# COPY ./prework_requirements.txt ./app/prework_requirements.txt
RUN ls

WORKDIR ./app
RUN apt-get update && apt-get install gcc libffi-dev -y

RUN pip install -r requirements.txt
RUN pip install pandas fsspec gcsfs kfp==1.8.14

