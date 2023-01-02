# pull official base image
FROM python:3.7-slim

# install system dependencies
RUN apt-get clean && apt-get update -y

# copy contents
COPY ./src ./src
COPY ./requirements.txt ./
COPY ./tests ./tests
COPY ./tests/dev_tests ./tests/dev_tests
COPY ./tests/stage_tests ./tests/stage_tests
COPY ./settings.yml ./settings.yml

# install system dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

# set work directory
# WORKDIR ./src/tests
WORKDIR ./tests

RUN ["ls"]
# command
CMD ["pytest"]
