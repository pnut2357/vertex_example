
FROM hub.docker.prod.walmart.com/google/cloud-sdk:slim
WORKDIR /root

# EOF
RUN echo 'deb http://satellite-capsule.wal-mart.com/debian/deb.debian.org/debian/ buster main' >> /etc/apt/sources.list
RUN echo 'deb http://satellite-capsule.wal-mart.com/debian/security.debian.org/debian-security/ buster/updates main' >> /etc/apt/sources.list
RUN echo 'deb http://satellite-capsule.wal-mart.com/debian/deb.debian.org/debian/ buster-updates main' >> /etc/apt/sources.list

# install system dependencies
RUN pip3 install --upgrade pip
RUN apt-get clean && apt-get update -y
RUN apt-get install -y cmake
RUN apt-get install -y locales locales-all
RUN apt-get install net-tools

# pip Installs
RUN pip install google_cloud_pipeline_components==1.0.7
RUN pip install kfp_pipeline_spec==0.1.16
# RUN pip install kfp_pipeline_spec==0.1.13
RUN pip install pytest==6.2.3
RUN pip install kfp==1.8.12
RUN pip install xgboost==1.5.2
RUN pip install protobuf==3.19.4
RUN pip install fsspec==2022.7.1
RUN pip install gcsfs==2022.7.1
RUN python3 -m pip install ipykernel
RUN python3 -m ipykernel install --user

# Copy contents
COPY ./nosales_model_registry.py nosales_model_registry.py 
COPY ./version_transition.py version_transition.py
COPY ./utils.py utils.py 

# Env Variables
RUN pip install https://repository.walmart.com/content/repositories/pangaea_releases/com/walmart/analytics/platform/library/element-mlflow-plugins-release/0.0.497/element-mlflow-plugins-release-0.0.497.tar.gz
RUN pip install attrs==21.2.0
RUN pip install numpy==1.18.1
RUN pip install pandas==1.1.4
RUN pip install setuptools==45.2.0
RUN pip install joblib==1.1.0
# RUN pip install joblib==0.17.0
RUN pip install h5py==2.10.0
RUN pip install keras==2.3.1
RUN pip install scikit-learn==0.24.1
RUN pip install tensorflow==1.15.4
RUN pip install pickle5
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV DATASET_SERVICE_HOST="ml.prod.walmart.com" 
ENV DATASET_SERVICE_PORT="31500"
ENV PROJECT_TOKEN="ab57d8bba855c2a1103e2ee6900c518572e42c8539ce3edd46afa32994382bda"  
ENV PROJECT_ID="11112"  
ENV ELEMENT_TOKEN="ab57d8bba855c2a1103e2ee6900c518572e42c8539ce3edd46afa32994382bda"
ENV ELEMENT_DECRYPTION_KEY="9YRrxbzmPm5vy4BODlWvE1Mqc"

ENV ENV="DEV"
# DEV, STAGE, PROD

ENV NOTEBOOK_ID="37764"
ENV MLFLOW_TRACKING_URI="element://"
ENV CLUSTER_NAME="NA"
ENV CLUSTER_USER="NA"

# https://ml-gcp.prod.walmart.com:31950/mlflow/11112/ab57d8bba855c2a1103e2ee6900c518572e42c8539ce3edd46afa32994382bda/#/experiments/15220