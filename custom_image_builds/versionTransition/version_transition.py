from google.cloud import storage
import os
import argparse
import mlflow.pyfunc
import utils
import pickle
from tempfile import TemporaryFile
from mlflow.tracking.client import MlflowClient

def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()

    # # cmd line args
    # parser.add_argument("--PROJECT_ID", required=True, type=str)
    # parser.add_argument("--SERVICE_ACCOUNT", required=True, type=str)
    # parser.add_argument("--GCS_MODEL_PATH", required=True, type=str) # 
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    # parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    # parser.add_argument("--ENV", required=True, type=str) 
    # parser.add_argument("--MODEL_PREFIX", required=True, type=str)
    # parser.add_argument("--CURRENT_AUC_SCORE", required=False, type=str)
    args = parser.parse_args()
    return args


args = get_args()
# gcs_model_path = args.GCS_MODEL_PATH # LATEST_NOSALES_MODEL_PATH
# model_name = args.MODEL_REGISTRY_NAME # MODEL_REGISTRY_NAME
# experiment_name = args.MLFLOW_EXP_NAME # MLFLOW_EXP_NAME
# ENV = args.ENV
# MODEL_PREFIX = args.MODEL_PREFIX # "oyi_nosales_model"
# MODEL_REGISTRY_NAME = f"{MODEL_PREFIX}_{ENV}" #"oyi_nosales_model_dev",
MODEL_REGISTRY_NAME = args.MODEL_REGISTRY_NAME #  "oyi_nosales_model_stage" 
# CURRENT_AUC_SCORE_STACK = args.CURRENT_AUC_SCORE

# mlflow.set_experiment(experiment_name)

# Initialize client
c = MlflowClient()

mlflow_obj = utils.MLFlowFunc(c)
# Get the metrics from 
current_metrics, current_model_version = mlflow_obj.find_metrics(MODEL_REGISTRY_NAME)
production_metrics, production_model_version = mlflow_obj.find_metrics(MODEL_REGISTRY_NAME, "Production")


if current_metrics["stack_auc"] >= production_metrics["stack_auc"]:
    # Transition the latest version to Staging
    mlflow_obj.model_version_registry(MODEL_REGISTRY_NAME, current_model_version, "Staging")
    # Transition the latest version to Production
    mlflow_obj.model_version_registry(MODEL_REGISTRY_NAME, current_model_version, "Production")

# if __name__ == "__main__":
#     client = MlflowClient()
#     mlflow_obj = utils.MLFlowFunc(client) 
#     # Find the latest version among the versions that current_stage as "None" 
#     latest_version = mlflow_obj.find_latest_version(MODEL_NAME) #client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version 
#     # Transition the latest version to Staging
#     mlflow_obj.model_version_registry(MODEL_NAME, latest_version, "Staging")
#     # Transition the latest version to Production
#     mlflow_obj.model_version_registry(MODEL_NAME, latest_version, "Production")

    
