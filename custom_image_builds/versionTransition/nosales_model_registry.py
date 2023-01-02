from google.cloud import storage
import os
import argparse
import mlflow.pyfunc
import utils
import pickle
import joblib
from tempfile import TemporaryFile
from mlflow.tracking.client import MlflowClient
import tensorflow as tf
import gcsfs
import pickle5 as pk
from time import sleep

def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()

    # # cmd line args
    # parser.add_argument("--PROJECT_ID", required=True, type=str)
    # parser.add_argument("--SERVICE_ACCOUNT", required=True, type=str)
    parser.add_argument("--GCS_MODEL_PATH", required=True, type=str) 
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    # parser.add_argument("--CURRENT_AUC_SCORE", required=False, type=float)
    args = parser.parse_args()
    return args


args = get_args()
# PROJECT_ID = args.PROJECT_ID
GCS_MODEL_PATH = args.GCS_MODEL_PATH # LATEST_NOSALES_MODEL_PATH= 
GCS_MODEL_PATH = "gs://oyi-ds-vertex-pipeline-bucket-nonprod/latest_nosales_model_output_dev"
MODEL_REGISTRY_NAME = args.MODEL_REGISTRY_NAME # MODEL_REGISTRY_NAME="oyi_nosales_model_stage" 
MLFLOW_EXP_NAME = args.MLFLOW_EXP_NAME # MLFLOW_EXP_NAME="oyi_training_stage"
# CURRENT_AUC_SCORE_STACK = args.CURRENT_AUC_SCORE

# artifact_repository = "./mlflow-run"

# Initialize client
c = MlflowClient()

# Parameters
params = {"feature1": "1", "feature2": "2"}
run_name = "test_model"

# If experiment exist then grab the existing one
# else it will create a new experiment id and will use to to run the experiments

# Get the experiment id if it already exists
experiment_id = c.get_experiment_by_name(MLFLOW_EXP_NAME).experiment_id



# Launching Multiple Runs in One Program.This is easy to do because the ActiveRun object returned by mlflow.start_run() is a
# Python context manager. You can “scope” each run to just one block of code as follows:
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    # Get run id 
    run_id = run.info.run_uuid
    
    # Perform model training
    # with open(nosales_model_input.path, "rb") as handler:
    #     model = pickle.load(handler)
    
    # fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    # with fs.open(GCS_MODEL_PATH, 'rb') as handle:
    #     model = joblib.load(handle)
    
    # model = joblib.load(tf.io.gfile.GFile(GCS_MODEL_PATH, 'rb'))
    
    blob = storage.blob.Blob.from_string(GCS_MODEL_PATH, client=storage.Client())
    with TemporaryFile() as temp_file:
        blob.download_to_file(temp_file)
        temp_file.seek(0)
        model=pk.load(temp_file)
    #     print("model loaded")
    #     # Log model artifacts
    mlflow.pyfunc.log_model(python_model=utils.EnsembleClassifierWrapper(model=model,), artifact_path="model", registered_model_name=MODEL_REGISTRY_NAME)
    # wait 20 s for updating the version while loading
    sleep(20) 
    mlflow_obj = utils.MLFlowFunc(c)
    
    current_metrics, current_model_version = mlflow_obj.find_metrics(MODEL_REGISTRY_NAME)
    
    blob = storage.Client().bucket(GCS_MODEL_PATH.split('/')[2]).blob("stack_auc") 
    with blob.open("r") as f:
        current_metrics = f.read()
    
    # Set the notes for the run
    c.set_tag(run_id,
              "mlflow.note.content",
              "This is experiment for testing")

    # Define customer tag
    tags = {"Application": "Order Your Inventory",
            "tags_model_version": f"{current_model_version}",
            "tags_run_id": f"{run_id}"}

    # Set Tag
    mlflow.set_tags(tags)

    # Log python re details
    # mlflow.log_artifact('requirements.txt')

    # logging params
    mlflow.log_param("run_id", run_id)
    mlflow.log_params(params)
    
    # model_uri = f"runs:/{run_id}/model"
    # model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

    # log model run_id
    # Perform model evaluation 
    # log metrics
    mlflow.log_metrics({"stack_auc": float(current_metrics)})
