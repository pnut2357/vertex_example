from google.cloud import storage
import os
import argparse
import mlflow.pyfunc
import utils
import pickle
from tempfile import TemporaryFile
# from mlflow.tracking.client import MlflowClient

def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()

    # # cmd line args
    # parser.add_argument("--PROJECT_ID", required=True, type=str)
    # parser.add_argument("--SERVICE_ACCOUNT", required=True, type=str)
    parser.add_argument("--GCS_MODEL_PATH", required=True, type=str) # 
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    parser.add_argument("--CURRENT_AUC_SCORE", required=True, type=float)
    args = parser.parse_args()
    return args


args = get_args()
GCS_MODEL_PATH = args.GCS_MODEL_PATH # LATEST_NOSALES_MODEL_PATH=
MODEL_NAME = args.MODEL_REGISTRY_NAME # MODEL_REGISTRY_NAME="oyi_nosales_model_stage" 
EXPERIMENT_NAME = args.MLFLOW_EXP_NAME # MLFLOW_EXP_NAME="oyi_training_stage"
CURRENT_AUC_SCORE_STACK = args.CURRENT_AUC_SCORE

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    blob = storage.blob.Blob.from_string(gcs_model_path, client=storage.Client())
    with TemporaryFile() as temp_file:
        blob.download_to_file(temp_file)
        temp_file.seek(0)
        model=pickle.load(temp_file)
    mlflow.pyfunc.log_model(python_model=utils.EnsembleClassifierWrapper(model=model,), 
                            artifact_path="model",
                            registered_model_name=model_name)
    
    
    # data = model.predict()
    # metrics = model.eval()
    # mlflow.log_param(param1)
    # mlflow.log_param(param2)
    mlflow.log_metric(CURRENT_AUC_SCORE_STACK)
    # mlflow.log_metric(metrics2)
    
# artifact_repository = './mlflow-run'

# # Parameters
# version = '1'
# params = {"feature1": "1", "feature2": "2"}
# auc_score = 0.7
# run_name = "test_model"
# # Initialize client
# c = MlflowClient()
# # If experiment exist then grab the existing one
# # else it will create a new experiment id and will use to to run the experiments
# try:
#     # Get the experiment id if it already exists
#     experiment_id = c.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
# except:
#     # Create experiment 
#     experiment_id = c.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_repository)
    

# # Launching Multiple Runs in One Program.This is easy to do because the ActiveRun object returned by mlflow.start_run() is a 
# # Python context manager. You can “scope” each run to just one block of code as follows:
# with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
#     # Get run id 
#     run_id = run.info.run_uuid

#     # Set the notes for the run
#     c.set_tag(run_id,
#               "mlflow.note.content",
#               "This is experiment for testing")

#     # Define customer tag
#     tags = {"Application": "Order Your Inventory",
#             "release.candidate": "PMP",
#             "release.version": f"{version}"}

#     # Set Tag
#     mlflow.set_tags(tags)

#     # Log python re details
#     # mlflow.log_artifact('requirements.txt')

#     # logging params
#     mlflow.log_params(params)

#     # Perform model training
#     blob = storage.blob.Blob.from_string(GCS_MODEL_PATH, client=storage.Client())
#     with TemporaryFile() as temp_file:
#         blob.download_to_file(temp_file)
#         temp_file.seek(0)
#         model=pickle.load(temp_file)
#     # Log model artifacts
#     mlflow.pyfunc.log_model(python_model=utils.EnsembleClassifierWrapper(model=model,), 
#                             artifact_path="model",
#                             registered_model_name=MODEL_NAME)

#     # Perform model evaluation 

#     # log metrics
#     mlflow.log_metrics({"stack_auc": CURRENT_AUC_SCORE_STACK})

# #     # Plot and save feature importance details
# #     ax = plot_importance(lgb_clf, height=0.4)
# #     filename = './images/lgb_validation_feature_importance.png'
# #     plt.savefig(filename)
# #     # log model artifacts
# #     mlflow.log_artifact(filename)




    
