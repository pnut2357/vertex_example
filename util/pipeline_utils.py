from google.cloud import storage
import argparse

import os
import sys
from pathlib import Path
import yaml

def store_pipeline(storage_path, filename):
    """Uploads a file to the bucket."""
    print(filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(filename)

    print(
        "contents {} uploaded to {}.".format(
            filename, storage_path
        )
    )


def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--COMMIT_ID", required=True, type=str)
    parser.add_argument("--BRANCH", required=True, type=str)
    parser.add_argument("--is_prod", required=False, type=str)
    
    # parser.add_argument("--MODE", required=True, type=str)
    # parser.add_argument("--STAGE1_FLAG", required=True, type=str)
    # parser.add_argument("--ENSEMBLE_FLAG", required=True, type=str)
    # parser.add_argument("--RF_CLF_MODEL_PATH", required=True, type=str)
    # parser.add_argument("--LOGISTIC_CLF_MODEL_PATH", required=True, type=str)
    # parser.add_argument("--STAGE1_NN_MODEL_PATH", required=True, type=str)
    # parser.add_argument("--GNB_MODEL_PATH", required=True, type=str)
    # parser.add_argument("--STG1_FEATURE_SELECTOR_MODEL_PATH", required=True, type=str)
    # parser.add_argument("--NOSALES_MODEL_PATH", required=True, type=str)

    args = parser.parse_args()
    return args


def yaml_import(file_path):
    # Import yaml to local variables
    
    # file_path = path #os.path.join(path, 'settings.yaml')
    # file_path_ = os.path.join(file_path_, 'settings.yaml')
    # file_path_ = 'settings.yaml'

    # Converts yaml document to python object
    try:
        with open(file_path, 'r') as stream:
            try:
                dict_ = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
    except:
        print("hi")
        # with open(file_path_, 'r') as stream:
        #     try:
        #         dict_ = yaml.safe_load(stream)
        #     except yaml.YAMLError as e:
        #         print(e)
    return(dict_)
