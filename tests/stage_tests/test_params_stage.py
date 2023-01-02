import os
import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute().parent.parent))
sys.path.append(str(Path('.').absolute().parent.parent)+"/src/utils")


from pipeline_utils import *
import pytest

file_path = Path().resolve().parent.parent
file_path = os.path.join(file_path, 'settings.yml')
print("file_path:", file_path) 
with open(file_path, 'r') as stream:
    try:
        params=yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)
locals().update(params['envs']['stage'])

def test_compile_pipeline():
            
    assert isinstance(SERVICE_ACCOUNT, str), "Incorrect Type, Expected Type String"
    assert isinstance(PIPELINE_ROOT, str), "Incorrect Type, Expected Type String"
    assert isinstance(REGION, str), "Incorrect Type, Expected Type String"
    assert isinstance(PIPELINE_NAME, str), "Incorrect Type, Expected Type String"
    assert isinstance(MODEL_REGISTRY_NAME, str), "Incorrect Type, Expected Type String"
    assert isinstance(MLFLOW_EXP_NAME, str), "Incorrect Type, Expected Type String"
    assert isinstance(PIPELINE_JSON, str), "Incorrect Type, Expected Type String"
    assert isinstance(BASE_IMAGE, str), "Incorrect Type, Expected Type String"
    assert isinstance(MLFLOW_IMAGE, str), "Incorrect Type, Expected Type String"
    assert isinstance(LATEST_NOSALES_MODEL_PATH, str), "Incorrect Type, Expected Type String"
