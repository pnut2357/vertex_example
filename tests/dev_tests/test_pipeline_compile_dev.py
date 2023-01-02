import os
import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute().parent.parent))
sys.path.append(str(Path(".").absolute().parent.parent) + "/src/utils")
sys.path.append(str(Path(".").absolute().parent.parent) + "/src/pipeline")

from pipeline_utils import *
# from pipeline import pipeline
import pytest
from kfp.v2 import compiler
import time

params = yaml_import('settings.yml')

locals().update(params["envs"]["dev"])

# Test Pipeline Compile
# def compile_pipeline():
#     compiler.Compiler().compile(
#         pipeline_func=pipeline,
#         package_path=os.path.join(
#             "/tmp",
#             (PIPELINE_NAME.replace(".json", "") + ".json"),
#         ),
#     )
#     return True


# def test_compile_pipeline():
#     assert compile_pipeline()
