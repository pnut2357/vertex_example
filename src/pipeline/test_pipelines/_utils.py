
import json
from typing import Any, Dict

from google.cloud import storage
from google.cloud import scheduler_v1

# Load data from a JSON document
def load_json(path: str) -> Dict[str, Any]:
    """Loads data from a JSON document.

    Args:
      path: The path of the JSON document. It can be a local path or a GS URI.

    Returns:
      A deserialized Dict object representing the JSON document.
    """

    if path.startswith('gs://'):
        return _load_json_from_gs_uri(path)
    else:
        return _load_json_from_local_file(path)

# Load data from a JSON document referenced by a GS URI
def _load_json_from_gs_uri(uri: str) -> Dict[str, Any]:
    """Loads data from a JSON document referenced by a GS URI.

    Args:
      uri: The GCS URI of the JSON document.

    Returns:
      A deserialized Dict object representing the JSON document.

    Raises:
      google.cloud.exceptions.NotFound: If the blob is not found.
      json.decoder.JSONDecodeError: On JSON parsing problems.
      ValueError: If uri is not a valid gs URI.
    """
    storage_client = storage.Client()
    blob = storage.Blob.from_string(uri, storage_client)
    return json.loads(blob.download_as_bytes())

# Load data from a JSON local file
def _load_json_from_local_file(file_path: str) -> Dict[str, Any]:
    """Loads data from a JSON local file.

    Args:
      file_path: The local file path of the JSON document.

    Returns:
      A deserialized Dict object representing the JSON document.

    Raises:
      json.decoder.JSONDecodeError: On JSON parsing problems.
    """
    with open(file_path) as f:
        return json.load(f)

# Get the scheduler job details
def get_job(job_name):
    # Create a client
    client = scheduler_v1.CloudSchedulerClient()

    # Initialize request argument(s)
    request = scheduler_v1.GetJobRequest(
        name=job_name,
    )

    # Make the request
    response = client.get_job(request=request)

    # Handle the response
    #print(response)

# Delete scheduler job by job-name
def delete_job(job_name):
    # Create a client
    client = scheduler_v1.CloudSchedulerClient()

    # Initialize request argument(s)
    request = scheduler_v1.DeleteJobRequest(
        name=job_name,
    )

    # Make the request
    client.delete_job(request=request)
    
    return True
