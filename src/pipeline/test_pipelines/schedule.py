
import base64
import hashlib
import json
import logging
import pathlib
import re
import tempfile
from typing import Any, Mapping, Optional
import zipfile

import googleapiclient
from googleapiclient import discovery
import requests
from google.cloud import resourcemanager_v3

from kfp.v2.google.client import client_utils
from kfp.v2.google.client import runtime_config_builder

_PROXY_FUNCTION_NAME = 'rahul_templated_http_request-v1'
# _PROXY_FUNCTION_FILENAME = '_cloud_function_templated_http_request.py'

_CAIPP_ENDPOINT_WITHOUT_REGION = 'aiplatform.googleapis.com'
_CAIPP_API_VERSION = 'v1beta1'

def _get_cloud_functions_api():
    functions_service = discovery.build(
        'cloudfunctions', 'v1', cache_discovery=False)
    functions_api = functions_service.projects().locations().functions()
    return functions_api

def _create_or_get_cloud_function(
    name: str,
    cloud_scheduler_service_account: str,
    cloud_function_project_id: str,
    region: str,
    runtime: str = 'python37',
):
    
    """Creates Google Cloud Function."""
    functions_api = _get_cloud_functions_api()

    project_location_path = 'projects/{}/locations/{}'.format(
        cloud_function_project_id, region)
    function_full_name = project_location_path + '/functions/' + name
    
    print("CLOUD FUNCTION FULL NAME :", function_full_name)
    
    # Returning early if the function already exists.
    try:
        function_get_response = functions_api.get(
            name=function_full_name).execute()
        
        # print("[BEFORE]: function_get_response", function_get_response)
        
        return function_get_response
    except googleapiclient.errors.HttpError as err:
        raise_error = True
        if err.resp['status'] == '404':
            # The function does not exist, which is expected.
            logging.info('["ERROR"] Cloud Function: name=%s does not exist', function_full_name)
            raise_error = False
        if raise_error:
            raise err
        return False
    
    return function_get_response

def _get_proxy_cloud_function_endpoint(
    cloud_function_project_id: str,
    cloud_scheduler_service_account: str,
    region: str = 'us-central1',
):
    function_dict = _create_or_get_cloud_function(
        name=_PROXY_FUNCTION_NAME,
        cloud_function_project_id=cloud_function_project_id,
        region=region,
        runtime='python37',
        cloud_scheduler_service_account=cloud_scheduler_service_account,
    )

    if not function_dict:
        print("Exiting the code")
        import sys
        sys.exit()
        
    else:
        endpoint_url = function_dict['httpsTrigger']['url']
        return endpoint_url

def _enable_required_apis(project_id: str,):
    """Enables necessary APIs."""
    serviceusage_service = discovery.build('serviceusage', 'v1', cache_discovery=False)
    services_api = serviceusage_service.services()

    required_services = [
        'cloudfunctions.googleapis.com',
        'cloudscheduler.googleapis.com',
        'appengine.googleapis.com',  # Required by the Cloud Scheduler.
    ]
    project_path = 'projects/' + project_id
    for service_name in required_services:
        service_path = project_path + '/services/' + service_name
        services_api.enable(name=service_path).execute()
        
def _create_scheduler_job(project_location_path: str,
                          job_body: Mapping[str, Any]) -> str:
    """Creates a scheduler job.

    Args:
      project_location_path: The project location path.
      job_body: The scheduled job dictionary object.

    Returns:
      The response from scheduler service.
    """
    # We cannot use google.cloud.scheduler_v1.CloudSchedulerClient since
    # it's not available internally.
    scheduler_service = discovery.build(
        'cloudscheduler', 'v1', cache_discovery=False)
    scheduler_jobs_api = scheduler_service.projects().locations().jobs()
    response = scheduler_jobs_api.create(
        parent=project_location_path,
        body=job_body,
    ).execute()
    return response


def _create_from_pipeline_dict(
    schedule: str,
    project_id: str,
    cloud_function_project_id: str,
    cloud_scheduler_service_account: str,
    pipeline_dict: dict = None,
    parameter_values: Optional[Mapping[str, Any]] = None,
    pipeline_root: Optional[str] = None,
    service_account: Optional[str] = None,
    app_engine_region: Optional[str] = None,
    scheduler_job_name: Optional[str] = None,
    region: str = 'us-central1',
    time_zone: str = 'US/Pacific'
) -> dict:
    """Creates schedule for compiled pipeline dictionary."""

    _enable_required_apis(project_id=project_id)

    # If appengine region is not provided, use the pipeline region.
    app_engine_region = app_engine_region or region

    proxy_function_url = _get_proxy_cloud_function_endpoint(
        cloud_function_project_id=cloud_function_project_id,
        region=region,
        cloud_scheduler_service_account=cloud_scheduler_service_account,
    )

    if parameter_values or pipeline_root:
        config_builder = runtime_config_builder.RuntimeConfigBuilder.from_job_spec_json(
            pipeline_dict)
        config_builder.update_runtime_parameters(
            parameter_values=parameter_values)
        config_builder.update_pipeline_root(pipeline_root=pipeline_root)
        updated_runtime_config = config_builder.build()
        pipeline_dict['runtimeConfig'] = updated_runtime_config

    # Creating job creation request to get the final request URL
    pipeline_jobs_api_url = f'https://{region}-{_CAIPP_ENDPOINT_WITHOUT_REGION}/{_CAIPP_API_VERSION}/projects/{project_id}/locations/{region}/pipelineJobs'
    
    # Preparing the request body for the Cloud Function processing
    pipeline_name = pipeline_dict['pipelineSpec']['pipelineInfo']['name']
    full_pipeline_name = 'projects/{}/pipelineJobs/{}'.format(project_id, pipeline_name)
    pipeline_display_name = pipeline_dict.get('displayName')
    time_format_suffix = "-{{$.scheduledTime.strftime('%Y-%m-%d-%H-%M-%S')}}"
    if 'name' in pipeline_dict:
        pipeline_dict['name'] += time_format_suffix
    if 'displayName' in pipeline_dict:
        pipeline_dict['displayName'] += time_format_suffix

    pipeline_dict['_url'] = pipeline_jobs_api_url
    pipeline_dict['_method'] = 'POST'

    if service_account is not None:
        pipeline_dict['serviceAccount'] = service_account

    pipeline_text = json.dumps(pipeline_dict)
    pipeline_data = pipeline_text.encode('utf-8')

#================ CUSTOM ==============================
    # pipeline_dict = {"pipeline_spec_uri":"gs://pipeline-schedule/intro_pipeline.json"}
    # pipeline_text = json.dumps(pipeline_dict)
    # pipeline_data = pipeline_text.encode('utf-8')

# ======================================================

    project_location_path = 'projects/{}/locations/{}'.format(
        project_id, app_engine_region)
    scheduled_job_full_name = '{}/jobs/{}'.format(project_location_path,
                                                  scheduler_job_name)
    service_account_email = cloud_scheduler_service_account or '{}@appspot.gserviceaccount.com'.format(
        project_id)

    scheduled_job = dict(
        name=scheduled_job_full_name,  # Optional. Only used for readable names.
        schedule=schedule,
        time_zone=time_zone,
        http_target=dict(
            http_method='POST',
            uri=proxy_function_url,
            # Warning: when using google.cloud.scheduler_v1, the type of body is
            # bytes or string. But when using the API through discovery, the body
            # needs to be base64-encoded.
            body=base64.b64encode(pipeline_data).decode('utf-8'),
            oidc_token=dict(service_account_email=service_account_email,audience = proxy_function_url),
        ),
        # TODO(avolkov): Add labels once Cloud Scheduler supports them
        # labels={
        #     'google.cloud.ai-platform.pipelines.scheduling': 'v1alpha1',
        # },
    )

    try:
        response = _create_scheduler_job(
            project_location_path=project_location_path,
            job_body=scheduled_job,
        )
        return response
    except googleapiclient.errors.HttpError as err:
        # Handling the case where the exact schedule already exists.
        if err.resp.get('status') == '409':
            raise RuntimeError(
                'The exact same schedule already exists') from err
        raise err
