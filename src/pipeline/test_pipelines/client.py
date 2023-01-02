
from typing import Any, Dict, List, Mapping, Optional
from schedule import _create_from_pipeline_dict
from utils import load_json, get_job, delete_job

class aipclient_custom:
    def __init__(self, project_id: str = None,region: str = None):
        self.project_id = project_id
        self.region = region
    
    def create_schedule_from_job_spec(self,
        job_spec_path: str = None,
        schedule: str = "* * * * *",
        time_zone: str = 'US/Pacific',
        pipeline_root: Optional[str] = None,
        parameter_values: Optional[Mapping[str, Any]] = None,
        cloud_function_project_id: str = None,
        service_account: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        app_engine_region: Optional[str] = None,
        cloud_scheduler_service_account: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> dict:
        """Creates schedule for compiled pipeline file.

        This function creates scheduled job which will run the provided pipeline on
        schedule. This is implemented by creating a Google Cloud Scheduler Job.
        The job will be visible in https://console.google.com/cloudscheduler and can
        be paused/resumed and deleted.

        To make the system work, this function also creates a Google Cloud Function
        which acts as an intermediary between the Scheduler and Pipelines. A single
        function is shared between all scheduled jobs.
        The following APIs will be activated automatically:
        * cloudfunctions.googleapis.com
        * cloudscheduler.googleapis.com
        * appengine.googleapis.com

        Args:
          job_spec_path: Path of the compiled pipeline file.
          schedule: Schedule in cron format. Example: "45 * * * *"
          time_zone: Schedule time zone. Default is 'US/Pacific'
          parameter_values: Arguments for the pipeline parameters
          pipeline_root: Optionally the user can override the pipeline root
            specified during the compile time.
          service_account: The service account that the pipeline workload runs as.
          enable_caching: Whether or not to enable caching for the run.
            If not set, defaults to the compile time settings, which are True for all
            tasks by default, while users may specify different caching options for
            individual tasks.
            If set, the setting applies to all tasks in the pipeline -- overrides
            the compile time settings.
          app_engine_region: The region that cloud scheduler job is created in.
          cloud_scheduler_service_account: The service account that Cloud Scheduler job and the proxy cloud function use.
            this should have permission to call AI Platform API and the proxy function.
            If not specified, the functions uses the App Engine default service account.

        Returns:
          Created Google Cloud Scheduler Job object dictionary.
        """
        if job_spec_path is not None:
            job_spec = load_json(job_spec_path)
        
        if cloud_function_project_id is None:
            cloud_function_project_id = self.project_id
        if enable_caching is not None:
            _set_enable_caching_value(job_spec['pipelineSpec'], enable_caching)
        
        # To check if a job already exists with the same name
        self.check_job(job_name)
        
        scheduler_job_name = job_name.split("/")[-1]
        # Create new job
        return _create_from_pipeline_dict(
            pipeline_dict=job_spec,
            schedule=schedule,
            project_id=self.project_id,
            cloud_function_project_id = cloud_function_project_id, #cloud-function_project_id
            region=self.region,
            time_zone=time_zone,
            parameter_values=parameter_values,
            pipeline_root=pipeline_root,
            service_account=service_account,
            app_engine_region=app_engine_region,
            cloud_scheduler_service_account=cloud_scheduler_service_account,
            scheduler_job_name = scheduler_job_name)
    
    def check_job(self,job_name):
        if job_name is not None:
            try:
                get_job(job_name)
                print("Job already exists. Deleting..")
                delete_job(job_name)
                print("Creating a new scheduler job")

            except Exception as e:
                print("Job doesn't exist")
        
        else:
            print("Creating a new scheduler job")
