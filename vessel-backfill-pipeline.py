# Service
import modal

# Environment check
import sys
import os

# Nice printing
from pprint import pprint

# Nice logging
import nice_log as NiceLog
from nice_log import BGColors

# My own exceptions (makes it nicer and quicker in the reports from Model)
from project_exceptions import *

# IDE help
from hsfs import feature_store, feature_group, feature_view
from hopsworks import project as HopsworksProject
from hsfs.constructor import query as hsfs_query
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult
from hopsworks.core.job_api import JobsApi as HopsworksJobsApi
from hopsworks.job import Job as HopsworksJob
from hopsworks.execution import Execution as HopsworksExecution

# Error help
from hopsworks import RestAPIError

# Data
import pandas as pd

# Settings
# - Modal
modal_stub_name = "vessel-backfill-pipeline"
modal_image_libraries = ["hopsworks", "joblib"]
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# Names
# - Models
models_dir = "models"
model_bridge_name = "bridge_model"
model_bridge_version = 1
# - Feature Groups
fg_vessel_name = "vessel"
fg_vessel_version = 1
# - Feature Views
fw_vessel_name = "vessel"
fw_vessel_version = 1

# - GitHub
github_backfill_csv_url = "https://raw.githubusercontent.com/martenb-se/id2223-project/main/backfill/vessel_data.csv"

LOCAL = True

# Running REMOTELY in Modal's environment
if "MODAL_ENVIRONMENT" in os.environ:
    NiceLog.info(f"Running in {BGColors.HEADER}REMOTE{BGColors.ENDC} Modal environment")
    LOCAL = False
# Running LOCALLY with 'modal run' to deploy to Modal's environment
elif "/modal" in os.environ["_"]:
    from dotenv import load_dotenv

    if sys.argv[1] == "run":
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal run' to run stub once in Modal's "
                     f"remote environment")

    elif sys.argv[1] == "deploy":
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal deploy' to deploy to Modal's "
                     f"remote environment")
    else:
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal {sys.argv[1]}'")

    LOCAL = False
# Running LOCALLY in Python
else:
    from dotenv import load_dotenv

    NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} in Python environment.")


def get_last_job_info(project: HopsworksProject.Project, fg_name: str) -> dict:
    """
    Get the last job execution for a job.
    :param project: The Hopsworks project to get the job from.
    :param fg_name: The name of the Feature Group to get the last job execution for.
    :return: A dictionary with the last job execution time, whether the job has finished, the last job status and the
    """
    from datetime import datetime

    last_job_execution_time = None
    last_job_finished = False
    last_job_status = None
    last_job_success = None

    jobs: HopsworksJobsApi = project.get_jobs_api()
    hw_jobs = jobs.get_jobs()

    if type(hw_jobs) is list:
        # print("list")
        # print(len(hw_jobs))

        # ERROR: This is not working due to bug in Hopsworks...
        # Traceback (most recent call last):
        #   File "hopsworks/util.py", line 27, in default
        #     return obj.to_dict()
        # AttributeError: 'Job' object has no attribute 'to_dict'
        #
        # Get first job
        # hw_job = [0]
        # print(hw_job)

        # Stupid workaround
        hw_jobs_string = str(hw_jobs)
        # print(hw_jobs_string)

        # Strip brackets
        hw_jobs_string = hw_jobs_string[1:-1]

        # Strip first "Job("
        hw_jobs_string = hw_jobs_string[4:]

        # Strip last ")"
        hw_jobs_string = hw_jobs_string[:-1]

        # Split into list
        hw_jobs_list = hw_jobs_string.split("), Job(")

        # Loop through list
        for hw_job in hw_jobs_list:
            # Split into name, type
            hw_job_list = hw_job.split(", ")
            hw_job_name = hw_job_list[0]

            # Strip quotes
            hw_job_name = hw_job_name[1:-1]

            # Only care about jobs with fg_vessel_name in the name
            if fg_name not in hw_job_name:
                continue

            # Only care about "materialization" jobs
            if "materialization" not in hw_job_name:
                continue

            print(hw_job_name)

            cur_job: HopsworksJob = jobs.get_job(hw_job_name)

            print("Created: ", cur_job.creation_time)
            # print("Schedule: ", cur_job.job_schedule)
            # print(cur_job.get_executions())

            print("Executions: ")

            cur_job_execution = cur_job.get_executions()  # Apparently this returns ALL executions,
            #                                               not just for the current job...
            if type(cur_job_execution) is list:
                # print("list")
                # print(len(cur_job_execution))

                # Stupid workaround
                cur_job_execution_string = str(cur_job_execution)
                # print(cur_job_execution_string)

                # Strip brackets
                cur_job_execution_string = cur_job_execution_string[1:-1]

                # Strip first "Execution("
                cur_job_execution_string = cur_job_execution_string[10:]

                # Strip last ")"
                cur_job_execution_string = cur_job_execution_string[:-1]

                # Split into list
                cur_job_execution_list = cur_job_execution_string.split("), Execution(")

                # Loop through list
                for cur_job_execution in cur_job_execution_list:
                    # Split into name, type
                    cur_job_execution_list = cur_job_execution.split(", ")
                    cur_job_execution_success = cur_job_execution_list[0]
                    cur_job_execution_status = cur_job_execution_list[1]
                    cur_job_execution_time = cur_job_execution_list[2]
                    cur_job_execution_command = cur_job_execution_list[3]

                    # Strip quotes
                    cur_job_execution_success = cur_job_execution_success[1:-1]
                    cur_job_execution_status = cur_job_execution_status[1:-1]
                    cur_job_execution_time = cur_job_execution_time[1:-1]
                    cur_job_execution_command = cur_job_execution_command[1:-1]

                    # Check that command includes hw_job_name:
                    if hw_job_name in cur_job_execution_command:
                        print(" >", cur_job_execution_time)
                        print(" - - Success: ", cur_job_execution_success)
                        print(" - - Status: ", cur_job_execution_status)

                        # Update last job execution time if last_job_execution_time is None or cur_job_execution_time is more recent
                        cur_job_execution_time_datetime = \
                            datetime.strptime(cur_job_execution_time, '%Y-%m-%dT%H:%M:%S.%fZ')
                        if last_job_execution_time is None or cur_job_execution_time_datetime > last_job_execution_time:
                            last_job_execution_time = cur_job_execution_time_datetime
                            last_job_status = cur_job_execution_status
                            last_job_success = cur_job_execution_success

    if last_job_status == "APP_MASTER_START_FAILED":
        last_job_finished = True
    elif last_job_status == "FINISHED":
        last_job_finished = True

    return {
        "execution_time": last_job_execution_time,
        "finished": last_job_finished,
        "lstatus": last_job_status,
        "success": last_job_success
    }


def g():
    import hopsworks
    import joblib
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    NiceLog.header(f"Running function to Backfill Vessel Data")

    if "HOPSWORKS_API_KEY" not in os.environ:
        NiceLog.error(f"Failed to log in to Hopsworks. HOPSWORKS_API_KEY is not in the current environment.")
        raise HopsworksNoAPIKey()

    # Log in
    NiceLog.info("Logging in to Hopsworks...")
    try:
        # As long as os.environ["HOPSWORKS_API_KEY"] is set, Hopsworks should not ask for user input
        project: HopsworksProject.Project = hopsworks.login()
    except RestAPIError as e:
        NiceLog.error(f"Failed to log in to Hopsworks. Reason: {e}")
        raise HopsworksLoginError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to log in to Hopsworks. Reason: {e}")
        raise HopsworksLoginError()
    NiceLog.success("Logged in to Hopsworks!")

    # Get the feature store
    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} feature store...")
    try:
        fs: feature_store.FeatureStore = project.get_feature_store()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature store from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureStoreError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature store from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureStoreError()

    NiceLog.success("Gotten feature store!")
    NiceLog.info(f"Feature store is named: {BGColors.HEADER}{fs.name}{BGColors.ENDC})")

    # Get Feature Group
    NiceLog.info(f"Getting {BGColors.HEADER}{fg_vessel_name}{BGColors.ENDC} "
                 f"(version {fg_vessel_version}) feature group...")
    try:
        vessel_fg: feature_group.FeatureGroup = \
            fs.get_or_create_feature_group(name=fg_vessel_name, version=fg_vessel_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{vessel_fg.name}{BGColors.ENDC} "
                 f"({vessel_fg.description})")

    # Get last job info
    NiceLog.info(f"Getting last job info for {BGColors.HEADER}{vessel_fg.name}{BGColors.ENDC} "
                 f"(version {vessel_fg.version}) feature group...")

    last_job_info = get_last_job_info(project, fg_vessel_name)
    # If job has not finished, wait for it to finish by not running the rest of the function
    if not last_job_info["finished"]:
        NiceLog.info(f"Job has not finished. Waiting for job to finish...")
        return

    else:
        NiceLog.info(f"Job has finished. Continuing...")

    # Get latest data from Feature Store
    NiceLog.info(f"Getting latest data from Feature Store...")
    try:
       fs_latest_data = vessel_fg.read()
    except Exception as e:
       fs_latest_data = pd.DataFrame()

    # Get Backfill CSV data from GitHub
    NiceLog.info(f"Getting Backfill CSV data from GitHub...")
    try:
        backfill_csv = pd.read_csv(github_backfill_csv_url)
    except Exception as e:
        NiceLog.error(f"Failed to get Backfill CSV data from GitHub. Reason: {e}")
        raise GitHubGetBackfillCSVError()

    NiceLog.success(f"Got Backfill CSV data from GitHub!")

    # Drop bad columns
    NiceLog.info(f"Dropping bad columns from Backfill CSV...")
    filtered_backfill_csv = \
        backfill_csv.drop(columns=['ship_name', 'eta_day', 'eta_hour', 'eta_minute', 'eta_month', 'destination'])

    # Get rows that are not in latest data
    NiceLog.info(f"Getting rows that are not in latest data...")
    filtered_backfill_csv_new = filtered_backfill_csv

    # Go through all rows in fs_latest_data and print out ship_id and time
    # for index, row in fs_latest_data.iterrows():
    #     ship_id = row['ship_id']
    #     time = row['time']
    #     # print(index, ship_id, time)
    #
    #     # If ship_id and time is in filtered_backfill_csv_new, remove it
    #     filtered_backfill_csv_new = \
    #         filtered_backfill_csv_new[~((filtered_backfill_csv_new['ship_id'] == ship_id) &
    #                                     (filtered_backfill_csv_new['time'] == time))]
    # Merging with an indicator to find common rows
    common_rows = fs_latest_data.merge(filtered_backfill_csv_new, on=['ship_id', 'time'], how='inner', indicator=True)

    # Filtering out the common rows from filtered_backfill_csv_new
    filtered_backfill_csv_new = filtered_backfill_csv_new[~filtered_backfill_csv_new.index.isin(common_rows.index)]

    # Print how many rows are already added from backfill
    NiceLog.info(f"Number of rows already added from backfill: {len(common_rows.index)}")

    # Print how many more rows there are to add from backfill
    NiceLog.info(f"Number of rows left to add from backfill: {len(filtered_backfill_csv_new.index)}")

    # If there are no more rows to add, stop
    if len(filtered_backfill_csv_new.index) == 0:
        NiceLog.info(f"No more rows to add from backfill. Stopping...")
        return

    # Print
    # NiceLog.info(f"Latest data tail:")
    # pprint(fs_latest_data.tail())

    # Get first 300 rows from Backfill CSV
    # NiceLog.info(f"Getting first 300 rows from Backfill CSV...")
    # backfill_csv = filtered_backfill_csv_new.head(300)

    # Inspect data head
    # NiceLog.info(f"Backfill CSV data head:")
    # pprint(backfill_csv.head())

    # Push data to Feature Store
    vessel_fg.insert(filtered_backfill_csv_new)


# Initialize
if not LOCAL:
    NiceLog.info(f"Setting Modal stub name to: {BGColors.HEADER}{modal_stub_name}{BGColors.ENDC}")
    stub = modal.Stub(modal_stub_name)

    NiceLog.info(f"Creating a Modal image with Python libraries: {BGColors.HEADER}{', '.join(modal_image_libraries)}"
                 f"{BGColors.ENDC}")
    image = modal.Image.debian_slim().pip_install(modal_image_libraries)

    if sys.argv[1] == "run":
        NiceLog.info(f"But this is just a {BGColors.HEADER}one time{BGColors.ENDC} test.")


    @stub.function(
        image=image,
        secrets=[
            modal.Secret.from_name(hopsworks_api_key_modal_secret_name),
        ]
    )
    def f():
        g()

# Load local environment
else:
    NiceLog.info("Loading local environment...")
    if load_dotenv() and 'HOPSWORKS_API_KEY' in os.environ:
        NiceLog.success("Loaded variables from .env file!")
    else:
        if 'HOPSWORKS_API_KEY' not in os.environ:
            NiceLog.error("Add add HOPSWORKS_API_KEY to your .env file!")
        else:
            NiceLog.error("Failed to load .env file!")
        exit(1)

if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        stub.deploy(modal_stub_name)
        with stub.run():
            f()
