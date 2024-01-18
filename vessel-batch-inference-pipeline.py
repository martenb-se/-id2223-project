# Service
import re
from datetime import timedelta, time

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
from project_exceptions import (
    HopsworksNoAPIKey, HopsworksLoginError, HopsworksGetFeatureStoreError, HopsworksGetFeatureGroupError,
    HopsworksQueryReadError, HopsworksFeatureGroupInsertError)

# IDE help
from hsfs import feature_store, feature_group, feature_view
from hopsworks import project as HopsworksProject
from hsfs.constructor import query as hsfs_query
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult
from hsml import model_registry
from hsml.python import model as hsml_model
# Error help
from hopsworks import RestAPIError

# Settings
# - Modal
modal_stub_name = "vessel-batch-inference-pipeline"
modal_image_libraries = ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image", "pytz", "xgboost"]
model_run_every_n_hours = 6
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# Names
# - Models
models_dir = "models"
model_bridge_name = "bridge_model"
model_bridge_version = 1
# - Feature Groups
fg_vessel_name = "vessel_processed"
fg_vessel_version = 1
# - Feature Views
fw_vessel_name = "vessel_processed"
fw_vessel_version = 1
# - Monitor
fg_monitor_name = "bridge_predictions"
fg_monitor_version = 1
dir_bridge_saves = "latest_bridge"
file_bridge_predict_save = f"latest_bridge.png"
file_bridge_actual_save = f"actual_bridge.png"
hopsworks_images_location = f"Resources/images/{dir_bridge_saves}"
file_dataframe_save = "df_recent.png"
num_monitor_entries_to_export = 4

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


def g():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import hopsworks
    import joblib
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    # from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import pytz
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report

    def bridge_state_url(cur_bridge_state):
        return ("https://raw.githubusercontent.com/martenb-se/id2223-project/main/images/" +
                ("open" if cur_bridge_state == 1 else "closed") + ".png")

    # Make sure directory exists
    Path(dir_bridge_saves).mkdir(parents=True, exist_ok=True)

    NiceLog.header(f"Running function to predict the bridge opening, save latest prediction, get a table and upload "
                   f"data to Hopsworks!")
    NiceLog.info("Logging in to Hopsworks...")

    try:
        project: HopsworksProject.Project = hopsworks.login()
    except RestAPIError as e:
        NiceLog.error(f"Failed to log in to Hopsworks. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to log in to Hopsworks. Reason: {e}")
        return

    NiceLog.success("Logged in to Hopsworks!")
    NiceLog.info(f"Active hopsworks project: {BGColors.HEADER}{project.name}{BGColors.ENDC} ({project.description})")
    NiceLog.info(f"Project created at: {BGColors.HEADER}{project.created}{BGColors.ENDC}")

    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} feature store...")
    try:
        fs: feature_store.FeatureStore = project.get_feature_store()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature store from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature store from Hopsworks project. Reason: {e}")
        return

    NiceLog.success("Gotten feature store!")
    NiceLog.info(f"Feature store is named: {BGColors.HEADER}{fs.name}{BGColors.ENDC})")

    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} model registry...")
    try:
        mr: model_registry.ModelRegistry = project.get_model_registry()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get model registry from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get model registry from Hopsworks project. Reason: {e}")
        return

    #
    # ----------------------- BRIDGE - MODEL
    #

    # TODO: Enable me, should be used instead of the fake prediction below
    NiceLog.info(f"Getting model named {BGColors.HEADER}{model_bridge_name}{BGColors.ENDC} (version {model_bridge_version})...")
    try:
        model_bridge: hsml_model.Model = mr.get_model(model_bridge_name, version=model_bridge_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get model from Hopsworks model registry. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get model from Hopsworks model registry. Reason: {e}")
        return
    NiceLog.info(
        f"Model description: {BGColors.HEADER}{model_bridge.description}{BGColors.ENDC} "
        f"(training accuracy: {model_bridge.training_metrics['accuracy']})")

    NiceLog.info(f"Downloading model...")
    model_dir = model_bridge.download()
    NiceLog.success(f"Model downloaded to: {BGColors.HEADER}{model_dir}{BGColors.ENDC}")

    bridge_local_model = joblib.load(model_dir + f"/{model_bridge_name}.pkl")
    NiceLog.ok(f"Initialized locally downloaded model "
               f"({BGColors.HEADER}{model_dir + f'/{model_bridge_name}.pkl'}{BGColors.ENDC})")


    #
    # ----------------------- LATEST ADDITION
    #

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

    NiceLog.info(f"Getting feature group as a DataFrame...")
    try:
        vessel_df: pd.DataFrame = vessel_fg.read()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project as DataFrame. "
                      f"Reason: {e}")
        return
    NiceLog.success(f"Gotten feature group as a DataFrame.")

    # Order vessel_df by time and get the latest row
    vessel_df = vessel_df.sort_values(by=['time'], ascending=False)

    # Get latest value in vessel_df
    vessel_df_latest = vessel_df.head(1)

    # Drop column "bridge_status" and "time" before running inference
    vessel_batch_data = vessel_df_latest.drop(columns=['bridge_status', 'time'])

    lag_count = 1

    # Show all data from vessel_batch_data
    NiceLog.info(f"All data from vessel_batch_data:")
    for column in vessel_df_latest:
        NiceLog.info(f" - {column}: {vessel_df_latest[column].values[0]}")

        # Find "_lag(\d+)" and get the largest X
        match = re.search(r'_lag(\d+)', column)
        if match:
            lag = int(match.group(1))
            if lag > lag_count:
                lag_count = lag

    # Print info about lag
    NiceLog.info(f"Lag count: {lag_count}")

    #
    # ----------------------- BRIDGE - PREDICT
    #

    # TODO: Enable me
    NiceLog.info(f"Predicting bridge opening stored in feature view using model {BGColors.HEADER}{model_bridge_name}{BGColors.ENDC} "
                 f"(version {model_bridge_version})...")

    bridge_y_pred = bridge_local_model.predict(vessel_batch_data)
    NiceLog.ok("Done")

    offset = 1
    bridge_open = bridge_y_pred[bridge_y_pred.size - offset]

    # Print info about prediction
    NiceLog.info(f"Latest bridge state prediction is: {BGColors.HEADER}{bridge_open}{BGColors.ENDC}")

    if bridge_open == 1:
        NiceLog.info(f"Latest bridge state prediction: {BGColors.HEADER}OPEN{BGColors.ENDC}")
    else:
        NiceLog.info(f"Latest bridge state prediction: {BGColors.HEADER}CLOSED{BGColors.ENDC}")

    # Image
    NiceLog.info(f"Saving an image of latest predicted bridge state to: {file_bridge_predict_save}")
    img = Image.open(requests.get(bridge_state_url(bridge_open), stream=True).raw)
    img.save(dir_bridge_saves + "/" + file_bridge_predict_save)
    NiceLog.ok("Done")

    trials = 3
    upload_done = False
    while trials > 0 and not upload_done:
        NiceLog.info(f"Uploading image of latest predicted bridge state to Hopsworks at: {hopsworks_images_location}")
        dataset_api = project.get_dataset_api()
        try:
            dataset_api.upload(dir_bridge_saves + "/" + file_bridge_predict_save, hopsworks_images_location, overwrite=True)
            upload_done = True
        except RestAPIError as e:
            NiceLog.error(f"Failed to upload latest predicted bridge state to Hopsworks. Reason: {e}")

            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return

        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload latest predicted bridge state to Hopsworks. Reason: {e}")

            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return

    NiceLog.success(f"Latest bridge state prediction uploaded to Hopsworks")

    # Save "bridge_status" column
    bridge_state = vessel_df_latest['bridge_status'].values[0]

    NiceLog.info(f"Bridge state: {bridge_state}")

    if bridge_state == 1:
        NiceLog.info(f"Latest bridge state was actually: {BGColors.HEADER}OPEN{BGColors.ENDC}")
    else:
        NiceLog.info(f"Latest bridge state was actually: {BGColors.HEADER}CLOSED{BGColors.ENDC}")

    # Image
    NiceLog.info(f"Saving an image of latest predicted bridge state to: {file_bridge_actual_save}")
    img = Image.open(requests.get(bridge_state_url(bridge_state), stream=True).raw)
    img.save(dir_bridge_saves + "/" + file_bridge_actual_save)


    trials = 3
    upload_done = False
    while trials > 0 and not upload_done:
        NiceLog.info(f"Uploading image of latest bridge state ground truth to Hopsworks at: {hopsworks_images_location}")
        dataset_api = project.get_dataset_api()
        try:
            dataset_api.upload(dir_bridge_saves + "/" + file_bridge_actual_save, hopsworks_images_location, overwrite=True)
            upload_done = True
        except RestAPIError as e:
            NiceLog.error(f"Failed to upload latest bridge state ground truth to Hopsworks. Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return

        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload latest bridge state ground truth to Hopsworks. "
                          f"Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return

    NiceLog.success(f"Latest bridge state ground truth uploaded to Hopsworks")

    #
    # ----------------------- MONITOR - INSERT
    #

    # Lag time
    time_now = vessel_df_latest['time'].values[0].astype(datetime)

    # Lag times
    # - End time of the period
    lag_time_to = time_now
    aggegation_time_frame = 15  # minutes
    # - Calculate the total time used for inference
    total_time_minutes = aggegation_time_frame * lag_count
    # - Calculate the start time of the period
    lag_time_from = lag_time_to - timedelta(minutes=total_time_minutes)

    # Inference times
    time_from = time_now
    time_to = time_from + timedelta(minutes=30)

    # Print info about time frame
    NiceLog.info(f"History frame: {lag_time_from} to {lag_time_to}")
    NiceLog.info(f"Inference frame: {time_from} to {time_to}")

    monitor_fg: feature_group.FeatureGroup = fs.get_or_create_feature_group(
        name=fg_monitor_name,
        version=fg_monitor_version,
        primary_key=["datetime_from", "datetime_to"],
        description="Bridge Opening Prediction/Outcome Monitoring"
    )
    NiceLog.ok(
        f"Got or created feature group: {BGColors.HEADER}{monitor_fg.name}{BGColors.ENDC} ({monitor_fg.description})")

    monitor_data = {
        'prediction': 'open' if bridge_open == 1 else 'closed',
        'outcome': 'open' if bridge_state == 1 else 'closed',
        'datetime_from': [time_from.strftime("%Y-%m-%d %H:%M:%S.%f %z")],
        'datetime_to': [time_to.strftime("%Y-%m-%d %H:%M:%S.%f %z")],
    }
    monitor_df = pd.DataFrame(monitor_data)
    NiceLog.ok(f"Created a DataFrame from latest prediction and ground truth.")
    pprint(monitor_df)

    NiceLog.info(f"Inserting the bridge prediction and ground truth to the "
                 f"{BGColors.HEADER}{fg_monitor_name}{BGColors.ENDC} feature group "
                 f"{BGColors.HEADER}asynchronously{BGColors.ENDC}...")
    fg_insert_info = monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})
    fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

    #
    # ----------------------- MONITOR - READ + ADD
    #

    NiceLog.info(f"Getting monitoring history feature group as a DataFrame...")
    try:
        history_df: pd.DataFrame = monitor_fg.read()
    except RestAPIError as e:
        NiceLog.error(
            f"Failed to get monitoring history feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get monitoring history feature group from Hopsworks project as "
                      f"DataFrame. Reason: {e}")
        return
    NiceLog.success(f"Gotten monitoring history feature group as a DataFrame.")
    # pprint(history_df.describe())

    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])
    NiceLog.ok(f"Added prediction to the monitoring history feature group:")
    pprint(history_df)

    #
    # ----------------------- MONITOR - MOST RECENT FETCH and UPLOAD
    #

    df_recent = history_df.tail(4)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent entries in monitoring history DataFrame:")
    pprint(df_recent)

    NiceLog.info(f"Exporting {num_monitor_entries_to_export} most recent entries to: {file_dataframe_save}")
    dfi.export(df_recent, dir_bridge_saves + "/" + file_dataframe_save, table_conversion = 'matplotlib')

    trials = 3
    upload_done = False
    while trials > 0 and not upload_done:
        NiceLog.info(f"Uploading image of {num_monitor_entries_to_export} most recent monitor entries to: "
                     f"{hopsworks_images_location}")
        dataset_api = project.get_dataset_api()
        try:
            dataset_api.upload(dir_bridge_saves + "/" + file_dataframe_save, hopsworks_images_location, overwrite=True)
            upload_done = True
        except RestAPIError as e:
            NiceLog.error(f"Failed to upload image of most recent monitor entries to Hopsworks. Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return
        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload image of most recent monitor entries to Hopsworks. "
                          f"Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
                time.sleep(3)
            else:
                return

    NiceLog.success(f"Image of most recent monitor entries uploaded to Hopsworks")

    #
    # ----------------------- MONITOR - MOST RECENT INFO
    #

    monitor_predictions = history_df[['prediction']]
    monitor_outcome = history_df[['outcome']]
    NiceLog.info(f"{num_monitor_entries_to_export} most recent predictions:")
    pprint(monitor_predictions)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent outcomes:")
    pprint(monitor_outcome)


# Initialize
if not LOCAL:
    NiceLog.info(f"Setting Modal stub name to: {BGColors.HEADER}{modal_stub_name}{BGColors.ENDC}")
    stub = modal.Stub(modal_stub_name)

    NiceLog.info(f"Creating a Modal image with Python libraries: {BGColors.HEADER}{', '.join(modal_image_libraries)}"
                 f"{BGColors.ENDC}")
    image = modal.Image.debian_slim().pip_install(modal_image_libraries)

    NiceLog.info(f"Stub should run every: {BGColors.HEADER}{model_run_every_n_hours}{BGColors.ENDC} hour(s)")

    if sys.argv[1] == "run":
        NiceLog.info(f"But this is just a {BGColors.HEADER}one time{BGColors.ENDC} test.")


    @stub.function(image=image, schedule=modal.Period(hours=model_run_every_n_hours),
                   secret=modal.Secret.from_name(hopsworks_api_key_modal_secret_name))
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
