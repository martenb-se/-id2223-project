# Service
from datetime import timedelta

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
modal_stub_name = "vessel-backfill-pipeline"
modal_image_libraries = ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image", "pytz"]
model_run_every_n_days = 1
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
    # NiceLog.info(f"Getting model named {BGColors.HEADER}{model_bridge_name}{BGColors.ENDC} (version {model_bridge_version})...")
    # try:
    #     model_bridge: hsml_model.Model = mr.get_model(model_bridge_name, version=model_bridge_version)
    # except RestAPIError as e:
    #     NiceLog.error(f"Failed to get model from Hopsworks model registry. Reason: {e}")
    #     return
    # except Exception as e:
    #     NiceLog.error(f"Unexpected error when trying to get model from Hopsworks model registry. Reason: {e}")
    #     return
    # NiceLog.info(
    #     f"Model description: {BGColors.HEADER}{model_bridge.description}{BGColors.ENDC} "
    #     f"(training accuracy: {model_bridge.training_metrics['accuracy']})")
    # 
    # NiceLog.info(f"Downloading model...")
    # model_dir = model_bridge.download()
    # NiceLog.success(f"Model downloaded to: {BGColors.HEADER}{model_dir}{BGColors.ENDC}")
    # 
    # bridge_local_model = joblib.load(model_dir + f"/{model_bridge_name}.pkl")
    # NiceLog.ok(f"Initialized locally downloaded model "
    #            f"({BGColors.HEADER}{model_dir + '/bridge_model.pkl'}{BGColors.ENDC})")

    #
    # ----------------------- VESSEL - FW
    #

    # TODO: Enable me, should be used instead of the Feature Group below
    # # Get Feature View for vessel data
    # NiceLog.info(f"Getting {BGColors.HEADER}{fw_vessel_name}{BGColors.ENDC} "
    #              f"(version {fw_vessel_version}) feature view...")
    # try:
    #     vessel_fw: feature_view.FeatureView = fs.get_feature_view(name=fw_vessel_name, version=fw_vessel_version)
    # except RestAPIError as e:
    #     NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
    #     return
    # except Exception as e:
    #     NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
    #     return
    # NiceLog.info(f"Feature group is named: {BGColors.HEADER}{vessel_fw.name}{BGColors.ENDC} ({vessel_fw.description})")
    # vessel_batch_data: pd.DataFrame = vessel_fw.get_batch_data()
    # NiceLog.ok(f"Gotten feature view as a DataFrame.")
    # pprint(vessel_batch_data.describe())

    #
    # ----------------------- PROCESS VESSEL DATA
    # ----------------------- GET CURRENT TIME SPAN
    #

    # Get current date and time
    date_time = datetime.now(pytz.utc)

    # TODO: Remove
    # Set time to 2024-01-05 XX:12:24
    date_time = datetime(2024, 1, 5, np.random.randint(0, 24), 12, 24, tzinfo=pytz.utc)

    # Adjust the minute to the nearest half-hour mark
    adjusted_minute = date_time.minute - (date_time.minute % 30)
    adjusted_date_time = date_time.replace(minute=adjusted_minute, second=0, microsecond=0)

    # Adjust the date to the nearest full hour mark
    adusted_hour_date_time = date_time.replace(minute=0, second=0, microsecond=0)

    # Adjust the date to the nearest 2 hour mark
    adjusted_2_hour = date_time.hour - (date_time.hour % 2)
    adjusted_2_hour_date_time = date_time.replace(hour=adjusted_2_hour, minute=0, second=0, microsecond=0)

    # Adjust the date to the nearest day mark
    adjusted_day_date_time = date_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Define time spans
    thirty_minute_span = (adjusted_date_time - timedelta(minutes=30), adjusted_date_time)
    one_hour_span = (adusted_hour_date_time, adusted_hour_date_time + timedelta(hours=1))
    two_hour_span = (adjusted_2_hour_date_time, adjusted_2_hour_date_time + timedelta(hours=2))
    today_span = (adjusted_day_date_time, adjusted_day_date_time + timedelta(days=1))

    # Print time spans
    NiceLog.info(f"Thirty minute span: {thirty_minute_span[0]} to {thirty_minute_span[1]}")
    NiceLog.info(f"One hour span: {one_hour_span[0]} to {one_hour_span[1]}")
    NiceLog.info(f"Two hour span: {two_hour_span[0]} to {two_hour_span[1]}")
    NiceLog.info(f"Today span: {today_span[0]} to {today_span[1]}")

    # TODO: Enable for one, two or today spans maybe..

    #
    # ----------------------- LATEST ADDITION
    #

    # TODO: Should actually use Feature View later (uses the Feature Group now)
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
    # pprint(vessel_df.describe())
    # pprint(vessel_df.tail())

    # Alter timestamps
    vessels_timestamp_fix = vessel_df.copy()
    vessels_timestamp_fix['time'] = pd.to_datetime(vessels_timestamp_fix['time'].str.replace(" UTC", ""),
                                                   format='%Y-%m-%d %H:%M:%S.%f %z')

    # Get entries in vessels_timestamp_fix that are in the last 30 minutes
    vessels_latest_min_30 = \
        vessels_timestamp_fix[(vessels_timestamp_fix['time'] > thirty_minute_span[0]) &
                              (vessels_timestamp_fix['time'] <= thirty_minute_span[1])]

    # Print info about timestamps in vessels_latest_min_30
    NiceLog.info(f"Number of entries in vessels_latest_min_30: {len(vessels_latest_min_30)}")

    if len(vessels_latest_min_30) > 0:
        NiceLog.info(f"First entry in vessels_latest_min_30: {vessels_latest_min_30.iloc[0]['time']}")
        NiceLog.info(f"Last entry in vessels_latest_min_30: {vessels_latest_min_30.iloc[-1]['time']}")

    else:
        NiceLog.info(f"No entries in vessels_latest_min_30, there's nothing to run the model on.")
        return

    # Imported from training_pipeline.ipynb
    def calculate_angle(boat_lat, boat_long, bridge_lat, bridge_long):
        delta_longitude = bridge_long - boat_long
        x = np.cos(np.radians(bridge_lat)) * np.sin(np.radians(delta_longitude))
        y = np.cos(np.radians(boat_lat)) * np.sin(np.radians(bridge_lat)) - np.sin(np.radians(boat_lat)) * np.cos(
            np.radians(bridge_lat)) * np.cos(np.radians(delta_longitude))
        angle = np.degrees(np.arctan2(x, y))
        return angle

    # Modified from training_pipeline.ipynb to aggregate all data to one row
    def process_vessels_data(df):
        df = df.copy()

        df['is_moving'] = df['ship_speed'] > 0
        df['angle_to_bridge'] = calculate_angle(df['latitude'], df['longitude'], 59.19985816310993, 17.628248643747433)
        df['is_moving_towards_bridge'] = np.abs(
            df['ship_heading'] - df['angle_to_bridge']) <= 10  # within +/- 10 degrees

        # Aggregate necessary data for the entire DataFrame, and
        # transpose to turn the aggregated Series into a single-row DataFrame
        aggregated_data = df.agg({
            'is_moving': 'sum',
            'ship_id': 'nunique',
            'is_moving_towards_bridge': 'sum',
            'width': 'mean',
            'length': 'mean'
        }).to_frame().T

        # Rename columns
        aggregated_data.rename(columns={
            'ship_id': 'unique_boats_count',
            'width': 'average_width',
            'length': 'average_length',
            'is_moving': 'moving_boats_count',
            'is_moving_towards_bridge': 'boats_moving_towards_bridge_count'
        }, inplace=True)

        return aggregated_data

    # Aggregate all rows in vessels_latest_min_30 to one row
    aggregated_latest_min_30_data = process_vessels_data(vessels_latest_min_30)

    NiceLog.info(f"Data to be used for input to the model:")
    NiceLog.info(f" - Moving boats count: {aggregated_latest_min_30_data['moving_boats_count'].values[0]}")
    NiceLog.info(f" - Unique boats count: {aggregated_latest_min_30_data['unique_boats_count'].values[0]}")
    NiceLog.info(f" - Boats moving towards bridge count: {aggregated_latest_min_30_data['boats_moving_towards_bridge_count'].values[0]}")
    NiceLog.info(f" - Average width: {aggregated_latest_min_30_data['average_width'].values[0]}")
    NiceLog.info(f" - Average length: {aggregated_latest_min_30_data['average_length'].values[0]}")

    # TODO: Add operation to delete all images on Hopsworks because the overwrite=True is buggy...

    #
    # ----------------------- BRIDGE - PREDICT
    #

    # TODO: Enable me
    # NiceLog.info(f"Predicting bridge opening stored in feature view using model {BGColors.HEADER}{model_bridge_name}{BGColors.ENDC} "
    #              f"(version {model_bridge_version})...")
    #
    # bridge_y_pred = bridge_local_model.predict(vessel_batch_data)
    # NiceLog.ok("Done")
    #
    # offset = 1
    # bridge_open = bridge_y_pred[bridge_y_pred.size - offset]

    # TODO: Remove me
    # Perform a random fake prediction between 0 and 1 but with 90% chance of 0
    import random
    bridge_open = random.choices([0, 1], weights=[0.9, 0.1], k=1)[0]

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
            else:
                return

        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload latest predicted bridge state to Hopsworks. Reason: {e}")

            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
            else:
                return

    NiceLog.success(f"Latest bridge state prediction uploaded to Hopsworks")

    #
    # ----------------------- BRIDGE - ACTUAL
    #

    bridge_df = pd.read_json('https://api.sodertalje.se/getAllBridgestat?start=2024-01-03&end=2025-12-31')

    # Get the timezone for Stockholm, Sweden
    stockholm_tz = pytz.timezone('Europe/Stockholm')

    # Convert 'formatted_time' to datetime objects in Stockholm timezone
    bridge_df['formatted_time'] = pd.to_datetime(bridge_df['formatted_time']).dt.tz_localize(stockholm_tz)

    # Convert the Stockholm times to UTC
    bridge_df['time'] = bridge_df['formatted_time'].dt.tz_convert('UTC')

    # Get entries in vessels_timestamp_fix that are in the last 30 minutes
    bridge_latest_min_30 = \
        bridge_df[(bridge_df['time'] > thirty_minute_span[0]) &
                  (bridge_df['time'] <= thirty_minute_span[1])]

    # Print how many entries were in the last 30 minutes
    NiceLog.info(f"Number of entries in bridge_latest_min_30: {len(bridge_latest_min_30)}")

    bridge_state = 0

    if len(bridge_latest_min_30) > 0:
        # See if any entry has "state" = 1
        bridge_open = bridge_latest_min_30[bridge_latest_min_30['state'] == 1]

        # If there are any entries with "state" = 1, the bridge is open
        if len(bridge_open) > 0:
            bridge_state = 1

    # If there are no entries in the last 30 minutes, get the latest entry before that
    elif len(bridge_latest_min_30) == 0:
        bridge_before_latest_min_30 = bridge_df[(bridge_df['time'] <= thirty_minute_span[1])]

        # Get the entry with the latest time
        latest_bridge = bridge_before_latest_min_30.iloc[bridge_before_latest_min_30['time'].idxmax()]

        # Get bridge "state"
        bridge_state = latest_bridge['state']

    # Print info about bridge state (0 = closed, 1 = open)
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
            else:
                return

        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload latest bridge state ground truth to Hopsworks. "
                          f"Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
            else:
                return

    NiceLog.success(f"Latest bridge state ground truth uploaded to Hopsworks")

    #
    # ----------------------- MONITOR - INSERT
    #

    time_from = thirty_minute_span[0].strftime("%Y-%m-%d %H:%M:%S.%f %z")
    time_to = thirty_minute_span[1].strftime("%Y-%m-%d %H:%M:%S.%f %z")

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
        'datetime_from': [time_from],
        'datetime_to': [time_to],
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
            else:
                return
        except Exception as e:
            NiceLog.error(f"Unexpected error and failed to upload image of most recent monitor entries to Hopsworks. "
                          f"Reason: {e}")
            if trials > 0:
                NiceLog.error(f"Trying to upload to Hopsworks again...")
                trials -= 1
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

    NiceLog.info(f"Stub should run every: {BGColors.HEADER}{model_run_every_n_days}{BGColors.ENDC} day(s)")

    if sys.argv[1] == "run":
        NiceLog.info(f"But this is just a {BGColors.HEADER}one time{BGColors.ENDC} test.")


    @stub.function(image=image, schedule=modal.Period(days=model_run_every_n_days),
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
