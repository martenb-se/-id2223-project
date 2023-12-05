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

# Error help
from hopsworks import RestAPIError

# AISStream
import asyncio
import websockets
import json
from datetime import datetime, timezone

# Data
import pandas as pd

# Settings
# - Modal
modal_stub_name = "aisstream-data-collector"
modal_image_libraries = ["hopsworks"]
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# Names
# - Feature Groups
fg_vessel_name = "vessel"
fg_vessel_version = 1

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


async def connect_ais_stream(api_key, skip_message_types=None, timeout=None, vessel_fg=None):
    if skip_message_types is None:
        skip_message_types = []

    vessel_data_log = []

    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": api_key,
            "BoundingBoxes": [[
                [61.345101156944445, 15.048625225719368],
                [57.11950799373074, 21.76127140685429]
            ]]
        }

        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        # Log time of start
        start_time = datetime.now(timezone.utc)
        push_time = start_time
        log_time = start_time

        async for message_json in websocket:
            # Vessel data
            vessel_data = {
                "ship_id": None,
                "ship_name": None,
                "latitude": None,
                "longitude": None,
                "time": None,
                "dimension": None,
                "length": None,
                "eta": None,
                "destination": None,
                "ship_type": None,
                "message_type": None
            }

            message = json.loads(message_json)
            message_type = message["MessageType"]

            if message_type in skip_message_types:
                continue

            vessel_data["message_type"] = message_type

            if "MetaData" in message and \
                    all(key in message['MetaData'] for key in
                        ['MMSI', 'ShipName', 'latitude', 'longitude', 'time_utc']):
                vessel_data["ship_id"] = message['MetaData']['MMSI']
                vessel_data["ship_name"] = message['MetaData']['ShipName']
                vessel_data["latitude"] = message['MetaData']['latitude']
                vessel_data["longitude"] = message['MetaData']['longitude']
                vessel_data["time"] = message['MetaData']['time_utc']

            if "Message" in message and "ShipStaticData" in message['Message']:
                if "Eta" in message['Message']['ShipStaticData']:
                    vessel_data["eta"] = message['Message']['ShipStaticData']['Eta']

                if "Destination" in message['Message']['ShipStaticData']:
                    vessel_data["destination"] = message['Message']['ShipStaticData']['Destination']

                if "Type" in message['Message']['ShipStaticData']:
                    vessel_data["ship_type"] = message['Message']['ShipStaticData']['Type']

            elif "Message" in message and "AidsToNavigationReport" in message['Message']:
                if "Type" in message['Message']['AidsToNavigationReport']:
                    vessel_data["dimension"] = message['Message']['AidsToNavigationReport']['Dimension']
                    vessel_data["length"] = \
                        message['Message']['AidsToNavigationReport']['Dimension']['A'] + \
                        message['Message']['AidsToNavigationReport']['Dimension']['B']

            # print(vessel_data)

            # Add to log
            vessel_data_log.append(vessel_data)

            # TODO: Should be 30 later when not testing
            # Every 10 seconds, print how many messages have been received
            if (datetime.now(timezone.utc) - log_time).total_seconds() > 10:
                log_time = datetime.now(timezone.utc)
                NiceLog.info(f"Received {len(vessel_data_log)} messages")

            # TODO: Should be 300 seconds later when not testing
            # Every 30 seconds, save the log to Hopsworks
            if (datetime.now(timezone.utc) - push_time).total_seconds() > 30:
                push_time = datetime.now(timezone.utc)
                NiceLog.info(f"Saving {len(vessel_data_log)} vessel info to Hopsworks...")

                # Print header
                print(pd.DataFrame(vessel_data_log).head())

                # Saving is disabled for now
                # fg_insert_info = vessel_fg.insert(pd.DataFrame(vessel_data_log))
                # fg_insert_job: feature_group.Job = fg_insert_info[0]
                # fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

                # # TODO: Use Great Expectations to validate the data in the future
                # if fg_insert_validation_info is None:
                #     NiceLog.info(f"Check job {fg_insert_job.name} manually at the provided link.")
                # else:
                #     if fg_insert_validation_info.success:
                #         NiceLog.success("Wine inserted into the feature group.")
                #     else:
                #         NiceLog.error("Could not insert wine into group! More info: ")
                #         pprint(fg_insert_validation_info)
                #         raise HopsworksFeatureGroupInsertError()

                # # Reset log
                vessel_data_log = []

            # If timeout seconds have passed, stop
            if timeout is not None and (datetime.now(timezone.utc) - start_time).total_seconds() > timeout:
                raise TimeoutException(f"{timeout} seconds have passed!")

def g():
    import hopsworks

    NiceLog.header(f"Running function to collect AISStream data and save it to Hopsworks")

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

    NiceLog.info(f"Connecting to AISStream with API key")
    NiceLog.info(f"Starting to listen for AISStream messages...")
    api_key = os.environ["AISSTREAM_API_KEY"]
    try:
        # TODO: This is the actual call when we are ready to run
        # asyncio.run(asyncio.run(connect_ais_stream(api_key, vessel_fg=vessel_fg)))

        # For testing:
        # - Run for 60 seconds
        asyncio.run(asyncio.run(connect_ais_stream(api_key, vessel_fg=vessel_fg, timeout=180)))

        # Skip these message types
        # message_type_filter = \
        #     ["PositionReport", "UnknownMessage", "DataLinkManagementMessage", "StandardClassBPositionReport"]
        # asyncio.run(asyncio.run(connect_ais_stream(api_key, message_type_filter)))

    except TimeoutException as e:
        NiceLog.info(e)


# Initialize
if not LOCAL:
    NiceLog.info(f"Setting Modal stub name to: {BGColors.HEADER}{modal_stub_name}{BGColors.ENDC}")
    stub = modal.Stub(modal_stub_name)

    NiceLog.info(f"Creating a Modal image with Python libraries: {BGColors.HEADER}{', '.join(modal_image_libraries)}"
                 f"{BGColors.ENDC}")
    image = modal.Image.debian_slim().pip_install(modal_image_libraries)

    if sys.argv[1] == "run":
        NiceLog.info(f"But this is just a {BGColors.HEADER}one time{BGColors.ENDC} test.")

    @stub.function(image=image, secret=modal.Secret.from_name(hopsworks_api_key_modal_secret_name))
    def f():
        g()

# Load local environment
else:
    NiceLog.info("Loading local environment...")
    if load_dotenv() and 'AISSTREAM_API_KEY' in os.environ:
        NiceLog.success("Loaded variables from .env file!")
    else:
        if 'AISSTREAM_API_KEY' not in os.environ:
            NiceLog.error("Add add AISSTREAM_API_KEY to your .env file!")
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
