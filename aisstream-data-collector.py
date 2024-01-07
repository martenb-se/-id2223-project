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
modal_image_libraries = ["hopsworks", "websockets"]
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# - AISStream
aisstream_api_key_modal_secret_name = "aisstream-api-key"  # Load secret to environment
# Names
# - Feature Groups
fg_vessel_name = "vessel"
fg_vessel_version = 1

# Data
# - Filter
filter_only_get_messages_with_type = ["PositionReport"]

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


async def connect_ais_stream(api_key, skip_message_types=None, filter_message_type=None, timeout=None, vessel_fg=None):
    if skip_message_types is None:
        skip_message_types = []

    if filter_message_type is None:
        filter_message_type = []

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

        log_every = 60
        push_every = 600
        total_logs = 0
        logs_before_push = push_every / log_every
        if logs_before_push % 1 != 0:
            logs_before_push = int(logs_before_push) + 1
        else:
            logs_before_push = int(logs_before_push)

        # Log updated and added ships
        added_ships = {}

        total_updates = 0

        async for message_json in websocket:
            # Vessel data
            vessel_data = {
                "ship_id": None,
                "ship_name": None,
                "latitude": None,
                "longitude": None,
                "time": None,
                # "dimension": None,
                "dim_a": None,
                "dim_b": None,
                "dim_c": None,
                "dim_d": None,
                "length": None,
                "width": None,
                # "eta": None,
                "eta_day": None,
                "eta_hour": None,
                "eta_minute": None,
                "eta_month": None,
                "destination": None,
                "ship_type": None,
                "ship_heading": None,
                "ship_speed": None,
            }

            message = json.loads(message_json)
            message_type = message["MessageType"]

            # Use skip or...
            if message_type in skip_message_types:
                continue

            # ...use filter (either should be used)
            if len(filter_message_type) > 0 and message_type not in filter_message_type:
                continue

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
                    vessel_data["eta_day"] = message['Message']['ShipStaticData']['Eta']['Day']
                    vessel_data["eta_hour"] = message['Message']['ShipStaticData']['Eta']['Hour']
                    vessel_data["eta_minute"] = message['Message']['ShipStaticData']['Eta']['Minute']
                    vessel_data["eta_month"] = message['Message']['ShipStaticData']['Eta']['Month']

                if "Destination" in message['Message']['ShipStaticData']:
                    vessel_data["destination"] = message['Message']['ShipStaticData']['Destination']

                if "Type" in message['Message']['ShipStaticData']:
                    vessel_data["ship_type"] = message['Message']['ShipStaticData']['Type']

                if "Dimension" in message['Message']['ShipStaticData']:
                    vessel_data["dim_a"] = message['Message']['ShipStaticData']['Dimension']['A']
                    vessel_data["dim_b"] = message['Message']['ShipStaticData']['Dimension']['B']
                    vessel_data["dim_c"] = message['Message']['ShipStaticData']['Dimension']['C']
                    vessel_data["dim_d"] = message['Message']['ShipStaticData']['Dimension']['D']

                    vessel_data["length"] = \
                        message['Message']['ShipStaticData']['Dimension']['A'] + \
                        message['Message']['ShipStaticData']['Dimension']['B']

                    vessel_data["width"] = \
                        message['Message']['ShipStaticData']['Dimension']['C'] + \
                        message['Message']['ShipStaticData']['Dimension']['D']

            elif "Message" in message and "AidsToNavigationReport" in message['Message']:
                if "Dimension" in message['Message']['AidsToNavigationReport']:
                    vessel_data["dim_a"] = message['Message']['AidsToNavigationReport']['Dimension']['A']
                    vessel_data["dim_b"] = message['Message']['AidsToNavigationReport']['Dimension']['B']
                    vessel_data["dim_c"] = message['Message']['AidsToNavigationReport']['Dimension']['C']
                    vessel_data["dim_d"] = message['Message']['AidsToNavigationReport']['Dimension']['D']

                    vessel_data["length"] = \
                        message['Message']['AidsToNavigationReport']['Dimension']['A'] + \
                        message['Message']['AidsToNavigationReport']['Dimension']['B']

                    vessel_data["width"] = \
                        message['Message']['AidsToNavigationReport']['Dimension']['C'] + \
                        message['Message']['AidsToNavigationReport']['Dimension']['D']

                if "Type" in message['Message']['AidsToNavigationReport']:
                    vessel_data["ship_type"] = message['Message']['AidsToNavigationReport']['Type']

            elif "Message" in message and "PositionReport" in message['Message']:
                if "TrueHeading" in message['Message']['PositionReport']:
                    vessel_data["ship_heading"] = message['Message']['PositionReport']['TrueHeading']

                if "Sog" in message['Message']['PositionReport']:
                    vessel_data["ship_speed"] = message['Message']['PositionReport']['Sog']

            elif "Message" in message and "StandardClassBPositionReport" in message['Message']:
                if "TrueHeading" in message['Message']['StandardClassBPositionReport']:
                    vessel_data["ship_heading"] = message['Message']['StandardClassBPositionReport']['TrueHeading']

                if "Sog" in message['Message']['StandardClassBPositionReport']:
                    vessel_data["ship_speed"] = message['Message']['StandardClassBPositionReport']['Sog']

            elif "Message" in message and "ExtendedClassBPositionReport" in message['Message']:
                if "Dimension" in message['Message']['ExtendedClassBPositionReport']:
                    vessel_data["dim_a"] = message['Message']['ExtendedClassBPositionReport']['Dimension']['A']
                    vessel_data["dim_b"] = message['Message']['ExtendedClassBPositionReport']['Dimension']['B']
                    vessel_data["dim_c"] = message['Message']['ExtendedClassBPositionReport']['Dimension']['C']
                    vessel_data["dim_d"] = message['Message']['ExtendedClassBPositionReport']['Dimension']['D']

                    vessel_data["length"] = \
                        message['Message']['ExtendedClassBPositionReport']['Dimension']['A'] + \
                        message['Message']['ExtendedClassBPositionReport']['Dimension']['B']

                    vessel_data["width"] = \
                        message['Message']['ExtendedClassBPositionReport']['Dimension']['C'] + \
                        message['Message']['ExtendedClassBPositionReport']['Dimension']['D']

                if "Type" in message['Message']['ExtendedClassBPositionReport']:
                    vessel_data["ship_type"] = message['Message']['ExtendedClassBPositionReport']['Type']

                if "TrueHeading" in message['Message']['ExtendedClassBPositionReport']:
                    vessel_data["ship_heading"] = message['Message']['ExtendedClassBPositionReport']['TrueHeading']

            elif "Message" in message and "StaticDataReport" in message['Message']:
                if "ReportB" in message['Message']['StaticDataReport']:
                    if "Dimension" in message['Message']['StaticDataReport']["ReportB"]:
                        vessel_data["dim_a"] = message['Message']['StaticDataReport']["ReportB"]['Dimension']['A']
                        vessel_data["dim_b"] = message['Message']['StaticDataReport']["ReportB"]['Dimension']['B']
                        vessel_data["dim_c"] = message['Message']['StaticDataReport']["ReportB"]['Dimension']['C']
                        vessel_data["dim_d"] = message['Message']['StaticDataReport']["ReportB"]['Dimension']['D']

                        vessel_data["length"] = \
                            message['Message']['StaticDataReport']["ReportB"]['Dimension']['A'] + \
                            message['Message']['StaticDataReport']["ReportB"]['Dimension']['B']

                        vessel_data["width"] = \
                            message['Message']['StaticDataReport']["ReportB"]['Dimension']['C'] + \
                            message['Message']['StaticDataReport']["ReportB"]['Dimension']['D']

                    if "ShipType" in message['Message']['StaticDataReport']["ReportB"]:
                        vessel_data["ship_type"] = message['Message']['StaticDataReport']["ReportB"]['ShipType']

            elif "Message" in message and "LongRangeAisBroadcastMessage" in message['Message']:
                if "ShipType" in message['Message']['LongRangeAisBroadcastMessage']:
                    vessel_data["ship_type"] = message['Message']['LongRangeAisBroadcastMessage']['ShipType']

            # print(vessel_data)

            # See if ship_id is in the log
            if vessel_data["ship_id"] in [vessel["ship_id"] for vessel in vessel_data_log]:
                # If it is, update the log
                for vessel in vessel_data_log:
                    if vessel["ship_id"] == vessel_data["ship_id"]:
                        if "message_type_counts" not in vessel:
                            vessel["message_type_counts"] = {message_type: 1}
                        elif message_type not in vessel["message_type_counts"]:
                            vessel["message_type_counts"][message_type] = 1
                        else:
                            vessel["message_type_counts"][message_type] += 1

                        # Update where value is not None
                        for key in vessel_data:
                            if vessel_data[key] is not None:
                                # If updated name does not match the old name, print error
                                if key == "ship_name" and vessel[key] is not None and vessel[key] != vessel_data[key]:
                                    NiceLog.warn(f"[{vessel['ship_id']}] "
                                                 f"Ship name has changed from {vessel[key]} to {vessel_data[key]}")
                                    if vessel_data[key].strip() == "":
                                        NiceLog.error(f"[{vessel['ship_id']}] "
                                                      f"Updated ship name is empty, keeping old name: {vessel[key]}")
                                        continue
                                    elif "@" in vessel_data[key]:
                                        NiceLog.error(f"[{vessel['ship_id']}] "
                                                      f"Updated ship name contains '@', keeping old name: {vessel[key]}")
                                        continue

                                # If updated ship_type does not match the old ship_type, print error
                                elif key == "ship_type" and vessel[key] is not None and vessel[key] != vessel_data[key]:
                                    NiceLog.warn(f"[{vessel['ship_id']}] "
                                                 f"Ship type has changed from {vessel[key]} to {vessel_data[key]}")
                                    if vessel[key] != 0 and vessel_data[key] == 0:
                                        NiceLog.error(f"[{vessel['ship_id']}] "
                                                      f"Updated ship type is 0, keeping old ship type: {vessel[key]}")
                                        continue

                                # If updated length or dimensions does not match the old values, print error
                                elif (key == "length" or key == "dim_a" or key == "dim_b") and \
                                        vessel["length"] is not None and vessel[key] != vessel_data[key]:
                                    NiceLog.warn(f"[{vessel['ship_id']}] "
                                                 f"Ship {key} has changed from {vessel[key]} to {vessel_data[key]}")
                                    if vessel[key] != 0 and vessel_data[key] == 0:
                                        NiceLog.error(f"[{vessel['ship_id']}] "
                                                      f"Updated ship length is 0, "
                                                      f"keeping old ship {key}: {vessel[key]}")
                                        continue

                                # If updated width or dimensions does not match the old values, print error
                                elif (key == "width" or key == "dim_c" or key == "dim_d") and \
                                        vessel["width"] is not None and vessel[key] != vessel_data[key]:
                                    NiceLog.warn(f"[{vessel['ship_id']}] "
                                                 f"Ship {key} has changed from {vessel[key]} to {vessel_data[key]}")
                                    if vessel[key] != 0 and vessel_data[key] == 0:
                                        NiceLog.error(f"[{vessel['ship_id']}] "
                                                      f"Updated ship width is 0, "
                                                      f"keeping old ship {key}: {vessel[key]}")
                                        continue

                                vessel[key] = vessel_data[key]

            else:
                # If it is not, add it to the log
                vessel_data["message_type_counts"] = {message_type: 1}
                vessel_data_log.append(vessel_data)

            total_updates += 1

            # Update added ships
            if vessel_data["ship_id"] in added_ships:
                added_ships[vessel_data["ship_id"]]["updates"] += 1
            else:
                added_ships[vessel_data["ship_id"]] = {
                    'updates': 1,
                    'id': vessel_data["ship_id"],
                    'name': vessel_data["ship_name"]
                }

            time_now = datetime.now(timezone.utc)

            # TODO: Should be 30 later when not testing
            # Print how many messages have been received
            if (time_now - log_time).total_seconds() >= log_every:
                log_time = time_now
                NiceLog.info(
                    f"Have received {len(vessel_data_log)} unique vessels "
                    f"(from a total of {total_updates} updates) since {push_time}")

                most_updates = sorted(added_ships.items(), key=lambda x: x[1]['updates'], reverse=True)
                NiceLog.info(f"Top 3 ships with most updates:")
                for i in range(min(3, len(most_updates))):
                    NiceLog.info(f"    {i+1}. Ship {most_updates[i][1]['id']} "
                                 f"has {most_updates[i][1]['updates']} updates (name: {most_updates[i][1]['name']})")

                total_logs += 1

                # TODO: Should be 300 seconds later when not testing
                # Save the log to Hopsworks
                if total_logs >= logs_before_push:
                    push_time = time_now

                    # Print header
                    # print(pd.DataFrame(vessel_data_log).head())

                    ### # Get ships with most updates
                    ### vessel_data_log_top = []
                    ### most_updates = sorted(added_ships.items(), key=lambda x: x[1]['updates'], reverse=True)
                    ### NiceLog.info(f"Top 10 ships with most updates:")
                    ### for i in range(min(10, len(most_updates))):
                    ###     NiceLog.info(f"    {i+1}. Ship {most_updates[i][1]['id']} "
                    ###                  f"has {most_updates[i][1]['updates']} updates (name: {most_updates[i][1]['name']})")
                    ###
                    ###     # Find vessel in vessel_data_log
                    ###     for vessel in vessel_data_log:
                    ###         if vessel["ship_id"] == most_updates[i][1]['id']:
                    ###             vessel_data_log_top.append(vessel)
                    ###
                    ### # Print top as a table
                    ### print(pd.DataFrame(vessel_data_log_top))

                    vessel_df = pd.DataFrame(vessel_data_log)

                    # Filter out vessels with complete information
                    vessel_df_filter = vessel_df[
                        vessel_df["ship_id"].notnull() &
                        # Name is not necessary (but nice to have)
                        # vessel_df["ship_name"].notnull() &
                        # (vessel_df["ship_name"] != "") &
                        vessel_df["length"].notnull() &
                        vessel_df["width"].notnull() &
                        (vessel_df["length"] != 0) &
                        (vessel_df["width"] != 0) &
                        # It seems that ETA is not always reliable, so ignore it as a necessary field
                        # vessel_df["eta_day"].notnull() &
                        # vessel_df["eta_hour"].notnull() &
                        # vessel_df["eta_minute"].notnull() &
                        # vessel_df["eta_month"].notnull() &
                        # ((vessel_df["eta_day"] != 0) | (vessel_df["eta_hour"] != 0) |
                        #  (vessel_df["eta_minute"] != 0) | (vessel_df["eta_month"] != 0)) &
                        # Destination is not necessary (but nice to have)
                        # vessel_df["destination"].notnull() &
                        # (vessel_df["destination"] != "") &
                        # Type might be necessary...
                        vessel_df["ship_type"].notnull() &
                        (vessel_df["ship_type"] != 0) &
                        vessel_df["ship_heading"].notnull() &
                        vessel_df["ship_speed"].notnull()
                        ]

                    # Print info
                    NiceLog.info(f"Have {len(vessel_df_filter)} vessels with complete information out of "
                                 f"{len(vessel_df)} vessels.")

                    # Remove "message_type_counts" column before pushing to Hopsworks
                    vessel_df_push = vessel_df_filter.drop(columns=["message_type_counts"])

                    # Print first 10 rows
                    pd.set_option('display.max_rows', 500)
                    pd.set_option('display.max_columns', 500)
                    pd.set_option('display.width', 1000)
                    print(vessel_df_push.head(10))

                    NiceLog.info(f"Saving {len(vessel_df_push)} vessel info to Hopsworks...")

                    # TODO: Temporary fix for saving to CSV
                    # Save to CSV file if it does not exist
                    if not os.path.exists("./data/vessel_data_tmp.csv"):
                        vessel_df_push.to_csv("./data/vessel_data_tmp.csv", header=True, index=False)
                    else:
                        vessel_df_push.to_csv("./data/vessel_data_tmp.csv", mode='a', header=False, index=False)

                    # Saving is disabled for now
                    # fg_insert_info = vessel_fg.insert(pd.DataFrame(vessel_df_filter))
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

                    # Reset log
                    vessel_data_log = []
                    total_logs = 0
                    total_updates = 0

                    # Reset added ships
                    added_ships = {}

            # If timeout seconds have passed, stop
            if timeout is not None and (datetime.now(timezone.utc) - start_time).total_seconds() > timeout:
                raise TimeoutException(f"{timeout} seconds have passed!")


def g():
    import hopsworks

    NiceLog.header(f"Running function to collect AISStream data and save it to Hopsworks")

    if "HOPSWORKS_API_KEY" not in os.environ:
        NiceLog.error(f"Failed to log in to Hopsworks. HOPSWORKS_API_KEY is not in the current environment.")
        raise HopsworksNoAPIKey()

    if "AISSTREAM_API_KEY" not in os.environ:
        NiceLog.error(f"Failed to log in to Hopsworks. AISSTREAM_API_KEY is not in the current environment.")
        raise AISStreamNoAPIKey()

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
        asyncio.run(asyncio.run(connect_ais_stream(api_key, vessel_fg=vessel_fg)))

        # For testing:
        # - Run for 60 seconds
        # asyncio.run(asyncio.run(connect_ais_stream(
        #     api_key, vessel_fg=vessel_fg, timeout=None,
        #     # filter_message_type=filter_only_get_messages_with_type
        # )))

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

    @stub.function(
        image=image,
        secrets=[
            modal.Secret.from_name(hopsworks_api_key_modal_secret_name),
            modal.Secret.from_name(aisstream_api_key_modal_secret_name),
        ]
    )
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
