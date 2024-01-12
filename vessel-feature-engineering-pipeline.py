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

# Data
import pandas as pd

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Settings
# - Modal
modal_stub_name = "vessel-train-pipeline"
modal_image_libraries = ["hopsworks", "joblib"]
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# Names
# - Feature Groups
# Raw data
fg_vessel_name = "vessel"
fg_vessel_version = 1
# Processed data
fg_vessel_processed_name = "vessel_processed"
fg_vessel_processed_version = 1

# Bridge data start date and end date
bridge_data_start = "2024-01-03"
bridge_data_end = "2025-12-31"

LOCAL = True

"""
Resamples time series data at given intervals (timestep) for each unique vessel,
filling missing values by backfilling from the previous available data point.
    
:param df: DataFrame containing the timestamped vessel data
:param df: String defining the time step, default '15T' 
:return: DataFrame containing the resampled data.
"""
def resample_and_fill_missing_values(df, time_step='15T'):
    unique_ships = df['ship_id'].unique()
    df = df.set_index('time')
    resampled_data = pd.DataFrame()

    for ship_id in unique_ships:
        ship_data = df[df['ship_id'] == ship_id]
        resampled_ship_data = ship_data.resample(time_step).bfill().reset_index()
        resampled_data = pd.concat([resampled_data, resampled_ship_data], ignore_index=True)

    return resampled_data

def is_within_area(row, area_polygon):
    boat_location = Point(row['longitude'], row['latitude'])
    return area_polygon.contains(boat_location)

def calculate_angle(boat_lat, boat_long, bridge_lat, bridge_long):
    delta_longitude = bridge_long - boat_long
    x = np.cos(np.radians(bridge_lat)) * np.sin(np.radians(delta_longitude))
    y = np.cos(np.radians(boat_lat)) * np.sin(np.radians(bridge_lat)) - np.sin(np.radians(boat_lat)) * np.cos(
        np.radians(bridge_lat)) * np.cos(np.radians(delta_longitude))
    angle = np.degrees(np.arctan2(x, y))

    return angle

def calculate_distance(boat_lat, boat_long, bridge_lat, bridge_long):
    # Calculate the distance using the Haversine formula https://en.wikipedia.org/wiki/Haversine_formula
    phi_1 = radians(boat_lat)
    phi_2 = radians(boat_long)

    delta_phi = radians(bridge_long - boat_long)
    delta_lambda = radians(bridge_lat - boat_lat)

    a = sin(delta_phi / 2) ** 2 + cos(phi_1) * cos(phi_2) * sin(delta_lambda / 2) ** 2
    
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    earth_radius = 6371000 # Earth radius in meters

    distance = earth_radius * c
    
    return distance

def process_vessels_data(df, time_step='15T'):
    df = df.copy()

    df['is_moving'] = df['ship_speed'] > 0
    df['angle_to_bridge'] = calculate_angle(df['latitude'], df['longitude'], 59.19985816310993, 17.628248643747433)
    df['is_moving_towards_bridge'] = np.abs(df['ship_heading'] - df['angle_to_bridge']) <= 10  # within +/- 10 degrees
    df['distance_to_bridge_meters'] = df.apply(lambda row: calculate_distance(row['latitude'], row['longitude'], 59.19985816310993, 17.628248643747433), axis=1)
    df['is_closer_than_1km'] = (df['distance_to_bridge_meters'] < 1000)
    df['larger_ship_type'] = ((df['ship_type'].between(70, 79)) | (df['ship_type'].between(80, 88)) | (df['ship_type'] == 36))

    # Group by given intervals
    grouped = df.groupby(pd.Grouper(key='time', freq=time_step))
    
    # Aggregate necessary data
    aggregated_data = grouped.agg({
        'is_moving': 'sum',
        'ship_id': 'nunique',
        'is_moving_towards_bridge': 'sum',
        'width': 'mean',
        'length': 'mean',
        'is_closer_than_1km': 'sum',
        'larger_ship_type': 'sum'
    }).reset_index()

    # Rename columns
    aggregated_data.rename(columns={
        'ship_id': 'unique_boats_count',
        'width': 'average_width',
        'length': 'average_length',
        'is_moving': 'moving_boats_count',
        'is_moving_towards_bridge': 'boats_moving_towards_bridge_count',
        'is_closer_than_1km': 'boats_closer_than_1km',
        'larger_ship_type': 'larger_ship_count'
    }, inplace=True)
    
    return aggregated_data

def check_bridge_status(row, df):
        current_time = row['time']

        future_time = current_time + pd.Timedelta(minutes=30)

        open_status = df.loc[
            (df['time'] >= current_time) & (df['time'] <= future_time) & (df['state'] == 1)
        ]

        bridge_was_opened = int(len(open_status) > 0)
    
        return bridge_was_opened

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
    import hopsworks

    NiceLog.header(f"Running function for Feature Engineering of the raw data")

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

    # Select the features we are interested in from the raw data
    vessel_query: hsfs_query.Query = vessel_fg.select_except(["dim_a", "dim_b", "dim_c", 
                                                              "dim_d", "eta_day", "eta_hour", 
                                                              "eta_minute", "eta_month", "destination"])
    
    vessel_data: pd.DataFrame = vessel_query.read()

    NiceLog.info(f"Running Feature Engineering...")

    vessels_timestamp_fix = vessel_data.copy()
    vessels_timestamp_fix['time'] = pd.to_datetime(vessel_data['time'].str.replace(" UTC", ""), format='%Y-%m-%d %H:%M:%S.%f %z')

    bridge_df = pd.read_json(f'https://api.sodertalje.se/getAllBridgestat?start={bridge_data_start}&end={bridge_data_end}')
    bridge_df['time'] = pd.to_datetime(bridge_df['formatted_time'])

    # Fix the timezone by converting from GMT+1 to GMT
    bridge_df['time'] = bridge_df['time'].dt.tz_localize('Etc/GMT+1')
    bridge_df['time'] = bridge_df['time'].dt.tz_convert('GMT')

    # Resample at 15 min time intervals
    vessels_resampled_df = resample_and_fill_missing_values(vessels_timestamp_fix, '15T')
    
    # Define the area where we are interested
    area_coordinates = [(17.09776409369716, 59.39450752033877), (17.58890720737646, 59.4169265974516), 
                        (18.413654438791557, 58.977659362589186), (17.28941977865246, 58.70128810203788)]
    area_polygon = Polygon(area_coordinates)

    # Filter to keep only boats within the area
    vessels_filtered_df = vessels_resampled_df[vessels_resampled_df.apply(is_within_area, axis=1, area_polygon=area_polygon)]

    vessels_aggregated_df = process_vessels_data(vessels_filtered_df, '15T')

    vessels_aggregated_df['bridge_status'] = vessels_aggregated_df.apply(check_bridge_status, axis=1, df=bridge_df)

    #vessels_aggregated_df.set_index('time', inplace=True)

    processed_data = vessels_aggregated_df.copy()

    # Create lag features for selected columns
    lag = 5  # Number of lag steps
    lag_cols = ['moving_boats_count', 'unique_boats_count', 'boats_moving_towards_bridge_count', 'average_width', 
                'average_length', 'boats_closer_than_1km', 'larger_ship_count']
    for col in lag_cols:
        for i in range(1, lag + 1):
            processed_data[f'{col}_lag{i}'] = processed_data[col].shift(i)

    # Drop rows with NaN
    processed_data.dropna(inplace=True)

    # Create or get (if it exists) the feature group for processed vessel data
    try:
        NiceLog.info("Creating new feature group...")
        vessel_processed_fg = fs.get_or_create_feature_group(
                name=fg_vessel_processed_name,
                version=fg_vessel_processed_version,
                primary_key=["time"],
                description="Vessel pre-processed dataset"
            )

    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{vessel_processed_fg.name}{BGColors.ENDC} "
                 f"({vessel_processed_fg.description})")
    
    try:
        NiceLog.info(f"Saving vessel pre-processed data to Hopsworks...")
        vessel_processed_fg.insert(processed_data)

        NiceLog.success(f"Saved pre-processed data to Hopsworks!")

    except Exception as e:
        NiceLog.error(f"Failed to insert data into feature group. Reason: {e}")


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
