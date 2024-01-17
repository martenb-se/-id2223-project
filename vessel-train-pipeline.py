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

# Model
from xgboost import XGBClassifier

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Settings
# - Modal
modal_stub_name = "vessel-train-pipeline"
modal_image_libraries = ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image", "pytz", "xgboost"]
model_run_every_n_hours = 12
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
    import hopsworks
    import joblib
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    NiceLog.header(f"Running function to Train Model")

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

    # Get Feature View for vessel data
    NiceLog.info(f"Getting {BGColors.HEADER}{fw_vessel_name}{BGColors.ENDC} "
                 f"(version {fw_vessel_version}) feature view...")
    try:
        vessel_fw: feature_view.FeatureView = fs.get_feature_view(name=fw_vessel_name, version=fw_vessel_version)
        NiceLog.info(f"Vessel feature fw exists.")
    except Exception as e:
        NiceLog.info(f"Vessel feature fw must be created.")
        vessel_query: hsfs_query.Query = vessel_fg.select_except(features=['time'])

        vessel_fw: feature_view.FeatureView = (
            fs.get_or_create_feature_view(
                name=fw_vessel_name,
                version=fw_vessel_version,
                description="Read from pre-processed vessel data",
                labels=["bridge_status"],
                query=vessel_query))

    NiceLog.info(f"Training model...")
    
    # Train model on vessel data
    vessel_train_X, vessel_test_X, vessel_train_y, vessel_test_y = vessel_fw.train_test_split(0.2)

    model_bridge = XGBClassifier(learning_rate=0.1, max_depth=5, scale_pos_weight=8)
    model_bridge.fit(vessel_train_X, vessel_train_y.values.ravel())

    # Evaluate on test data
    vessel_y_pred = model_bridge.predict(vessel_test_X)
    vessel_metrics = classification_report(vessel_test_y, vessel_y_pred, output_dict=True)
    vessel_results = confusion_matrix(vessel_test_y, vessel_y_pred)
    vessel_accuracy = accuracy_score(vessel_test_y, vessel_y_pred)

    # Error rate on test data
    vessel_total_errors = (vessel_test_y.values.ravel() != vessel_y_pred).sum()
    vessel_total_predictions = len(vessel_y_pred)
    vessel_error_rate = vessel_total_errors / vessel_total_predictions

    # TODO: Evaluate the accuracy of the model
    # TODO: Only save model if accuracy is above a certain threshold

    # Get an object for the model registry.
    mr = project.get_model_registry()

    # Save bridge model
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    # Save bridge model
    joblib.dump(model_bridge, models_dir + f"/{model_bridge_name}.pkl")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    vessel_input_schema = Schema(vessel_train_X)
    vessel_output_schema = Schema(vessel_train_y)
    bridge_model_schema = ModelSchema(vessel_input_schema, vessel_output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    bridge_model = mr.python.create_model(
        name=model_bridge_name,
        metrics={"accuracy": vessel_metrics['accuracy'],
                 "f1-score": vessel_metrics['weighted avg']['f1-score']},
        model_schema=bridge_model_schema,
        description="Bridge Status Type Predictor"
    )

    bridge_model.save(models_dir)

    NiceLog.success(f"Saved model to Hopsworks!")


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
