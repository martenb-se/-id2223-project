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
from huggingface_hub.hf_api import HfApi as HuggingfaceApi
from huggingface_hub.hf_api import SpaceInfo as HuggingfaceSpaceInfo
from huggingface_hub.hf_api import SpaceRuntime as HuggingfaceSpaceRuntime

# Error help
from requests.exceptions import MissingSchema, HTTPError

# Settings
# - Modal
modal_stub_name = "huggingface-restart-service"
modal_image_libraries = ["huggingface-hub"]
model_run_every_n_hours = 12
# - Huggingface
huggingface_access_token_modal_secret_name = "huggingface-access-token"  # Load secret to environment
huggingface_api_endpoint = "https://huggingface.co"
huggingface_repo_id = "GroupSix/bridge"

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
    from huggingface_hub import HfApi

    NiceLog.header(f"Running function to restart Huggingface service at {huggingface_repo_id}")

    if "HUGGINGFACE_ACCESS_TOKEN" not in os.environ:
        NiceLog.error(f"Failed to log in to Huggingface. HUGGINGFACE_ACCESS_TOKEN is not in the current environment.")
        raise HuggingfaceNoAccessToken()

    # Log in
    NiceLog.info("Logging in to Huggingface...")
    try:
        hf_api: HuggingfaceApi = HfApi(
            endpoint=huggingface_api_endpoint,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
        )

        hf_whoami: dict = hf_api.whoami()

    except HTTPError as e:
        NiceLog.error(f"Failed to log in to Huggingface. Reason: {e}")
        raise HuggingfaceLoginError()
    except MissingSchema as e:
        NiceLog.error(f"Failed to log in to Huggingface. Reason: {e}")
        raise HuggingfaceLoginError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to log in to Huggingface. Reason: {e}")
        raise HuggingfaceLoginError()
    NiceLog.success(f"Logged in to Huggingface as: {BGColors.HEADER}{hf_whoami['name']}{BGColors.ENDC}")

    # Get info about space
    NiceLog.info(f"Getting info about space: {BGColors.HEADER}{huggingface_repo_id}{BGColors.ENDC}")
    try:
        hf_space_info: HuggingfaceSpaceInfo = hf_api.space_info(repo_id=huggingface_repo_id)

    except HTTPError as e:
        NiceLog.error(f"Failed to get info about space: {huggingface_repo_id}. Reason: {e}")
        raise HuggingfaceGetSpaceInfoError()

    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get info about space: {huggingface_repo_id}. Reason: {e}")
        raise HuggingfaceGetSpaceInfoError()

    NiceLog.success(f"Got info about space: {BGColors.HEADER}{huggingface_repo_id}{BGColors.ENDC}")
    NiceLog.info(f"Space was last modified: {BGColors.HEADER}{hf_space_info.last_modified}{BGColors.ENDC}")

    # Restart service
    NiceLog.info(f"Restarting service: {BGColors.HEADER}{huggingface_repo_id}{BGColors.ENDC}")
    try:
        hf_restart_space: HuggingfaceSpaceRuntime = hf_api.restart_space(repo_id="GroupSix/bridge")

    except HTTPError as e:
        NiceLog.error(f"Failed to restart service: {huggingface_repo_id}. Reason: {e}")
        raise HuggingfaceGetSpaceInfoError()

    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to restart service: {huggingface_repo_id}. Reason: {e}")
        raise HuggingfaceGetSpaceInfoError()

    NiceLog.success(f"Restarted service: {BGColors.HEADER}{huggingface_repo_id}{BGColors.ENDC}")
    NiceLog.info(f"Current space stage: {BGColors.HEADER}{hf_restart_space.stage}{BGColors.ENDC}")


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
        schedule=modal.Period(hours=model_run_every_n_hours),
        secrets=[
            modal.Secret.from_name(huggingface_access_token_modal_secret_name),
        ]
    )
    def f():
        g()

# Load local environment
else:
    NiceLog.info("Loading local environment...")
    if load_dotenv() and 'HUGGINGFACE_ACCESS_TOKEN' in os.environ:
        NiceLog.success("Loaded variables from .env file!")
    else:
        if 'HUGGINGFACE_ACCESS_TOKEN' not in os.environ:
            NiceLog.error("Add add HUGGINGFACE_ACCESS_TOKEN to your .env file!")
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
