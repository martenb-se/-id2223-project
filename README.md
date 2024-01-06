# Bridge Opening Prediction Model

## Project Overview
This project is focused on predicting the opening times of the 
[Södertälje Mälarbro](https://www.sjofartsverket.se/sv/tjanster/kanaler-slussar-broar/sodertalje---malarbron/) 
in Sweden. It utilizes Automatic Identification System (AIS) data from nearby vessels to forecast bridge opening events. 
This model aims to assist in traffic management and planning for both maritime and vehicular traffic.

## Data Collection
The `aisstream-data-collector.py` script is used for collecting AIS data. This data includes vessel locations, speeds, 
directions, and identification details.

## Environment Setup
Before running the data collection script, you need to set up your environment:

1. **API Keys**: 
   - **aisstream.io API Key**: You need an API key from aisstream.io to access AIS data. 
   - **Hopsworks API Key**: To store the collected data, you will need an API key from Hopsworks.

2. **.env File**: 
   - Create a `.env` file in the project root directory.
   - Add the following lines to the file:
     ```
     AISSTREAM_API_KEY=your_aisstream_api_key_here
     HOPSWORKS_API_KEY=your_hopsworks_api_key_here
     ```
   - Replace `your_aisstream_api_key_here` and `your_hopsworks_api_key_here` with your actual API keys.

## Data Collection
Data is collected from the [aisstream.io](https://aisstream.io/) API. The data collection script is responsible for
retrieving data from the API and storing it in a Hopsworks dataset.

### Manually Running the Data Collector
#### Installation and Dependencies
Install the required dependencies listed in `requirements.txt` using:

```bash
pip install -r requirements.txt
```

#### Running the Data Collector Locally
To run the data collector script locally for testing and development, use the following command:

```bash
./launch_aisstream-data-collector.sh --local
```

#### Running the Data Collector Remotely
This script will later be adapted for remote deployment.

Initially it was intended to be run on Modal, but due to its limitations on continuous execution, it will be run on 
Microsoft Azure or Google Cloud instead. It is not yet clear which cloud provider is best for our use case.

### Building and Running the Data Collector Locally using Docker
To run the data collector script using Docker, use the following command:

```bash
docker compose up -d --build datacollector
```

#### Subscribe to logs
To subscribe to the logs of the data collector container, use the following command:
```bash
docker logs -f datacollector
```

#### Stop container
To stop the data collector container, use the following command:
```bash
docker container stop datacollector
```

## Model Description
The machine learning model processes AIS data to identify patterns indicative of bridge opening events, considering 
vessel type, size, speed, and proximity to the bridge.

## Contributing
Contribution guidelines will cover coding standards, pull request processes, and contact information for 
project maintainers.

## License
Details about the project's license and usage rights will be provided here.

## Acknowledgements
Acknowledgements to contributors, supporting organizations, and institutions.

---

*Note: This README is subject to updates as the project evolves.*
