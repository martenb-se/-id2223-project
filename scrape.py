import os
from dotenv import load_dotenv

import asyncio
import websockets
import json
from datetime import datetime, timezone


async def connect_ais_stream(api_key):
    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {"APIKey": api_key,  # Required !
                             "BoundingBoxes": [
                                 [[61.345101156944445, 15.048625225719368], [57.11950799373074, 21.76127140685429]]],
                             # Required!
                             # "FiltersShipMMSI": ["368207620", "367719770", "211476060"],  # Optional!
                             # "FilterMessageTypes": ["PositionReport"]  # Optional!
                             }

        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        async for message_json in websocket:
            message = json.loads(message_json)
            message_type = message["MessageType"]

            if message_type == "PositionReport" or \
                    message_type == "UnknownMessage" or \
                    message_type == "DataLinkManagementMessage" or \
                    message_type == "StandardClassBPositionReport":
                continue

            print(f"[{datetime.now(timezone.utc)}]")
            print(f" - Type: {message_type}")

            if "MetaData" in message and \
                    all(key in message['MetaData'] for key in
                        ['MMSI', 'MMSI_String', 'ShipName', 'latitude', 'longitude']):
                print(f" - ShipId: {message['MetaData']['MMSI']}")
                # print(f" - ShipName: {message['MetaData']['ShipName']}")
                print(f" - Latitude: {message['MetaData']['latitude']}")
                print(f" - Latitude: {message['MetaData']['longitude']}")
                print(f" - Time: {message['MetaData']['time_utc']}")

            if "Message" in message:
                if "ShipStaticData" in message['Message'] and "Dimension" in message['Message']['ShipStaticData']:
                    print(f" - Dimension: {message['Message']['ShipStaticData']['Dimension']}")
                    print(
                        f" - Length: {message['Message']['ShipStaticData']['Dimension']['A'] + message['Message']['ShipStaticData']['Dimension']['B']}")
                elif "AidsToNavigationReport" in message['Message'] and "Dimension" in message['Message'][
                    'AidsToNavigationReport']:
                    print(f" - Dimension: {message['Message']['AidsToNavigationReport']['Dimension']}")
                    print(
                        f" - Length: {message['Message']['AidsToNavigationReport']['Dimension']['A'] + message['Message']['AidsToNavigationReport']['Dimension']['B']}")

            if "ShipStaticData" in message['Message'] and "Eta" in message['Message']['ShipStaticData']:
                print(f" - ETA: {message['Message']['ShipStaticData']['Eta']}")

            if "ShipStaticData" in message['Message'] and "Destination" in message['Message']['ShipStaticData']:
                print(f" - Destination: {message['Message']['ShipStaticData']['Destination']}")

            if "ShipStaticData" in message['Message'] and "Type" in message['Message']['ShipStaticData']:
                print(f" - Type: {message['Message']['ShipStaticData']['Type']}")

            print(message)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('API_KEY')
    asyncio.run(asyncio.run(connect_ais_stream(api_key)))
