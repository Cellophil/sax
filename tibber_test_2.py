#%%
import asyncio, json
from websockets.client import connect

API_TOKEN = 'EEE07436AD7347807083C7321542A3DA1CDF197D22A7424210F7FAA8F52029C8-1'
HOME_ID = "2f81b209-5371-4e4e-b93c-5f332e428721"

WS_URL = "wss://api.tibber.com/v1-beta/gql/subscriptions"  # or fetch viewer.websocketSubscriptionUrl first

async def main():
    async with connect(
        WS_URL,
        subprotocols=["graphql-transport-ws"],   # required
        open_timeout=15
    ) as ws:
        # 1) Authenticate in the WS INIT payload (not via HTTP headers)
        await ws.send(json.dumps({
            "type": "connection_init",
            "payload": {"token": API_TOKEN, "version": "1"}   # version is optional; token is required
        }))

        # (Optional) wait for ack before subscribing
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") in ("connection_ack", "ka", "ping", "pong"):
                break

        # 2) Start subscription
        query = """
        subscription($homeId: ID!) {
          liveMeasurement(homeId: $homeId) {
            timestamp
            power
            accumulatedConsumption
            accumulatedCost
          }
        }"""
        await ws.send(json.dumps({
            "id": "1",
            "type": "subscribe",
            "payload": {"query": query, "variables": {"homeId": HOME_ID}}
        }))

        # 3) Read stream
        async for raw in ws:
            data = json.loads(raw)
            if data.get("type") in ("next", "data"):
                print(data)
            elif data.get("type") in ("error", "complete"):
                print("Done/error:", data)
                break

await main()

#%%