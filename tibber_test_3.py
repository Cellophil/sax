import asyncio, json, requests
from websockets.client import connect

TOKEN = 'EEE07436AD7347807083C7321542A3DA1CDF197D22A7424210F7FAA8F52029C8-1'
HOME_ID = "2f81b209-5371-4e4e-b93c-5f332e428721"

WS_URL = "wss://api.tibber.com/v1-beta/gql/subscriptions"  # or fetch viewer.websocketSubscriptionUrl first

GRAPHQL_HTTP = "https://api.tibber.com/v1-beta/gql"
FEATURES_QUERY = """
query {
  viewer {
    websocketSubscriptionUrl
    homes {
      id
      appNickname
      address { address1 }
      features {
        realTimeConsumptionEnabled
        realTimeConsumptionSupported
      }
    }
  }
}
"""

async def subscribe_realtime(ws_url: str, home_id: str):
    print(f"Connecting to: {ws_url}  home: {home_id}")
    async with connect(ws_url, subprotocols=["graphql-transport-ws"], open_timeout=15) as ws:
        # INIT with token in payload (required by Tibber)
        await ws.send(json.dumps({"type": "connection_init", "payload": {"token": TOKEN}}))

        # Wait for server ACK; reply to ping if sent first
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            print("<<", msg)
            t = msg.get("type")
            if t == "connection_ack":
                break
            if t == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        # Start the subscription
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
            "payload": {"query": query, "variables": {"homeId": home_id}}
        }))

        # Stream messages
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "next":    # data frame per graphql-transport-ws
                print("DATA:", msg["payload"]["data"])
            elif t == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            else:
                print("EVT:", msg)

def fetch_ws_url_and_home():
    # 1) Fetch dynamic WebSocket URL + homes/features via HTTPS GraphQL
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "User-Agent": "tibber-realtime-minimal/1.0",
        "Content-Type": "application/json",
    }
    r = requests.post(GRAPHQL_HTTP, headers=headers, json={"query": FEATURES_QUERY}, timeout=15)
    r.raise_for_status()
    data = r.json()

    viewer = data["data"]["viewer"]
    ws_url = viewer["websocketSubscriptionUrl"]

    # Pick a home that actually has realtime enabled
    homes = viewer["homes"]
    enabled = [h for h in homes if h["features"]["realTimeConsumptionEnabled"]]
    if not enabled:
        raise RuntimeError("No home with realTimeConsumptionEnabled=true on this token.")
    # If you have multiple, pick the first (or add your own selection logic)
    home_id = enabled[0]["id"]
    print("Using home:", enabled[0].get("appNickname") or enabled[0].get("address", {}).get("address1"), "→", home_id)
    return ws_url, home_id

if __name__ == "__main__":
    ws_url, home_id = fetch_ws_url_and_home()
    asyncio.run(subscribe_realtime(ws_url, home_id))