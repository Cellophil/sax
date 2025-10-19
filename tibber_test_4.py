import asyncio, json, requests
from websockets.client import connect

TOKEN = 'EEE07436AD7347807083C7321542A3DA1CDF197D22A7424210F7FAA8F52029C8-1'.strip()
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
      }
    }
  }
}
"""

SUBSCRIPTION = """
subscription($homeId: ID!) {
  liveMeasurement(homeId: $homeId) {
    timestamp
    power
    accumulatedConsumption
    accumulatedCost
  }
}
"""

def fetch_ws_url_and_home():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "tibber-realtime-minimal/1.0",
    }
    r = requests.post(GRAPHQL_HTTP, headers=headers,
                      json={"query": FEATURES_QUERY, "variables": {}}, timeout=20)
    # Helpful print on error
    if r.status_code != 200:
        print("HTTP error:", r.status_code, r.text)
        r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    viewer = data["data"]["viewer"]
    ws_url = viewer["websocketSubscriptionUrl"]

    homes = viewer["homes"]
    enabled = [h for h in homes if h["features"]["realTimeConsumptionEnabled"]]
    if not enabled:
        raise RuntimeError("No home with realTimeConsumptionEnabled=true on this token.")
    home = enabled[0]
    print("Using home:", home.get("appNickname") or home.get("address", {}).get("address1"), "→", home["id"])
    print("WS URL:", ws_url)
    return ws_url, home["id"]

async def subscribe_realtime(ws_url: str, home_id: str):
    async with connect(ws_url, subprotocols=["graphql-transport-ws"], open_timeout=15) as ws:
        # 1) INIT with token in payload
        await ws.send(json.dumps({"type": "connection_init", "payload": {"token": TOKEN}}))

        # 2) Wait for connection_ack (reply to ping if needed)
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            t = msg.get("type")
            print("<<", msg)
            if t == "connection_ack":
                break
            if t == "ping":
                await ws.send(json.dumps({"type": "pong"}))

        # 3) Start subscription
        await ws.send(json.dumps({
            "id": "1",
            "type": "subscribe",
            "payload": {"query": SUBSCRIPTION, "variables": {"homeId": home_id}}
        }))

        # 4) Read stream
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "next":
                print("DATA:", msg["payload"]["data"])
            elif t == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            else:
                print("EVT:", msg)

if __name__ == "__main__":
    ws_url, home_id = fetch_ws_url_and_home()
    asyncio.run(subscribe_realtime(ws_url, home_id))