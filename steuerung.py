#%%

import numpy as np
import os
from pathlib import Path

import tibber
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import time
import inspect
from logging.handlers import TimedRotatingFileHandler
try:
    from .config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, TIBBER_TOKEN, get_effective_soc_limits,
        TIBBER_USER_AGENT,
    )
    from .controller import BatteryFleetController
    from .strategy import Strategy, StrategyContext, decide_strategy, normalize_prices_15
    from .fast_control import FastController, FastControlConfig
    from .tibber_ws import TibberWSClient
except ImportError:
    from config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, TIBBER_TOKEN, get_effective_soc_limits,
        TIBBER_USER_AGENT,
    )
    from controller import BatteryFleetController
    from strategy import Strategy, StrategyContext, decide_strategy, normalize_prices_15
    from fast_control import FastController, FastControlConfig
    from tibber_ws import TibberWSClient

#%%

N_battery = N_BATTERIES

async def tibber_check_realtime_capability(token: str, home_id: str | None = None) -> dict:
    """Query Tibber GraphQL features for realtime capability.

    Returns a dict with enabled/supported flags for homes and logs details.
    """
    url = "https://api.tibber.com/v1-beta/gql"
    query = {
        "query": (
            "query ViewerHomesFeatures {\n"
            "  viewer { homes { id appNickname: appNickname features { realTimeConsumptionEnabled } } }\n"
            "}"
        )
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": str(TIBBER_USER_AGENT or "sax")
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=query, headers=headers, timeout=20) as resp:
                data = await resp.json(content_type=None)
        homes = (((data or {}).get("data") or {}).get("viewer") or {}).get("homes") or []
        if not homes:
            tibber_logger.info("GraphQL feature check returned no homes")
            return {"homes": []}
        # Prefer matching home_id if provided
        chosen = None
        if home_id:
            for h in homes:
                if str(h.get("id")) == str(home_id):
                    chosen = h
                    break
        if chosen is None:
            chosen = homes[0]
        feats = (chosen.get("features") or {})
        enabled = bool(feats.get("realTimeConsumptionEnabled", False))
        tibber_logger.info(
            "Realtime capability (home id=%s nick=%s): enabled=%s",
            chosen.get("id"), chosen.get("appNickname"), enabled,
        )
        if not enabled:
            tibber_logger.warning("Realtime not enabled according to API; subscription may stay silent.")
        return {"homes": homes, "chosen": chosen, "enabled": enabled}
    except Exception as e:
        tibber_logger.warning(f"GraphQL feature check failed: {e}")
        return {"error": str(e)}


async def tibber_fetch_ws_url_and_home(token: str) -> tuple[str | None, str | None, bool | None]:
    """Fetch viewer.websocketSubscriptionUrl and find an enabled home id.

    Returns (ws_url, home_id, enabled) where enabled is the feature flag for that home.
    """
    url = "https://api.tibber.com/v1-beta/gql"
    query = {
        "query": (
            "query ViewerWS {\n"
            "  viewer {\n"
            "    websocketSubscriptionUrl\n"
            "    homes { id appNickname: appNickname features { realTimeConsumptionEnabled } }\n"
            "  }\n"
            "}"
        ),
        "variables": {},
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": str(TIBBER_USER_AGENT or "sax"),
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=query, headers=headers, timeout=20) as resp:
                data = await resp.json(content_type=None)
        viewer = (((data or {}).get("data") or {}).get("viewer") or {})
        ws_url = viewer.get("websocketSubscriptionUrl")
        homes = viewer.get("homes") or []
        chosen = None
        for h in homes:
            feats = (h.get("features") or {})
            if feats.get("realTimeConsumptionEnabled"):
                chosen = h
                break
        if chosen is None and homes:
            chosen = homes[0]
        hid = chosen.get("id") if chosen else None
        enabled = (chosen.get("features") or {}).get("realTimeConsumptionEnabled") if chosen else None
        return ws_url, hid, enabled
    except Exception as e:
        tibber_logger.info(f"Viewer websocketSubscriptionUrl query (direct HTTP) failed: {e}")
        return None, None, None


async def tibber_check_realtime_capability_via_tc(tc_obj, home_id: str | None = None) -> dict:
    """Same as tibber_check_realtime_capability but using pyTibber's execute (shared session/headers)."""
    query = (
        "query ViewerHomesFeatures {\n"
        "  viewer { homes { id appNickname: appNickname features { realTimeConsumptionEnabled } } }\n"
        "}"
    )
    try:
        data = await tc_obj.execute(query)
        homes = (((data or {}).get("data") or {}).get("viewer") or {}).get("homes") or []
        if not homes:
            tibber_logger.info("GraphQL feature check (tc) returned no homes")
            return {"homes": []}
        chosen = None
        if home_id:
            for h in homes:
                if str(h.get("id")) == str(home_id):
                    chosen = h
                    break
        if chosen is None:
            chosen = homes[0]
        feats = (chosen.get("features") or {})
        enabled = bool(feats.get("realTimeConsumptionEnabled", False))
        tibber_logger.info(
            "Realtime capability (home id=%s nick=%s): enabled=%s",
            chosen.get("id"), chosen.get("appNickname"), enabled,
        )
        if not enabled:
            tibber_logger.warning("Realtime not enabled according to API; subscription may stay silent.")
        return {"homes": homes, "chosen": chosen, "enabled": enabled}
    except Exception as e:
        tibber_logger.warning(f"GraphQL feature check via tc failed: {e}")
        return {"error": str(e)}

async def update_soc():
    global soc, fleet
    if fleet is None:
        fleet = BatteryFleetController()
    while True:
        try:
            per = fleet.read_soc_all()
            if per:
                soc = {**per, 'total': float(np.mean([per[b] for b in BATTERIES if b in per]))}
        except Exception:
            logger.warning('Could not read SOCs via fleet; retain old value.')
        await asyncio.sleep(60)
    

def set_power(p):
    global soc
    global fleet

    # Initialize fleet controller lazily
    if fleet is None:
        fleet = BatteryFleetController()

    # Fleet expects the total setpoint; it will distribute and respect SOC bounds
    # Fleet uses its persisted SOC bounds; they default to weekly overrides
    fleet.set_total_power(int(p), {b: soc.get(b) for b in BATTERIES})


async def update_power():
    global p_t, current_p, fleet
    if fleet is None:
        fleet = BatteryFleetController()
    current_p = {}
    while True:
        try:
            per = fleet.read_power_all()
            for b in BATTERIES:
                if b in per:
                    current_p[b] = per[b]
                else:
                    # best guess if missing
                    current_p[b] = int(current_p.get(b, 0) * 0.9)
            current_p['total'] = sum(current_p.get(b, 0) for b in BATTERIES)
            current_p['t'] = datetime.now(ZoneInfo("Europe/Berlin"))
            p_t.append(current_p.copy())
        except Exception as e:
            logger.warning(f'Could not read power via fleet; retaining last values. err={e}')
        await asyncio.sleep(2)


async def strategy_loop():
    """Periodic high-level decision loop producing a total fleet setpoint.

    Runs every few minutes, consuming 15-min price series, SOC, and current power.
    """
    global soc, current_p, prices_series_15, fast_ctrl, fleet
    first_run = True
    while True:
        try:
            # First run immediately, then align to quarter-hour boundaries
            if not first_run:
                now = pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin"))
                next_q = now.ceil("15min")
                sleep_s = max(0.0, (next_q - now).total_seconds())
                await asyncio.sleep(sleep_s)
            else:
                first_run = False

            now = pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin"))
            # On first run, give prices task up to 2 seconds to populate
            if first_run and (('prices_series_15' not in globals()) or prices_series_15 is None):
                await asyncio.sleep(2)
            prices_norm = normalize_prices_15(prices_series_15 if 'prices_series_15' in globals() else None)
            # Ensure ctx.now is a pandas.Timestamp so .floor works reliably
            now_pd = pd.Timestamp(now)
            ctx = StrategyContext(
                now=now_pd,
                prices_15=prices_norm,
                soc_total=float(soc.get('total', 0) or 0),
                soc_per={b: float(soc.get(b, 0) or 0) for b in BATTERIES},
                current_total_power_w=float(current_p.get('total', 0) or 0),
            )
            strat, maybe_energy_Wh = decide_strategy(ctx)

            # Ensure fleet and fast controller exist
            if fleet is None:
                fleet = BatteryFleetController()
            if 'fast_ctrl' not in globals() or fast_ctrl is None:
                fast_ctrl = FastController(fleet)
                # Measurement function: use Tibber net reading; invert sign to make positive=export
                async def measure_net():
                    try:
                        return -float(measurement[-1][1]) if measurement else 0.0
                    except Exception:
                        return 0.0
                asyncio.create_task(fast_ctrl.run(measure_net))

            # Update controller mode based on strategy
            fast_ctrl.update_from_strategy(strat, maybe_energy_Wh, now=now)
            # Strategy log: current 15-min price, max price next 24h, SOC
            try:
                price_now = None
                price_max = None
                price_max_t = None
                if prices_norm is not None and len(prices_norm) > 0:
                    q_start = now.floor("15min")
                    # current price: nearest in index (it should be aligned already)
                    idxer = prices_norm.index.get_indexer([q_start], method="nearest")
                    if idxer is not None and idxer.size == 1 and idxer[0] != -1:
                        price_now = float(prices_norm.iloc[idxer[0]])
                    future = prices_norm.loc[q_start : q_start + pd.Timedelta(hours=24)]
                    if len(future) > 0:
                        price_max = float(future.max())
                        price_max_t = future.idxmax()
                strategy_logger.info(
                    "strategy=%s energy_Wh=%s soc=%.1f price_now=%s price_max_24h=%s at=%s",
                    strat.name,
                    f"{maybe_energy_Wh:.0f}" if isinstance(maybe_energy_Wh, (int, float)) else "-",
                    ctx.soc_total,
                    f"{price_now:.3f}" if price_now is not None else "-",
                    f"{price_max:.3f}" if price_max is not None else "-",
                    f"{price_max_t}" if price_max_t is not None else "-",
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Strategy loop error")
        # loop continues; we'll align to next quarter at the top



#%%

async def get_tibber():
    global measurement, tc, home
    global tibber_connect_lock, last_tibber_attempt
    global rt_msg_count, last_rt_message_ts
    global tibber_session
    global tibber_ws_client
    global tibber_home_id

    if len(measurement) > 1200*24:
        measurement = measurement[-1200*24:]
    # We will use raw WS directly; ignore pyTibber realtime state

    # Establish once, with exponential backoff on failures
    backoff = 60  # start with 60s
    max_backoff = 900  # cap at 15 minutes
    # Create a lock lazily to prevent concurrent connect attempts
    if 'tibber_connect_lock' not in globals() or tibber_connect_lock is None:
        tibber_connect_lock = asyncio.Lock()

    min_interval = 30.0  # seconds between attempts to avoid flood
    if 'last_tibber_attempt' not in globals():
        last_tibber_attempt = 0.0

    while True:
        try:
            # Rate-limit attempts across restarts of this coroutine
            now_m = time.monotonic()
            elapsed = now_m - float(last_tibber_attempt or 0.0)
            if elapsed < min_interval:
                sleep_left = min_interval - elapsed
                await asyncio.sleep(sleep_left)

            async with tibber_connect_lock:
                last_tibber_attempt = time.monotonic()
            if not TIBBER_TOKEN:
                raise RuntimeError("TIBBER_TOKEN is not set. Set it in config.py (TIBBER_TOKEN=...) or via environment and restart.")

            tibber_logger.info(f'Connecting to Tibber with user_agent="{TIBBER_USER_AGENT}"')
            # Maintain a persistent aiohttp session across retries to avoid 'Session is closed'
            try:
                if 'tibber_session' not in globals() or tibber_session is None or tibber_session.closed:
                    tibber_session = aiohttp.ClientSession()
            except Exception:
                tibber_session = None
            # Use raw GraphQL WS directly
            # 1) Determine WS URL and home id (prefer enabled home)
            # Try via pyTibber execute first for consistency
            tc = tibber.Tibber(TIBBER_TOKEN, user_agent=TIBBER_USER_AGENT, websession=tibber_session)
            ws_url = None
            hid = None
            try:
                gql = (
                    "query ViewerWS {\n"
                    "  viewer {\n"
                    "    websocketSubscriptionUrl\n"
                    "    homes { id features { realTimeConsumptionEnabled } }\n"
                    "  }\n"
                    "}"
                )
                data = await tc.execute(gql)
                viewer = (data or {}).get("data", {}).get("viewer", {})
                ws_url = viewer.get("websocketSubscriptionUrl")
                homes_v = viewer.get("homes") or []
                chosen = None
                for h in homes_v:
                    feats = (h.get("features") or {})
                    if feats.get("realTimeConsumptionEnabled"):
                        chosen = h
                        break
                if chosen is None and homes_v:
                    chosen = homes_v[0]
                hid = chosen.get("id") if chosen else None
            except Exception:
                pass
            if not ws_url or not hid:
                ws_url2, hid2, enabled2 = await tibber_fetch_ws_url_and_home(TIBBER_TOKEN)
                ws_url = ws_url or ws_url2
                hid = hid or hid2
                if enabled2 is False:
                    tibber_logger.warning("Realtime disabled according to viewer query; stream may be silent.")
            if not hid:
                raise RuntimeError("Could not determine Tibber home id for realtime subscription")
            if not ws_url:
                raise RuntimeError("viewer.websocketSubscriptionUrl missing; cannot start realtime without dynamic URL")
            tibber_logger.info(f"Using Tibber websocketSubscriptionUrl={ws_url}")

            # Remember home id for price updates
            tibber_home_id = str(hid)

            # Initialize `home` for price updates via pyTibber (realtime handled separately)
            try:
                await tc.update_info()
                homes_list = tc.get_homes() or []
                chosen_home = None
                for h in homes_list:
                    try:
                        h_id = getattr(h, 'id', getattr(h, 'home_id', None))
                        if str(h_id) == str(hid):
                            chosen_home = h
                            break
                    except Exception:
                        continue
                if chosen_home is None and homes_list:
                    chosen_home = homes_list[0]
                if chosen_home is not None:
                    globals()['home'] = chosen_home
            except Exception as e_init_home:
                tibber_logger.info(f"Could not initialize Tibber home for price updates yet: {e_init_home}")

            # 2) Start WS client
            tibber_logger.info("Starting raw GraphQL WS client…")
            rt_msg_count = 0
            last_rt_message_ts = None
            tibber_ws_client = TibberWSClient(
                TIBBER_TOKEN,
                str(hid),
                session=tibber_session,
                logger=tibber_logger,
                prefer_legacy=False,
                user_agent=TIBBER_USER_AGENT,
                ws_url=ws_url,
            )
            await tibber_ws_client.start(lambda pkg: _callback(measurement, pkg))
            await asyncio.sleep(45)
            tibber_logger.info(f"raw_ws running={tibber_ws_client.running} msg_count={rt_msg_count} last_msg_ts={last_rt_message_ts}")
            if tibber_ws_client.running and (rt_msg_count and rt_msg_count > 0):
                tibber_logger.info('Raw GraphQL websocket subscription started.')
                backoff = 60
                return
            else:
                try:
                    await tibber_ws_client.stop()
                except Exception:
                    pass
                raise RuntimeError('Raw GraphQL websocket silent; will back off and retry')
        except Exception as e:
            # Common pitfall: Tibber API returns text/plain when token is invalid or URL is wrong
            msg = str(e)
            if 'Unexpected content type: text/plain' in msg:
                tibber_logger.error('Tibber auth/content-type error. This usually means your TIBBER_TOKEN is invalid or the API endpoint changed. '\
                             'Please verify the token (Account -> Developer in Tibber app).')
            # If we got rate limited (HTTP 429), back off more aggressively
            if '429' in msg or 'Too Many Requests' in msg or 'rate limit' in msg.lower():
                backoff = min(max_backoff, max(120, int(backoff * 1.7)))
                tibber_logger.warning(f'Tibber rate-limited or connection allowance reached. Backing off for {backoff}s')
            else:
                backoff = min(max_backoff, max(60, int(backoff * 1.4)))
                tibber_logger.warning(f'Tibber setup failed, will retry in {backoff}s: {e}')
            # Do not close shared session here
            # add jitter +/- 20% to spread reconnects
            jitter = 0.2 * backoff
            sleep_for = max(15, backoff + np.random.uniform(-jitter, jitter))
            await asyncio.sleep(sleep_for)


async def update_price():
    global prices_series_15, tibber_home_id
    prices_series_15 = None

    while True:
        try:
            # Wait until home id is initialized
            if 'tibber_home_id' not in globals() or not tibber_home_id:
                await asyncio.sleep(5)
                continue

            # Fetch price info via GraphQL HTTP directly
            async def _fetch_prices(resolution: str):
                gql = (
                    "query Price($homeId: ID!, $res: PriceInfoResolution!) {\n"
                    "  viewer {\n"
                    "    home(id: $homeId) {\n"
                    "      currentSubscription {\n"
                    "        priceInfo(resolution: $res) {\n"
                    "          today { total startsAt }\n"
                    "          tomorrow { total startsAt }\n"
                    "        }\n"
                    "      }\n"
                    "    }\n"
                    "  }\n"
                    "}"
                )
                headers = {
                    "Authorization": f"Bearer {TIBBER_TOKEN}",
                    "Content-Type": "application/json",
                    "User-Agent": str(TIBBER_USER_AGENT or "sax"),
                }
                payload = {"query": gql, "variables": {"homeId": tibber_home_id, "res": resolution}}
                sess = tibber_session if ('tibber_session' in globals() and tibber_session and not tibber_session.closed) else aiohttp.ClientSession()
                owns = sess is not tibber_session
                try:
                    async with sess.post("https://api.tibber.com/v1-beta/gql", json=payload, headers=headers, timeout=20) as resp:
                        data = await resp.json(content_type=None)
                    if not data or 'errors' in data:
                        raise RuntimeError(str(data.get('errors'))) if isinstance(data, dict) else RuntimeError("GraphQL error")
                    vi = (((data or {}).get("data") or {}).get("viewer") or {}).get("home") or {}
                    sub = (vi.get("currentSubscription") or {}).get("priceInfo") or {}
                    today = sub.get("today") or []
                    tomorrow = sub.get("tomorrow") or []
                    items = today + tomorrow
                    entries = []
                    for p in items:
                        try:
                            ts = pd.to_datetime(p.get("startsAt"))
                            if ts.tzinfo is None:
                                ts = ts.tz_localize("UTC")
                            ts = ts.tz_convert(ZoneInfo("Europe/Berlin"))
                            val = float(p.get("total"))
                            entries.append((ts, val))
                        except Exception:
                            continue
                    return entries
                finally:
                    if owns:
                        await sess.close()

            res_used = "QUARTER_HOURLY"
            entries = []
            try:
                entries = await _fetch_prices("QUARTER_HOURLY")
            except Exception as e_qh:
                tibber_logger.warning(f"15-min price fetch failed; leaving prices unset: {e_qh}")
                entries = []

            s = None
            if entries:
                s = pd.Series([v for (_, v) in entries], index=[t for (t, _) in entries]).sort_index()
                # If we got hourly, upsample to 15-min with forward-fill
                try:
                    freq = pd.infer_freq(s.index)
                except Exception:
                    freq = None
                if freq is None or freq.upper() not in ("15T", "15MIN"):
                    s = s.resample("15min").ffill()
                prices_series_15 = normalize_prices_15(s)
            else:
                prices_series_15 = None

            tibber_logger.info(
                "Price update done (resolution=%s, points=%s)",
                res_used,
                len(prices_series_15) if prices_series_15 is not None else 0,
            )
        except Exception as e:
            prices_series_15 = None
            tibber_logger.warning(f'Price problems: {e}')

        # Tibber updates at most hourly; 5 minutes poll is fine
        await asyncio.sleep(300)


#%%

def get_robust_reading(N=10, which='last'):
    global measurement

    try:
        data = measurement[-N:]
        assert (datetime.now(ZoneInfo("Europe/Berlin")) - data[0][0]) < pd.Timedelta(minutes=3)
        # throw out any exact zero
        data = [d for d in data if (d[1]!=0)]
        if len(data) == 0:
            logger.warning('Could not get robust reading... sending 0.')
            return (datetime.now(ZoneInfo("Europe/Berlin")), 0)
        else:
            if which=='last':
                return (datetime.now(ZoneInfo("Europe/Berlin")), data[-1][1])
            elif which=='max':
                return (datetime.now(ZoneInfo("Europe/Berlin")), np.array([x[1] for x in data]).max())
            elif which=='min':
                return (datetime.now(ZoneInfo("Europe/Berlin")), np.array([x[1] for x in data]).min())
            elif which=='median':
                return (datetime.now(ZoneInfo("Europe/Berlin")), np.median(np.array([x[1] for x in data])))

    except:
        logger.warning('Could not get robust reading... sending 0.')
        return (datetime.now(ZoneInfo("Europe/Berlin")), 0)


def _callback(collect, pkg):
    global rt_msg_count, last_rt_message_ts
    data = pkg.get("data")
    if data is None:
        return
    # Track that we saw a realtime message
    try:
        last_rt_message_ts = datetime.now(ZoneInfo("Europe/Berlin"))
        rt_msg_count = (rt_msg_count or 0) + 1
    except Exception:
        pass
    collect.append(
        (pd.to_datetime(data.get("liveMeasurement")['timestamp']), data.get("liveMeasurement")['power'] -data.get("liveMeasurement")["powerProduction"])
    )



#%%

measurement = []
soc = {
    'total':0,
    'A':0,
    'B':0,
    'C':0
}
current_p = {
    'A':0,
    'B':0,
    'C':0,
    'total':0,
    't':'now'
}
p_t = []

# Shared aiohttp session for Tibber to prevent 'Session is closed' during watchdog/resubscribe
tibber_session: aiohttp.ClientSession | None = None

async def main():
    global measurement
    global soc
    global current_p, p_t
    global tc, home
    global logger
    tc = None
    home = None

    logger.info('Starting background tasks: update_soc, update_power, strategy_loop')
    asyncio.create_task(update_soc())
    asyncio.create_task(update_power())
    asyncio.create_task(strategy_loop())
        

    tibber_logger.info('Initializing Tibber realtime client...')
    await get_tibber()
    # Start price task right away after realtime is up
    tibber_logger.info('Starting price update task')
    asyncio.create_task(update_price())
    await asyncio.sleep(3)
    # Keep running indefinitely; strategy loop handles decisions every 15 min
    logger.info('Service started. Entering idle wait loop.')
    # Heartbeat: log every 60s a small status
    try:
        while True:
            try:
                logger.info(
                    "heartbeat soc_total=%.1f p_total=%s meas=%s tasks=ok",
                    float(soc.get('total', 0) or 0),
                    current_p.get('total', 'n/a'),
                    len(measurement) if isinstance(measurement, list) else 'n/a',
                )
            except Exception:
                logger.debug('heartbeat failed')
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.info('Main loop cancelled, shutting down...')
        pass

# %%
logger = logging.getLogger('sax')
# Allow overriding log level with env (e.g., SAX_LOG_LEVEL=DEBUG)
_lvl = os.getenv('SAX_LOG_LEVEL', 'INFO').upper()
try:
    logger.setLevel(getattr(logging, _lvl, logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)

# Verbose format includes timestamp and module for journald and file
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s:%(process)d] %(message)s')

# Console (journald captures stdout/stderr from systemd)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

# Rotating file in repo dir for easy tailing
date = datetime.today().strftime('%Y-%m-%d')
log_file = Path(__file__).with_name(f"steuerung_{date}.log")
fileHandler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info('SAX control starting up...')
try:
    eff_min, eff_max = get_effective_soc_limits()
    masked_ips = {k: (v.rsplit('.', 1)[0] + '.x') for k, v in BATTERY_IPS.items()}
    logger.info(
        'startup batteries=%d ips=%s soc_window=%s power_limits=[%d..%d] log_level=%s',
        N_BATTERIES,
        masked_ips,
        (eff_min, eff_max),
        POWER_LIMIT_DISCHARGE_MIN_W,
        POWER_LIMIT_CHARGE_MAX_W,
        logging.getLevelName(logger.level),
    )
except Exception:
    pass

fleet: BatteryFleetController | None = None
fast_ctrl: FastController | None = None

# Dedicated strategy logger
strategy_logger = logging.getLogger('sax.strategy')
strategy_logger.setLevel(logging.INFO)
strategy_log_file = Path(__file__).with_name("strategy.log")
try:
    s_file = TimedRotatingFileHandler(strategy_log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8")
    s_file.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    strategy_logger.addHandler(s_file)
except Exception:
    pass

# Dedicated Tibber logger (inherits handlers from 'sax' so it logs to the same files/console)
tibber_logger = logging.getLogger('sax.tibber')
tibber_logger.setLevel(logging.INFO)

#%%
#await main()
try:
    asyncio.run(main())
except Exception:
    logging.getLogger('sax').exception('Fatal error in main')
    raise
#%%
#systemd-run --unit=sax --collect python ~/sax/steuerung.py^