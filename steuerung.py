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

#%%

N_battery = N_BATTERIES

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
    while True:
        try:
            # Align to quarter-hour boundaries
            now = pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin"))
            next_q = now.ceil("15min")
            sleep_s = max(0.0, (next_q - now).total_seconds())
            await asyncio.sleep(sleep_s)

            now = pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin"))
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

    if len(measurement) > 1200*24:
        measurement = measurement[-1200*24:]
    # Already subscribed and running? Nothing to do.
    if home is not None and getattr(home, 'rt_subscription_running', False):
        return

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
            tc = tibber.Tibber(TIBBER_TOKEN, user_agent=TIBBER_USER_AGENT)
            await tc.update_info()
            homes = tc.get_homes()
            if not homes:
                raise RuntimeError("Tibber account has no homes associated with this token.")
            home = homes[0]
            tibber_logger.info(f"Found {len(homes)} home(s); subscribing to realtime for the first home…")
            # Reset counters before subscribe
            rt_msg_count = 0
            last_rt_message_ts = None
            await home.rt_subscribe(lambda pkg: _callback(measurement, pkg))
            # Give the subscription some time to flip the running flag
            tibber_logger.info("Realtime subscribed; waiting up to 30s for running flag…")
            await asyncio.sleep(30)
            running_flag = getattr(home, 'rt_subscription_running', False)
            tibber_logger.info(f"rt_subscription_running={running_flag} msg_count={rt_msg_count} last_msg_ts={last_rt_message_ts}")
            # Treat reception of any message as success, even if flag didn't flip
            if running_flag or (rt_msg_count and rt_msg_count > 0):
                tibber_logger.info('Tibber realtime subscription started.')
                backoff = 60
                return
            else:
                # Trigger retry/backoff path handled below
                raise RuntimeError('Tibber realtime subscription not running after subscribe call')
        except Exception as e:
            # Common pitfall: Tibber API returns text/plain when token is invalid or URL is wrong
            msg = str(e)
            if 'Unexpected content type: text/plain' in msg:
                tibber_logger.error('Tibber auth/content-type error. This usually means your TIBBER_TOKEN is invalid or the API endpoint changed. '\
                             'Please verify the token (Account -> Developer in Tibber app).')
            # If we got rate limited (HTTP 429), back off more aggressively
            if '429' in msg or 'Too Many Requests' in msg:
                backoff = min(max_backoff, max(120, backoff * 2))
                tibber_logger.warning(f'Tibber rate-limited (429). Backing off for {backoff}s')
            else:
                backoff = min(max_backoff, max(60, backoff * 2))
                tibber_logger.warning(f'Tibber setup failed, will retry in {backoff}s: {e}')
            # Best-effort: close any underlying aiohttp session to avoid warnings
            try:
                if tc is not None:
                    for cand in ('close', 'close_connection', 'close_session'):
                        fn = getattr(tc, cand, None)
                        if callable(fn):
                            res = fn()
                            if inspect.isawaitable(res):
                                await res
                    sess = getattr(tc, 'session', None)
                    if sess is not None:
                        try:
                            res = sess.close()
                            if inspect.isawaitable(res):
                                await res
                        except Exception:
                            pass
            except Exception:
                pass
            await asyncio.sleep(backoff)


async def update_price():
    global home, prices_series_15
    prices_series_15 = None

    while True:
        try:
            # Wait until home is initialized
            if home is None:
                await asyncio.sleep(5)
                continue

            # Request 15-min prices when supported; fall back to hourly
            res_used = "QUARTER_HOURLY"
            try:
                await home.update_price_info(resolution="QUARTER_HOURLY")
            except Exception as e:
                tibber_logger.info(f"15-min prices not available, fallback to hourly: {e}")
                await home.update_price_info(resolution="HOURLY")
                res_used = "HOURLY"

            # Extract price entries from Tibber price_info (today + tomorrow)
            info = getattr(home, 'price_info', None)
            entries = []
            if info is not None:
                for arr in [getattr(info, 'today', []) or [], getattr(info, 'tomorrow', []) or []]:
                    for p in arr:
                        try:
                            ts_raw = getattr(p, 'startsAt', getattr(p, 'starts_at', None))
                            ts = pd.to_datetime(ts_raw)
                            if ts.tzinfo is None:
                                ts = ts.tz_localize("UTC")
                            ts = ts.tz_convert(ZoneInfo("Europe/Berlin"))
                            val = float(getattr(p, 'total'))
                            entries.append((ts, val))
                        except Exception:
                            pass

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
                # Fallback to legacy attribute if price_info missing
                _price = pd.Series(getattr(home, 'price_total', []))
                prices_series_15 = normalize_prices_15(_price)

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
    await asyncio.sleep(10)
    tibber_logger.info('Starting price update task')
    asyncio.create_task(update_price())
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