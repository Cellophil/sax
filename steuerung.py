#%%

import numpy as np
import os
from pathlib import Path

import tibber.const
import tibber
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
from logging.handlers import TimedRotatingFileHandler
try:
    from .config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, TIBBER_TOKEN, get_effective_soc_limits,
    )
    from .controller import BatteryFleetController
    from .strategy import Strategy, StrategyContext, decide_strategy, strategy_to_setpoint, normalize_prices_15
    from .fast_control import FastController, FastControlConfig
except ImportError:
    from config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, TIBBER_TOKEN, get_effective_soc_limits,
    )
    from controller import BatteryFleetController
    from strategy import Strategy, StrategyContext, decide_strategy, strategy_to_setpoint, normalize_prices_15
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
        except Exception:
            logger.warning('Could not read power via fleet; retaining last values.')
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
            ctx = StrategyContext(
                now=now.to_pydatetime(),
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
            logger.warning(f"Strategy loop error: {e}")
        # loop continues; we'll align to next quarter at the top



#%%

async def get_tibber():
    global measurement, tc, home

    if len(measurement) > 1200*24:
        measurement = measurement[-1200*24:]
    # Already subscribed and running? Nothing to do.
    if home is not None and getattr(home, 'rt_subscription_running', False):
        return

    # Establish once, minimal retry without closing
    while True:
        try:
            tc = tibber.Tibber(TIBBER_TOKEN, user_agent="SAX")
            await tc.update_info()
            home = tc.get_homes()[0]
            await home.rt_subscribe(lambda pkg: _callback(measurement, pkg))
            await asyncio.sleep(5)
            if home.rt_subscription_running:
                logger.info('Tibber realtime subscription started.')
                return
        except Exception as e:
            logger.warning(f'Tibber setup failed, retrying in 120s: {e}')
            await asyncio.sleep(120)


async def update_price():
    global home, prices_series_15
    prices_series_15 = None

    while True:
        try:
            # Keep full horizon; normalize to 15-min tz-aware series
            _price = pd.Series(home.price_total)
            prices_series_15 = normalize_prices_15(_price)
        except Exception as e:
            prices_series_15 = None
            logger.warning(f'Price problems: {e}')

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
    data = pkg.get("data")
    if data is None:
        return
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
    tc = None
    home = None

    asyncio.create_task(update_soc())
    asyncio.create_task(update_power())
    asyncio.create_task(strategy_loop())
        
    logger = logging.getLogger('sax')

    await get_tibber()
    await asyncio.sleep(10)
    asyncio.create_task(update_price())
    # Keep running indefinitely; strategy loop handles decisions every 15 min
    await asyncio.Event().wait()

# %%
logger = logging.getLogger('sax')
logger.setLevel('WARNING')

console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
formatter =  logging.Formatter('%(levelname)s : %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

date = datetime.today().strftime('%Y-%m-%d')
log_file = Path(__file__).with_name(f"steuerung_{date}.log")
fileHandler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.debug('test message')

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

#%%
#await main()
asyncio.run(main())
#%%
#systemd-run --unit=sax --collect python ~/sax/steuerung.py^