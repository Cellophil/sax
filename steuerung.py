#%%

from pyModbusTCP.client import ModbusClient
from pyModbusTCP.utils import long_list_to_word
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
import time
import matplotlib.pyplot as plt

from astral import LocationInfo
from astral.sun import sun
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
def set_up_plot():
    fig, ax = plt.subplots(3,3)

def update_plots(measurements):
    pass

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
    

def _set_power(p, battery):
    # Deprecated: use fleet.set_total_power instead
    return


def set_power(p):
    global soc
    global fleet

    # Initialize fleet controller lazily
    if fleet is None:
        fleet = BatteryFleetController()

    # Fleet expects the total setpoint; it will distribute and respect SOC bounds
    # Fleet uses its persisted SOC bounds; they default to weekly overrides
    fleet.set_total_power(int(p), {b: soc.get(b) for b in BATTERIES})


def _get_power(battery):
    # Deprecated: use fleet.read_power_all / read_total_power
    return -9999

def get_power():
    # Deprecated: use fleet.read_total_power
    return 0

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


async def soll_power():
    # Deprecated: replaced by strategy_loop
    while True:
        await asyncio.sleep(5)

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


def test_battery(battery):
    # ramp up and down with different communication intervals

    global soc
    soc = {'total':50, 'A':50, 'B':50, 'C':50}

    _set_power(1500, battery)
    p_t = []
    for i in range(30):
        p_t.append((datetime.now(), get_power()))
        print(p_t[-1])
        time.sleep(1)
    
    _set_power(0, battery)
    for i in range(30):
        p_t.append((datetime.now(), get_power()))
        print(p_t[-1])
        time.sleep(1)

    t = datetime.now()
    for i in range(60):

        sin_y = np.sin((datetime.now() - t).seconds / 25 * 2 * np.pi) 
        _set_power(sin_y*200, battery)
        p_t.append((datetime.now(), get_power()))
        print(p_t[-1])
        time.sleep(1)    

    _set_power(0, battery)

    p_t = pd.DataFrame(data=p_t, columns=['t','p']).set_index('t')
    p_t.plot()

    return p_t


#%%

async def get_tibber():
    global measurement, tc, home

    if len(measurement) > 1200*24:
        measurement = measurement[-1200*24:]

    if home is not None:
        # nothing to do
        if home.rt_subscription_running:
            return

    if tc is not None:
        try:
            await tc.close_connection()
            await asyncio.sleep(10)
        except:
            pass
    
    timeout = 0
    while True:
        timeout += 60
        logger.warning(f'Trying to setup Tibber connection - wait for {timeout}s.')
        await asyncio.sleep(timeout)

        try:
            tc = tibber.Tibber(TIBBER_TOKEN, user_agent="Andreas")
            await tc.update_info()
            print(tc.name)
            home = tc.get_homes()[0]
            await home.rt_subscribe(lambda pkg: _callback(measurement, pkg))
            await asyncio.sleep(10)
            assert home.rt_subscription_running
            break
        except:
            logger.warning('Retrying to reestablish tibber connection...')
            try:
                await tc.close_connection()
                await asyncio.sleep(10)
            except:
                pass

    return


async def update_price():
    global prices, home, prices_series_15
    prices=np.array([30.]*11)
    prices_series_15 = None

    while True:

        try:
            _price = pd.Series(home.price_total)
            # Normalize to tz-aware 15-min series for strategy
            prices_series_15 = normalize_prices_15(_price)
            # Backward compatible numpy fallback for legacy code paths
            _price_idx = pd.to_datetime(_price.index)
            _price = _price.loc[[t for t in _price_idx if t + pd.Timedelta('1h') > datetime.now(ZoneInfo("Europe/Berlin"))]]
        except Exception:
            _price = pd.Series([])
            prices_series_15 = None
            logger.warning(f'Price problems...')

        logger.debug(f'Raw prices: {_price}')
        

        if len(_price) >= 11:
            # ok
            prices = np.array(_price.values[:11])
            #prices[0] = prices[0] - 0.04
            
        else:
            # need to fetch

            # artificial prices in the meantime
            if len(_price) > 0:
                prices = np.array(list(_price.values) + [30.]*11)
            else:
                prices = np.array([30.]*11)

            try:
                await home.update_price_info()
            except:
                pass

        await asyncio.sleep(60)


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
    global current_p, p_t, set_p
    global tc, home
    global prices
    prices = np.array([30.]*11)
    tc = None
    home = None

    asyncio.create_task(update_soc())
    asyncio.create_task(update_power())
    set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
    asyncio.create_task(soll_power())
    asyncio.create_task(strategy_loop())
        
    logger = logging.getLogger('sax')

    await get_tibber()
    await asyncio.sleep(20)
    
    asyncio.create_task(update_price())
    
    # Optionally start a fast controller in aggressive times. For now, keep it idle; you can
    # start it conditionally from decide_strategy (by flipping a flag) or here.
    # Example hook (commented):
    # fleet_ctrl = FastController(fleet)
    # asyncio.create_task(fleet_ctrl.run(lambda: measurement[-1][1] if measurement else 0, mode="aggressive"))

    while True:
        
        #prices = get_next_prices(home)
        #if prices is None:
        #    try:
        #        await update_price(home)
        #        prices = get_next_prices(home)
        #    except:
        #        tc, home = await get_tibber()
        #        continue


        #if not home.rt_subscription_running:
        #    await get_tibber()
        #    continue
    
        try:
            t, reading = get_robust_reading(N=10, which='median')
            t, reading_min = get_robust_reading(N=3, which='min')
            t, reading_max = get_robust_reading(N=3, which='max')
        except:
            logger.warning('Could not get current measurement...')
            set_p = (datetime.now(ZoneInfo("Europe/Berlin")), 0.)
            await asyncio.sleep(10)
            continue

        consumption = reading + current_p['total']
        charging_power = reading_max + current_p['total']
        if charging_power > 0:
            charging_power = 0
        compensation_power = reading_min + current_p['total']# - 100
        if compensation_power < 0:
            compensation_power = 0

        logger.debug(f't={t}: Current reading {reading}W, battery {current_p["total"]}W ({current_p["A"]}/{current_p["B"]}/{current_p["C"]}), set point {set_p["p"]}W, estimated total consumption {consumption}W')
        
        if reading == 0:
            # could be an error that reading... idle
            await asyncio.sleep(1)
            logger.info(f'Reading says exactly 0 - could be correct, but assuming a communication error.')

        elif consumption < -10:

            if soc["total"] < 95:
                logger.debug(f'PV production of {consumption/1000:.2}kW - soc {soc["total"]}, charging...')
                set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':charging_power}
            else:
                # full, idle
                logger.debug(f'PV production of {consumption/1000:.2}kW - soc {soc["total"]}, idle')
                set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
                await asyncio.sleep(20)

        elif consumption > 10:

            # any cheap hours followed by expensive?
            #prices = get_next_prices(home)
            logger.debug(f'Prices for the next 8h: {prices}')
            maxp = np.max(prices)
            minp = np.min(prices)
            perc_per_hour = 10 if maxp > 40 else 20
            hours_supply = (soc["total"]-15)/perc_per_hour
            price_non_supplied = np.sort(prices)[-int(np.round(hours_supply))-1]
            current_p_index = np.where(prices[0] == np.sort(prices))[0][-1]
            hours_charging = (100-soc["total"])/35 # with 2.5kW
            cheaper_hours = np.where(prices[0] == np.sort(prices))[0][0]
            more_expensive_hours = 11-np.where(prices[0] == np.sort(prices))[0][-1]

            logger.debug(f'soc: {soc["total"]}, hours of supply est: {hours_supply:.2}, hours of charging: {hours_charging:.2}, cheaper hours: {cheaper_hours}, more exp hours: {more_expensive_hours}')

            if (hours_supply > more_expensive_hours) or (price_non_supplied - prices[0] < 0.02):
                # discharge and compensate
                if soc["total"] > 15:
                    set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':compensation_power}
                    logger.debug(f'Current price {prices[0]} - max next 8h is {maxp} - soc {soc["total"]} - Discharging at {compensation_power}W...')
                    
                else:
                    logger.debug(f'Current price {prices[0]} - max next 8h is {maxp} - soc {soc["total"]} - Low soc, idle...')
                    set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
                    await asyncio.sleep(20)
            else:
                # we are currently in a cheap hour
                if hours_charging >= cheaper_hours:
                    if price_non_supplied - prices[0] >= 0.05:
                        # pump up battery
                        # how many opportunities are there to do that?

                        if (soc["total"] < 85):
                            logger.debug(f'Current price {prices[0]} - max next 8h is {maxp} - soc {soc["total"]} - Charging from grid...')
                            set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':-4500}
                            await asyncio.sleep(20)
                        else:
                            logger.debug(f'Current price {prices[0]} - max next 8h is {maxp} - min next 8h is {minp} - soc {soc["total"]} - Waiting...')
                            set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
                            await asyncio.sleep(20)
                    else:
                        logger.debug(f'Current price {prices[0]} - max next 8h is {maxp} - min next 8h is {minp} - soc {soc["total"]} - Price spread not worth charging...')
                        set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
                        await asyncio.sleep(20)
                else:
                    # not worth it to store up, do nothing
                    logger.info(f'Current price {prices[0]} - max next 8h is {maxp} - soc {soc["total"]} - Idle...')
                    set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}
                    await asyncio.sleep(20)
        else:
            set_p = {'t':datetime.now(ZoneInfo("Europe/Berlin")), 'p':0.}

        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break

    await tc.close_connection()
    return home, reading

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