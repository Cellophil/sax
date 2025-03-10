#%%

from pyModbusTCP.client import ModbusClient
from pyModbusTCP.utils import long_list_to_word
import numpy as np

import tibber.const
import tibber
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import time
import matplotlib.pyplot as plt

from astral import LocationInfo
from astral.sun import sun

#%%
def set_up_plot():
    fig, ax = plt.subplots(3,3)

def update_plots(measurements):
    pass

#%%

N_battery = 3

def get_tcp_client(battery):

    if battery == 'A':
        ip="192.168.178.127"
    elif battery == 'B':
        ip="192.168.178.130"
    elif battery == 'C':
        ip="192.168.178.132"
    try:
        client = ModbusClient(host=ip, port=502, unit_id=64, auto_open=True, auto_close=True)
        return client
    except:
        return None

def get_soc(battery=None):
    
    soc = 0

    if battery is not None:
        try:
            client = get_tcp_client(battery)
            soc += client.read_holding_registers(46, 1)[0]
            return soc
        except:
            return -1

    try:
        for battery in ['A','B','C']:
            client = get_tcp_client(battery)
            soc += client.read_holding_registers(46, 1)[0]
        return soc / N_battery
    except:
        return -1

async def update_soc():
    global soc
    while True:

        error = False
        new_soc = []
        for battery in ['A','B','C']:
            try:
                client = get_tcp_client(battery)
                _soc = client.read_holding_registers(46, 1)[0]
                client.close()
            except:
                _soc = soc[battery]
                logger.warning(f'Could not read SOC of {battery}... retry in 60s. Retain old value.')
            new_soc.append(_soc)
            

        soc = {'total':(new_soc[0]+new_soc[1]+new_soc[2])/3, 'A':new_soc[0], 'B':new_soc[1], 'C':new_soc[2]}    

        await asyncio.sleep(60)
    

def _set_power(p, battery):
    p = int(p)
    assert battery in ['A','B','C']

    try:
        client = get_tcp_client(battery)
        _soc = soc[battery]

        if p > 3700:
            logger.warning(f'Power limit (per battery) {3.7:.2}kW')
            p = 3700
        elif p < -2400:
            logger.warning(f'Power limit (per battery) -{2.4:.2}kW')
            p = -2400

    
        if (p > 0) and (_soc < 15):
            p = 0
            logger.warning('Overwriting power here to save battery - low soc.')
        if (p < 0) and (_soc > 95):
            logger.warning('Overwriting power here to save battery - high soc.')
            p = 0
        
        
        if p >= 0:
            client.write_multiple_registers(41, [p])
        else:
            client.write_multiple_registers(41, [65536+p])
    except:
        logger.warning(f'Could not communicate with battery {battery}.')


def set_power(p):
    global soc

    p /= N_battery
    p = int(p)

    for battery in ['A','B','C']:
        _set_power(p, battery)


def _get_power(battery):

    try:
        client = get_tcp_client(battery)
        attempts = 0
        _p = -9999
        while _p == -9999:
            _p = client.read_holding_registers(47, 1)[0]
            if _p is None:
                _p = -9999
            if _p > 0:
                break
            attempts += 1
            if attempts > 10:
                logger.warning(f'Could not read current power level of battery {battery}.')
                client.close()
                return -9999
            time.sleep(1)
        _p = (_p - 32768//2)

        client.close()
        return _p
    except:
        return -9999

def get_power():
    p = 0
    N = 0

    for battery in ['A','B','C']:
        _p = _get_power(battery)
        if not (_p==-9999):
            p += _p
            N += 1
    #p /= (N+1e-6)
    return p

async def update_power():

    global p_t, current_p
    current_p = {}

    while True:

        for battery in ['A', 'B', 'C']:
            _p = _get_power(battery)
            if _p == -9999:
                # best guess: old value * 0.9
                _p = current_p[battery] * 0.9
            current_p[battery] = _p
        current_p['total'] = current_p['A']+current_p['B']+current_p['C']
        current_p['t'] = datetime.now(ZoneInfo("Europe/Berlin"))
        p_t.append(current_p)

        await asyncio.sleep(2)


async def soll_power():
    
    global set_p, soc
    battery_i = 0
    battery_index = {0:'A',1:'B',2:'C'}

    while True:

        # check if set_p is up to date
        if (datetime.now(ZoneInfo("Europe/Berlin")) - set_p['t']).seconds > 30:
            set_p['t'] = datetime.now(ZoneInfo("Europe/Berlin"))
            set_p['p'] = 0.
            logger.warning('Didnt get up to date setpoint of batteries - 30s old - setting zero.')

        battery_i += 1
        battery_i = battery_i % 3
        battery = battery_index[battery_i]

        _set_power(set_p['p']/3., battery)
        await asyncio.sleep(1)


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
        logger.info(f'Trying to setup Tibber connection - wait for {timeout}s.')
        await asyncio.sleep(timeout)

        try:
            tc = tibber.Tibber('VlMXTgqKRqgY2ZYdai-WzKZ4h8Go1FbdIj-RqRffYjU', user_agent="Andreas")
            await tc.update_info()
            print(tc.name)
            home = tc.get_homes()[0]
            await home.rt_subscribe(lambda pkg: _callback(measurement, pkg))
            await asyncio.sleep(10)
            assert home.rt_subscription_running
            break
        except:
            logger.info('Retrying to establish tibber connection...')
            try:
                await tc.close_connection()
                await asyncio.sleep(10)
            except:
                pass

    return


async def update_price():
    global prices, home
    prices=np.array([30.]*11)

    while True:

        try:
            _price = pd.Series(home.price_total)
            _price.index = pd.to_datetime(_price.index)
            _price = _price.loc[[t for t in _price.index if t + pd.Timedelta('1h') > datetime.now(ZoneInfo("Europe/Berlin"))]]
        except:
            _price = pd.Series([])
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
        
    logger = logging.getLogger('sax')

    await get_tibber()
    await asyncio.sleep(20)
    
    asyncio.create_task(update_price())

    while True:
        
        #prices = get_next_prices(home)
        #if prices is None:
        #    try:
        #        await update_price(home)
        #        prices = get_next_prices(home)
        #    except:
        #        tc, home = await get_tibber()
        #        continue


        if not home.rt_subscription_running:
            await get_tibber()
            continue
    
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
            asyncio.sleep(1)
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
fileHandler = logging.FileHandler(f"/home/andreas/sax/steuerung_{date}.log")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.debug('test message')

#%%
#await main()
asyncio.run(main())
#%%
#systemd-run --unit=sax --collect python ~/sax/steuerung.py^