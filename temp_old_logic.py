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