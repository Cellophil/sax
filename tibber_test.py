#%%
import tibber.const
import tibber
import asyncio
import aiohttp
import pandas as pd
#%%

async def start():
  tibber_connection = tibber.Tibber('VlMXTgqKRqgY2ZYdai-WzKZ4h8Go1FbdIj-RqRffYjU', user_agent="Anton")
  await tibber_connection.update_info()
  print(tibber_connection.name)

  home = tibber_connection.get_homes()[0]
  await home.fetch_consumption_data()
  await home.update_info()
  print(home.address1)

  await home.update_price_info()
  print(home.current_price_info)
  price = home.price_total
  price.index = pd.to_datetime(price.index)
  print(price)

  await tibber_connection.close_connection()

loop = asyncio.run(start())

#%%
def _callback(pkg):
    print(pkg)
    data = pkg.get("data")
    if data is None:
        return
    print(data.get("liveMeasurement"))

#%%

async def run():
    
    tibber_connection = tibber.Tibber('EEE07436AD7347807083C7321542A3DA1CDF197D22A7424210F7FAA8F52029C8-1', user_agent="Andreas")
    await tibber_connection.update_info()
    home = tibber_connection.get_homes()[0]
    await home.rt_subscribe(_callback)

    while True:
        
        await asyncio.sleep(10)

#%%
#tibber_connection = tibber.Tibber('VlMXTgqKRqgY2ZYdai-WzKZ4h8Go1FbdIj-RqRffYjU', user_agent="Anton")


#%%

await run()


# %%
