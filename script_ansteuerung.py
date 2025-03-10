#%%
from pyModbusTCP.client import ModbusClient
from pyModbusTCP.utils import long_list_to_word
import numpy as np
import time

"3600 / 502"
client = ModbusClient(host="192.168.178.127", port=502, unit_id=64, auto_open=True, auto_close=True)
#%%
soc = client.read_holding_registers(46, 1)[0]
print(soc)
#%%
p = client.read_holding_registers(47, 1)[0]
print(p-32768//2)
# %%
state = client.read_holding_registers(45, 1)
print(state)
#%%
state = client.write_multiple_registers(45, [2])
print(state)
# %%
p_run = client.write_multiple_registers(41, [0])
print(p_run)
#%%
# %%
p_run = client.write_multiple_registers(41, [65536-500])
print(p_run)

#%%
while True:

    time.sleep(30)
    try:

        soc = client.read_holding_registers(46, 1)[0]
        if soc < 30:
            break
        print(soc)
        p_run = client.write_multiple_registers(41, [800])
        #p_run = client.write_multiple_registers(41, [500])
        time.sleep(2)
        p = client.read_holding_registers(47, 1)[0]
        print(p-32768//2)
    except:
        client = ModbusClient(host="192.168.178.127", port=502, unit_id=64, auto_open=True, auto_close=True)

p_run = client.write_multiple_registers(41, [0])

# %%



# %%
