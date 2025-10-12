from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import socket
from pyModbusTCP.client import ModbusClient

try:
    from .config import (
        REG_POWER_CMD, REG_SOC, REG_POWER_MEAS,
    )
except ImportError:
    from config import (
        REG_POWER_CMD, REG_SOC, REG_POWER_MEAS,
    )


@dataclass
class BatteryReading:
    soc: Optional[int] = None  # 0..100
    power_w: Optional[int] = None  # signed W, + discharge, - charge


class BatteryClient:
    """Thin adapter around Modbus for a single battery.

    Handles unsigned register conventions and small retry loops.
    """
    def __init__(self, host: str, unit_id: int = 64, port: int = 502):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.client = ModbusClient(host=host, port=port, unit_id=unit_id, auto_open=True, auto_close=True)

    def _retry(self, fn, attempts: int = 3, delay_s: float = 0.5):
        last_exc = None
        for _ in range(attempts):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                time.sleep(delay_s)
        if last_exc:
            raise last_exc

    def read_soc(self) -> Optional[int]:
        def _op():
            vals = self.client.read_holding_registers(REG_SOC, 1)
            if not vals:
                return None
            return int(vals[0])
        try:
            return self._retry(_op)
        except Exception:
            return None

    def read_power_w(self) -> Optional[int]:
        def _op():
            vals = self.client.read_holding_registers(REG_POWER_MEAS, 1)
            if not vals:
                return None
            raw = int(vals[0])
            # Device-specific mapping observed in existing code
            return raw - (32768 // 2)
        try:
            return self._retry(_op)
        except Exception:
            return None

    def set_power_w(self, watts: int) -> bool:
        # Positive means discharge to house; negative means charge from house
        watts = int(watts)
        def _op():
            if watts >= 0:
                return self.client.write_multiple_registers(REG_POWER_CMD, [watts])
            else:
                return self.client.write_multiple_registers(REG_POWER_CMD, [65536 + watts])
        try:
            ok = self._retry(_op)
            return bool(ok)
        except Exception:
            return False

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
