from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

try:
    # Package import
    from .config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, get_effective_soc_limits,
    )
    from .modbus_io import BatteryClient
except ImportError:
    # Script-level import fallback
    from config import (
        BATTERY_IPS, BATTERIES, N_BATTERIES,
        POWER_LIMIT_CHARGE_MAX_W, POWER_LIMIT_DISCHARGE_MIN_W,
        SOC_MIN, SOC_MAX, get_effective_soc_limits,
    )
    from modbus_io import BatteryClient


@dataclass
class FleetState:
    soc: Dict[str, int]  # per battery soc 0..100


class BatteryFleetController:
    """Controls a fleet of identical batteries as if it was one big battery.

    Distributes a total setpoint across batteries while:
    - Respecting per-battery power limits and safe SOC window
    - Balancing SOCs to keep them in sync
    - Favoring charging of lower-SOC batteries and discharging of higher-SOC ones
    """

    def __init__(self):
        self.clients: Dict[str, BatteryClient] = {
            b: BatteryClient(BATTERY_IPS[b]) for b in BATTERIES
        }
        self.last_weights: Dict[str, Dict[str, float]] = {"charge": {}, "discharge": {}}
        # Persist effective SOC bounds (e.g., Friday/Sunday overrides)
        self.soc_bounds: Tuple[int, int] = get_effective_soc_limits()

    def set_soc_bounds(self, soc_min: int, soc_max: int) -> None:
        """Explicitly set effective SOC bounds for distribution logic."""
        self.soc_bounds = (int(soc_min), int(soc_max))

    def compute_weights(self, socs: Dict[str, int]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (charge_weights, discharge_weights) per battery.

        - Unknown/invalid SOC => zero weight (battery will be set to 0 W)
        - Charging weights increase with room to max SOC
        - Discharging weights increase with room to min SOC
        """
        eff_min, eff_max = self.soc_bounds
        w_charge: Dict[str, float] = {}
        w_discharge: Dict[str, float] = {}
        for b in BATTERIES:
            s = socs.get(b)
            try:
                s_f = float(s) if s is not None else None
            except Exception:
                s_f = None
            if s_f is None or not (0.0 <= s_f <= 100.0):
                w_charge[b] = 0.0
                w_discharge[b] = 0.0
                continue
            w_charge[b] = max(0.0, eff_max - s_f)
            w_discharge[b] = max(0.0, s_f - eff_min)
        self.last_weights = {"charge": w_charge.copy(), "discharge": w_discharge.copy()}
        return w_charge, w_discharge

    def read_soc_all(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for b, cli in self.clients.items():
            s = cli.read_soc()
            if s is None:
                continue
            out[b] = s
        return out

    def read_power_all(self) -> Dict[str, int]:
        """Read current power [W] per battery.

        Returns a dict with available readings; missing/unavailable entries are omitted.
        Positive means discharge to house/grid, negative means charging.
        """
        out: Dict[str, int] = {}
        for b, cli in self.clients.items():
            p = cli.read_power_w()
            if p is None:
                continue
            out[b] = int(p)
        return out

    def read_total_power(self) -> int:
        """Read and sum current power [W] across all batteries.

        Sums available per-battery values; if none are available, returns 0.
        """
        per = self.read_power_all()
        if not per:
            return 0
        return int(sum(per.values()))

    def set_total_power(self, p_total_w: int, socs: Dict[str, int]) -> Dict[str, int]:
        """Distribute total power across batteries with SOC balancing.

        Positive p_total_w discharges; negative charges.
        Returns dict of per-battery setpoints actually sent.
        """
        if not socs:
            # No SOC information available: safest is to set all to 0 W
            res = {b: 0 for b in BATTERIES}
            for b in BATTERIES:
                self.clients[b].set_power_w(0)
            return res
        # Use persisted SOC bounds
        eff_min, eff_max = self.soc_bounds

        # Compute symmetric weights
        w_charge, w_discharge = self.compute_weights(socs)
        total = float(p_total_w)
        p_sign = 0 if total == 0 else (1 if total > 0 else -1)
        weights = w_discharge if p_sign > 0 else w_charge
        total_w = sum(weights.values())

        # Allocate proportionally; accept small rounding error
        res: Dict[str, int] = {b: 0 for b in BATTERIES}
        if total == 0 or total_w == 0:
            # Idle or no usable weights: set zeros
            for b in BATTERIES:
                res[b] = 0
        else:
            for b in BATTERIES:
                share = total * (weights[b] / total_w)
                v = int(round(share))
                # SOC-bound enforcement at allocation time
                s = socs.get(b, None)
                if s is None:
                    v = 0
                else:
                    try:
                        s_f = float(s)
                    except Exception:
                        v = 0
                    else:
                        if v > 0 and s_f <= eff_min:
                            v = 0
                        if v < 0 and s_f >= eff_max:
                            v = 0
                # Per-battery power caps
                v = max(min(v, POWER_LIMIT_CHARGE_MAX_W), POWER_LIMIT_DISCHARGE_MIN_W)
                res[b] = v

        # Send without micro-corrections; small error (<~10–20 W) is acceptable
        for b in BATTERIES:
            self.clients[b].set_power_w(res[b])

        return res

    def close(self):
        for cli in self.clients.values():
            cli.close()
