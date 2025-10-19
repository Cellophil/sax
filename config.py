from __future__ import annotations
import os
from typing import Dict, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# Battery network configuration
BATTERY_IPS: Dict[str, str] = {
    'A': "192.168.178.127",
    'B': "192.168.178.130",
    'C': "192.168.178.132",
}
BATTERIES: Tuple[str, ...] = tuple(BATTERY_IPS.keys())
N_BATTERIES: int = len(BATTERIES)

# Modbus register addresses
REG_POWER_CMD = 41
REG_STATE = 45
REG_SOC = 46
REG_POWER_MEAS = 47

# Power limits per battery (Watts)
POWER_LIMIT_CHARGE_MAX_W = 3700      # +3700 W discharge to house/grid
POWER_LIMIT_DISCHARGE_MIN_W = -2400  # -2400 W charge from house/grid

# Safe SOC window
SOC_MIN = 15
SOC_MAX = 90

# External integrations
# IMPORTANT: No fallback token. Provide TIBBER_TOKEN via environment (e.g. systemd EnvironmentFile)
#_tok = os.getenv('TIBBER_TOKEN', '').strip()
#TIBBER_TOKEN: Optional[str] = _tok if _tok else None
TIBBER_TOKEN = 'EEE07436AD7347807083C7321542A3DA1CDF197D22A7424210F7FAA8F52029C8-1'

# Tibber API user agent (some users report different behavior depending on UA).
# Make it configurable; default to 'Andreas' as previously working value.
TIBBER_USER_AGENT: str = (os.getenv('TIBBER_USER_AGENT', 'Andreas').strip() or 'Andreas')


def get_effective_soc_limits(now: Optional[datetime] = None) -> Tuple[int, int]:
    """Return (soc_min, soc_max) with weekly overrides.

    - Friday: allow discharge to 0% (min=0)
    - Sunday: allow charge to 100% (max=100)
    Other days: use SOC_MIN..SOC_MAX.
    """
    if now is None:
        now = datetime.now(ZoneInfo("Europe/Berlin"))
    weekday = now.weekday()  # Monday=0 ... Sunday=6
    eff_min, eff_max = SOC_MIN, SOC_MAX
    if weekday == 4:  # Friday
        eff_min = 0
    if weekday == 6:  # Sunday
        eff_max = 100
    return eff_min, eff_max

# Fast-control default tuning (can be tweaked without code changes)
FAST_CONTROL_DEFAULTS = {
    # sampling and tolerances
    "sample_period_s": 3.0,
    "tol_normal_w": 150,
    "tol_aggressive_w": 50,
    "max_step_w": 500,
    "max_total_abs_w": 9000,
    # filtering and pulse handling
    "ema_alpha_normal": 0.5,
    "ema_alpha_passive": 0.2,
    "pulse_threshold_w": 350,
    "pulse_detect_window": 8,
    "pulse_detect_min_hits": 2,
    "passive_cooldown_s": 45.0,
    "sustained_follow_secs": 10.0,
    "tol_passive_w": 300,
    "max_step_passive_w": 250,
    # strategy nudges
    "discharge_bias_w": 1500,
    "discharge_bias_aggr_w": 3000,
    # charge behavior
    "charge_flat": True,
}
