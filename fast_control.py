from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from zoneinfo import ZoneInfo
from datetime import datetime
import logging

# This module implements a fast control loop that runs every ~3 seconds,
# adjusting the total setpoint to keep net import/export near zero depending on
# aggressiveness. It relies solely on the Fleet API exposed by controller.

try:
    from .controller import BatteryFleetController
    from .config import BATTERIES, FAST_CONTROL_DEFAULTS
    from .strategy import Strategy
except ImportError:
    from controller import BatteryFleetController
    from config import BATTERIES, FAST_CONTROL_DEFAULTS
    from strategy import Strategy


@dataclass
class FastControlConfig:
    # Positive means export to grid, negative means import from grid
    # We try to keep measured_net near the target within a tolerance.
    sample_period_s: float = 3.0
    # Tolerances for normal/aggressive modes (W)
    tol_normal_w: int = 150
    tol_aggressive_w: int = 50
    # Max step change in total setpoint per cycle to avoid oscillations (W)
    max_step_w: int = 500
    # Safety cap for absolute total setpoint change (W)
    max_total_abs_w: int = 9000
    # Filtering: exponential moving average to avoid chasing pulsing loads
    ema_alpha_normal: float = 0.5
    ema_alpha_passive: float = 0.2
    # Pulse detection: if |raw-ema| exceeds a threshold repeatedly, switch to passive mode
    pulse_threshold_w: int = 350
    pulse_detect_window: int = 8
    pulse_detect_min_hits: int = 2
    passive_cooldown_s: float = 45.0
    # Treat deviations sustained longer than this as real loads we should follow (seconds)
    sustained_follow_secs: float = 10.0
    # Passive mode also widens tolerance and reduces step size
    tol_passive_w: int = 300
    max_step_passive_w: int = 250
    # Discharge bias baseline
    discharge_bias_w: int = 1500
    discharge_bias_aggr_w: int = 3000


class FastController:
    def __init__(self, fleet: BatteryFleetController, cfg: Optional[FastControlConfig] = None):
        self.logger = logging.getLogger('sax.fast')
        self.fleet = fleet
        # Apply defaults from config, then override with provided cfg
        base = FastControlConfig()
        # Load from config defaults if present
        try:
            for k, v in FAST_CONTROL_DEFAULTS.items():
                if hasattr(base, k):
                    setattr(base, k, v)
        except Exception:
            pass
        if cfg:
            # overlay provided cfg fields
            for f in base.__dataclass_fields__.keys():
                if hasattr(cfg, f):
                    setattr(base, f, getattr(cfg, f))
        self.cfg = base
        self._last_target_w: int = 0
        self._ema_net: Optional[float] = None
        self._pulse_hits = []
        self._passive_until = None
        self._mode = "balance"  # balance | charge | discharge
        self._aggressive = False
        self._requested_energy_Wh = 0.0
        self._remaining_energy_Wh = 0.0
        self._interval_end = None
        self._last_ts = None
        # Heartbeat & measurement staleness
        self._last_heartbeat = None
        self._meas_prev: Optional[float] = None
        self._meas_still_since: Optional[datetime] = None

    def set_mode(
        self,
        mode: str,
        *,
        aggressive: bool = False,
        charge_energy_Wh: Optional[float] = None,
        interval_end: Optional[datetime] = None,
        discharge_bias_w: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """Configure behavior: balance | charge | discharge with optional aggressiveness.

        If mode=='charge' and charge_energy_Wh is set, the controller will try to
        draw that energy from the grid over the remaining time of the current
        15-minute interval (or provided interval_end), blended with balancing.
    """
        from datetime import timedelta

        self._mode = mode
        self._aggressive = bool(aggressive)
        ref_now = now or datetime.now(ZoneInfo("Europe/Berlin"))
        if interval_end is None:
            # next 15-min boundary
            floored = ref_now.replace(minute=(ref_now.minute // 15) * 15, second=0, microsecond=0)
            interval_end = floored + timedelta(minutes=15)
        self._interval_end = interval_end

        if mode == "charge":
            self._requested_energy_Wh = float(charge_energy_Wh or 0.0)
            self._remaining_energy_Wh = max(self._requested_energy_Wh, 0.0)
        else:
            self._requested_energy_Wh = 0.0
            self._remaining_energy_Wh = 0.0

        if discharge_bias_w is not None:
            self.cfg.discharge_bias_w = int(discharge_bias_w)

        # reset integrator for new interval
        self._last_ts = None

    def update_from_strategy(self, strat: Strategy, maybe_energy_Wh: Optional[float], now: Optional[datetime] = None) -> None:
        aggressive = strat in (Strategy.DISCHARGE_AGGRESSIVE, Strategy.BALANCE_AGGRESSIVE)
        if strat in (Strategy.IDLE, Strategy.BALANCE, Strategy.BALANCE_AGGRESSIVE):
            self.set_mode("balance", aggressive=aggressive, now=now)
        elif strat in (Strategy.DISCHARGE, Strategy.DISCHARGE_AGGRESSIVE):
            bias = self.cfg.discharge_bias_aggr_w if aggressive else self.cfg.discharge_bias_w
            self.set_mode("discharge", aggressive=aggressive, discharge_bias_w=bias, now=now)
        elif strat in (Strategy.CHARGE, getattr(Strategy, 'CHARGE_GRID', Strategy.CHARGE)):
            self.set_mode("charge", aggressive=aggressive, charge_energy_Wh=(maybe_energy_Wh or 0.0), now=now)

    def _clip_total(self, total_w: int) -> int:
        m = self.cfg.max_total_abs_w
        return max(min(total_w, m), -m)

    async def run(self, measure_func: Callable[[], float | int | object]):
        """Run the fast control loop.

        measure_func: async or sync callable returning current net power in W
          - Positive means exporting to grid (PV surplus)
          - Negative means importing from grid (consumption > production)
        """
        import asyncio
        while True:
            try:
                val = measure_func()
                if asyncio.iscoroutine(val):
                    measured_net = await val
                else:
                    measured_net = val
            except Exception:
                measured_net = 0

            # Timestamp and filtered signal
            now_ts = datetime.now(ZoneInfo("Europe/Berlin"))
            raw = float(measured_net or 0.0)
            # Track measurement staleness (no change over time)
            if self._meas_prev is None:
                self._meas_prev = raw
                self._meas_still_since = now_ts
            else:
                if abs(raw - self._meas_prev) < 1.0:
                    # essentially unchanged
                    if self._meas_still_since is None:
                        self._meas_still_since = now_ts
                else:
                    self._meas_prev = raw
                    self._meas_still_since = None
            if self._ema_net is None:
                self._ema_net = raw

            # Passive mode window due to pulsing loads
            in_passive = self._passive_until is not None and now_ts < self._passive_until
            alpha = self.cfg.ema_alpha_passive if in_passive else self.cfg.ema_alpha_normal
            self._ema_net = alpha * raw + (1.0 - alpha) * self._ema_net

            # Pulse detection
            deviation = abs(raw - self._ema_net)
            self._pulse_hits.append(deviation >= self.cfg.pulse_threshold_w)
            if len(self._pulse_hits) > self.cfg.pulse_detect_window:
                self._pulse_hits = self._pulse_hits[-self.cfg.pulse_detect_window:]
            # Compute length of the most recent consecutive hits
            consec = 0
            for hit in reversed(self._pulse_hits):
                if hit:
                    consec += 1
                else:
                    break
            sustained_min_consec = max(1, int(self.cfg.sustained_follow_secs / self.cfg.sample_period_s + 0.5))

            if (not in_passive
                and sum(self._pulse_hits) >= self.cfg.pulse_detect_min_hits
                and consec < sustained_min_consec):
                from datetime import timedelta
                self._passive_until = now_ts + timedelta(seconds=self.cfg.passive_cooldown_s)
                in_passive = True
                self._pulse_hits.clear()
                self.logger.info("fast: spike detected → entering passive for %.0fs", self.cfg.pulse_cooldown_s if hasattr(self.cfg, 'pulse_cooldown_s') else self.cfg.passive_cooldown_s)

            # Tolerances and step sizes
            if in_passive:
                tol = self.cfg.tol_passive_w
                max_step = self.cfg.max_step_passive_w
            else:
                tol = self.cfg.tol_aggressive_w if self._aggressive else self.cfg.tol_normal_w
                max_step = self.cfg.max_step_w

            # Balancing target using filtered net
            error = -self._ema_net
            desired = self._last_target_w + max(min(error, max_step), -max_step)

            # Deadband decay
            if abs(self._ema_net) <= tol:
                desired = int(self._last_target_w * 0.8)

            # Mode-specific nudges
            if self._mode == "discharge":
                bias = self.cfg.discharge_bias_aggr_w if self._aggressive else self.cfg.discharge_bias_w
                desired += bias
            elif self._mode == "charge" and self._interval_end is not None:
                # Plan to draw remaining energy over remaining time
                remaining_s = max((self._interval_end - now_ts).total_seconds(), 1.0)
                remaining_h = remaining_s / 3600.0
                if self._remaining_energy_Wh > 0:
                    if getattr(self.cfg, "charge_flat", True):
                        # Flat charging: aim for constant power that delivers remaining energy over time
                        p_plan = - self._remaining_energy_Wh / remaining_h
                        desired = p_plan  # take precedence during charge intervals
                    else:
                        # Blend planning term with balancing
                        p_plan = - self._remaining_energy_Wh / remaining_h
                        desired += float(p_plan)

            desired = self._clip_total(int(desired))

            # Apply via fleet
            socs = self.fleet.read_soc_all()
            self.fleet.set_total_power(desired, socs)
            self._last_target_w = desired

            # Integrate energy actually delivered by the fleet
            try:
                per = self.fleet.read_power_all()
                batt_total_w = sum(per.get(b, 0) for b in BATTERIES)
            except Exception:
                batt_total_w = desired

            if self._mode == "charge" and self._interval_end is not None:
                if self._last_ts is not None:
                    dt_s = (now_ts - self._last_ts).total_seconds()
                    # Charging power is negative; subtract delivered Wh from remaining
                    self._remaining_energy_Wh = max(0.0, self._remaining_energy_Wh - max(-batt_total_w, 0.0) * dt_s / 3600.0)
                # Stop when interval ends or energy satisfied
                if now_ts >= self._interval_end or self._remaining_energy_Wh <= 1.0:
                    self._requested_energy_Wh = 0.0
                    self._remaining_energy_Wh = 0.0
                    self._interval_end = None

            self._last_ts = now_ts

            # Heartbeat every ~30s with useful context
            try:
                if self._last_heartbeat is None:
                    self._last_heartbeat = now_ts
                hb_dt = (now_ts - self._last_heartbeat).total_seconds()
                if hb_dt >= 30.0:
                    stale_s = (now_ts - self._meas_still_since).total_seconds() if self._meas_still_since else 0.0
                    self.logger.info(
                        "fast hb mode=%s aggr=%s passive=%s tol=%d max_step=%d raw=%.1f ema=%.1f desired=%d batt=%d pulses(win=%d hits=%d consec=%d) meas_stale_s=%.0f",
                        self._mode,
                        self._aggressive,
                        in_passive,
                        (self.cfg.tol_passive_w if in_passive else (self.cfg.tol_aggressive_w if self._aggressive else self.cfg.tol_normal_w)),
                        (self.cfg.max_step_passive_w if in_passive else self.cfg.max_step_w),
                        raw,
                        self._ema_net,
                        desired,
                        batt_total_w,
                        self.cfg.pulse_detect_window,
                        sum(self._pulse_hits),
                        consec,
                        stale_s,
                    )
                    self._last_heartbeat = now_ts
            except Exception:
                pass
            await asyncio.sleep(self.cfg.sample_period_s)
