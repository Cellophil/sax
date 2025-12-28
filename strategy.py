#%%
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
from zoneinfo import ZoneInfo
import numpy as np
import pypsa
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

strategy_logger = logging.getLogger('sax.strategy')


class Strategy(Enum):
    IDLE = auto()
    DISCHARGE = auto()
    DISCHARGE_AGGRESSIVE = auto()

    BALANCE = auto()
    BALANCE_AGGRESSIVE = auto()

    CHARGE = auto()
    CHARGE_GRID = auto()


@dataclass
class StrategyContext:
    now: datetime
    soc_total: float
    # Make SOC bounds optional with safe defaults; can be overridden by caller
    soc_limit_lower: float = 15.0
    soc_limit_upper: float = 90.0

    prices_15: Optional[pd.Series] = None  # index tz-aware datetimes, values in €/kWh or similar
    # Optional extended context
    soc_per: Optional[Dict[str, float]] = None
    current_total_power_w: Optional[float] = None
    outdoor_temp_c: Optional[float] = None
    pv_forecast_w: Optional[pd.Series] = None
    load_forecast_w: Optional[pd.Series] = None



def normalize_prices_15(prices: Optional[pd.Series]) -> Optional[pd.Series]:
    """Ensure prices are tz-aware (Europe/Berlin), future-looking, and sorted.

    Accepts None or a pandas Series with datetime-like index.
    Returns a cleaned Series or None.
    """
    if prices is None or len(prices) == 0:
        return None
    try:
        idx = pd.to_datetime(prices.index, utc=True).tz_convert(ZoneInfo("Europe/Berlin"))
    except Exception:
        try:
            idx = pd.to_datetime(prices.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC").tz_convert(ZoneInfo("Europe/Berlin"))
            else:
                idx = idx.tz_convert(ZoneInfo("Europe/Berlin"))
        except Exception:
            return None
    s = pd.Series(prices.values, index=idx).sort_index()
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    s = s[s.index >= (now - pd.Timedelta(minutes=1))]
    if len(s) == 0:
        return None
    return s


def get_default_expected_pv(t):
    """Return a simple default expected PV generation profile (W) according to the time index t.
    
    """

    season_strength = max(0, 8 - 2*np.abs(t[0].month - 6.5))*1000
    profile = pd.Series(index=t, data=0.0)

    for offset in [-24, 0, 24]:
        dt = (t - pd.Timestamp(t[0].replace(hour=13, minute=0, second=0, microsecond=0)))/pd.Timedelta(hours=1) + offset
        profile += pd.Series(index=t, data =
                np.exp(-dt**2 / 4) * season_strength)

    return profile

def get_default_expected_consumption(t):
    """Return a simple default expected household consumption profile (W) according to the time index t.
    
    """

    base_level = 500 + 200*np.sin(2 * np.pi * (t.hour + t.minute/60) / 24 - np.pi/2)  # daily cycle
    evening_peak = 1000 * np.exp(-((t.hour + t.minute/60 - 19)**2) / 4)               # evening peak around 19:00
    night_increase = 200 * np.exp(-((t.hour + t.minute/60 - 3)**2) / 4)               # small night increase around 03:00
    morning_peak = 300 * np.exp(-((t.hour + t.minute/60 - 8)**2) / 4)               # morning peak around 07:00
    profile = pd.Series(index=t, data=base_level + evening_peak + night_increase + morning_peak)

    # add heat pump depending on month

    if t[0].month in [10, 11, 12, 1, 2, 3, 4]:
        delta_coldest = min(np.abs(t[0].month - 1), np.abs(t[0].month - 12))
        profile += 2400 - 800*delta_coldest


    return profile

def get_default_prices(t):
    """Return a simple default price profile (€/kWh) according to the time index t.
    
    """

    base_price = 0.30 + 0.05*np.sin(2 * np.pi * (t.hour + t.minute/60) / 24 - np.pi/2)  # daily cycle
    profile = pd.Series(index=t, data=base_price)

    for offset in [-24, 0, 24]:

        evening_peak = 0.20 * np.exp(-((t.hour + t.minute/60 - 19 + offset)**2) / 4)               # evening peak around 19:00
        night_dip = -0.10 * np.exp(-((t.hour + t.minute/60 - 3 + offset)**2) / 4)                  # small night dip around 03:00
        profile += evening_peak + night_dip

    profile = profile.clip(lower=0.05)  # minimum price

    return profile

def strategy_plot(ctx: StrategyContext, n: pypsa.Network, out_dir: Optional[str] = None, filename_prefix: str = "strategy") -> Path:
    """Save an interactive Plotly HTML showing battery dispatch (W) and SOC (%).

    - Left axis: battery dispatch (W), positive = discharge, negative = charge.
    - Right axis: SOC (%).
    - The first 15-min window is lightly highlighted.

    Returns the path to the saved HTML file.
    """
    # Choose output directory
    if out_dir is None:
        out_dir = str(Path(__file__).resolve().parent / "plots")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Time index: PyPSA snapshots are tz-naive; interpret as UTC and convert for plotting
    idx_utc = pd.DatetimeIndex(n.snapshots)
    idx_local = idx_utc.tz_localize("UTC").tz_convert(ZoneInfo("Europe/Berlin"))

    # Battery dispatch already in W (model uses Watts)
    p_series_w = n.storage_units_t.p.loc[:, "battery"]

    # SOC in Wh; convert to percent based on capacity (p_nom [W] * max_hours [h])
    su = n.storage_units.loc["battery"]
    capacity_wh = float(su.p_nom) * float(su.max_hours) / 0.25
    soc_series_wh = n.storage_units_t.state_of_charge.loc[:, "battery"]
    soc_pct = (soc_series_wh / max(capacity_wh, 1e-9)) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx_local,
            y=p_series_w.values,
            name="Battery dispatch [W]",
            mode="lines",
            line=dict(color="#1f77b4"),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx_local,
            y=soc_pct.values,
            name="SOC [%]",
            mode="lines",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=idx_local,
            y=n.generators_t['p'].loc[:, 'grid'].values,
            name="Energy from grid [W]",
            mode="lines",
            line=dict(color="#0f630c"),
            yaxis="y1",
        )
    )

    now_local = ctx.now.astimezone(ZoneInfo("Europe/Berlin"))
    # Highlight next 15 min
    start = idx_local[0]
    end = start + pd.Timedelta(minutes=15)
    fig.add_vrect(x0=start, x1=end, fillcolor="#1f77b4", opacity=0.08, line_width=0, layer="below")

    fig.update_layout(
        title=(
            f"Battery plan next 24h — {now_local:%Y-%m-%d %H:%M %Z}<br>"
            f"SOC now {ctx.soc_total:.0f}% | bounds {ctx.soc_limit_lower:.0f}–{ctx.soc_limit_upper:.0f}%"
        ),
        xaxis=dict(title="Time", showgrid=True),
        yaxis=dict(title="Power [W]", zeroline=True, zerolinewidth=1, zerolinecolor="#999"),
        yaxis2=dict(title="SOC [%]", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=50, t=80, b=40),
        template="plotly_dark",
    )

    html_path = out_path / f"{filename_prefix}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    return html_path



def decide_strategy(ctx: StrategyContext) -> Tuple[Strategy, Optional[int]]:
    """Return a (strategy, optional_total_power_w) tuple.

    This is a placeholder for your decision logic. For now, we default to IDLE with 0W.
    You can implement your price-aware, SOC-aware strategy here and optionally
    return a concrete target power in watts. If you return None for power, the
    caller can translate Strategy to a setpoint using strategy_to_setpoint().
    """


    # check context and add missing data from default estimates
    t = pd.date_range(ctx.now.floor("15min"), ctx.now + pd.Timedelta(hours=24), freq='15min', tz=ZoneInfo("Europe/Berlin"))
    if ctx.prices_15 is None or len(ctx.prices_15) == 0:
        ctx.prices_15 = get_default_prices(t)
    if ctx.pv_forecast_w is None or len(ctx.pv_forecast_w) == 0:
        ctx.pv_forecast_w = get_default_expected_pv(t)
    if ctx.load_forecast_w is None or len(ctx.load_forecast_w) == 0:
        ctx.load_forecast_w = get_default_expected_consumption(t)
    
    # convert to UTC tz-naive for PyPSA
    t = t.tz_convert(ZoneInfo("UTC")).tz_localize(None)
    ctx.prices_15 = ctx.prices_15.tz_convert(ZoneInfo("UTC")).tz_localize(None)
    ctx.pv_forecast_w.index = ctx.pv_forecast_w.index.tz_convert(ZoneInfo("UTC")).tz_localize(None)
    ctx.load_forecast_w.index = ctx.load_forecast_w.index.tz_convert(ZoneInfo("UTC")).tz_localize(None)

    # Optionally log a compact price summary (disabled noisy head listing)
    try:
        s = ctx.prices_15
        if s is not None and len(s) > 0:
            summary = (
                f"prices: len={len(s)} window=[{s.index.min()} .. {s.index.max()}] "
                f"min={float(s.min()):.3f} max={float(s.max()):.3f}"
            )
            strategy_logger.info('%s', summary)
    except Exception as _e:
        strategy_logger.debug('Price logging failed: %s', _e)
    ctx.prices_15 = ctx.prices_15.fillna(0.30)

    if ctx.soc_limit_lower >= ctx.soc_total:
        ctx.soc_limit_lower = max(0.0, ctx.soc_total - 2.0)
    if ctx.soc_limit_upper <= ctx.soc_total:
        ctx.soc_limit_upper = min(100.0, ctx.soc_total + 2.0)
    strategy_logger.info("Received SOC limits: %.1f .. %.1f %% (current %.1f %%)", ctx.soc_limit_lower, ctx.soc_limit_upper, ctx.soc_total)

    # set up a small pypsa optimization model to decide on strategy
    
    n = pypsa.Network()
    n.set_snapshots(t)
    n.add("Carrier", "AC")
    n.add("Bus", "bus", carrier="AC")
    n.add("Load", "load", bus="bus", carrier="AC")

    # Load in Watts
    n.loads_t["p_set"].loc[:, "load"] = ctx.load_forecast_w

    # PV in Watts (availability via p_max_pu)
    pv_p_nom_w = 8000.0  # 8 kW
    n.add("Generator", "pv", bus="bus", carrier="AC", p_nom=pv_p_nom_w, marginal_cost=0)
    n.generators_t["p_max_pu"].loc[:, "pv"] = (ctx.pv_forecast_w / pv_p_nom_w).clip(lower=0.0)

    # Battery in Watts/Wh
    batt_p_nom_w = 6000.0  # 6 kW
    batt_max_hours = 2.5 # 15 min intervals
    batt_capacity_wh = batt_p_nom_w * batt_max_hours
    batt_capacity_wh_pypsa = batt_capacity_wh / 0.25  # PyPSA expects max_hours in hours relative to snapshot length (15min = 0.25h)
    n.add(
        "StorageUnit",
        "battery",
        bus="bus",
        p_nom=batt_p_nom_w,
        max_hours=batt_max_hours / 0.25,  # PyPSA expects max_hours in hours relative to snapshot length (15min = 0.25h)
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        state_of_charge_initial=(ctx.soc_total / 100.0) * batt_capacity_wh_pypsa,
        marginal_cost=0,
    )

    # Grid generator; price must be €/Wh (prices were €/kWh)
    n.add("Generator", "grid", bus="bus", carrier="AC", p_nom=50000.0)
    n.generators_t["marginal_cost"].loc[:, "grid"] = ctx.prices_15 / 1000.0

    # Persist artifacts in the repository folder (next to this file)
    base_dir = Path(__file__).resolve().parent
    n.export_to_netcdf(str(base_dir / "plots/strategy_model_bfmodel.nc"))

    n.optimize.create_model()

    if False:
        # Limits are unlikely to change battery behavior
        # need to enter the soc limits as custom constraints
        n.model.add_constraints(
            n.model.variables['StorageUnit-state_of_charge'],
            ">=",
            (ctx.soc_limit_lower / 100.0) * batt_capacity_wh_pypsa,
            'battery_soc_min'
        )
        strategy_logger.info("Battery SOC limits: %.1f .. %.1f %%", ctx.soc_limit_lower, ctx.soc_limit_upper)
        n.model.add_constraints(
            n.model.variables['StorageUnit-state_of_charge'],
            "<=",
            (ctx.soc_limit_upper / 100.0) * batt_capacity_wh_pypsa,
            'battery_soc_max'
        )

    n.export_to_netcdf(str(base_dir / "plots/strategy_model.nc"))

    n.optimize.solve_model(solver_name="highs", log_fn=str(base_dir / "plots/highs.log"))

    n.export_to_netcdf(str(base_dir / "plots/strategy_model_solved.nc"))


    try:
        _su_p = n.storage_units_t['p']
        if 'battery' not in _su_p.columns or _su_p.empty:
            strategy_logger.warning("Optimization produced no 'battery' results; returning IDLE")
            return Strategy.IDLE, 0
        strategy_logger.debug("storage unit power head:\n%s", _su_p.head().to_string())
    except Exception:
        strategy_logger.warning("No storage unit results available; returning IDLE")
        return Strategy.IDLE, 0

    # Save an interactive result plot for visibility
    strategy_plot(ctx, n)

    if n.storage_units_t['p'].loc[t[0], 'battery'] > 100:
        # if there's currently pv in the system, balance carefully
        if n.generators_t['p'].loc[t[0], 'pv'] > 0:
            # if pv is present, balance aggressively
            return Strategy.BALANCE, 0
        else:
            # pure discharge, possibly over night
            #return Strategy.DISCHARGE, 0
            return Strategy.BALANCE, 0

    if n.storage_units_t['p'].loc[t[0], 'battery'] < -100:
        # are we charging above PV generation?
        if -n.storage_units_t['p'].loc[t[0], 'battery'] >= n.generators_t['p'].loc[t[0], 'pv'] + 200:
            # yes, we are charging above PV generation
            # Convert W at the first snapshot to Wh over 15 minutes: Wh = W * 0.25h
            return Strategy.CHARGE, int(-n.storage_units_t['p'].loc[t[0], 'battery'] * 0.25)
        # just charge normally
        else:
            # are prices currently very low? allow balancing from battery?
            uncaptured_pv_8h = (n.generators_t['p'].loc[t[:32], 'pv'] + n.storage_units_t['p'].loc[t[:32], 'battery']).sum()

            if uncaptured_pv_8h > 1000:
                return Strategy.BALANCE, 0

            if ctx.prices_15.loc[t[0]] < ctx.prices_15.min() + 0.05:
                return Strategy.CHARGE, 0
            
            return Strategy.BALANCE, 0
            

    return Strategy.IDLE, 0




if __name__ == "__main__":
    pass


    #%%

    ctx = StrategyContext(
        now = pd.Timestamp.now(tz=ZoneInfo("Europe/Berlin")),
        soc_total = 50.0,
        soc_limit_lower = 20.0,
        soc_limit_upper = 80.0,
        prices_15 = None,
        pv_forecast_w = None,
        load_forecast_w = None,
    )

    decide_strategy(ctx)

    #%%

    plt.plot(ctx.pv_forecast_w.index, ctx.pv_forecast_w.values)
    plt.title("Default Expected PV Generation Profile")
    plt.xlabel("Time")
    plt.ylabel("PV Generation (W)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    plt.plot(ctx.load_forecast_w.index, ctx.load_forecast_w.values)
    plt.title("Default Expected Household Consumption Profile")
    plt.xlabel("Time")
    plt.ylabel("Household Consumption (W)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    plt.plot(ctx.prices_15.index, ctx.prices_15.values)
    plt.title("Default Expected Electricity Prices")
    plt.xlabel("Time")
    plt.ylabel("Electricity Price (€/Wh)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
    #%%