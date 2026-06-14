import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import date, datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm

OPTIONABLE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD","NFLX", "JPM", "BAC", "GS", "XOM", "CVX", "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO"]

CORE_GREEK_NAMES = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Volga", "Vanna"]
ADVANCED_GREEK_NAMES = ["Charm", "Speed", "Color", "Zomma", "Veta", "Vera", "Ultima"]
GREEK_NAMES = CORE_GREEK_NAMES + ADVANCED_GREEK_NAMES

PRICE_NAME = "Price"
FIRST_ORDER_GREEKS = ["Delta", "Vega", "Theta", "Rho"]
SECOND_ORDER_GREEKS = ["Gamma", "Volga", "Vanna", "Charm", "Veta", "Vera"]
THIRD_ORDER_GREEKS = ["Speed", "Color", "Zomma", "Ultima"]

VANILLA_TYPES = ["Calls", "Puts"]

DIGITAL_TYPES = ["Digital Calls Cash or Nothing","Digital Puts Cash or Nothing","Digital Calls Asset or Nothing","Digital Puts Asset or Nothing"]

BARRIER_TYPES = ["Calls Up and In", "Calls Up and Out", "Calls Down and In", "Calls Down and Out","Puts Up and In", "Puts Up and Out", "Puts Down and In", "Puts Down and Out"]

EXTRA_STRIKE_TYPES = ["Gap Calls", "Gap Puts", "Capped Calls", "Capped Puts"]

EXOTIC_OPTION_TYPES = VANILLA_TYPES + DIGITAL_TYPES + BARRIER_TYPES + EXTRA_STRIKE_TYPES

DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_VOLATILITY = 0.20
DEFAULT_DIVIDEND_YIELD = 0.00

FIXED_MONTE_CARLO_SEED = 42
FIXED_STEPS_PER_YEAR = 252
MONTE_CARLO_BATCH_SIZE = 50_000

S_GRID_MIN_MULTIPLIER = 0.0
S_GRID_MAX_MULTIPLIER = 2.0
DEFAULT_S_GRID_POINTS = 50
DEFAULT_MONTE_CARLO_SIMULATIONS = 50_000

SPOT_BUMP_RELATIVE = 0.01
VOL_BUMP_ABSOLUTE = 0.02
VOL_BUMP_RELATIVE = 0.10
RATE_BUMP_ABSOLUTE = 0.001
THETA_DAY = 1 / 365.0

MONTE_CARLO_SPOT_BUMP_RELATIVE = 0.04
MONTE_CARLO_STRIKE_BUMP_RELATIVE = 0.01
MONTE_CARLO_VOL_BUMP_ABSOLUTE = 0.04
MONTE_CARLO_VOL_BUMP_RELATIVE = 0.20
MONTE_CARLO_RATE_BUMP_ABSOLUTE = 0.0025
MONTE_CARLO_TIME_BUMP_DAYS = 7

def safe_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if np.isfinite(value) else default

def format_price(value, currency):
    if value is None or not np.isfinite(value):
        return "N/A"

    suffix = f" {currency}" if currency else ""
    return f"{value:,.4f}{suffix}"

def format_percent(value):
    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value * 100:.2f}%"

def format_strike(value):
    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value:.2f}".rstrip("0").rstrip(".")

def as_datetime(maturity):
    if isinstance(maturity, datetime):
        return maturity

    if isinstance(maturity, date):
        return datetime.combine(maturity, datetime.min.time())

    return datetime.strptime(str(maturity), "%Y-%m-%d")

def years_to_maturity(maturity):
    maturity_dt = as_datetime(maturity)
    return (maturity_dt - datetime.today()).days / 365.0

def valid_base_inputs(spot, strike, tau, volatility, risk_free_rate, dividend_yield):
    inputs = [spot, strike, tau, volatility, risk_free_rate, dividend_yield]

    if not all(np.isfinite(x) for x in inputs):
        return False

    return spot > 0 and strike > 0 and tau > 0 and volatility > 0


def valid_monte_carlo_inputs(spot, strike, tau, volatility, risk_free_rate, dividend_yield):
    inputs = [spot, strike, tau, volatility, risk_free_rate, dividend_yield]

    if not all(np.isfinite(x) for x in inputs):
        return False

    return spot >= 0 and strike > 0 and tau > 0 and volatility > 0

def tuple_to_dict(result):
    return {name: value for name, value in zip(GREEK_NAMES, result)}


def build_selected_greek_names(include_first_order=True, include_second_order=True, include_third_order=True):
    selected = {PRICE_NAME}

    if include_first_order:
        selected.update(FIRST_ORDER_GREEKS)

    if include_second_order:
        selected.update(SECOND_ORDER_GREEKS)

    if include_third_order:
        selected.update(THIRD_ORDER_GREEKS)

    return [greek_name for greek_name in GREEK_NAMES if greek_name in selected]


def infer_greek_order_defaults(greek_names):
    selected = set(greek_names or GREEK_NAMES)

    return (bool(selected & set(FIRST_ORDER_GREEKS)),bool(selected & set(SECOND_ORDER_GREEKS)),bool(selected & set(THIRD_ORDER_GREEKS)),)


def render_greek_order_checkboxes(key_prefix,default_first_order=True,default_second_order=True,default_third_order=True,):
    include_first_order = st.checkbox("Greeks de premier ordre",value=default_first_order,key=f"{key_prefix}_first_order",help=", ".join(FIRST_ORDER_GREEKS),)
    include_second_order = st.checkbox("Greeks de deuxième ordre",value=default_second_order,key=f"{key_prefix}_second_order",help=", ".join(SECOND_ORDER_GREEKS),)
    include_third_order = st.checkbox("Greeks de troisième ordre",value=default_third_order,key=f"{key_prefix}_third_order",help=", ".join(THIRD_ORDER_GREEKS),)

    selected_greeks = build_selected_greek_names(include_first_order=include_first_order,include_second_order=include_second_order,include_third_order=include_third_order,)

    return selected_greeks

def nan_result_tuple():
    return tuple(np.nan for _ in GREEK_NAMES)

def copy_parameters(parameters, **updates):
    updated = dict(parameters)
    updated.update(updates)
    return updated

def load_spot_price(symbol):
    ticker = yf.Ticker(symbol)

    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        try:
            history = ticker.history(period="5d")
            spot = history["Close"].dropna().iloc[-1] if not history.empty else np.nan
        except Exception:
            spot = np.nan

    spot = safe_float(spot)
    return spot if np.isfinite(spot) and spot > 0 else np.nan

def load_option_data(symbol, quote_option_type):
    ticker = yf.Ticker(symbol)
    rows = []
    today = datetime.today()

    try:
        maturities = ticker.options
    except Exception:
        maturities = []

    for maturity_string in maturities:
        try:
            maturity_dt = datetime.strptime(maturity_string, "%Y-%m-%d")
        except ValueError:
            continue

        if maturity_dt <= today + timedelta(days=1):
            continue

        try:
            chain = ticker.option_chain(maturity_string)
        except Exception:
            continue

        if quote_option_type == "Calls":
            chain_df = pd.DataFrame(chain.calls)
            chain_df["option_type"] = "Calls"
        else:
            chain_df = pd.DataFrame(chain.puts)
            chain_df["option_type"] = "Puts"

        if chain_df.empty:
            continue

        required_columns = ["strike", "bid", "ask", "currency", "option_type"]
        optional_columns = ["lastPrice", "volume", "openInterest", "impliedVolatility"]
        available_optional_columns = [col for col in optional_columns if col in chain_df.columns]

        clean_df = chain_df[required_columns + available_optional_columns].copy()
        clean_df["maturity"] = maturity_string
        rows.append(clean_df)

    if not rows:
        return pd.DataFrame(columns=["strike", "bid", "ask", "currency", "option_type", "maturity", "mid"])

    data = pd.concat(rows, ignore_index=True)
    data["bid"] = pd.to_numeric(data["bid"], errors="coerce")
    data["ask"] = pd.to_numeric(data["ask"], errors="coerce")
    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["mid"] = (data["bid"] + data["ask"]) / 2
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["strike", "bid", "ask", "mid"])
    data = data[(data["strike"] > 0) & (data["bid"] >= 0) & (data["ask"] > 0) & (data["mid"] > 0)]

    return data.reset_index(drop=True)

def load_risk_free_rate(currency):
    try:
        if currency == "USD":
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
            df = pd.read_csv(url)
            df = df[df["SOFR"] != "."]
            return float(df["SOFR"].iloc[-1]) / 100

        if currency == "EUR":
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBESTRVOLWGTTRMDMNRT"
            df = pd.read_csv(url)
            df = df[df["ECBESTRVOLWGTTRMDMNRT"] != "."]
            return float(df["ECBESTRVOLWGTTRMDMNRT"].iloc[-1]) / 100

        if currency == "GBP":
            url = (
                "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?"
                "CSVF=TT&DAT=RNG&FD=1&FM=Jan&FY=2024&TD=31&TM=Dec&TY=2030&"
                "FNY=&Filter=N&FromSeries=1&ToSeries=50&SeriesCodes=IUDSOIA&UsingCodes=Y&VPD=Y"
            )
            df = pd.read_csv(url)
            df = df[df["IUDSOIA"] != "."]
            return float(df["IUDSOIA"].iloc[-1]) / 100
    except Exception:
        pass

    return DEFAULT_RISK_FREE_RATE

def load_dividend_yield(symbol, spot):
    ticker = yf.Ticker(symbol)
    dividend_yield = np.nan
    annual_dividend = np.nan
    method = "No dividend data found"

    try:
        dividends = ticker.dividends
        if dividends is not None and len(dividends) > 0:
            dividends = dividends.dropna()
            dividends.index = pd.to_datetime(dividends.index)
            one_year_ago = pd.Timestamp.today(tz=dividends.index.tz) - pd.DateOffset(years=1)
            recent_dividends = dividends[dividends.index >= one_year_ago]

            if len(recent_dividends) > 0 and np.isfinite(spot) and spot > 0:
                annual_dividend = float(recent_dividends.sum())
                dividend_yield = annual_dividend / spot
                method = "Trailing 12-month dividends divided by spot"
    except Exception:
        pass

    if not np.isfinite(dividend_yield):
        try:
            info = ticker.info
            raw_dividend_yield = info.get("dividendYield", np.nan)
            raw_dividend_yield = safe_float(raw_dividend_yield)

            if np.isfinite(raw_dividend_yield):
                dividend_yield = raw_dividend_yield / 100 if raw_dividend_yield > 1 else raw_dividend_yield
                method = "Yahoo Finance dividendYield field"
        except Exception:
            pass

    if not np.isfinite(dividend_yield):
        dividend_yield = DEFAULT_DIVIDEND_YIELD

    return float(dividend_yield), annual_dividend, method

def vanilla_price(S, K, tau, sigma, r, q, option_type):
    if not valid_base_inputs(S, K, tau, sigma, r, q):
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type == "Calls":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    if option_type == "Puts":
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

    return np.nan

def price_formula_from_parameters(parameters):
    return vanilla_price(S=parameters["spot"],K=parameters["strike"],tau=years_to_maturity(parameters["maturity"]),sigma=parameters["IV"],r=parameters["rf"],q=parameters.get("dividend_yield", 0.0),option_type=parameters["option_type"],)

def calculate_formula_greeks(parameters):
    price = price_formula_from_parameters(parameters)

    if not np.isfinite(price):
        return nan_result_tuple()

    spot = parameters["spot"]
    volatility = parameters["IV"]
    maturity = parameters["maturity"]

    hS = max(spot * SPOT_BUMP_RELATIVE, 1e-4)
    hV = max(min(max(volatility * VOL_BUMP_RELATIVE, VOL_BUMP_ABSOLUTE), volatility * 0.50), 1e-4)
    hR = RATE_BUMP_ABSOLUTE

    p_S_up = price_formula_from_parameters(copy_parameters(parameters, spot=spot + hS))
    p_S_down = price_formula_from_parameters(copy_parameters(parameters, spot=max(spot - hS, 1e-8)))
    delta = (p_S_up - p_S_down) / (2 * hS) if np.isfinite(p_S_up) and np.isfinite(p_S_down) else np.nan
    gamma = (p_S_up - 2 * price + p_S_down) / (hS**2) if np.isfinite(p_S_up) and np.isfinite(p_S_down) else np.nan

    p_v_up = price_formula_from_parameters(copy_parameters(parameters, IV=volatility + hV))
    p_v_down = price_formula_from_parameters(copy_parameters(parameters, IV=max(volatility - hV, 1e-8)))
    vega = (p_v_up - p_v_down) / (2 * hV) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan
    volga = (p_v_up - 2 * price + p_v_down) / (hV**2) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan

    p_r_up = price_formula_from_parameters(copy_parameters(parameters, rf=parameters["rf"] + hR))
    p_r_down = price_formula_from_parameters(copy_parameters(parameters, rf=parameters["rf"] - hR))
    rho = (p_r_up - p_r_down) / (2 * hR) if np.isfinite(p_r_up) and np.isfinite(p_r_down) else np.nan

    theta = np.nan
    if years_to_maturity(maturity) > THETA_DAY:
        p_tomorrow = price_formula_from_parameters(copy_parameters(parameters, maturity=as_datetime(maturity) - timedelta(days=1)))
        theta = p_tomorrow - price if np.isfinite(p_tomorrow) else np.nan

    p_up_up = price_formula_from_parameters(copy_parameters(parameters, spot=spot + hS, IV=volatility + hV))
    p_up_down = price_formula_from_parameters(copy_parameters(parameters, spot=spot + hS, IV=max(volatility - hV, 1e-8)))
    p_down_up = price_formula_from_parameters(copy_parameters(parameters, spot=max(spot - hS, 1e-8), IV=volatility + hV))
    p_down_down = price_formula_from_parameters(copy_parameters(parameters, spot=max(spot - hS, 1e-8), IV=max(volatility - hV, 1e-8)))

    if all(np.isfinite(x) for x in [p_up_up, p_up_down, p_down_up, p_down_down]):
        vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hV)
    else:
        vanna = np.nan

    return price, delta, gamma, vega, theta, rho, volga, vanna

def monte_carlo_price_single_batch(parameters, num_simulations, seed):
    strike = parameters["strike"]
    maturity = parameters["maturity"]
    volatility = parameters["IV"]
    spot = parameters["spot"]
    risk_free_rate = parameters["rf"]
    dividend_yield = parameters["dividend_yield"]
    option_type = parameters["option_type"]
    barrier = parameters.get("barrier")
    cash_payout = parameters.get("cash_payout", 1.0)
    extra_strike = parameters.get("extra_strike")

    tau = years_to_maturity(maturity)

    if not valid_monte_carlo_inputs(spot, strike, tau, volatility, risk_free_rate, dividend_yield):
        return np.nan

    n_steps = max(1, int(np.ceil(tau * FIXED_STEPS_PER_YEAR)))
    dt = tau / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((int(num_simulations), n_steps))
    log_returns = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    paths = spot * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack((np.full(int(num_simulations), spot), paths))
    terminal_spot = paths[:, -1]

    call_payoff = np.maximum(terminal_spot - strike, 0.0)
    put_payoff = np.maximum(strike - terminal_spot, 0.0)

    if option_type == "Calls":
        payoffs = call_payoff
    elif option_type == "Puts":
        payoffs = put_payoff
    elif option_type == "Digital Calls Cash or Nothing":
        payoffs = cash_payout * (terminal_spot > strike)
    elif option_type == "Digital Puts Cash or Nothing":
        payoffs = cash_payout * (terminal_spot < strike)
    elif option_type == "Digital Calls Asset or Nothing":
        payoffs = terminal_spot * (terminal_spot > strike)
    elif option_type == "Digital Puts Asset or Nothing":
        payoffs = terminal_spot * (terminal_spot < strike)
    elif option_type in BARRIER_TYPES:
        if barrier is None or barrier <= 0:
            return np.nan

        is_call = "Calls" in option_type
        is_up = "Up" in option_type
        is_in = "In" in option_type
        vanilla_payoff = call_payoff if is_call else put_payoff

        if is_up:
            barrier_touched = np.max(paths, axis=1) >= barrier
        else:
            barrier_touched = np.min(paths, axis=1) <= barrier

        payoffs = np.where(barrier_touched, vanilla_payoff, 0.0) if is_in else np.where(~barrier_touched, vanilla_payoff, 0.0)
    elif option_type == "Gap Calls":
        if extra_strike is None or extra_strike <= 0:
            return np.nan
        payoffs = np.where(terminal_spot > strike, terminal_spot - extra_strike, 0.0)
    elif option_type == "Gap Puts":
        if extra_strike is None or extra_strike <= 0:
            return np.nan
        payoffs = np.where(terminal_spot < strike, extra_strike - terminal_spot, 0.0)
    elif option_type == "Capped Calls":
        cap = extra_strike
        if cap is None or cap <= strike:
            return np.nan
        payoffs = np.minimum(np.maximum(terminal_spot - strike, 0.0), cap - strike)
    elif option_type == "Capped Puts":
        floor = extra_strike
        if floor is None or floor >= strike or floor <= 0:
            return np.nan
        payoffs = np.minimum(np.maximum(strike - terminal_spot, 0.0), strike - floor)
    else:
        return np.nan

    return np.exp(-risk_free_rate * tau) * np.mean(np.asarray(payoffs, dtype=float))

def monte_carlo_price_batched(parameters: dict, num_simulations: int, seed=FIXED_MONTE_CARLO_SEED) -> float:
    total_simulations = int(num_simulations)

    if total_simulations <= 0:
        return np.nan

    batch_size = min(MONTE_CARLO_BATCH_SIZE, total_simulations)
    weighted_price_sum = 0.0
    completed_simulations = 0
    batch_index = 0

    while completed_simulations < total_simulations:
        current_batch_size = min(batch_size, total_simulations - completed_simulations)
        batch_price = monte_carlo_price_single_batch(parameters, current_batch_size, seed + batch_index)

        if not np.isfinite(batch_price):
            return np.nan

        weighted_price_sum += batch_price * current_batch_size
        completed_simulations += current_batch_size
        batch_index += 1

    return weighted_price_sum / completed_simulations

def calculate_monte_carlo_greeks(parameters, num_simulations, greek_names=None):
    requested_greeks = set(greek_names or GREEK_NAMES)
    base_spot = float(parameters["spot"])
    strike = float(parameters["strike"])
    base_volatility = float(parameters["IV"])
    base_rate = float(parameters["rf"])
    base_maturity = parameters["maturity"]

    price_cache = {}

    def price_at(spot_value=base_spot, volatility_value=base_volatility, risk_free_rate_value=base_rate, maturity_value=base_maturity):
        spot_value = max(float(spot_value), 0.0)
        volatility_value = max(float(volatility_value), 1e-8)
        risk_free_rate_value = float(risk_free_rate_value)
        maturity_value = as_datetime(maturity_value)

        cache_key = (round(spot_value, 12),round(volatility_value, 12),round(risk_free_rate_value, 12),maturity_value.strftime("%Y-%m-%d %H:%M:%S"),)

        if cache_key not in price_cache:
            price_cache[cache_key] = monte_carlo_price_batched(copy_parameters(parameters,spot=spot_value,IV=volatility_value,rf=risk_free_rate_value,maturity=maturity_value,),num_simulations)

        return price_cache[cache_key]

    values = {greek_name: np.nan for greek_name in GREEK_NAMES}
    price = price_at()

    if not np.isfinite(price):
        return nan_result_tuple()

    values["Price"] = price

    hS = max(abs(base_spot) * MONTE_CARLO_SPOT_BUMP_RELATIVE, strike * MONTE_CARLO_STRIKE_BUMP_RELATIVE, 1e-4)
    hV = max(min(max(base_volatility * MONTE_CARLO_VOL_BUMP_RELATIVE, MONTE_CARLO_VOL_BUMP_ABSOLUTE),max(base_volatility * 0.50, MONTE_CARLO_VOL_BUMP_ABSOLUTE)),1e-8)
    hR = MONTE_CARLO_RATE_BUMP_ABSOLUTE

    def spot_derivatives(volatility_value=base_volatility,risk_free_rate_value=base_rate,maturity_value=base_maturity,compute_delta=False,compute_gamma=False,compute_speed=False,):
        delta = np.nan
        gamma = np.nan
        speed = np.nan

        needs_first_grid = compute_delta or compute_gamma or compute_speed
        if not needs_first_grid:
            return delta, gamma, speed

        f0 = price_at(base_spot, volatility_value, risk_free_rate_value, maturity_value)
        f_up_1 = price_at(base_spot + hS, volatility_value, risk_free_rate_value, maturity_value)

        f_down_1 = np.nan
        f_up_2 = np.nan

        if base_spot >= hS:
            f_down_1 = price_at(base_spot - hS, volatility_value, risk_free_rate_value, maturity_value)

            if compute_delta and all(np.isfinite(x) for x in [f_up_1, f_down_1]):
                delta = (f_up_1 - f_down_1) / (2 * hS)

            if compute_gamma and all(np.isfinite(x) for x in [f_up_1, f0, f_down_1]):
                gamma = (f_up_1 - 2 * f0 + f_down_1) / (hS**2)
        else:
            f_up_2 = price_at(base_spot + 2 * hS, volatility_value, risk_free_rate_value, maturity_value)

            if compute_delta and all(np.isfinite(x) for x in [f0, f_up_1, f_up_2]):
                delta = (-3 * f0 + 4 * f_up_1 - f_up_2) / (2 * hS)

            if compute_gamma and all(np.isfinite(x) for x in [f0, f_up_1, f_up_2]):
                gamma = (f0 - 2 * f_up_1 + f_up_2) / (hS**2)

        if compute_speed:
            if base_spot >= 2 * hS:
                if not np.isfinite(f_down_1):
                    f_down_1 = price_at(base_spot - hS, volatility_value, risk_free_rate_value, maturity_value)
                f_up_2 = price_at(base_spot + 2 * hS, volatility_value, risk_free_rate_value, maturity_value)
                f_down_2 = price_at(base_spot - 2 * hS, volatility_value, risk_free_rate_value, maturity_value)

                if all(np.isfinite(x) for x in [f_up_2, f_up_1, f_down_1, f_down_2]):
                    speed = (f_up_2 - 2 * f_up_1 + 2 * f_down_1 - f_down_2) / (2 * hS**3)
            else:
                if not np.isfinite(f_up_2):
                    f_up_2 = price_at(base_spot + 2 * hS, volatility_value, risk_free_rate_value, maturity_value)
                f_up_3 = price_at(base_spot + 3 * hS, volatility_value, risk_free_rate_value, maturity_value)

                if all(np.isfinite(x) for x in [f0, f_up_1, f_up_2, f_up_3]):
                    speed = (-f0 + 3 * f_up_1 - 3 * f_up_2 + f_up_3) / (hS**3)

        return delta, gamma, speed

    needs_base_delta = bool(requested_greeks & {"Delta", "Charm"})
    needs_base_gamma = bool(requested_greeks & {"Gamma", "Color"})
    needs_base_speed = "Speed" in requested_greeks
    needs_vol_prices = bool(requested_greeks & {"Vega", "Volga", "Ultima", "Veta"})
    needs_vol_spot_delta = "Vanna" in requested_greeks
    needs_vol_spot_gamma = "Zomma" in requested_greeks
    needs_time_shift = bool(requested_greeks & {"Theta", "Charm", "Color", "Veta"})

    delta = gamma = speed = np.nan
    if needs_base_delta or needs_base_gamma or needs_base_speed:
        delta, gamma, speed = spot_derivatives(compute_delta=needs_base_delta,compute_gamma=needs_base_gamma,compute_speed=needs_base_speed)

        if "Delta" in requested_greeks:
            values["Delta"] = delta
        if "Gamma" in requested_greeks:
            values["Gamma"] = gamma
        if "Speed" in requested_greeks:
            values["Speed"] = speed

    p_v_up = p_v_down = np.nan
    if needs_vol_prices:
        p_v_up = price_at(volatility_value=base_volatility + hV)
        p_v_down = price_at(volatility_value=base_volatility - hV)

    vega = np.nan
    if "Vega" in requested_greeks or "Veta" in requested_greeks:
        vega = (p_v_up - p_v_down) / (2 * hV) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan
        if "Vega" in requested_greeks:
            values["Vega"] = vega

    if "Volga" in requested_greeks:
        values["Volga"] = (p_v_up - 2 * price + p_v_down) / (hV**2) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan

    if "Ultima" in requested_greeks:
        p_v_2_up = price_at(volatility_value=base_volatility + 2 * hV)
        p_v_2_down = price_at(volatility_value=base_volatility - 2 * hV)
        if all(np.isfinite(x) for x in [p_v_2_up, p_v_up, p_v_down, p_v_2_down]):
            values["Ultima"] = (p_v_2_up - 2 * p_v_up + 2 * p_v_down - p_v_2_down) / (2 * hV**3)

    if needs_vol_spot_delta or needs_vol_spot_gamma:
        delta_v_up, gamma_v_up, _ = spot_derivatives(volatility_value=base_volatility + hV,compute_delta=needs_vol_spot_delta,compute_gamma=needs_vol_spot_gamma,)
        delta_v_down, gamma_v_down, _ = spot_derivatives(volatility_value=base_volatility - hV,compute_delta=needs_vol_spot_delta,compute_gamma=needs_vol_spot_gamma,)

        if "Vanna" in requested_greeks:
            values["Vanna"] = (delta_v_up - delta_v_down) / (2 * hV) if np.isfinite(delta_v_up) and np.isfinite(delta_v_down) else np.nan

        if "Zomma" in requested_greeks:
            values["Zomma"] = (gamma_v_up - gamma_v_down) / (2 * hV) if np.isfinite(gamma_v_up) and np.isfinite(gamma_v_down) else np.nan

    if "Rho" in requested_greeks:
        p_r_up = price_at(risk_free_rate_value=base_rate + hR)
        p_r_down = price_at(risk_free_rate_value=base_rate - hR)
        values["Rho"] = (p_r_up - p_r_down) / (2 * hR) if np.isfinite(p_r_up) and np.isfinite(p_r_down) else np.nan

    if "Vera" in requested_greeks:
        p_r_up_v_up = price_at(volatility_value=base_volatility + hV, risk_free_rate_value=base_rate + hR)
        p_r_up_v_down = price_at(volatility_value=base_volatility - hV, risk_free_rate_value=base_rate + hR)
        p_r_down_v_up = price_at(volatility_value=base_volatility + hV, risk_free_rate_value=base_rate - hR)
        p_r_down_v_down = price_at(volatility_value=base_volatility - hV, risk_free_rate_value=base_rate - hR)

        if all(np.isfinite(x) for x in [p_r_up_v_up, p_r_up_v_down, p_r_down_v_up, p_r_down_v_down]):
            values["Vera"] = (p_r_up_v_up - p_r_up_v_down - p_r_down_v_up + p_r_down_v_down) / (4 * hR * hV)

    if needs_time_shift and years_to_maturity(base_maturity) > THETA_DAY:
        time_bump_days = min(MONTE_CARLO_TIME_BUMP_DAYS, max(1, days_to_maturity(base_maturity) - 1))
        shifted_maturity = as_datetime(base_maturity) - timedelta(days=time_bump_days)

        if "Theta" in requested_greeks:
            p_shifted = price_at(maturity_value=shifted_maturity)
            values["Theta"] = (p_shifted - price) / time_bump_days if np.isfinite(p_shifted) else np.nan

        if "Charm" in requested_greeks or "Color" in requested_greeks:
            if "Charm" in requested_greeks and not np.isfinite(delta):
                delta, _, _ = spot_derivatives(compute_delta=True)

            if "Color" in requested_greeks and not np.isfinite(gamma):
                _, gamma, _ = spot_derivatives(compute_gamma=True)

            delta_shifted, gamma_shifted, _ = spot_derivatives(maturity_value=shifted_maturity,compute_delta="Charm" in requested_greeks,compute_gamma="Color" in requested_greeks,)

            if "Charm" in requested_greeks:
                values["Charm"] = (delta_shifted - delta) / time_bump_days if np.isfinite(delta_shifted) and np.isfinite(delta) else np.nan

            if "Color" in requested_greeks:
                values["Color"] = (gamma_shifted - gamma) / time_bump_days if np.isfinite(gamma_shifted) and np.isfinite(gamma) else np.nan

        if "Veta" in requested_greeks:
            if not np.isfinite(vega):
                vega = (p_v_up - p_v_down) / (2 * hV) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan
            p_shifted_v_up = price_at(volatility_value=base_volatility + hV, maturity_value=shifted_maturity)
            p_shifted_v_down = price_at(volatility_value=base_volatility - hV, maturity_value=shifted_maturity)
            vega_shifted = (p_shifted_v_up - p_shifted_v_down) / (2 * hV) if np.isfinite(p_shifted_v_up) and np.isfinite(p_shifted_v_down) else np.nan
            values["Veta"] = (vega_shifted - vega) / time_bump_days if np.isfinite(vega_shifted) and np.isfinite(vega) else np.nan

    return tuple(values[greek_name] for greek_name in GREEK_NAMES)

def infer_quote_option_type(option_type):
    return "Puts" if "Puts" in option_type else "Calls"

def implied_volatility_from_mid(row, spot, risk_free_rate, dividend_yield):
    strike = safe_float(row.get("strike"))
    mid = safe_float(row.get("mid"))
    maturity = row.get("maturity")
    option_type = row.get("option_type")
    tau = years_to_maturity(as_datetime(maturity))

    if not valid_base_inputs(spot, strike, tau, DEFAULT_VOLATILITY, risk_free_rate, dividend_yield):
        return np.nan, "Invalid inputs"

    if not np.isfinite(mid) or mid <= 0:
        return np.nan, "Invalid market mid"

    def objective(volatility):
        return vanilla_price(spot, strike, tau, volatility, risk_free_rate, dividend_yield, option_type) - mid

    try:
        low_value = objective(1e-6)
        high_value = objective(5.0)

        if not np.isfinite(low_value) or not np.isfinite(high_value) or low_value * high_value > 0:
            return np.nan, "Unavailable from listed mid"

        return float(brentq(objective, 1e-6, 5.0, maxiter=200)), "Brent root from bid/ask mid"
    except Exception:
        return np.nan, "Unavailable from listed mid"

def build_iv_points_for_maturity(option_data,maturity_string,spot,risk_free_rate,dividend_yield):
    maturity_data = option_data[option_data["maturity"] == maturity_string].copy()
    points = []

    for _, row in maturity_data.iterrows():
        implied_vol, source = implied_volatility_from_mid(row, spot, risk_free_rate, dividend_yield)
        strike = safe_float(row.get("strike"))

        if np.isfinite(strike) and strike > 0 and np.isfinite(implied_vol) and implied_vol > 0:
            points.append({"strike": float(strike), "iv": float(implied_vol), "source": source})

    if not points:
        return pd.DataFrame(columns=["strike", "iv", "source"])

    iv_points = pd.DataFrame(points)
    iv_points = iv_points.sort_values("strike").drop_duplicates(subset=["strike"], keep="first")

    return iv_points.reset_index(drop=True)

def interpolate_iv_at_strike(iv_points, target_strike):
    target_strike = safe_float(target_strike)

    if not np.isfinite(target_strike) or target_strike <= 0 or iv_points is None or iv_points.empty:
        return np.nan, "Unavailable"

    strikes = iv_points["strike"].to_numpy(dtype=float)
    vols = iv_points["iv"].to_numpy(dtype=float)

    if len(strikes) == 1:
        return float(vols[0]), "Only one usable IV point"

    exact_match = np.where(np.isclose(strikes, target_strike, rtol=0.0, atol=1e-10))[0]
    if len(exact_match) > 0:
        return float(vols[exact_match[0]]), "Exact listed/manual point"

    if target_strike < strikes[0]:
        return float(vols[0]), "Nearest point below available range"

    if target_strike > strikes[-1]:
        return float(vols[-1]), "Nearest point above available range"

    return float(np.interp(target_strike, strikes, vols)), "Linear interpolation"

def weighted_average_available(weighted_vols):
    clean_items = [(weight, vol) for weight, vol in weighted_vols if weight > 0 and np.isfinite(vol)]
    total_weight = sum(weight for weight, _ in clean_items)

    if total_weight <= 0:
        return np.nan

    return sum(weight * vol for weight, vol in clean_items) / total_weight

def calculate_smile_adjusted_volatility(option_type,strike,spot,barrier,extra_strike,iv_points,base_iv):
    diagnostics = []

    def get_point(label, level):
        iv, source = interpolate_iv_at_strike(iv_points, level)
        diagnostics.append({"Reference point": label, "Level": level, "Vanilla IV": iv, "Method": source})
        return iv

    strike_iv = get_point("Pricing strike", strike)
    atm_iv = get_point("ATM / spot", spot)

    if not np.isfinite(strike_iv):
        strike_iv = base_iv

    if not np.isfinite(atm_iv):
        atm_iv = base_iv

    effective_iv = strike_iv
    formula_label = "Vanilla IV at pricing strike"

    if option_type in BARRIER_TYPES:
        barrier_iv = get_point("Barrier", barrier)
        effective_iv = weighted_average_available([(0.50, strike_iv), (0.30, barrier_iv), (0.20, atm_iv)])
        formula_label = "50% strike IV + 30% barrier IV + 20% ATM IV"
    elif option_type in ["Gap Calls", "Gap Puts"]:
        payoff_iv = get_point("Payoff strike", extra_strike)
        effective_iv = weighted_average_available([(0.50, strike_iv), (0.50, payoff_iv)])
        formula_label = "50% trigger strike IV + 50% payoff strike IV"
    elif option_type in ["Capped Calls", "Capped Puts"]:
        cap_floor_iv = get_point("Cap / floor", extra_strike)
        effective_iv = weighted_average_available([(0.50, strike_iv), (0.50, cap_floor_iv)])
        formula_label = "50% strike IV + 50% cap/floor IV"
    elif option_type in DIGITAL_TYPES:
        effective_iv = strike_iv
        formula_label = "Vanilla IV at digital strike"

    if not np.isfinite(effective_iv) or effective_iv <= 0:
        effective_iv = base_iv
        formula_label = "Fallback to base IV"

    diagnostics_df = pd.DataFrame(diagnostics)
    return float(effective_iv), formula_label, diagnostics_df

def get_extra_inputs(option_type, spot, strike):
    barrier = None
    cash_payout = 1.0
    extra_strike = None

    if option_type in BARRIER_TYPES:
        default_barrier = spot * 1.1 if "Up" in option_type else spot * 0.9
        barrier = st.number_input("Barrier", min_value=0.0001, value=float(default_barrier), step=max(spot * 0.01, 0.5))

    if option_type in DIGITAL_TYPES:
        cash_payout = st.number_input("Cash payout", min_value=0.0001, value=1.0, step=0.5)

    if option_type in EXTRA_STRIKE_TYPES:
        label = "Payoff strike / cap / floor"
        default_extra = strike * 1.1 if option_type in ["Gap Calls", "Capped Calls"] else strike * 0.9
        extra_strike = st.number_input(label, min_value=0.0001, value=float(default_extra), step=max(strike * 0.01, 0.5))

    return barrier, float(cash_payout), extra_strike

def build_manual_iv_points(option_type,spot,strike,barrier,extra_strike,base_iv):
    volatility_mode = st.radio("Volatility model",["Manual constant volatility", "Manual smile-adjusted proxy"],horizontal=True,help="The smile proxy lets you enter IVs at key levels and interpolates between them.")

    if volatility_mode == "Manual constant volatility":
        return float(base_iv), volatility_mode, "Manual constant volatility", pd.DataFrame(), pd.DataFrame()

    st.caption("Enter manual IVs at the key levels used by the smile-adjusted proxy.")
    vol_cols = st.columns(3)
    strike_iv = vol_cols[0].number_input("Strike IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
    atm_iv = vol_cols[1].number_input("ATM / spot IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")

    manual_points = [{"strike": float(strike), "iv": float(strike_iv), "source": "Manual strike IV"},{"strike": float(spot), "iv": float(atm_iv), "source": "Manual ATM IV"}]

    if option_type in BARRIER_TYPES:
        level_iv = vol_cols[2].number_input("Barrier IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
        if barrier is not None:
            manual_points.append({"strike": float(barrier), "iv": float(level_iv), "source": "Manual barrier IV"})
    elif option_type in ["Gap Calls", "Gap Puts"]:
        level_iv = vol_cols[2].number_input("Payoff strike IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
        if extra_strike is not None:
            manual_points.append({"strike": float(extra_strike), "iv": float(level_iv), "source": "Manual payoff strike IV"})
    elif option_type in ["Capped Calls", "Capped Puts"]:
        level_iv = vol_cols[2].number_input("Cap / floor IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
        if extra_strike is not None:
            manual_points.append({"strike": float(extra_strike), "iv": float(level_iv), "source": "Manual cap/floor IV"})

    iv_points = pd.DataFrame(manual_points).sort_values("strike").reset_index(drop=True)
    effective_iv, volatility_formula, diagnostics = calculate_smile_adjusted_volatility(option_type, strike, spot, barrier, extra_strike, iv_points, base_iv)

    return effective_iv, volatility_mode, volatility_formula, iv_points, diagnostics

def get_manual_parameters():
    st.subheader("Manual option setup")

    option_type = st.selectbox("Option type", EXOTIC_OPTION_TYPES, index=0)

    input_cols = st.columns(3)
    spot = input_cols[0].number_input("Spot price", min_value=0.0001, value=100.0, step=1.0)
    strike = input_cols[1].number_input("Strike", min_value=0.0001, value=100.0, step=1.0)
    maturity_date = input_cols[2].date_input("Maturity",value=datetime.today().date() + timedelta(days=90),min_value=datetime.today().date() + timedelta(days=1))

    market_cols = st.columns(3)
    base_iv = market_cols[0].number_input("Base volatility / IV", min_value=0.0001, value=DEFAULT_VOLATILITY, step=0.01, format="%.4f")
    risk_free_rate = market_cols[1].number_input("Risk-free rate", value=DEFAULT_RISK_FREE_RATE, step=0.005, format="%.4f")
    dividend_yield = market_cols[2].number_input("Dividend yield", min_value=0.0, value=DEFAULT_DIVIDEND_YIELD, step=0.005, format="%.4f")

    barrier, cash_payout, extra_strike = get_extra_inputs(option_type, float(spot), float(strike))
    effective_iv, volatility_mode, volatility_formula, iv_points, diagnostics = build_manual_iv_points(option_type, float(spot), float(strike), barrier, extra_strike, float(base_iv))

    return {"source": "Manual inputs","symbol": "Manual input","option_type": option_type,"spot": float(spot),"strike": float(strike),"maturity": datetime.combine(maturity_date, datetime.min.time()),"IV": float(effective_iv),"base_IV": float(base_iv),"rf": float(risk_free_rate),"dividend_yield": float(dividend_yield),"currency": "","volatility_mode": volatility_mode,"volatility_formula": volatility_formula,"volatility_diagnostics": diagnostics,"iv_points": iv_points,"barrier": barrier,"cash_payout": cash_payout,"extra_strike": extra_strike}


def get_yahoo_parameters():
    st.subheader("Yahoo Finance option setup")

    ticker_source = st.radio("Ticker source", ["Liquid ticker universe", "Custom ticker"], horizontal=True)

    setup_cols = st.columns(2)

    with setup_cols[0]:
        if ticker_source == "Liquid ticker universe":
            default_index = OPTIONABLE_TICKERS.index("AAPL") if "AAPL" in OPTIONABLE_TICKERS else 0
            symbol = st.selectbox("Ticker", OPTIONABLE_TICKERS, index=default_index)
        else:
            symbol = st.text_input("Ticker", value="AAPL")

    with setup_cols[1]:
        option_type = st.selectbox("Option type", EXOTIC_OPTION_TYPES, index=0)

    symbol = symbol.strip().upper()

    if not symbol:
        st.info("Enter a ticker to begin.")
        return None

    quote_option_type = infer_quote_option_type(option_type)

    with st.spinner("Loading Yahoo Finance options and spot data..."):
        option_data = load_option_data(symbol, quote_option_type)
        spot = load_spot_price(symbol)

    if option_data.empty:
        st.warning(f"No usable {quote_option_type.lower()} quotes are available for {symbol} after bid/ask filtering.")
        return None

    if not np.isfinite(spot):
        st.warning(f"Unable to retrieve a valid spot price for {symbol}.")
        return None

    currency = option_data["currency"].dropna().iloc[0] if "currency" in option_data.columns and not option_data["currency"].dropna().empty else ""
    available_maturities = sorted(option_data["maturity"].dropna().unique())

    snapshot_cols = st.columns(3)
    snapshot_cols[0].metric("Spot", format_price(spot, currency))
    snapshot_cols[1].metric("Ticker", symbol)
    snapshot_cols[2].metric("Currency", currency if currency else "N/A")

    setup_cols = st.columns(2)
    maturity_string = setup_cols[0].selectbox("Maturity", available_maturities)
    maturity = datetime.strptime(maturity_string, "%Y-%m-%d")

    strikes = sorted(option_data.loc[option_data["maturity"] == maturity_string, "strike"].dropna().unique())
    strike = setup_cols[1].selectbox("Strike", strikes, format_func=format_strike)
    strike = float(strike)

    selected_rows = option_data[(option_data["maturity"] == maturity_string) & (option_data["strike"] == strike)].copy()

    if selected_rows.empty:
        st.warning("No quote was found for this strike and maturity.")
        return None

    selected_quote = selected_rows.iloc[0]

    with st.spinner("Preparing rates, dividends and implied volatility..."):
        risk_free_rate = load_risk_free_rate(currency)
        dividend_yield, annual_dividend, dividend_method = load_dividend_yield(symbol, spot)
        base_iv, base_iv_source = implied_volatility_from_mid(selected_quote, spot, risk_free_rate, dividend_yield)

    if not np.isfinite(base_iv):
        base_iv = DEFAULT_VOLATILITY
        base_iv_source = "Fallback default IV"

    barrier, cash_payout, extra_strike = get_extra_inputs(option_type, float(spot), float(strike))

    volatility_mode = st.radio("Volatility input",["Single vanilla IV", "Smile-adjusted proxy"],horizontal=True,help="Single vanilla IV uses the selected listed option. Smile-adjusted proxy builds a same-maturity IV curve from listed vanilla quotes.")

    iv_points = pd.DataFrame()
    diagnostics = pd.DataFrame()
    effective_iv = base_iv
    volatility_formula = base_iv_source

    if volatility_mode == "Smile-adjusted proxy":
        with st.spinner("Building volatility smile from vanilla quotes..."):
            iv_points = build_iv_points_for_maturity(option_data, maturity_string, spot, risk_free_rate, dividend_yield)
            effective_iv, volatility_formula, diagnostics = calculate_smile_adjusted_volatility(option_type, strike, spot, barrier, extra_strike, iv_points, base_iv)

    market_cols = st.columns(3)
    market_cols[0].metric("Volatility input", format_percent(effective_iv))
    market_cols[1].metric("Risk-free rate", format_percent(risk_free_rate))
    market_cols[2].metric("Dividend yield", format_percent(dividend_yield))

    st.caption(f"Volatility method: {volatility_mode}. {volatility_formula}.")

    return {"source": "Yahoo Finance","symbol": symbol,"option_type": option_type,"spot": float(spot),"strike": float(strike),"maturity": maturity,"IV": float(effective_iv),"base_IV": float(base_iv),"rf": float(risk_free_rate),"dividend_yield": float(dividend_yield),"currency": currency,"annual_dividend": annual_dividend,"dividend_method": dividend_method,"volatility_mode": volatility_mode,"volatility_formula": volatility_formula,"volatility_diagnostics": diagnostics,"iv_points": iv_points,"barrier": barrier,"cash_payout": cash_payout,"extra_strike": extra_strike}

def parameters_for_spot(parameters: dict, spot: float) -> dict:
    return copy_parameters(parameters, spot=float(spot))

def calculate_result(parameters, num_simulations, greek_names=None):
    return calculate_monte_carlo_greeks(parameters, num_simulations, greek_names=greek_names)


def calculate_forward_price(parameters):
    tau = years_to_maturity(parameters["maturity"])
    spot = parameters["spot"]
    risk_free_rate = parameters["rf"]
    dividend_yield = parameters.get("dividend_yield", 0.0)

    if not all(np.isfinite(x) for x in [spot, tau, risk_free_rate, dividend_yield]) or tau <= 0:
        return np.nan

    return float(spot * np.exp((risk_free_rate - dividend_yield) * tau))


def build_greek_curves(parameters, s_min, s_max, number_of_points, num_simulations, greek_names=None):
    selected_greek_names = list(greek_names or GREEK_NAMES)
    s_grid = np.linspace(float(s_min), float(s_max), int(number_of_points))
    rows = []
    progress = st.progress(0)

    for index, spot_value in enumerate(s_grid):
        curve_parameters = parameters_for_spot(parameters, spot_value)
        result = calculate_result(curve_parameters, num_simulations, greek_names=selected_greek_names)
        result_dict = tuple_to_dict(result)
        row = {"S": float(spot_value), "IV used": curve_parameters["IV"]}
        row.update({greek_name: result_dict.get(greek_name, np.nan) for greek_name in selected_greek_names})
        rows.append(row)
        progress.progress((index + 1) / len(s_grid))

    progress.empty()
    return pd.DataFrame(rows)

def interpolate_curve_value(curve_df, x_value, y_column):
    if not np.isfinite(x_value) or curve_df.empty or y_column not in curve_df.columns:
        return np.nan

    clean_df = curve_df[["S", y_column]].replace([np.inf, -np.inf], np.nan).dropna()

    if clean_df.empty:
        return np.nan

    x_values = clean_df["S"].to_numpy(dtype=float)
    y_values = clean_df[y_column].to_numpy(dtype=float)

    if x_value < x_values.min() or x_value > x_values.max():
        return np.nan

    return float(np.interp(x_value, x_values, y_values))


def add_forward_marker(figure, curve_df, forward_price, y_column, marker_label="Forward", y_transform=lambda x: x):
    forward_value = interpolate_curve_value(curve_df, forward_price, y_column)

    if not np.isfinite(forward_price) or not np.isfinite(forward_value):
        return

    figure.add_trace(go.Scatter(x=[forward_price],y=[y_transform(forward_value)],mode="markers+text",text=[marker_label],textposition="top center",name=marker_label,marker=dict(size=10, symbol="diamond"),hovertemplate="Forward S=%{x:.4f}<br>Value=%{y:.6f}<extra></extra>",))


def create_greek_curve_figure(curve_df,greek_name,current_spot,strike,forward_price,barrier,currency,comparison_curve_df=None,comparison_forward_price=None,comparison_label="Scenario",):
    figure = go.Figure()

    if greek_name in curve_df.columns:
        figure.add_trace(go.Scatter(x=curve_df["S"],y=curve_df[greek_name],mode="lines",name="Initial curve",hovertemplate="S=%{x:.4f}<br>Initial=%{y:.6f}<extra></extra>"))
        add_forward_marker(figure, curve_df, forward_price, greek_name, marker_label="Initial forward")

    if comparison_curve_df is not None and greek_name in comparison_curve_df.columns:
        figure.add_trace(go.Scatter(x=comparison_curve_df["S"],y=comparison_curve_df[greek_name],mode="lines",name=comparison_label,hovertemplate=f"S=%{{x:.4f}}<br>{comparison_label}=%{{y:.6f}}<extra></extra>"))
        if comparison_forward_price is not None:
            add_forward_marker(figure, comparison_curve_df, comparison_forward_price, greek_name, marker_label="Scenario forward")

    figure.add_vline(x=current_spot, line_dash="dash", annotation_text="Current spot", annotation_position="top left")
    figure.add_vline(x=strike, line_dash="dot", annotation_text="Strike", annotation_position="top right")

    if barrier is not None and np.isfinite(barrier):
        figure.add_vline(x=barrier, line_dash="dashdot", annotation_text="Barrier", annotation_position="bottom right")

    y_label = f"{greek_name} ({currency})" if greek_name == "Price" and currency else greek_name

    figure.update_layout(template="plotly_dark",title=f"{greek_name} as a function of underlying price S",xaxis_title="Underlying price S",yaxis_title=y_label,height=520,margin=dict(l=40, r=40, t=75, b=40))

    return figure

def display_parameter_summary(parameters):
    rows = [{"Input": "Source", "Value": parameters["source"]},{"Input": "Symbol", "Value": parameters["symbol"]},{"Input": "Option type", "Value": parameters["option_type"]},{"Input": "Spot", "Value": format_price(parameters["spot"], parameters.get("currency", ""))}, {"Input": "Strike", "Value": format_price(parameters["strike"], parameters.get("currency", ""))}, {"Input": "Maturity", "Value": as_datetime(parameters["maturity"]).strftime("%Y-%m-%d")},{"Input": "Volatility", "Value": format_percent(parameters["IV"])},{"Input": "Volatility model", "Value": parameters.get("volatility_mode", "Single volatility")},{"Input": "Volatility formula", "Value": parameters.get("volatility_formula", "N/A")},{"Input": "Risk-free rate", "Value": format_percent(parameters["rf"])},{"Input": "Dividend yield", "Value": format_percent(parameters["dividend_yield"])}]

    if parameters.get("barrier") is not None:
        rows.append({"Input": "Barrier", "Value": format_price(parameters["barrier"], parameters.get("currency", ""))})

    if parameters.get("extra_strike") is not None:
        rows.append({"Input": "Payoff strike / cap / floor", "Value": format_price(parameters["extra_strike"], parameters.get("currency", ""))})

    if parameters.get("cash_payout") not in [None, 1.0] and parameters["option_type"] in DIGITAL_TYPES:
        rows.append({"Input": "Cash payout", "Value": format_price(parameters["cash_payout"], parameters.get("currency", ""))})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

METRIC_LABELS = {"Theta": "Theta / day","Charm": "Charm / day","Color": "Color / day","Veta": "Veta / day",}

def format_metric_value(metric_name, value, currency):
    if metric_name == "Price":
        return format_price(value, currency)

    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value:.6f}"

def display_metrics(result, currency, greek_names=None, comparison_result=None):
    result_dict = tuple_to_dict(result)
    comparison_dict = tuple_to_dict(comparison_result) if comparison_result is not None else {}
    selected_metric_names = [greek_name for greek_name in (greek_names or GREEK_NAMES) if greek_name in GREEK_NAMES]

    for start_index in range(0, len(selected_metric_names), 4):
        metric_cols = st.columns(4)
        row_names = selected_metric_names[start_index:start_index + 4]

        for col, greek_name in zip(metric_cols, row_names):
            label = METRIC_LABELS.get(greek_name, greek_name)
            value = result_dict.get(greek_name, np.nan)
            delta_value = None

            if greek_name in comparison_dict and np.isfinite(value) and np.isfinite(comparison_dict[greek_name]):
                raw_delta = value - comparison_dict[greek_name]
                delta_value = format_metric_value(greek_name, raw_delta, currency)

            col.metric(label, format_metric_value(greek_name, value, currency), delta=delta_value)


def days_to_maturity(maturity):
    days = (as_datetime(maturity) - datetime.today()).days
    return max(1, int(days))

def build_scenario_parameters_from_sliders(parameters, key_prefix="scenario"):
    base_spot = float(parameters["spot"])
    base_strike = float(parameters["strike"])
    base_iv = float(parameters["IV"])
    base_rf = float(parameters["rf"])
    base_dividend_yield = float(parameters["dividend_yield"])
    base_maturity_days = days_to_maturity(parameters["maturity"])

    slider_cols_1 = st.columns(3)
    scenario_spot_pct = slider_cols_1[0].slider("Spot as % of initial spot",min_value=25,max_value=200,value=100,step=1,format="%d%%",key=f"{key_prefix}_spot_pct")
    scenario_iv_pct = slider_cols_1[1].slider("Volatility / IV",min_value=1.0,max_value=200.0,value=float(np.clip(base_iv * 100, 1.0, 200.0)),step=1.0,format="%.1f%%",key=f"{key_prefix}_iv_pct")
    scenario_maturity_days = slider_cols_1[2].slider("Maturity",min_value=1,max_value=max(1825, int(base_maturity_days * 2)),value=int(np.clip(base_maturity_days, 1, max(1825, int(base_maturity_days * 2)))),step=1,format="%d days",key=f"{key_prefix}_maturity_days")

    slider_cols_2 = st.columns(3)
    scenario_rf_pct = slider_cols_2[0].slider("Risk-free rate",min_value=-20.0,max_value=50.0,value=float(np.clip(base_rf * 100, -20.0, 50.0)),step=0.25,format="%.2f%%",key=f"{key_prefix}_rf_pct")
    scenario_dividend_yield_pct = slider_cols_2[1].slider("Dividend yield",min_value=0.0,max_value=50.0,value=float(np.clip(base_dividend_yield * 100, 0.0, 50.0)),step=0.25,format="%.2f%%",key=f"{key_prefix}_dividend_yield_pct")
    scenario_strike_pct = slider_cols_2[2].slider("Strike as % of initial strike",min_value=25,max_value=200,value=100,step=1,format="%d%%",key=f"{key_prefix}_strike_pct")

    scenario_parameters = copy_parameters(parameters,spot=max(base_spot * scenario_spot_pct / 100, 1e-8),strike=max(base_strike * scenario_strike_pct / 100, 1e-8),IV=max(scenario_iv_pct / 100, 1e-8),base_IV=max(scenario_iv_pct / 100, 1e-8),rf=scenario_rf_pct / 100,dividend_yield=scenario_dividend_yield_pct / 100,maturity=datetime.today() + timedelta(days=int(scenario_maturity_days)))

    return scenario_parameters

def render_greek_curve_tabs(curve_df,greek_names,current_spot,strike,forward_price,barrier,currency,comparison_curve_df=None,comparison_forward_price=None,comparison_label="Scenario curve"):
    available_greek_names = [greek_name for greek_name in greek_names if greek_name in curve_df.columns]

    if comparison_curve_df is not None:
        available_greek_names = [greek_name for greek_name in greek_names if greek_name in curve_df.columns or greek_name in comparison_curve_df.columns]

    if not available_greek_names:
        st.info("No Greek curve is available for the current selection.")
        return

    tabs = st.tabs(available_greek_names)

    for tab, greek_name in zip(tabs, available_greek_names):
        with tab:
            figure = create_greek_curve_figure(curve_df=curve_df,greek_name=greek_name,current_spot=current_spot,strike=strike,forward_price=forward_price,barrier=barrier,currency=currency,comparison_curve_df=comparison_curve_df,comparison_forward_price=comparison_forward_price,comparison_label=comparison_label)
            st.plotly_chart(figure, use_container_width=True)

def render_interactive_curve_slider_tab(parameters, number_of_points, num_simulations, default_greek_names=None):
    st.divider()
    slider_tab = st.tabs(["Greek curve sliders"])[0]

    with slider_tab:
        with st.form("scenario_curve_form"):
            st.markdown("**Greeks to recalculate**")
            scenario_default_greeks = st.session_state.get("scenario_curve_greeks",default_greek_names or ["Price", "Delta", "Gamma", "Vega", "Theta"])
            default_first_order, default_second_order, default_third_order = infer_greek_order_defaults(scenario_default_greeks)
            selected_greeks = render_greek_order_checkboxes(key_prefix="scenario_greek_order",default_first_order=default_first_order,default_second_order=default_second_order,default_third_order=default_third_order)

            scenario_parameters = build_scenario_parameters_from_sliders(parameters, key_prefix="curve_scenario")
            submitted = st.form_submit_button("Update scenario curves", type="primary", use_container_width=True)

        if submitted:
            scenario_spot = float(scenario_parameters["spot"])
            scenario_s_min = scenario_spot * S_GRID_MIN_MULTIPLIER
            scenario_s_max = scenario_spot * S_GRID_MAX_MULTIPLIER

            with st.spinner("Recalculating selected scenario Greek curves..."):
                scenario_curve_df = build_greek_curves(parameters=scenario_parameters,s_min=scenario_s_min,s_max=scenario_s_max,number_of_points=int(number_of_points),num_simulations=int(num_simulations),greek_names=selected_greeks)

            st.session_state["scenario_curve_df"] = scenario_curve_df
            st.session_state["scenario_parameters"] = scenario_parameters
            st.session_state["scenario_curve_greeks"] = selected_greeks
            st.session_state["scenario_forward_price"] = calculate_forward_price(scenario_parameters)

def display_smile_diagnostics(parameters):
    diagnostics = parameters.get("volatility_diagnostics")
    iv_points = parameters.get("iv_points")

    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        with st.expander("Smile-adjusted volatility diagnostics"):
            display_df = diagnostics.copy()
            display_df["Level"] = display_df["Level"].map(lambda x: f"{x:.4f}" if np.isfinite(safe_float(x)) else "N/A")
            display_df["Vanilla IV"] = display_df["Vanilla IV"].map(format_percent)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    if isinstance(iv_points, pd.DataFrame) and not iv_points.empty:
        with st.expander("IV points used by the smile proxy"):
            display_df = iv_points.copy()
            display_df["strike"] = display_df["strike"].map(lambda x: f"{x:.4f}")
            display_df["iv"] = display_df["iv"].map(format_percent)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title="Greeks Curves Dashboard", layout="wide")
    st.title("Greeks Curves Dashboard")
    st.caption("Choose manual inputs or Yahoo Finance data, then visualize Price and Greeks as functions of the underlying price S.")

    with st.sidebar:
        st.header("Dashboard settings")
        input_source = st.radio("Input source", ["Manual inputs", "Yahoo Finance"], horizontal=False, index=1)
        number_of_points = st.slider("Number of S points", min_value=10, max_value=100, value=DEFAULT_S_GRID_POINTS, step=10)
        num_simulations = st.slider("Monte Carlo simulations per point",min_value=10_000,max_value=100_000,value=DEFAULT_MONTE_CARLO_SIMULATIONS,step=10_000,format="%d")

        st.subheader("Greeks to calculate")
        selected_greek_names = render_greek_order_checkboxes(key_prefix="base_greek_order",default_first_order=True,default_second_order=True,default_third_order=False)


    parameters = get_manual_parameters() if input_source == "Manual inputs" else get_yahoo_parameters()

    if parameters is None:
        return

    if not all(np.isfinite(parameters[key]) for key in ["spot", "strike", "IV", "rf", "dividend_yield"]):
        st.warning("At least one required pricing input is invalid.")
        return

    current_spot = parameters["spot"]
    strike = parameters["strike"]
    currency = parameters.get("currency", "")
    forward_price = calculate_forward_price(parameters)
    s_min = current_spot * S_GRID_MIN_MULTIPLIER
    s_max = current_spot * S_GRID_MAX_MULTIPLIER

    display_smile_diagnostics(parameters)

    st.divider()
    st.subheader("Current price and Greeks")

    with st.spinner("Calculating current price and selected Greeks..."):
        base_result = calculate_result(parameters, int(num_simulations), greek_names=selected_greek_names)

    display_metrics(base_result, currency, greek_names=selected_greek_names)

    st.divider()
    if st.button("Generate Greek curves", type="primary", use_container_width=True):
        with st.spinner("Calculating Greek curves..."):
            curve_df = build_greek_curves(parameters=parameters,s_min=s_min,s_max=s_max,number_of_points=int(number_of_points),num_simulations=int(num_simulations),greek_names=selected_greek_names)

        st.session_state["base_curve_df"] = curve_df
        st.session_state["base_curve_parameters"] = parameters
        st.session_state["base_curve_currency"] = currency
        st.session_state["base_curve_number_of_points"] = int(number_of_points)
        st.session_state["base_curve_num_simulations"] = int(num_simulations)
        st.session_state["base_curve_forward_price"] = forward_price
        st.session_state["base_curve_greeks"] = selected_greek_names
        st.session_state.pop("scenario_curve_df", None)
        st.session_state.pop("scenario_parameters", None)
        st.session_state.pop("scenario_curve_greeks", None)
        st.session_state.pop("scenario_forward_price", None)

    if "base_curve_df" not in st.session_state:
        return

    curve_df = st.session_state["base_curve_df"]
    base_parameters = st.session_state["base_curve_parameters"]
    base_currency = st.session_state.get("base_curve_currency", currency)
    base_number_of_points = st.session_state.get("base_curve_number_of_points", int(number_of_points))
    base_num_simulations = st.session_state.get("base_curve_num_simulations", int(num_simulations))
    base_forward_price = st.session_state.get("base_curve_forward_price", calculate_forward_price(base_parameters))
    base_curve_greeks = st.session_state.get("base_curve_greeks", selected_greek_names)

    st.subheader("Greek curves")
    greek_curves_container = st.container()

    render_interactive_curve_slider_tab(base_parameters,int(base_number_of_points),int(base_num_simulations),default_greek_names=base_curve_greeks)

    scenario_curve_df = st.session_state.get("scenario_curve_df")
    scenario_forward_price = st.session_state.get("scenario_forward_price")
    scenario_curve_greeks = st.session_state.get("scenario_curve_greeks", [])
    display_greek_names = [greek_name for greek_name in GREEK_NAMES if greek_name in set(base_curve_greeks) | set(scenario_curve_greeks)]

    with greek_curves_container:
        render_greek_curve_tabs(curve_df=curve_df,greek_names=display_greek_names,current_spot=float(base_parameters["spot"]),strike=float(base_parameters["strike"]),forward_price=base_forward_price,barrier=base_parameters.get("barrier"),currency=base_currency,comparison_curve_df=scenario_curve_df,comparison_forward_price=scenario_forward_price,comparison_label="Scenario curve")

if __name__ == "__main__":
    main()