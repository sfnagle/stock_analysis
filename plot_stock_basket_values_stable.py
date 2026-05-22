import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from dateutil.relativedelta import relativedelta
import time
import requests
import os
import warnings

# Suppress numerical solver warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# === CONFIG ===
USE_ALPHA_VANTAGE = True
USE_POLYGON = True
USE_TIINGO = True
USE_FINNHUB = True
ALPHA_VANTAGE_KEY = "YICWOPZ29LQHT3ASN"
POLYGON_API_KEY = "yGPdfTwOulL2Lg7LpqDfPGcpKlcSgCJF"
TIINGO_API_KEY = "f7d733feb23e47cb403a1ff606624ddcfd0cecb1"
FINNHUB_API_KEY = "d0dbqm1r01qhd59vgtggd0dbqm1r01qhd59vgth0"


def load_basket(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX files are supported.")

    required_columns = {'Ticker', 'Transaction Date', 'Transaction Type', 'Quantity'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input file must include the columns: {required_columns}")

    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    # Validate Transaction Type
    valid_types = {'Buy', 'Sell'}
    invalid_types = set(df['Transaction Type'].unique()) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid Transaction Type values: {invalid_types}. Must be 'Buy' or 'Sell'")

    # Sort by transaction date (chronological order)
    df = df.sort_values('Transaction Date').reset_index(drop=True)

    return df


def try_yfinance(tickers, start_date, end_date):
    print("🔁 Trying Yahoo Finance...")
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            threads=False,
            auto_adjust=True,
            progress=False
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs('Close', axis=1, level=1)
            return data.dropna(how='all')
    except Exception as e:
        print(f"❌ yfinance failed: {e}")
    return pd.DataFrame()


def try_alpha_vantage(tickers, start_date, end_date):
    if not USE_ALPHA_VANTAGE:
        return pd.DataFrame()

    print("🔁 Trying Alpha Vantage...")
    base_url = "https://www.alphavantage.co/query"
    all_data = {}

    for ticker in tickers:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "datatype": "json",
            "apikey": ALPHA_VANTAGE_KEY
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            continue
        raw = response.json().get("Time Series (Daily)", {})
        df = pd.DataFrame.from_dict(raw, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        if "5. adjusted close" not in df.columns:
            print(f"⚠️ Alpha Vantage data for {ticker} missing '5. adjusted close'. Skipping.")
            continue
        df = df.rename(columns={"5. adjusted close": ticker})[[ticker]].astype(float)
        all_data[ticker] = df
        time.sleep(12)  # Respect rate limit

    if all_data:
        merged = pd.concat(all_data.values(), axis=1, join='outer')
        return merged.dropna(how='all')
    return pd.DataFrame()


def try_polygon(tickers, start_date, end_date):
    if not USE_POLYGON:
        return pd.DataFrame()

    print("🔁 Trying Polygon.io...")
    base_url = "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}"
    real_time_url = "https://api.polygon.io/v2/last/nbbo/{}"
    all_data = {}

    for ticker in tickers:
        url = base_url.format(ticker, start_date, end_date)
        params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"⚠️ Polygon failed for {ticker}: {response.status_code}")
            continue
        results = response.json().get("results", [])
        if not results:
            print(f"ℹ️ No daily data for {ticker}, trying real-time...")
            rt_url = real_time_url.format(ticker)
            rt_params = {"apiKey": POLYGON_API_KEY}
            rt_response = requests.get(rt_url, params=rt_params)
            if rt_response.status_code == 429:
                print(f"🚫 Polygon rate limit hit for real-time {ticker}. Skipping.")
                continue
            if rt_response.status_code == 200:
                rt_data = rt_response.json().get("results")
                if isinstance(rt_data, dict) and "P" in rt_data:
                    now = pd.Timestamp.now().normalize()
                    df = pd.DataFrame({ticker: [rt_data["P"]]}, index=[now])
                    all_data[ticker] = df
            continue
        df = pd.DataFrame(results)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df = df.rename(columns={"c": ticker})[[ticker]]
        all_data[ticker] = df
        time.sleep(1)

    if all_data:
        merged = pd.concat(all_data.values(), axis=1, join='outer')
        return merged.dropna(how='all')
    return pd.DataFrame()


def try_tiingo(tickers, start_date, end_date):
    if not USE_TIINGO:
        return pd.DataFrame()

    print("🔁 Trying Tiingo...")
    all_data = {}
    headers = {"Content-Type": "application/json"}

    for ticker in tickers:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "token": TIINGO_API_KEY
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            continue
        raw = response.json()
        df = pd.DataFrame(raw)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.set_index('date', inplace=True)
        df = df[["adjClose"]].rename(columns={"adjClose": ticker})
        all_data[ticker] = df
        time.sleep(1)

    if all_data:
        merged = pd.concat(all_data.values(), axis=1, join='outer')
        return merged.dropna(how='all')
    return pd.DataFrame()


def download_stock_data(tickers, start_date, end_date):
    print(f"📥 Downloading data for tickers: {tickers}")
    data = try_tiingo(tickers, start_date, end_date)
    if data.empty:
        data = try_polygon(tickers, start_date, end_date)
    if data.empty:
        data = try_alpha_vantage(tickers, start_date, end_date)
    if data.empty:
        data = try_yfinance(tickers, start_date, end_date)
    if data.empty:
        raise RuntimeError("No data could be downloaded from any source.")
    return data



def build_position_history(basket_df, date_index):
    """
    Build a time series of stock holdings based on transaction history.

    The initial positions (from the first transaction date) are projected backwards
    across the entire data range to show hypothetical historical performance.

    Args:
        basket_df: DataFrame with columns 'Ticker', 'Transaction Date', 'Transaction Type', 'Quantity'
        date_index: DatetimeIndex from price data

    Returns:
        DataFrame with index=date_index, columns=tickers, values=quantity held at each date
    """
    # Get unique tickers
    tickers = basket_df['Ticker'].unique()

    # Initialize position history with zeros
    position_history = pd.DataFrame(0.0, index=date_index, columns=tickers)

    # Find the earliest transaction date (initial purchase date)
    first_txn_date = basket_df['Transaction Date'].min()

    # Build initial positions from first transaction date
    initial_positions = {}
    for ticker in tickers:
        ticker_transactions = basket_df[basket_df['Ticker'] == ticker].copy()

        # Get all transactions on the first transaction date
        initial_txns = ticker_transactions[ticker_transactions['Transaction Date'] == first_txn_date]
        initial_position = 0.0
        for _, txn in initial_txns.iterrows():
            quantity = float(txn['Quantity'])
            if txn['Transaction Type'] == 'Buy':
                initial_position += quantity
            elif txn['Transaction Type'] == 'Sell':
                initial_position -= quantity
        initial_positions[ticker] = initial_position

    # Project initial positions backwards across entire date range
    for ticker in tickers:
        position_history[ticker] = initial_positions[ticker]

    # Now apply subsequent transactions (after the first transaction date)
    for ticker in tickers:
        ticker_transactions = basket_df[basket_df['Ticker'] == ticker].copy()

        # Process transactions after the first date
        subsequent_txns = ticker_transactions[ticker_transactions['Transaction Date'] > first_txn_date]
        for _, txn in subsequent_txns.iterrows():
            txn_date = txn['Transaction Date']
            quantity = float(txn['Quantity'])
            txn_type = txn['Transaction Type']

            # Apply the transaction to all dates >= transaction date
            mask = position_history.index >= txn_date
            if txn_type == 'Buy':
                position_history.loc[mask, ticker] += quantity
            elif txn_type == 'Sell':
                position_history.loc[mask, ticker] -= quantity

    return position_history


def build_sold_amounts_history(basket_df, date_index):
    """
    Build a time series of cumulative sold amounts based on transaction history.

    Args:
        basket_df: DataFrame with columns 'Ticker', 'Transaction Date', 'Transaction Type', 'Quantity'
        date_index: DatetimeIndex from price data

    Returns:
        DataFrame with index=date_index, columns=tickers, values=cumulative dollar value sold at each date
    """
    tickers = basket_df['Ticker'].unique()
    sold_history = pd.DataFrame(0.0, index=date_index, columns=tickers)

    for ticker in tickers:
        ticker_transactions = basket_df[basket_df['Ticker'] == ticker].copy()
        cumulative_sold = 0.0

        for _, txn in ticker_transactions.iterrows():
            if txn['Transaction Type'] == 'Sell':
                txn_date = txn['Transaction Date']
                quantity = float(txn['Quantity'])

                # Use Transaction Price if available
                if 'Transaction Price' in txn.index and pd.notna(txn['Transaction Price']):
                    price = float(txn['Transaction Price'])
                    sold_amount = quantity * price
                    cumulative_sold += sold_amount

                    # Apply this cumulative sold amount to all dates >= transaction date
                    mask = sold_history.index >= txn_date
                    sold_history.loc[mask, ticker] = cumulative_sold

    return sold_history


def compute_trailing_6mo_annualized_pct(series, start_date=pd.Timestamp('2025-10-10')):
    """Compute trailing 6-month percent change, annualized to 1 year, for dates >= start_date."""
    result = pd.Series(np.nan, index=series.index)
    for dt in series.index:
        if dt < start_date:
            continue
        target = dt - pd.DateOffset(months=6)
        idx = series.index.asof(target)
        if pd.isna(idx):
            continue
        base_val = series.loc[idx]
        if base_val != 0:
            six_mo_return = series.loc[dt] / base_val - 1
            annualized = (1 + six_mo_return) ** 2 - 1
            result.loc[dt] = annualized * 100
    return result


def compute_trailing_1yr_pct(series, start_date=pd.Timestamp('2026-04-10')):
    """Compute trailing 1-year percent change for dates >= start_date."""
    result = pd.Series(np.nan, index=series.index)
    for dt in series.index:
        if dt < start_date:
            continue
        target = dt - pd.DateOffset(years=1)
        idx = series.index.asof(target)
        if pd.isna(idx):
            continue
        base_val = series.loc[idx]
        if base_val != 0:
            result.loc[dt] = (series.loc[dt] / base_val - 1) * 100
    return result


def compute_basket_value(basket_df, data, include_sold_value=False):
    import pandas as pd

    # Build position history over time
    position_history = build_position_history(basket_df, data.index)

    # Build sold amounts history if requested
    sold_history = None
    if include_sold_value:
        sold_history = build_sold_amounts_history(basket_df, data.index)

    stock_values = pd.DataFrame(index=data.index)
    stock_prices = pd.DataFrame(index=data.index)

    # Extract prices and calculate values for each ticker
    for ticker in position_history.columns:
        if ticker not in data.columns:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            price_series = data[ticker]['Close']
        else:
            price_series = data[ticker]

        # Value = position × price at each date
        value_series = position_history[ticker] * price_series

        stock_values[ticker] = value_series
        stock_prices[ticker] = price_series

    stock_values.fillna(0, inplace=True)
    basket_value = stock_values.sum(axis=1)

    # Add sold amounts if requested (for display only, not for MWR calculation)
    display_basket_value = basket_value.copy()
    display_stock_values = stock_values.copy()
    if include_sold_value and sold_history is not None:
        # Add sold amounts to display values
        display_stock_values = display_stock_values + sold_history
        display_basket_value = display_stock_values.sum(axis=1)

    # Calculate simple percentage gain:
    # % gain = (current_holdings_value + cumulative_sell_proceeds) / total_invested - 1
    stock_pct = pd.DataFrame(0.0, index=data.index, columns=stock_prices.columns)
    total_invested = pd.Series(0.0, index=data.index)
    total_sold = pd.Series(0.0, index=data.index)

    print("📊 Calculating percentage gains...")

    for ticker in stock_prices.columns:
        ticker_txns = basket_df[basket_df['Ticker'] == ticker].sort_values('Transaction Date')
        cum_invested = 0.0
        cum_sold = 0.0
        txn_events = []  # (date, cum_invested, cum_sold)

        for _, txn in ticker_txns.iterrows():
            quantity = float(txn['Quantity'])
            if 'Transaction Price' in txn.index and pd.notna(txn['Transaction Price']):
                price = float(txn['Transaction Price'])
            elif txn['Transaction Date'] in stock_prices.index:
                price = stock_prices.loc[txn['Transaction Date'], ticker]
                if pd.isna(price):
                    continue
            else:
                continue
            if txn['Transaction Type'] == 'Buy':
                cum_invested += quantity * price
            elif txn['Transaction Type'] == 'Sell':
                cum_sold += quantity * price
            txn_events.append((txn['Transaction Date'], cum_invested, cum_sold))

        if not txn_events:
            continue

        # Build time series of cumulative invested/sold for this ticker
        inv_series = pd.Series(0.0, index=data.index)
        sold_series = pd.Series(0.0, index=data.index)
        for txn_date, ci, cs in txn_events:
            mask = data.index >= txn_date
            inv_series.loc[mask] = ci
            sold_series.loc[mask] = cs

        # Per-stock % gain
        has_investment = inv_series > 0
        stock_pct.loc[has_investment, ticker] = (
            (stock_values[ticker][has_investment] + sold_series[has_investment])
            / inv_series[has_investment] - 1
        ) * 100

        # Accumulate into basket-level totals
        total_invested += inv_series
        total_sold += sold_series

    # Basket-level % gain
    has_investment = total_invested > 0
    basket_pct = pd.Series(0.0, index=data.index)
    basket_pct.loc[has_investment] = (
        (basket_value[has_investment] + total_sold[has_investment])
        / total_invested[has_investment] - 1
    ) * 100

    return display_basket_value, display_stock_values, stock_prices, position_history, basket_pct, stock_pct

def make_hover_trace(total_series, name, stock_prices, stock_values, position_history, show_date=False, stock_pct=None, basket_pct=None, basket_1yr_pct=None, stock_1yr_pct=None, basket_6mo_pct=None, stock_6mo_pct=None, line_color=None):
    hover_data = []
    for dt in total_series.index:
        lines = []

        if show_date:
            lines.append(f"<b>{dt.strftime('%Y-%m-%d')}</b>")

        basket_total = total_series.at[dt]
        if basket_pct is not None:
            basket_change = basket_pct.at[dt]
            line = f"<b>{name}: ${basket_total:,.0f} ({basket_change:+.1f}%)"
            if basket_6mo_pct is not None and pd.notna(basket_6mo_pct.at[dt]):
                line += f" [6mo: {basket_6mo_pct.at[dt]:+.1f}%]"
            if basket_1yr_pct is not None and pd.notna(basket_1yr_pct.at[dt]):
                line += f" [1yr: {basket_1yr_pct.at[dt]:+.1f}%]"
            line += "</b>"
            lines.append(line)
        else:
            lines.append(f"<b>{name}: ${basket_total:,.0f}</b>")

        for ticker in stock_values.columns:
            price = stock_prices.at[dt, ticker] if dt in stock_prices.index else None
            value = stock_values.at[dt, ticker] if dt in stock_values.index else None
            position = position_history.at[dt, ticker] if dt in position_history.index and ticker in position_history.columns else None
            pct_change = stock_pct.at[dt, ticker] if stock_pct is not None and ticker in stock_pct.columns else None
            if pd.notna(price) and pd.notna(value) and pd.notna(position) and position != 0:
                line = f"{ticker}: {position:.2f} × ${price:.2f} = ${value:,.0f} ({pct_change:+.1f}%)"
                if stock_6mo_pct is not None and ticker in stock_6mo_pct.columns and pd.notna(stock_6mo_pct.at[dt, ticker]):
                    line += f" [6mo: {stock_6mo_pct.at[dt, ticker]:+.1f}%]"
                if stock_1yr_pct is not None and ticker in stock_1yr_pct.columns and pd.notna(stock_1yr_pct.at[dt, ticker]):
                    line += f" [1yr: {stock_1yr_pct.at[dt, ticker]:+.1f}%]"
                lines.append(line)

        hover_data.append("<br>".join(lines))

    line_kwargs = {}
    if line_color:
        line_kwargs['color'] = line_color

    return go.Scatter(
        x=total_series.index,
        y=total_series.values,
        mode='lines',
        name=name,
        hovertext=hover_data,
        hoverinfo='text',
        line=dict(**line_kwargs) if line_kwargs else None
    )


def main(include_sold_value=False):
    from datetime import date
    import pandas as pd

    start_date, end_date = date.today() - relativedelta(years=3), date.today().strftime("%Y-%m-%d")
    name1 = 'Basket A'
    name2 = 'Basket B'
    path = '~/Code/stock_analysis/'
    file1 = path + name1 + '.csv'
    file2 = path + name2 + '.csv'

    basket1 = load_basket(file1)
    basket2 = load_basket(file2)

    all_tickers = pd.concat([basket1, basket2])['Ticker'].unique().tolist()
    print("✅ Ready to download data for:", all_tickers)

    data = download_stock_data(all_tickers, start_date, end_date)

    value1, values1, prices1, positions1, pct1, stockpct1 = compute_basket_value(basket1, data, include_sold_value)
    value2, values2, prices2, positions2, pct2, stockpct2 = compute_basket_value(basket2, data, include_sold_value)

    # Compute trailing 6-month annualized percent change (starting Oct 10, 2025)
    sixmo_start = pd.Timestamp('2025-10-10')
    basket_6mo_1 = compute_trailing_6mo_annualized_pct(value1, sixmo_start)
    basket_6mo_2 = compute_trailing_6mo_annualized_pct(value2, sixmo_start)
    stock_6mo_1 = pd.DataFrame({col: compute_trailing_6mo_annualized_pct(prices1[col], sixmo_start) for col in prices1.columns})
    stock_6mo_2 = pd.DataFrame({col: compute_trailing_6mo_annualized_pct(prices2[col], sixmo_start) for col in prices2.columns})

    # Compute trailing 1-year percent change (starting April 10, 2026)
    yr_start = pd.Timestamp('2026-04-10')
    basket_1yr_1 = compute_trailing_1yr_pct(value1, yr_start)
    basket_1yr_2 = compute_trailing_1yr_pct(value2, yr_start)
    stock_1yr_1 = pd.DataFrame({col: compute_trailing_1yr_pct(prices1[col], yr_start) for col in prices1.columns})
    stock_1yr_2 = pd.DataFrame({col: compute_trailing_1yr_pct(prices2[col], yr_start) for col in prices2.columns})

    fig = go.Figure()

    fig.add_trace(make_hover_trace(value1, name1, prices1, values1, positions1, show_date=True, stock_pct=stockpct1, basket_pct=pct1, basket_1yr_pct=basket_1yr_1, stock_1yr_pct=stock_1yr_1, basket_6mo_pct=basket_6mo_1, stock_6mo_pct=stock_6mo_1, line_color='red'))
    fig.add_trace(make_hover_trace(value2, name2, prices2, values2, positions2, show_date=False, stock_pct=stockpct2, basket_pct=pct2, basket_1yr_pct=basket_1yr_2, stock_1yr_pct=stock_1yr_2, basket_6mo_pct=basket_6mo_2, stock_6mo_pct=stock_6mo_2, line_color='blue'))

    # Add percent gain traces on secondary y-axis
    fig.add_trace(go.Scatter(
        x=pct1.index,
        y=pct1.values,
        mode='lines',
        name=f"{name1} (% gain)",
        yaxis="y2",
        line=dict(dash='dot', color='red')
    ))
    fig.add_trace(go.Scatter(
        x=pct2.index,
        y=pct2.values,
        mode='lines',
        name=f"{name2} (% gain)",
        yaxis="y2",
        line=dict(dash='dot', color='blue')
    ))

    # Add trailing 6-month annualized percent change traces on secondary y-axis
    b6mo1 = basket_6mo_1.dropna()
    b6mo2 = basket_6mo_2.dropna()
    if not b6mo1.empty:
        fig.add_trace(go.Scatter(
            x=b6mo1.index,
            y=b6mo1.values,
            mode='lines',
            name=f"{name1} (6mo ann %)",
            yaxis="y2",
            line=dict(dash='dashdot', color='red')
        ))
    if not b6mo2.empty:
        fig.add_trace(go.Scatter(
            x=b6mo2.index,
            y=b6mo2.values,
            mode='lines',
            name=f"{name2} (6mo ann %)",
            yaxis="y2",
            line=dict(dash='dashdot', color='blue')
        ))

    # Add trailing 1-year percent change traces on secondary y-axis
    b1yr1 = basket_1yr_1.dropna()
    b1yr2 = basket_1yr_2.dropna()
    if not b1yr1.empty:
        fig.add_trace(go.Scatter(
            x=b1yr1.index,
            y=b1yr1.values,
            mode='lines',
            name=f"{name1} (1yr %)",
            yaxis="y2",
            line=dict(dash='dash', color='red')
        ))
    if not b1yr2.empty:
        fig.add_trace(go.Scatter(
            x=b1yr2.index,
            y=b1yr2.values,
            mode='lines',
            name=f"{name2} (1yr %)",
            yaxis="y2",
            line=dict(dash='dash', color='blue')
        ))

    title_suffix = " (Including Sold Amounts)" if include_sold_value else ""
    fig.update_layout(
        title=f"Stock Basket Value Over Time{title_suffix}",
        xaxis_title="Date",
        yaxis=dict(title="Total Value ($)", side="left"),
        yaxis2=dict(title="Percent Change (%)", overlaying='y', side='right'),
        legend_title="Basket",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        width=1000
    )

    fig.show()

    print('Goodbye, World')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot stock basket values over time')
    parser.add_argument('--include-sold', action='store_true',
                       help='Include dollar amounts from sold stocks in the plot (calculations still based on actual holdings)')
    args = parser.parse_args()

    main(include_sold_value=args.include_sold)
