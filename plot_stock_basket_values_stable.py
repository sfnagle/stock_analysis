import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from dateutil.relativedelta import relativedelta
import time
import requests
import os


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


def compute_basket_value(basket_df, data):
    import pandas as pd

    # Build position history over time
    position_history = build_position_history(basket_df, data.index)

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

    # Find baseline dates for percentage calculations
    # Basket baseline: earliest transaction date
    basket_baseline_date = basket_df['Transaction Date'].min()

    # Find first Buy transaction date for each ticker (for per-stock percentages)
    ticker_baselines = {}
    for ticker in position_history.columns:
        ticker_buys = basket_df[(basket_df['Ticker'] == ticker) & (basket_df['Transaction Type'] == 'Buy')]
        if not ticker_buys.empty:
            ticker_baselines[ticker] = ticker_buys['Transaction Date'].min()

    # Adjust baseline date if not in data range
    if basket_baseline_date not in basket_value.index:
        if basket_baseline_date < basket_value.index.min():
            print(f"⚠️ Warning: Earliest transaction date {basket_baseline_date.date()} is before available data range starting {basket_value.index.min().date()}. Using first available date instead.")
            basket_baseline_date = basket_value.index.min()
        elif basket_baseline_date > basket_value.index.max():
            print(f"⚠️ Warning: Earliest transaction date {basket_baseline_date.date()} is after available data range ending {basket_value.index.max().date()}. Using last available date instead.")
            basket_baseline_date = basket_value.index.max()
        else:
            # Use nearest available date if it's within range but not exactly matched (e.g., weekend)
            basket_baseline_date = basket_value.index[basket_value.index.get_indexer([basket_baseline_date], method='nearest')[0]]

    # Calculate basket percentage gain from baseline
    initial_value = basket_value.loc[basket_baseline_date]
    if initial_value > 0:
        basket_pct = (basket_value / initial_value - 1) * 100
    else:
        basket_pct = pd.Series(0, index=basket_value.index)

    # Calculate per-stock percentage gains from their respective baselines
    stock_pct = pd.DataFrame(index=data.index, columns=stock_prices.columns)
    for ticker in stock_prices.columns:
        if ticker in ticker_baselines:
            baseline_date = ticker_baselines[ticker]

            # Adjust baseline date if needed
            if baseline_date not in stock_prices.index:
                if baseline_date < stock_prices.index.min():
                    baseline_date = stock_prices.index.min()
                elif baseline_date > stock_prices.index.max():
                    baseline_date = stock_prices.index.max()
                else:
                    baseline_date = stock_prices.index[stock_prices.index.get_indexer([baseline_date], method='nearest')[0]]

            initial_price = stock_prices.loc[baseline_date, ticker]
            if pd.notna(initial_price) and initial_price > 0:
                stock_pct[ticker] = (stock_prices[ticker] / initial_price - 1) * 100
            else:
                stock_pct[ticker] = 0
        else:
            stock_pct[ticker] = 0

    return basket_value, stock_values, stock_prices, position_history, basket_pct, stock_pct

def make_hover_trace(total_series, name, stock_prices, stock_values, position_history, show_date=False, stock_pct=None, basket_pct=None):
    hover_data = []
    for dt in total_series.index:
        lines = []

        if show_date:
            lines.append(f"<b>{dt.strftime('%Y-%m-%d')}</b>")

        basket_total = total_series.at[dt]
        if basket_pct is not None:
            basket_change = basket_pct.at[dt]
            lines.append(f"<b>{name}: ${basket_total:,.0f} ({basket_change:+.1f}%)</b>")
        else:
            lines.append(f"<b>{name}: ${basket_total:,.0f}</b>")

        for ticker in stock_values.columns:
            price = stock_prices.at[dt, ticker] if dt in stock_prices.index else None
            value = stock_values.at[dt, ticker] if dt in stock_values.index else None
            position = position_history.at[dt, ticker] if dt in position_history.index and ticker in position_history.columns else None
            pct_change = stock_pct.at[dt, ticker] if stock_pct is not None and ticker in stock_pct.columns else None
            if pd.notna(price) and pd.notna(value) and pd.notna(position) and position != 0:
                lines.append(
                    f"{ticker}: {position:.2f} × ${price:.2f} = ${value:,.0f} ({pct_change:+.1f}%)"
                )

        hover_data.append("<br>".join(lines))

    return go.Scatter(
        x=total_series.index,
        y=total_series.values,
        mode='lines',
        name=name,
        hovertext=hover_data,
        hoverinfo='text'
    )


def main():
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

    value1, values1, prices1, positions1, pct1, stockpct1 = compute_basket_value(basket1, data)
    value2, values2, prices2, positions2, pct2, stockpct2 = compute_basket_value(basket2, data)

    fig = go.Figure()

    fig.add_trace(make_hover_trace(value1, name1, prices1, values1, positions1, show_date=True, stock_pct=stockpct1, basket_pct=pct1))
    fig.add_trace(make_hover_trace(value2, name2, prices2, values2, positions2, show_date=False, stock_pct=stockpct2, basket_pct=pct2))

    # Add percent gain traces on secondary y-axis
    fig.add_trace(go.Scatter(
        x=pct1.index,
        y=pct1.values,
        mode='lines',
        name=f"{name1} (% gain)",
        yaxis="y2",
        line=dict(dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=pct2.index,
        y=pct2.values,
        mode='lines',
        name=f"{name2} (% gain)",
        yaxis="y2",
        line=dict(dash='dot')
    ))

    fig.update_layout(
        title="Stock Basket Value Over Time",
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
    main()
