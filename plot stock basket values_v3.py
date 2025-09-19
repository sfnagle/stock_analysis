import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
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

    required_columns = {'Ticker', 'Purchase Date', 'Purchase Price', 'Quantity'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input file must include the columns: {required_columns}")

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
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

def compute_basket_value(basket_df, data, purchase_date):
    import pandas as pd

    stock_values = pd.DataFrame(index=data.index)
    stock_prices = pd.DataFrame(index=data.index)
    quantities = {}

    for _, row in basket_df.iterrows():
        ticker = str(row['Ticker']).strip()
        quantity = float(row['Quantity'])

        if ticker not in data.columns:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            price_series = data[ticker]['Close']
        else:
            price_series = data[ticker]

        value_series = price_series * quantity

        stock_values[ticker] = value_series
        stock_prices[ticker] = price_series
        quantities[ticker] = quantity

    stock_values.fillna(0, inplace=True)
    basket_value = stock_values.sum(axis=1)

    # Compute gains relative to the basket value at purchase_date
    if purchase_date not in basket_value.index:
        if purchase_date < basket_value.index.min():
            print(f"⚠️ Warning: Purchase date {purchase_date.date()} is before available data range starting {basket_value.index.min().date()}. Using first available date instead.")
            purchase_date = basket_value.index.min()
        elif purchase_date > basket_value.index.max():
            print(f"⚠️ Warning: Purchase date {purchase_date.date()} is after available data range ending {basket_value.index.max().date()}. Using last available date instead.")
            purchase_date = basket_value.index.max()
        else:
            # Use nearest available date if it's within range but not exactly matched (e.g., weekend)
            purchase_date = basket_value.index[basket_value.index.get_indexer([purchase_date], method='nearest')[0]]

    initial_value = basket_value.loc[purchase_date]
    basket_pct = (basket_value / initial_value - 1) * 100

    initial_prices = stock_prices.loc[purchase_date]
    stock_pct = (stock_prices.divide(initial_prices) - 1) * 100

    return basket_value, stock_values, stock_prices, quantities, basket_pct, stock_pct

def make_hover_trace(total_series, name, stock_prices, stock_values, quantities, show_date=False, stock_pct=None, basket_pct=None):
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
            quantity = quantities.get(ticker, None)
            pct_change = stock_pct.at[dt, ticker] if stock_pct is not None and ticker in stock_pct.columns else None
            if pd.notna(price) and pd.notna(value) and quantity:
                lines.append(
                    f"{ticker}: {quantity} × ${price:.2f} = ${value:,.0f} ({pct_change:+.1f}%)"
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

    start_date, end_date = '2024-06-01', date.today().strftime("%Y-%m-%d")
    name1 = 'Basket A'
    name2 = 'Basket B'
    path = './'
    file1 = path + name1 + '.csv'
    file2 = path + name2 + '.csv'

    basket1 = load_basket(file1)
    basket2 = load_basket(file2)

    # Determine and check purchase dates
    purchase_date1 = pd.to_datetime(basket1['Purchase Date'].iloc[0])
    purchase_date2 = pd.to_datetime(basket2['Purchase Date'].iloc[0])

    assert (basket1['Purchase Date'] == purchase_date1.strftime('%Y-%m-%d')).all(), "Inconsistent purchase dates in Basket A"
    assert (basket2['Purchase Date'] == purchase_date2.strftime('%Y-%m-%d')).all(), "Inconsistent purchase dates in Basket B"

    all_tickers = pd.concat([basket1, basket2])['Ticker'].unique().tolist()
    print("✅ Ready to download data for:", all_tickers)

    data = download_stock_data(all_tickers, start_date, end_date)

    value1, values1, prices1, qtys1, pct1, stockpct1 = compute_basket_value(basket1, data, purchase_date1)
    value2, values2, prices2, qtys2, pct2, stockpct2 = compute_basket_value(basket2, data, purchase_date2)

    fig = go.Figure()

    fig.add_trace(make_hover_trace(value1, name1, prices1, values1, qtys1, show_date=True, stock_pct=stockpct1, basket_pct=pct1))
    fig.add_trace(make_hover_trace(value2, name2, prices2, values2, qtys2, show_date=False, stock_pct=stockpct2, basket_pct=pct2))

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
