import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date


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

def download_stock_data(tickers, start_date, end_date):
    print(f"📥 Downloading data for tickers: {tickers}")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        threads=False,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise RuntimeError("No data downloaded. Check your ticker symbols or date range.")

    # Check for individual failures
    for ticker in tickers:
        if ticker not in data.columns:
            print(f"⚠️ Warning: No data found for ticker '{ticker}'. It may be invalid or delisted.")

    return data

def compute_basket_value(basket_df, data):
    stock_values = pd.DataFrame(index=data.index)
    stock_prices = pd.DataFrame(index=data.index)
    quantities = {}  # Ticker → Quantity

    for _, row in basket_df.iterrows():
        ticker = str(row['Ticker']).strip()
        quantity = float(row['Quantity'])
        purchase_date = pd.to_datetime(row['Purchase Date'])

        if ticker not in data.columns:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            price_series = data[ticker]['Close']
        else:
            price_series = data[ticker]

        price_series = price_series[price_series.index >= purchase_date]
        value_series = price_series * quantity

        stock_values[ticker] = value_series
        stock_prices[ticker] = price_series
        quantities[ticker] = quantity

    stock_values.fillna(0, inplace=True)
    basket_value = stock_values.sum(axis=1)

    basket_pct = (basket_value / basket_value.iloc[0] - 1) * 100
    stock_pct = (stock_values.divide(stock_values.iloc[0]) - 1) * 100

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
    # User input
    # start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    # end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    # file1 = input(f"Enter path to basket 1 file (CSV/XLSX): ").strip()
    # file2 = input(f"Enter path to basket 2 file (CSV/XLSX): ").strip()
    start_date = '2025-04-10'
    end_date  = date.today().strftime("%Y-%m-%d")
    end_date  =  '2025-05-05'
    name1 = 'Basket A'
    name2 = 'Basket B'
    path = '/Users/sfnagle/Code/sandbox/'
    file1 = path + name1 + '.csv'
    file2 = path + name2 + '.csv'

    # Load baskets
    basket1 = load_basket(file1)
    basket2 = load_basket(file2)

    # Combine all tickers to download once
    all_tickers = pd.concat([basket1, basket2])['Ticker'].unique().tolist()
    data = download_stock_data(all_tickers, start_date, end_date)

    # Compute values
    value1 = compute_basket_value(basket1, data)
    value2 = compute_basket_value(basket2, data)

    # After computing basket values and stock breakdowns:
    value1, values1, prices1, qtys1, pct1, stockpct1 = compute_basket_value(basket1, data)
    value2, values2, prices2, qtys2, pct2, stockpct2 = compute_basket_value(basket2, data)

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
