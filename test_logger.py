#from scripts.fetch_data import get_nifty_50_data
bse_stocks = {
    'TATAMOTORS.BO': 'Tata Motors',
    'RELIANCE.BO': 'Reliance Industries',
    'TCS.BO': 'TCS',
    'BAJFINANCE.BO': 'Bajaj Finance',
    'INFY.BO': 'Infosys',
    'HDFCBANK.BO': 'HDFC Bank',
    'TATASTEEL.BO': 'Tata Steel',
    'LT.BO': 'Larsen & Toubro',
    'ICICIBANK.BO': 'ICICI Bank',
    'INDUSINDBK.BO': 'IndusInd Bank',
    'SBIN.BO': 'SBI',
    'M&M.BO': 'Mahindra & Mahindra',
    'ZOMATO.BO': 'Zomato',
    'SUNPHARMA.BO': 'Sun Pharma',
    'ADANIPORTS.BO': 'Adani Ports',
    'HINDUNILVR.BO': 'Hindustan Unilever',
    'HCLTECH.BO': 'HCL Technologies',
    'AXISBANK.BO': 'Axis Bank',
    'BHARTIARTL.BO': 'Bharti Airtel',
    'POWERGRID.BO': 'Power Grid',
    'NTPC.BO': 'NTPC',
    'TECHM.BO': 'Tech Mahindra',
    'ITC.BO': 'ITC',
    'NESTLEIND.BO': 'Nestlé India',
    'ULTRACEMCO.BO': 'UltraTech Cement',
    'BAJAJFINSV.BO': 'Bajaj Finserv',
    'KOTAKBANK.BO': 'Kotak Mahindra Bank',
    'TITAN.BO': 'Titan Company',
    'MARUTI.BO': 'Maruti Suzuki'
}

dropdown_options = [{"label": name, "value": symbol} for symbol, name in bse_stocks.items()]
print(bse_stocks['ITC.BO'])
print(dropdown_options)#print(bse_stocks['TATAMOTORS.BO'])
#print(dir(bse_stocks))
def get_net_income(ticker, version="annual"):
    """
    Fetch Net Income from the income statement.
    
    Parameters:
    - ticker: yfinance.Ticker object
    - version: 'annual' or 'quarterly'
    
    Returns:
    - Net Income series or a fallback message
    """
    try:
        if version == "annual":
            income_stmt = ticker.income_stmt
        elif version == "quarterly":
            income_stmt = ticker.quarterly_income_stmt
        else:
            return "Invalid version specified. Use 'annual' or 'quarterly'."

        net_income = income_stmt.loc["Net Income"]
        return net_income
    except Exception as e:
        return f"Net income not available. Reason: {e}"
import yfinance as yf
print(get_net_income(yf.Ticker('TCS.BO')))

def get_stock_insights(ticker):
    insights = {
        "analyst_price_targets": ticker.analyst_price_targets,
        "earnings_estimate": ticker.earnings_estimate,
        "eps_trend": ticker.eps_trend,
        "eps_revisions": ticker.eps_revisions,
        "growth_estimates": ticker.growth_estimates,
        #"upgrades_downgrades": ticker.upgrades_downgrades,
        #"insider_transactions": ticker.insider_transactions,
        #"insider_purchases": ticker.insider_purchases,
        #"institutional_holders": ticker.institutional_holders,
        "sustainability": ticker.sustainability,
    }
    return insights

print(get_stock_insights(yf.Ticker('TCS.BO')))
