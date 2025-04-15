import yfinance as yf

def get_net_income(ticker, version="annual"):

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
    
print(get_net_income(yf.Ticker('TCS.BO')))