import yfinance as yf
def get_recommendations(ticker):
    try:
        rc = ticker.get_recommendations()
    except Exception:
        rc = "Recommendations not available."

    try:
        rcsummary = ticker.get_recommendations_summary()
    except Exception:
        rcsummary = "Recommendation summary not available."

    return rc

print(get_recommendations(yf.Ticker('TCS.BO')))