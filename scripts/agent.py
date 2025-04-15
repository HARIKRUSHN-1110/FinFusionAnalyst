import os
import requests
import json
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
)
 
def get_yfinance_news(company):
    stock = yf.Ticker(company)
    news_data = yf.Search(company, news_count=10).news  # Get latest 5 news articles
    
    if not news_data:
        return "No recent news available."

    formatted_news = []
    for news in news_data:
        try:
            title = news.get("title", "No Title")
            publisher = news.get("publisher", "Unknown Source")
            link = news.get("link", "#")
            formatted_news.append(f"- [{title}]({link}) ({publisher})")
        except Exception as e:
            formatted_news.append(f"- [Error retrieving news] ({str(e)})")

    return "\n".join(formatted_news)


def get_recommendations(ticker):
    try:
        rc = ticker.get_recommendations()
    except Exception:
        rc = "Recommendations not available."

    try:
        rcsummary = ticker.get_recommendations_summary()
    except Exception:
        rcsummary = "Recommendation summary not available."

    return rc, rcsummary

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

#def get_earnings_and_financial_reports(ticker):
    financials = {}


    try:
        financials["balance_sheet"] = ticker.balance_sheet
    except Exception:
        financials["balance_sheet"] = "Not available"

    try:
        financials["cashflow"] = ticker.cashflow
    except Exception:
        financials["cashflow"] = "Not available"

    try:
        financials["quarterly_cashflow"] = ticker.quarterly_cashflow
    except Exception:
        financials["quarterly_cashflow"] = "Not available"

    return financials

#def get_stock_insights(ticker):
    insights = {
        "analyst_price_targets": ticker.analyst_price_targets,
        "earnings_estimate": ticker.earnings_estimate,
        "eps_trend": ticker.eps_trend,
        "eps_revisions": ticker.eps_revisions,
        "growth_estimates": ticker.growth_estimates,
        "upgrades_downgrades": ticker.upgrades_downgrades,
        "insider_transactions": ticker.insider_transactions,
        "insider_purchases": ticker.insider_purchases,
        "institutional_holders": ticker.institutional_holders,
        "sustainability": ticker.sustainability,
    }
    return insights
    
def analyze_stock_with_FinFusionAI(company, company_symbol):

    ticker = yf.Ticker(company_symbol)
    
    # Fetch data
    news = get_yfinance_news(company)
    rc, rcsummary = get_recommendations(ticker)
    #financials = get_earnings_and_financial_reports(ticker)
    #stock_insights = get_stock_insights(ticker)
    annual_income = get_net_income(ticker, version="annual")
    quarterly_income = get_net_income(ticker, version="quarterly")


    #Format insights into a readable prompt chunk
    #insight_summary = "\n".join([
        #f"- Analyst Price Targets: {stock_insights.get('analyst_price_targets')}",
        #f"- EPS Trend: {stock_insights.get('eps_trend')}",
        #f"- EPS Revisions: {stock_insights.get('eps_revisions')}",
        #f"- Growth Estimates: {stock_insights.get('growth_estimates')}",
        #f"- Insider Transactions: {stock_insights.get('insider_transactions')}",
        #f"- Institutional Holdings: {stock_insights.get('institutional_holders')}",
        #f"- Sustainability Info: {stock_insights.get('sustainability')}"
    #])

    # Build prompt
    prompt = (
        f"Latest News for {company}:\n{news}\n\n"
        f"Stock Recommendations:\n{rc}\n\n"
        f"Summary of Recommendations:\n{rcsummary}\n\n"
        #f"Financial Reports & Earnings:\n{financials}\n\n"
        f"Annual and quarterly Income of Company:\n{annual_income}&{quarterly_income}\n\n"
        #f"Additional Stock Insights:\n{insight_summary}\n\n"
        f"Based on the above data, provide a clear and concise investment recommendation "
        f"on whether to **Buy, Hold, or Sell** the stock. Justify your recommendation "
        f"in 5-6 sentences (~60-70 words max), focusing on short-term to mid-term reasoning."
    )

    # AI Completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI financial analyst. Analyze stocks based on real-time financials from internet sources, "
                    "news, insider activities, analyst targets, and sustainability factors. Give insightful yet concise guidance."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model='deepseek-r1-distill-llama-70b'
    )

    return chat_completion.choices[0].message.content


#if __name__ == "__main__":
#    company = "TATAMOTORS.BO"
#    company_symbol = "TATAMOTORS.NS"  # Example: Reliance Industries (NSE)
#    recommendation = analyze_stock_with_news(company, company_symbol)
#    print(f"**Stock Recommendation for {company}:**\n\n{recommendation}")