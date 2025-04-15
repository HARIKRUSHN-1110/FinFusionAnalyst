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


def get_recommendations(company_symbol):
    ticker = yf.Ticker(company_symbol)  # Create a Ticker object
    rc = ticker.get_recommendations()  # Get recommendations
    rcsummary = ticker.get_recommendations_summary()  # Get recommendation summary
    return rc, rcsummary

def analyze_stock_with_news(company, company_symbol):
    news = get_yfinance_news(company)  # Ensure this function is correctly implemented
    rc, rcsummary = get_recommendations(company_symbol)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are an advanced AI financial analyst. Analyze {company_symbol} stock using the latest news and provide a final Buy/Hold/Sell recommendation with reasoning."
            },
            {
                "role": "user",
                "content": f"Latest News for {company}:\n{news}\n\n"
                           f"Stock Recommendations:\n{rc}\n\n"
                           f"Summary of Recommendations:\n{rcsummary}\n\n"
                           f"Based on this, provide a final Buy/Hold/Sell recommendation with reasoning."
            }
        ],
        model='deepseek-r1-distill-qwen-32b',
    )

    return chat_completion.choices[0].message.content

#if __name__ == "__main__":
#    company = "RELIANCE INDUSTRIES"
#    company_symbol = "RELIANCE.NS"  # Example: Reliance Industries (NSE)
#    recommendation = analyze_stock_with_news(company, company_symbol)
#    print(f"**Stock Recommendation for {company}:**\n\n{recommendation}")