import os
from dotenv import load_dotenv

load_dotenv()


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY")
#NEWS_API_KEY= os.getenv("NEWS_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

