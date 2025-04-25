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
print(bse_stocks)
print(dropdown_options)#print(bse_stocks['TATAMOTORS.BO'])
#print(dir(bse_stocks))

