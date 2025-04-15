import dash
import os
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
from dash import ctx
import dash_bootstrap_components as dbc
from scripts.fetch_data import get_nifty_50_data
from scripts.lstm import scale_and_save_data
# Import your model prediction functions
from scripts.lstm import predict_stock_price_lstm  
from scripts.monte_carlo import predict_stock_price_monte_carlo
from scripts.agent import analyze_stock_with_FinFusionAI


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment
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
# Define layout
app.layout = html.Div([
    html.H1("FinFusion Analyst", style={'textAlign': 'center'}),

    html.Label("Select Stock:"),
    dcc.Dropdown(
        id="stock-dropdown",
        options=dropdown_options,
        value="RELIANCE.NS",
        style={"width": "50%"}
    ),

    dcc.Loading(
    id="loading-stock-data",
    type="circle",
    children=[
        dcc.Graph(id="stock-chart")
        ]
    ),

    html.Div([
        html.Button("Predict (LSTM)", id="predict-lstm-button", n_clicks=0, className="btn btn-primary"),
        html.Button("Predict (Monte Carlo)", id="predict-mc-button", n_clicks=0, className="btn btn-secondary"),
        html.Button("Get AI Recommendation", id="ai-recommend-button", n_clicks=0, className="btn btn-success"),
    ], style={"margin": "20px 0"}),

    dcc.Loading(
        id="loading-prediction",
        type="circle",
        children=[
            dcc.Graph(id="prediction-chart")  # Prediction graph
        ]
    ),

    html.Div(id="ai-recommendation-container", style={"display": "none", "margin-top": "20px"}),
])

@app.callback(
    Output("stock-chart", "figure"),
    Input("stock-dropdown", "value"),
    
)
#def update_stock_chart(stock_symbol):
    #stock_data = yf.Ticker(stock_symbol).history(period="6mo")
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode="lines", name=stock_symbol))
    #fig.update_layout(title=f"{stock_symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (INR)")
    #return fig

def update_stock_chart(stock_symbol):
    # Automatically fetch and store raw data using Alpha Vantage
    get_nifty_50_data(stock_symbol)

    # Load the saved data from file (Alpha Vantage data)
    file_path = os.path.join("data/raw", f"{stock_symbol}.csv")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Create professional chart
    fig = go.Figure()

# Line chart for closing price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Close Price",
        line=dict(color="royalblue", width=2)
    ))

    fig.update_layout(
        title={
            "text": f"{stock_symbol} Stock Price (Alpha Vantage)",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Price (INR)", showgrid=True),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    return fig

# Callback to trigger prediction and update the prediction chart
@app.callback(
    Output("prediction-chart", "figure"),
    Input("predict-lstm-button", "n_clicks"),
    Input("predict-mc-button", "n_clicks"),
    Input("stock-dropdown", "value"),
)
def update_prediction_chart(n_clicks_lstm, n_clicks_mc, selected_stock):
    """Updates the chart based on which prediction button is clicked."""

    # Detect which button was clicked
    triggered_id = ctx.triggered_id  

    if triggered_id == "predict-lstm-button" and n_clicks_lstm > 0:
        return predict_stock_price_lstm(selected_stock)
    elif triggered_id == "predict-mc-button" and n_clicks_mc > 0:
        return predict_stock_price_monte_carlo(selected_stock)
    
    return go.Figure()  # Empty figure initially

@app.callback(
    Output("ai-recommendation-container", "children"),
    Output("ai-recommendation-container", "style"),  # Show the AI output div
    Input("ai-recommend-button", "n_clicks"),
    Input("stock-dropdown", "value"),
    prevent_initial_call=True  # Prevent execution on app load
)
def update_ai_recommendation(n_clicks_ai, selected_stock):
    """Fetches AI recommendation and displays it in a styled card format."""
    triggered_id = ctx.triggered_id
    # Get the AI-generated stock recommendation
    
    recommendation_text = analyze_stock_with_FinFusionAI(selected_stock, bse_stocks[selected_stock])
    # Define card layout with a professional look
    recommendation_card = dbc.Card(
        dbc.CardBody([
            html.H4(f"AI Stock Recommendation for {selected_stock}", className="card-title"),
            html.P(recommendation_text, className="card-text"),
        ]),
        style={"margin": "10px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "5px"}
    )
    
    if triggered_id == "ai-recommend-button" and n_clicks_ai > 0:
        
        return recommendation_card, {"display": "block"}

# Run app
if __name__ == '__main__':
    app.run(debug=True)