import dash
import os
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # Required for deployment
bse_stocks = {
    'TATAMOTORS.BO': 'Tata Motors',
    'RELIANCE.BO': 'Reliance Industries',
    'TCS.BO': 'Tata Consultancy Services',
    'BAJFINANCE.BO': 'Bajaj Finance',
    'INFY.BO': 'Infosys Ltd.',
    'HDFCBANK.BO': 'HDFC Bank',
    'TATASTEEL.BO': 'Tata Steel',
    'LT.BO': 'Larsen & Toubro India',
    'ICICIBANK.BO': 'ICICI Bank',
    'INDUSINDBK.BO': 'IndusInd Bank',
    'SBIN.BO': 'State Bank of India',
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
    'ITC.BO': 'ITC Ltd.',
    'NESTLEIND.BO': 'Nestlé India',
    'ULTRACEMCO.BO': 'UltraTech Cement',
    'BAJAJFINSV.BO': 'Bajaj Finserv',
    'KOTAKBANK.BO': 'Kotak Mahindra Bank',
    'TITAN.BO': 'Titan Company',
    'MARUTI.BO': 'Maruti Suzuki India Pvt. Ltd.'
}

dropdown_options = [{"label": name, "value": symbol} for symbol, name in bse_stocks.items()]
# Define layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="📊 FinFusion Analyst",
        color="dark",
        dark=True,
        fluid=True,
        className="mb-4",
        brand_style={"fontWeight": "bold", "fontSize": "24px", "margin": "auto"},
        style={"justifyContent": "center"},
    ),

    dbc.Row([
        dbc.Col([
            html.Label("🔍 Select Stock:", className="text-white fw-bold"),
            dcc.Dropdown(
                id="stock-dropdown",
                options=dropdown_options,
                placeholder="Choose a stock...",
                value="RELIANCE.BO",
                className="mb-4",
                style={"backgroundColor": "#2c2c2c", "color": "black"},
            ),
        ], lg=4),
    ], justify="center"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Stock Price Overview", className="text-white"),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="stock-chart"), type="circle"),
                ])
            ], color="dark", inverse=True),
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Button("🔮 Predict (LSTM)", id="predict-lstm-button", color="primary", className="me-2", n_clicks=0),
            dbc.Button("📊 Predict (Monte Carlo)", id="predict-mc-button", color="info", className="me-2", n_clicks=0),
            dbc.Button("🤖 AI Recommendation", id="ai-recommend-button", color="success", n_clicks=0),
        ], width="auto", className="mb-3 text-center")
    ], justify="center"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📉 Prediction Output", className="text-white"),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="prediction-chart"), type="circle"),
                ])
            ], color="dark", inverse=True)
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Div(id="ai-recommendation-container", style={"display": "none"})
        ])
    ]),
], fluid=True)
@app.callback(
    Output("stock-chart", "figure"),
    Input("stock-dropdown", "value"),
    
)

def update_stock_chart(stock_symbol):
    # Automatically fetch and store raw data using Alpha Vantage
    get_nifty_50_data(stock_symbol)
    if not stock_symbol:
        raise PreventUpdate
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
        "text": f"{bse_stocks[stock_symbol]} Stock Price (Alpha Vantage)",
        "x": 0.5,
        "xanchor": "center"
    },
    xaxis=dict(title="Date", showgrid=True),
    yaxis=dict(title="Price (INR)", showgrid=True),
    hovermode="x unified",
    template="plotly_dark",
    plot_bgcolor="#1e1e2f",
    paper_bgcolor="#1e1e2f",
    font=dict(color="white"),
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


    # Detect which button was clicked
    triggered_id = ctx.triggered_id  

    if triggered_id == "predict-lstm-button" and n_clicks_lstm > 0:
        return predict_stock_price_lstm(selected_stock)
    elif triggered_id == "predict-mc-button" and n_clicks_mc > 0:
        return predict_stock_price_monte_carlo(selected_stock)
    
    return go.Figure()  # Empty figure initially


@app.callback(
    Output("ai-recommendation-container", "children"),
    Output("ai-recommendation-container", "style"),
    Input("ai-recommend-button", "n_clicks"),
    Input("stock-dropdown", "value"),
    prevent_initial_call=True
)
def update_ai_recommendation(n_clicks_ai, selected_stock):
    # If no stock is selected, don't update
    if not selected_stock:
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    # Only generate recommendation if the button was clicked
    if triggered_id == "ai-recommend-button" and n_clicks_ai and n_clicks_ai > 0:
        recommendation_text = analyze_stock_with_FinFusionAI(bse_stocks[selected_stock], selected_stock)

        recommendation_card = dbc.Card(
            dbc.CardBody([
                html.H4(f"AI Stock Recommendation for {bse_stocks[selected_stock]}", className="card-title"),
                html.P(recommendation_text, className="card-text"),
            ]),
            style={"margin": "10px", "padding": "15px", "border": "1px solid #ddd", "border-radius": "5px"}
        )

        return recommendation_card, {"display": "block"}
    
    # Return default state if button not clicked (still need a valid return)
    return dash.no_update, dash.no_update


# Run app
if __name__ == '__main__':
    app.run(debug=True)