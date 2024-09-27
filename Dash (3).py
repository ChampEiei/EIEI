import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash.dash_table import DataTable
from prophet import Prophet

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and process the data
df = pd.read_excel("frank.xlsx")
future = pd.read_excel("future.xlsx")
t = pd.read_excel("C:/Users/ExPertComputer/Downloads/Book1.xlsx")
t['Growth Rate%']=t['Growth Rate%']*100
t['Growth Rate most likely']=[4.9,13.86,1.06,0.94,1.29]

# Group and process your data for Prophet forecasting
fil = df.groupby(['Start Date', 'P&L Type'])['margin'].sum().reset_index()
fil = fil[(fil['Start Date'].dt.year != 2024) & (fil['Start Date'].dt.year != 2025)]

# Group by 'Start Date' and sum the 'margin'
df = fil.copy()
df = df.groupby(['Start Date'])['margin'].sum().reset_index()

# Group by year and month
df = df.groupby(df['Start Date'].dt.to_period('M'))['margin'].sum().reset_index()

# Convert the period back to a timestamp for easier manipulation
df['Start Date'] = df['Start Date'].dt.to_timestamp()

# Extract unique activities from the P&L Type column
activities = fil['P&L Type'].unique()

# Define styles for the dashboard theme
theme_style = {
    'background-color': '#f0f8ff',
    'padding': '20px',
    'font-family': 'Arial, sans-serif'
}

header_style = {
    'font-size': '36px',
    'font-weight': 'bold',
    'color': '#1f77b4',
    'text-align': 'center',
}

table_style = {
    'background-color': 'white',
    'padding': '10px',
    'border': '1px solid #ddd'
}

# Create a layout with a dropdown and a callout for metrics
app.layout = html.Div([
    html.H1("Margin Forecast Dashboard", style=header_style),
    
    dcc.Dropdown(
        id='activity-dropdown',
        options=[{'label': activity, 'value': activity} for activity in activities] + [{'label': 'All', 'value': 'All'}],
        value='All',  # Default value is 'All'
        style={'width': '50%', 'margin-bottom': '20px'}
    ),
    
    # Callout to show total margin
    html.Div(id='callout-div', style={'font-size': '24px', 'color': 'black', 'margin-bottom': '30px'}),

    # Two-column layout for forecast graph and pie chart
    html.Div([
        dcc.Graph(id='forecast-graph', style={'width': '50%'}),
        dcc.Graph(id='pie-chart', style={'width': '50%'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    # Two-column layout for bar chart and table
    html.Div([
        dcc.Graph(id='growth-bar-chart', style={'width': '50%', 'padding-right': '20px', 'height': '600px'}),  # Adjusted height
        
        html.Div([
            html.H3("Revenue Data", style={'text-align': 'center'}),
            DataTable(
                id='revenue-table',
                columns=[{"name": i, "id": i} for i in t.columns],
                data=t.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#f0f8ff',
                    'fontWeight': 'bold',
                    'border': '1px solid black'
                },
                style_cell={
                    'textAlign': 'center',
                    'border': '1px solid black'
                }
            )
        ], style={'width': '50%', 'height': '600px'})  # Adjusted height
    ], style={'display': 'flex', 'flex-direction': 'row', 'margin-top': '50px'}),

    # Add assumptions text below the table
    html.Div([
        html.H3("Assumption Customer Data", style={'margin-top': '50px'}),
        html.P("The model includes both new and existing customers based on past trends."),
        html.H4("New Customers"),
        html.P("No adjustments are made for new customer growth as it is already factored into the forecast."),
        html.H4("Old Customer Retention"),
        html.P("Assumes a lower churn rate for old customers due to retention strategies."),
        html.H4("Revenue Growth"),
        html.P("Forecasted profits will be adjusted by the improved retention rate of old customers, without changing new customer impact.")
    ], style={'margin-top': '20px', 'padding': '10px', 'border': '1px solid #ddd', 'background-color': '#f9f9f9'})

], style=theme_style)

# Callback to update the forecast graph, pie chart, bar chart, and table based on selected activity
@app.callback(
    [Output('forecast-graph', 'figure'), 
     Output('callout-div', 'children'), 
     Output('pie-chart', 'figure'), 
     Output('growth-bar-chart', 'figure'), 
     Output('revenue-table', 'data')],
    [Input('activity-dropdown', 'value')]
)
def update_dashboard(selected_activity):
    if selected_activity == 'All' or not selected_activity:
        # If "All" is selected or no activity is selected, sum over all activities
        filtered_data = fil.groupby(fil['Start Date'].dt.to_period('M'))['margin'].sum().reset_index()
    else:
        # Filter data based on the selected activity
        filtered_data = fil[fil['P&L Type'] == selected_activity]
        filtered_data = filtered_data.groupby(filtered_data['Start Date'].dt.to_period('M'))['margin'].sum().reset_index()

    # Convert period back to timestamp for easier plotting
    filtered_data['Start Date'] = filtered_data['Start Date'].dt.to_timestamp()

    # Prepare the data for Prophet (Prophet requires columns named 'ds' and 'y')
    df_prophet = filtered_data.rename(columns={'Start Date': 'ds', 'margin': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df_prophet)
    
    # Create a dataframe with future dates for the next 12 months
    future_dates = model.make_future_dataframe(periods=12, freq='MS')
    
    # Forecast the future values
    forecast = model.predict(future_dates)
    
    # Separate the historical and predicted data
    historical_data = forecast[forecast['ds'] <= df_prophet['ds'].max()]
    predicted_data = forecast[forecast['ds'] > df_prophet['ds'].max()]
    predicted_data['yhat'] = predicted_data['yhat']*1.49
    forecast=forecast[:-12]
    # Create the scatter plot with the historical data in blue and the forecast in red
    fig = px.scatter(forecast, x='ds', y='yhat', title=f'Margin Forecast for {selected_activity}', trendline='ols')
    fig.add_scatter(x=predicted_data['ds'], y=predicted_data['yhat'], mode='markers', marker=dict(color='red'), name='Predicted')

    # Calculate a metric to display in the callout (e.g., the total forecast for the next 12 months)
    total_forecast = forecast['yhat'].sum()
    callout_text = f'Total forecasted margin for {selected_activity}: {total_forecast:,.2f}'

    # Prepare the data for the pie chart, showing the sum of forecasted margin by P&L Type for 2024
    future_2024 = future[future['ds'].dt.year == 2024]
    pie_fig = px.pie(future_2024, values='yhat', names='P&L Type', title='Forecasted Margin by P&L Type for 2024',hole=0.3)

    # Bar chart for comparing growth
    # Merge t with future_2024 on P&L Type to get the growth rate
    merged_data = pd.merge(future_2024, t, on='P&L Type')
    
    # Calculate values after applying the growth rate
    merged_data['Adjusted yhat'] = merged_data['yhat'] * (1 + merged_data['Growth Rate%'] / 100)
    #merged_data['Growth Rate most likely'] = merged_data['yhat']  * (1 + merged_data['Growth Rate most likely'] / 100)
    # Calculate the increase in margin
    #merged_data['Increasing'] = merged_data['Adjusted yhat'] - merged_data['yhat']
    merged_data['Growth Rate most likely'] = merged_data['yhat']  * (1 + merged_data['Growth Rate most likely'] / 100)

    # Create bar chart comparing original, adjusted values, and the increase
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=merged_data['P&L Type'], y=merged_data['yhat'], name='Original Forecast'))
    bar_fig.add_trace(go.Bar(x=merged_data['P&L Type'], y=merged_data['Adjusted yhat'], name='Adjusted Forecast'))
    bar_fig.add_trace(go.Bar(x=merged_data['P&L Type'], y=merged_data['Growth Rate most likely'], name='Adjusted Forecast'))
    #bar_fig.add_trace(go.Bar(x=merged_data['P&L Type'], y=merged_data['Increasing'], name='Increase'))

    # Set title and layout for bar chart
    bar_fig.update_layout(
        title="Comparison of Original, Adjusted Forecasts, and Increase by P&L Type",
        xaxis_title="P&L Type",
        yaxis_title="Margin",
        barmode='group'
    )

    # Update table data
    table_data = t.to_dict('records')

    return fig, callout_text, pie_fig, bar_fig, table_data

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=8066)