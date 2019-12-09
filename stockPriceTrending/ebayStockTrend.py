
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import pandas as pd

stockDF = pd.read_csv("../Data/NYStockMarket/prices.csv",sep=",")
ebayStocks = stockDF[stockDF["symbol"] == "EBAY"]
ebayStocks.drop(columns="symbol")
ebayStocks["date"] = pd.to_datetime(ebayStocks['Date'])
"""
On the New York Stock Exchange, 
I tried to modelize the stock market trends of eBay from 2010 to 2015 
and tried to forecast the trend graph in 2016 from the model I buiilt.
"""

#Metrics that I used to evaulate the model are MAPE(Mean Absolute Percentage Error)-
#MSE(Mean Square Error)-RMSE(Rooted Mean Square Error)
#Training set : 2010-01-01 to 2017-01-01
train_X = ebayStocks[ebayStocks["date"]]

features = ["date","close"] #ebayStocks.columns.tolist()

#Split
#Last-Value Method based prediction.
train_X= train_X[features]
forecastHorizon = 24
forecastModel = Prophet()
forecastModel.fit(train_X)

#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#Trend lines and confident intervals can be found in forecast object's above attribues

#Eliminate days that are weekend because stock market is closed.
future = forecastModel.make_future_dataframe(periods=40)
future['day'] = future['ds'].dt.weekday
future = future[future['day']<=4]
forecast = forecastModel.predict(future)

#Plot stock trend forecasted for specified horizon above.
forecastPlot = forecastModel.plot(forecast)
forecastComponentPlot = forecastModel.plot_components(forecast)
plotWithChangepoints = add_changepoints_to_plot(forecastPlot.gca(), forecastModel, forecast)

#in Prophet model constructor, change_prior_scale argument control how much flexibility is allowed for changepoints.
#Increasing that parameter means that model is that much more flexible, otherwise less flexible.
#This parameter have an effect on avoiding overfitting data. Therefore , It should be considered carefully.
#Prophet defaults that parameter to 0.05

change_point_scale = [0.00,0.25,0.50,0.75,1.00,1.25,1.50]
performances = []
for changePoint in change_point_scale:
    forecastModelCP = Prophet(changepoint_prior_scale=changePoint)
    forecastModelCP.fit(train_X)
    forecastCV = cross_validation(forecastModelCP, horizon='40 days')
    forecastPM = performance_metrics(forecastCV)
    performances.append((forecastPM,forecastCV))

#The plots in list below demonstrate delta in MSE error across specified horizon
performancesPlotAcrossHorizon = [plot_cross_validation_metric(model(1),metric="mse") for model in performances]

#These plots and performance metrics can help to decide which change_point_scale I should use


