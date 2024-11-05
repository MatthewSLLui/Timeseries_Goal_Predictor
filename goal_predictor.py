import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
file_path = '/Users/user/Downloads/Football Analytics/Football final/Fulham_Synthetic_Data.csv'  # Insert local path to Fulham_synthetic_data
data = pd.read_csv(file_path) #create a dataframe 


# Data preprocessing
data['date'] = pd.to_datetime(data['date'])  # Convert into date format 
data.set_index('date', inplace=True)  # Use 'date' as the index 

# Train / test splitting.
train_data = data[:'2021-07-31']  # First 3 seasons for training
test_data = data['2021-08-01':'2022-05-31']  # Data from August 2021 to end of season


# Extract opponent strength as an exogenous variable
exog_train = train_data[['opponent_strength']]
exog_test = test_data[['opponent_strength']]

#Run seasonal ARIMA model
print("Model Running ... Please wait")
model = SARIMAX(train_data['fulham_goals'], exog=exog_train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 52))
sarimax_result = model.fit()
print("Model fitted successfully.")
print(sarimax_result.summary())

# Generate forecast for 36 weeks from August 2021
print("Generating forecast for 36 weeks from August 2021...")
future_forecast = sarimax_result.get_forecast(steps=36, exog=exog_test)  # Forecast for 36 weeks
forecast_values = future_forecast.predicted_mean
conf_int = future_forecast.conf_int()


# Generate future dates starting from August 1, 2021 for indexing
future_index = pd.date_range(start="2021-08-01", periods=36, freq='W')

# Data visualisation:
plt.figure(figsize=(12, 6))

# Plot historical data (2018 to July 2021)
plt.plot(train_data.index, train_data['fulham_goals'], label='Historical Data (2018 to July 2021)', color='blue')

# Plot forecast from August 2021 to End of Season, including confidence level
plt.plot(future_index, forecast_values, label='Forecast for Aug 2021 to End of Season', color='green')
plt.fill_between(future_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='lightgreen', alpha=0.3)

# Plot actual data from August 2021 to end of season for comparison
plt.plot(test_data.index, test_data['fulham_goals'], 
         linestyle='-', color='red', label='Actual Data for Aug 2021 to End of Season')


plt.title('Times Series Forecasting of Fulham Goals')
plt.xlabel('Date')
plt.ylabel('Goals Scored')
plt.legend()
plt.grid(True)
plt.show()



