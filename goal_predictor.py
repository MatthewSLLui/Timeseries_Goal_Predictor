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


# Define hyperparameter ranges
p = d = q = range(0, 3)  # Autoregressive, differencing, and moving average orders
P = D = Q = range(0, 2)  # Seasonal orders
s = [40]  # Seasonal period 

# Generate all possible combinations of hyperparameters
parameters = list(itertools.product(p, d, q, P, D, Q, s))

# Track the best model
best_aic = float('inf')
best_params = None
best_model = None

# Grid search over different parameter combinations
for param in parameters:
    try:
        
        order = (param[0], param[1], param[2])
        seasonal_order = (param[3], param[4], param[5], param[6])
        
        # Define SARIMAX model
        model = SARIMAX(train_data['fulham_goals'], exog=exog_train, order=order, seasonal_order=seasonal_order)
        result = model.fit(disp=False)
        
        # Check if current model has the lowest AIC
        if result.aic < best_aic:
            best_aic = result.aic
            best_params = param
            best_model = result
            
        
    
    except Exception as e:
        print(f"Failed to fit SARIMAX{order}x{seasonal_order}. Error: {e}")
        continue

# Generate future dates starting from August 1, 2021 for indexing

print(f"\nBest SARIMAX{(best_params[0], best_params[1], best_params[2])}x{(best_params[3], best_params[4], best_params[5], best_params[6])} - AIC:{best_aic}")

print("Generating forecast for 36 weeks from August 2021...")
future_forecast = best_model.get_forecast(steps=36, exog=exog_test)  # Forecast for 36 weeks
forecast_values = future_forecast.predicted_mean
conf_int = future_forecast.conf_int()

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



