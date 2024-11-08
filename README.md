# ARIMA football goal predictor

This project uses a Seasonal ARIMA (SARIMA) model to forecast Fulham Football Club's weekly goal counts for an upcoming season. All data used is synthetic.

The 'Fulham_Synthetic_Data' file contains the dates, goals scored, opponent played against (along with their strenght ratings). It made the assumption that the order of play for each season is identical (e.g. first opponent of 2018 is Norwich, the corresponding first opponent in 2019 would also be the same). 

The SARIMA model generates the forecasted goals for the 2021 season based on the prehistoric training data from the years 2018-2020. The model uses opponent strength as an exogenous predictor.

