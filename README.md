🌫️ Air Pollution Forecast & Personal Exposure Risk Dashboard

This project is an end-to-end machine learning–based system that forecasts air pollutant levels and estimates personalized health risk using a novel Personal Exposure Index (PEI). Instead of only relying on city-level air quality indicators, the system incorporates individual lifestyle factors to provide more realistic pollution exposure insights.

🚀 Key Features

📈 Forecasting of air pollutant concentrations using LightGBM

🧪 Synthetic data generation to handle limited and imbalanced exposure samples

🧠 AI-assisted feature engineering and model experimentation

👤 Personal Exposure Index (PEI) to estimate individual pollution risk

📊 Interactive dashboard for real-time visualization

📍 Location-based pollutant monitoring station data

🧠 Machine Learning Pipeline

Data collection from air quality monitoring stations

Data cleaning and preprocessing

Synthetic data generation to improve model generalization

Feature engineering (pollutant levels, time outdoors, activity patterns, etc.)

Training LightGBM regression models for pollutant / exposure prediction

Model evaluation using MAE, RMSE, and R² metrics

Integration of predictions into an interactive dashboard

📊 Personal Exposure Index (PEI)

PEI is a composite score calculated using:

Forecasted pollutant concentration

Duration of outdoor exposure

Activity type (commute, walking, indoor/outdoor)

User-defined lifestyle parameters

This allows personalized risk categorization instead of generic city-level air quality indicators.

🛠️ Tech Stack

Python

LightGBM

Pandas, NumPy, Scikit-learn

Streamlit (for dashboard)

Plotly (for visualizations)

📌 Use Cases

Personal health risk awareness

Smart city pollution monitoring systems

Environmental health research prototypes

Decision-support dashboards for urban planning

🔮 Future Improvements

Integration of official AQI calculation (CPCB standard)

Real-time API-based data ingestion

Mobile-friendly frontend

Map-based visualization

Health advisory recommendations based on PEI
