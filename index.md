# ENSOPredict

Weather forecasting plays an important role in mitigating weather-related risks since our everyday wellbeing will always be affected by weather, whether it in short term or long term period. 

For example, weather impacts cross-boundary transportation schedule which has snowball effect on commodity prices. And then for business planning in agriculture, with seasons forecast, the government can make strategic decision for drought or flood prevention to minimize harvest failure risk or plan for better food sustainability. Likewise, weather forecast can also help government to plan for spikes in energy demand and anticipate availability of renewable energy.

So how do contemporary climate scientists forecast weather? It is by using theory based or physical models which are provided by only few agencies, one of the prominent one is NOAA-GEFS, but it is computationally expensive.

Therefore, in this EnsoPredict project, we tried to use machine learning models which can run prediction under offline environment. Hopefully with this project we can get some idea on machine learning performance as one alternative of physical based model. 

Just as the name suggest, EnsoPredict will focus on doing El-Nino prediction which is one of major weather anomaly that has widespread impact on people's socioeconomic. In short, it is a condition where a pool of warm water from western pacific ocean moves to eastern pacific, and this anomaly occurs every 2-7 years. This movement pattern is called El Nino Southern Oscillation (or ENSO) which is what we're trying to predict in this project.

EnsoPredict will try to monitor anomalies in equatorial Pacific ocean's Sea Surface Temperature by measuring its Nino3.4 index using regression Convolutional Neural Network model.

One thing to keep in mind is that EnsoPredict does not doing forecasting. It is doing hindcast or prediction on past data.

