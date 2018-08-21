Time series are everywhere around us. Every physical measurement can be listed in time order, which is effectively a time series.
By improving our ability to forecast these time series, we are directly improving our decisions that are based on these forecasts.
Time series are widely used in statistics, finance, meteorology, astronomy and energy sector. We will primarily focus on electricity consumption forecasting.

Stability and response of the electrical grid is directly connected with our ability to forecast electricity consumption.
Improved forecast accuracy leads to more effective electricity pricing and minimizes the chance of grid overload. Grid managers can 
order additional resources or motivate consumers to consume less in electricity in a variety of innovative products. [reference]

We based our analysis on the public data set to ensure easy reproducibility of the results. Data set includes 1000 consumer electricity consumption since October 2012 to March 2014. 
Data set also includes weather and holiday data for the same period. Consumer measurements in the data set were collected once everyday. 
This is currently the standard in energy sector, because the device technology is not advanced enough to stream data in real time.
This is an important disctintion, because we won't have access to n-1 data point in time series. New measurements will only be available the next day and we must take this into the account. Data interval is 30 minutes. 
We will focus on short term forecasting (one day ahead), because this setting of forecast accuracy and time to act is the most favored byy the grid managers.
Medium and long term forecasting will have lesser accuracy, but will provide grid managers more time to act. However they usually don't need more time, but they do need better accuracy.

We started our analysis on previously discovered[refenrence to diploma] best models for electricity consumption forecasting.


We used programming language Python 3.7.


Previous research came to conclusion that neural networks had too big time complexity to be effective on this size of data set.
We solved this problem by using external graphical processing units (3x 1080TI).

*long short-term memory network (LSTM), which learns both short-term and long-term memory by enforcing constant error
flow through the designed cell state.

* data dependency

* feature creation

* discrete napovedne spremenljivke

* trend as variable

* aggregated analysis (half hourly, daily, monthly)

* add typical day (1 = monday, 2 = tuesday, wednesday, thursday, 3=friday, 4=saturday, 5= sunday, holiday)

* relevant neural network: Leta 2002 je bil objavljen članek »Electric energy demand forecasting with neural
networks« [10], kjer je prikazana uporaba nevronskih mrež za napovedovanje porabe
električne energije. Rezultati nastavljene trinivojske nevronske mreže so v večini testov
dosegali relativno napako do 5% .

* describe preprocessing and timeseries cross validation
* consumer vs aggregated forecast

* Uporabili so tudi vrsto obdelav podatkov za odstranjevanje
vplivov sezonskosti, trendov in moˇcne variance, katere so se do danes izkazale
kot uspeˇsni pristopi za izboljˇsanje napovedi ˇcasovnih vrst z LSTM

* remove trend
* check stationarity = is ok?
* standardize between -1 and 1