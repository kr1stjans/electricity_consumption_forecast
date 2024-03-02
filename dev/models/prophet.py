import pandas as pd

from fbprophet import Prophet

from dev.models.abstract_model import AbstractModel


class ProphetModel(AbstractModel):

    def get_prediction(self, x_values, y_values, train_end_index, consumer_index):
        m = Prophet()
        m.fit(pd.DataFrame({'ds': y_values[:train_end_index].index, 'y': y_values[:train_end_index].values}))
        m.predict()
        future = m.make_future_dataframe(periods=48, freq='30min', include_history=False)
        forecast = m.predict(future)
        return forecast['yhat']

    def transform_data(self, df):
        return df['value'], df['value']

    def get_name(self):
        return "baseline"
