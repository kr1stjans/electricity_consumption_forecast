class AbstractModel:
    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):
        raise NotImplementedError("Should have implemented this")

    @staticmethod
    def transform_data(df):
        raise NotImplementedError("Should have implemented this")
