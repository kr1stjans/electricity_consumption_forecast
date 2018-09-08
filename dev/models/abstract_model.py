class AbstractModel:

    def __init__(self) -> None:
        self.model = None

    def get_prediction(self, x_values, y_values, train_end_index, consumer_index):
        pass

    def transform_data(self, df):
        pass


    def get_name(self):
        return "abstract"
