class Statistics(object):

    def __init__(self) -> None:
        self.average = 0
        self.average_cnt = 0

    def update_average(self, value):
        self.average += value
        self.average_cnt += 1

    def get_average(self):
        return self.average / self.average_cnt
