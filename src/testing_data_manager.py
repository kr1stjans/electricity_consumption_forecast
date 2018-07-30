from operator import itemgetter

import numpy as np
from src.tools import Connection


class DataManager(object):
    """
    DataManager is used for data retrieval. It returns either already preprocessed cached data or it fetches it from DB 
    and prepossesses it.
    """

    def __init__(self) -> None:
        self.connection = Connection()
        self.conn = self.connection.connection()
        self.curr = self.connection.cursor()

    @staticmethod
    def print_cooficients(columns, coef):
        result = list(zip(columns, [abs(x) for x in coef]))
        result.sort(key=itemgetter(1), reverse=True)
        result = [(x, '{0:.2f}'.format(y)) for (x, y) in result]
        for cooef in result:
            print(cooef)

    def get_data(self, mm_id):
        '''
        Method gets measurements and weather data from 1.1 2016 to 1.5.2017 sorted by date and concatenates them together.
        :param mm_id:
        :return:
        '''
        self.curr.execute("SELECT value FROM MERGED_DATA WHERE mm_id=? ORDER BY dt ASC", mm_id)
        measurements = np.array(self.curr.fetchall())

        self.curr.execute(
            'SELECT Cas, Temperatura, Obsevanje, Veter, smer_vetra, vetrni_potencial, relativna_vlaga '
            'FROM Vreme ORDER BY Cas ASC')
        weather = np.array(self.curr.fetchall())

        return np.hstack((measurements, weather))
