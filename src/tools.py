import pyodbc
import os


class Connection(object):
    """
    Wrapper for database related operations
    """

    def __init__(self):
        self.conn = pyodbc.connect(DRIVER=os.environ.get("FORECAST_API_DB_DRIVER"),
                                   SERVER=os.environ.get("FORECAST_API_DB_HOST") + "," + os.environ.get(
                                       "FORECAST_API_DB_PORT"),
                                   DATABASE=os.environ.get("FORECAST_API_DB_NAME"),
                                   UID=os.environ.get("FORECAST_API_DB_USER"),
                                   PWD=os.environ.get("FORECAST_API_DB_PASSWORD"))
        self.curr = self.conn.cursor()
        self.curr.fast_executemany = True

    def connection(self):
        return self.conn

    def cursor(self):
        return self.curr

    def close(self):
        if self.curr is not None:
            self.curr.close()
        if self.curr is not None:
            self.conn.close()
