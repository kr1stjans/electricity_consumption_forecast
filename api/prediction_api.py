import threading
from queue import Queue, Empty

import falcon
import json
import logging
from database.database_manager import DatabaseManager
from prediction.prediction_manager import PredictionManager


class PredictionApi:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # thread safe queue
        self.requests_queue = Queue()

        self.processing_thread = threading.Thread(target=self.__process)
        self.processing_thread.setDaemon(True)

    def on_post(self, req, resp):
        # convert data to json
        data = json.loads(req.stream.read().decode('utf-8'))

        # add to queue
        self.requests_queue.put(data)

        # run processor if stopped
        if not self.processing_thread.is_alive():
            self.processing_thread.start()

        self.logger.info("Received %s data sources", len(data['dataSourceValuesGroupedByDate']))
        resp.status = falcon.HTTP_200

    def __process(self):
        db_manager = DatabaseManager()
        prediction_manager = PredictionManager()

        while True:
            self.logger.info("%s items remaining", self.requests_queue.unfinished_tasks)

            try:
                data = self.requests_queue.get()
            except Empty:
                break

            # update data in the database
            db_manager.update_raw_data(data)

            # initiate prediction for updated measurement places
            prediction_manager.predict(list(data['dataSourceValuesGroupedByDate'].keys()), data['source'])
            self.requests_queue.task_done()


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filename='logs.log',
                    filemode='w')
api = falcon.API()
api.add_route('/newData', PredictionApi())
