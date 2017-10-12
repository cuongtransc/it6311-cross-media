#!/usr/bin/env python3

import datetime
import logging
import os
import sys
import time

from queue import Queue
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlretrieve

# Config logging
logger = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stderr, level=getattr(logging, 'DEBUG'))
FILE_LOG = 'logs/download_{}.log'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logging.basicConfig(filename=FILE_LOG, filemode='w', level=logging.INFO)


def download_image(image_url, image_local_path):
    try:
        urlretrieve(image_url, image_local_path)
    except FileNotFoundError as err:
        logger.error(err, image_url)   # something wrong with local path
    except HTTPError as err:
        logger.error(err, image_url)  # something wrong with url


class ProcessWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            cate, image_filename, image_url = self.queue.get()

            data_dir = '{}/{}'.format('images', cate)
            image_local_path = '{}/{}'.format(data_dir, image_filename)

            try:
                download_image(image_url, image_local_path)
            except Exception as e:
                logger.exception(
                    'Error when download image: {}'.format(image_url))

            self.queue.task_done()


def main(data_urls_filename):
    t_start = time.time()

    # create a queue to communicate with the worker threads
    queue = Queue()

    # create 8 worker threads
    N_THREADS = 8
    for x in range(N_THREADS):
        worker = ProcessWorker(queue)

        # Setting daemon to True will let the main thread exit even though the workers
        # are blocking
        worker.daemon = True
        worker.start()


    # Put the tasks into the queue as a tuple
    with open(data_urls_filename) as f:
        for line in f:
            # logger.info('Queueing {}'.format(row))
            queue.put(line.strip().split(','))

    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()

    t_end = time.time()
    print('Took {}s'.format(t_end - t_start))


if __name__ == '__main__':
    # data_urls_filename = 'train_urls.txt'
    data_urls_filename = 'train_urls_1.txt'

    main(data_urls_filename)
