#!/usr/bin/env python3

from multiprocessing.managers import BaseManager
from queue import Queue

class QueueManager(BaseManager):
    pass

image_queue = Queue()
QueueManager.register('get_image_queue', callable=lambda:image_queue)
manager = QueueManager(address=('', 4081), authkey=b'daisy')

print("Server Started")
server = manager.get_server()
server.serve_forever()
