#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import argparse
import zmq
import time

parser = argparse.ArgumentParser(description="n")
parser.add_argument("--n", default=None, type=int, 
    help="n")
args = parser.parse_args()

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:8888")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print(f"Sending request {request} …")
    socket.send(b"Hello")

    #  Get the reply.
    message = socket.re
    print(f"Received reply {request} [ {message} ]")
