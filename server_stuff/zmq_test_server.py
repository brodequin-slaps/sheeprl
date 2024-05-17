#!/usr/bin/env python

"""
synopsis:
    Create in-process clients and workers.
    Implement a "proxy" to pass messages from clients to workers and back.
    Original author: "Felipe Cruz <felipecruz@loogica.net>"
    Modified for async/ioloop: Dave Kuhlman <dkuhlman(at)davekuhlman(dot)org>
usage:
    python asyncsrv.py
"""

import sys
import zmq
from zmq.asyncio import Context, Poller, ZMQEventLoop
import asyncio
import threading

__author__ = "Felipe Cruz <felipecruz@loogica.net>"
__license__ = "MIT/X11"
FRONTEND_ADDR = 'tcp://*:9999'
#FRONTEND_ADDR = 'inproc://frontend'
BACKEND_ADDR = 'inproc://backend'


DEBUG = False

all_obs = {}

def printdbg(*args):
    if DEBUG:
        print(*args)


class Client(object):
    """A client that generates requests."""
    def __init__(self, context, id):
        self.context = context
        self.id = id

    @asyncio.coroutine
    def run_client(self):
        socket = self.context.socket(zmq.REQ)
        identity = u'client-%d' % self.id
        socket.connect(FRONTEND_ADDR)
        print('Client %s started' % (identity))
        reqs = 0
        while True:
            reqs = reqs + 1
            msg = 'request # {}.{}'.format(self.id, reqs)
            msg = msg.encode('utf-8')
            printdbg('Client {} before sending request: {}'.format(
                self.id, reqs))
            yield from socket.send(msg)
            print('Client {} sent request: {}'.format(self.id, reqs))
            msg = yield from socket.recv()
            print('Client %s received: %s' % (identity, msg))
            yield from asyncio.sleep(1)
            printdbg('(run_client) client {} after sleep'.format(self.id))
        #socket.close()
        #context.term()
        printdbg('(run_client) client {} exiting'.format(self.id))


class Server(object):
    """A server to set up and initialize clients and request handlers"""
    def __init__(self, loop, context):
        self.loop = loop
        self.context = context

    def run_server(self):
        printdbg('(Server.run) starting')
        tasks = []
        frontend = self.context.socket(zmq.ROUTER)
        frontend.bind(FRONTEND_ADDR)
        backend = self.context.socket(zmq.DEALER)
        backend.bind(BACKEND_ADDR)
        task = asyncio.ensure_future(run_proxy(frontend, backend))
        tasks.append(task)
        printdbg('(Server.run) started proxy')
        # Start up the workers.
        for idx in range(5):
            worker = Worker(self.context, idx)
            task = asyncio.ensure_future(worker.run_worker())
            tasks.append(task)
            printdbg('(Server.run) started worker {}'.format(idx))

        printdbg('(run_server) tasks: {}'.format(tasks))
        printdbg('(Server.run) after starting clients')
        #frontend.close()
        #backend.close()
        #self.context.term()
        return tasks

data_sent = asyncio.Event()
data_ready = asyncio.Event()
sent = 0

class Worker(object):
    """A request handler"""
    def __init__(self, context, idx):
        self.context = context
        self.idx = idx

    @asyncio.coroutine
    def run_worker(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect(BACKEND_ADDR)
        print('Worker {} started'.format(self.idx))
        while True:
            ident, part2, msg = yield from worker.recv_multipart()
            print('Worker %s received %s from %s' % (self.idx, msg, ident))
            print('from thread ' + str(threading.get_ident()))
            #yield from asyncio.sleep(0.5)
            #if all_obs[ident]:
            #    yield from data_sent.wait() # wait for signal previous data was sent
            #else:
            #    all_obs[ident] = msg

            all_obs[ident] = msg
            
            if len(list(all_obs.keys())) == 3 and sent % 3 == 0:
                print('all there')
                yield from asyncio.sleep(0.5) # run the model
                data_ready.set() # signal to other coroutines that data is ready
            else:
                print('waiting: ' + str(all_obs))
                data_ready.clear()
                yield from data_ready.wait() # wait for signal that data is ready
            
            yield from worker.send_multipart([ident, part2, msg])
            all_obs[ident] = None
            sent += 1
            
        worker.close()


@asyncio.coroutine
def run_proxy(socket_from, socket_to):
    poller = Poller()
    poller.register(socket_from, zmq.POLLIN)
    poller.register(socket_to, zmq.POLLIN)
    printdbg('(run_proxy) started')
    while True:
        events = yield from poller.poll()
        events = dict(events)
        if socket_from in events:
            msg = yield from socket_from.recv_multipart()
            printdbg('(run_proxy) received from frontend -- msg: {}'.format(
                msg))
            yield from socket_to.send_multipart(msg)
            printdbg('(run_proxy) sent to backend -- msg: {}'.format(msg))
        elif socket_to in events:
            msg = yield from socket_to.recv_multipart()
            printdbg('(run_proxy) received from backend -- msg: {}'.format(
                msg))
            yield from socket_from.send_multipart(msg)
            printdbg('(run_proxy) sent to frontend -- msg: {}'.format(msg))


def run(loop):
    printdbg('(run) starting')
    context = Context()
    server = Server(loop, context)
    tasks = server.run_server()
    loop.run_until_complete(asyncio.wait(tasks))
    printdbg('(run) finished')


def main():
    """main function"""
    print('(main) starting')
    args = sys.argv[1:]
    if len(args) != 0:
        sys.exit(__doc__)
    try:
        loop = ZMQEventLoop()
        asyncio.set_event_loop(loop)
        printdbg('(main) before starting run()')
        run(loop)
        printdbg('(main) after starting run()')
    except KeyboardInterrupt:
        print('\nFinished (interrupted)')


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    main()
