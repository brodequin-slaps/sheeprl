from sheeprl.cli import get_trained_agent_entrypoint_multienv
from omegaconf import DictConfig, OmegaConf, open_dict

import gymnasium as gym 
from sheeprl.utils.env import make_env
from sheeprl.utils.timer import timer
from torchmetrics import MaxMetric, MeanMetric, CatMetric
import pickle
import socketserver
import logging
import argparse
import threading
import numpy as np
import struct

import melee_env.env_v2
from melee_env.obs_space import obs_space_v1
from melee_env.act_space import act_space_v1

import time
import zmq

trained_checkpoint_path = "/home/sam/dev/sheeprl/trained_agent_checkpoint/2024-04-19_18-42-53_dreamer_v3_melee_7/version_0/checkpoint/ckpt_1550000_0.ckpt"

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="trained agent eval server (pass num envs)")
parser.add_argument("--num_envs", default=None, type=int, 
    help="how many obs to eval")
args = parser.parse_args()

num_envs = args.num_envs

global trained_agent
trained_agent = None

def get_formatted_obs(obs):
    formatted_obs = {}
    for obs_key in obs[0].keys():
        formatted_obs[obs_key] = np.array([obs[env_idx][obs_key] for env_idx in obs.keys()])
    
    return formatted_obs

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

obs_received_this_frame = 0
this_frame_all_obs = {}
this_frame_raw_actions = None


raw_actions_ready_lock = threading.Lock()
raw_actions_ready_cond = threading.Condition(lock=raw_actions_ready_lock)

this_frame_obs_lock = threading.Lock()
this_frame_obs_cond = threading.Condition(lock=this_frame_obs_lock)

class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
    

    def handle(self):
        global obs_received_this_frame
        global this_frame_all_obs
        global this_frame_raw_actions
        global raw_actions_ready_cond
        global raw_actions_ready_lock
        global this_frame_obs_lock
        global this_frame_obs_cond
        global trained_agent

        logger.info('handling')

        #data = self.request.recv(4096)
        data = recv_msg(self.request)

        #pickled_obs = self.rfile.readline().strip()
        obs = pickle.loads(data)
        env_idx = obs['env_idx']
        obs.pop('env_idx')

        with this_frame_obs_cond:
            if env_idx in this_frame_all_obs.keys():
                logger.info("Recieved 2 requests from same env before sending response")
                this_frame_obs_cond.wait()
                #this_frame_all_obs[env_idx] = None
                #raise RuntimeError("Recieved 2 requests from same env before sending response")
 
            this_frame_all_obs[env_idx] = obs
            obs_received_this_frame += 1

        if obs_received_this_frame == num_envs:
            formatted_obs = get_formatted_obs(this_frame_all_obs)
            this_frame_raw_actions = trained_agent.act_v2(formatted_obs)
            with raw_actions_ready_cond:
                obs_received_this_frame = 0
                this_frame_all_obs = {}
                raw_actions_ready_cond.notify_all()
        
        else:
            with raw_actions_ready_cond:
                raw_actions_ready_cond.wait()
        
        #this_request_action = {obs_key: this_frame_raw_actions[obs_key][env_idx] for obs_key in this_frame_raw_actions.keys()}
        this_request_action = this_frame_raw_actions[env_idx]
        send_msg(self.request, pickle.dumps(this_request_action, pickle.HIGHEST_PROTOCOL))
        with this_frame_obs_cond:
            this_frame_obs_cond.notify_all()
        #self.wfile.write(pickle.dumps(this_request_action, pickle.HIGHEST_PROTOCOL))



class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):    
    pass


def get_config(checkpoint_path, 
                capture_video = False,
                fabric_accelerator = "auto", 
                float32_matmul_precision = "high"):
    cfg = DictConfig({
        "disable_grads": True,
        "checkpoint_path": checkpoint_path,
        "env": {
            "capture_video": capture_video
        },
        "fabric": {
            "accelerator": fabric_accelerator
        },
        "float32_matmul_precision": float32_matmul_precision,
        "num_threads": 1,
        "seed": None
    })
    return cfg


def eval_server_v1():
    trained_agent_generator = get_trained_agent_entrypoint_multienv(get_config(trained_checkpoint_path), args.num_envs)
    trained_agent = trained_agent_generator(
        obs_space_v1.get_observation_space_v1(2),
        act_space_v1.get_action_space_v1(2))

    logging.basicConfig(filename='eval_server.log', level=logging.INFO)
    logger.info('eval_server started')

    HOST, PORT = "0.0.0.0", 9999

    # Create the server, binding to localhost on port 9999

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        #server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        #server_thread.daemon = True
        #server_thread.start()

        server.serve_forever()

def eval_server_v2():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print(f"Received request: {message}")

        #  Do some 'work'
        time.sleep(1)

        #  Send reply back to client
        socket.send(b"World")

if __name__ == "__main__":

    trained_agent_generator = get_trained_agent_entrypoint_multienv(get_config(trained_checkpoint_path), args.num_envs)
    trained_agent = trained_agent_generator(
        obs_space_v1.get_observation_space_v1(2),
        act_space_v1.get_action_space_v1(2))

    logging.basicConfig(filename='eval_server.log', level=logging.INFO)
    logger.info('eval_server started')

    HOST, PORT = "0.0.0.0", 9999

    # Create the server, binding to localhost on port 9999

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        #server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        #server_thread.daemon = True
        #server_thread.start()

        server.serve_forever()
    
    #server.shutdown()
    #logger.info('Finished')