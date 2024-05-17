import socket
import threading
import socketserver
import melee_env.env_v2
import pickle
import argparse
import hashlib
from sheeprl.utils.timer import timer
from torchmetrics import MaxMetric, MeanMetric
import time
import struct

parser = argparse.ArgumentParser(description="env num lolol")
parser.add_argument("--env_num", default=None, type=int, 
    help="the env num")
args = parser.parse_args()

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

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))

        print('sending: ' + str(message))
        send_msg(sock, message)
       
       
        #sock.sendall(message)

        response = recv_msg(sock)
        
        if response:
            unpickled = pickle.loads(response)
        else:
            unpickled = [0,0,0,0]
    
        print("Received: {}".format(unpickled))


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    obs = melee_env.env_v2.MeleeEnv_v2.get_observation_space_v1(2).sample()
    obs['env_idx'] = args.env_num

    message = pickle.dumps(obs, pickle.HIGHEST_PROTOCOL)
    
    print('msg size= ' + str(len(message)))
    
    print('msg hash=' + str(hashlib.md5(message).hexdigest()))

    for i in range(0,60):
        time.sleep(1)
        with timer("env_inference_time", MeanMetric):
            client(HOST, PORT, message)
        
    metrics = timer.compute()
    if "env_inference_time" in metrics:
        print('mean inference time = ' + str(metrics["env_inference_time"]))