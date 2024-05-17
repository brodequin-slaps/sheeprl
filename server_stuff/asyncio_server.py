import asyncio
import time

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

background_tasks = set()

trained_checkpoint_path = "/home/sam/dev/sheeprl/trained_agent_checkpoint/2024-04-19_18-42-53_dreamer_v3_melee_7/version_0/checkpoint/ckpt_1550000_0.ckpt"

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="trained agent eval server (pass num envs)")
parser.add_argument("--num_envs", default=None, type=int, 
    help="how many obs to eval")
args = parser.parse_args()

num_envs = args.num_envs

global trained_agent # remove global?
trained_agent = None

gamestate_data = {}
actions_ready = asyncio.Event()
all_actions = {}
current_connection_number = 0
max_connection_number = args.num_envs
obs_received_this_frame = 0
this_frame_raw_actions = None

def get_config(checkpoint_path, 
                capture_video = False,
                fabric_accelerator = "cpu", 
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

def get_formatted_obs(obs):
    formatted_obs = {}
    for obs_key in obs[0].keys():
        formatted_obs[obs_key] = np.array([obs[env_idx][obs_key] for env_idx in obs.keys()])
    
    return formatted_obs

async def lockstep_compute_n(reader, writer, n):
    global gamestate_data
    global actions_ready
    global all_actions
    global current_connection_number
    global obs_received_this_frame
    global this_frame_raw_actions
    
    while True:
        
        try:
            print(str(n) + ' recv')
            prefix = await reader.readline()
            data = await reader.readexactly(int(prefix))
        except:
            print('read op in handler ' + str(n) + ' no data, quitting')
            writer.close()
            await writer.wait_closed()
            current_connection_number -= 1

            metrics = timer.compute()
            print('max: ' + str(metrics["max_inference_time"]) + ' | mean: ' + str(metrics["mean_inference_time"]))
            return

        print(str(n) + ' pickle loads')
        obs = pickle.loads(data)

        if not obs:
            writer.close()
            print(str(n) + ' wait_closed')
            await writer.wait_closed()
            current_connection_number -= 1
            return

        gamestate_data[n] = obs
        obs_received_this_frame += 1

        if obs_received_this_frame == max_connection_number:
            formatted_obs = get_formatted_obs(gamestate_data)
            with timer("max_inference_time", MaxMetric), timer("mean_inference_time", MeanMetric):
                this_frame_raw_actions = trained_agent.act_v2(formatted_obs)
            actions_ready.set()
        else:
            actions_ready.clear()
            print(str(n) + ' waiting')
            await actions_ready.wait()

        obs_received_this_frame -= 1
        this_request_action = this_frame_raw_actions[n]
        print(this_request_action)
        pickled = pickle.dumps(this_request_action,  pickle.HIGHEST_PROTOCOL)
        sz_bytes = b'%d\n' % len(pickled)
        writer.write(sz_bytes)
        writer.write(pickled)
        print(str(n) + ' drain')
        await writer.drain()

async def dummy_compute_n(reader, writer, n):
    global gamestate_data
    global actions_ready
    global all_actions
    global current_connection_number
    global obs_received_this_frame
    global this_frame_raw_actions
    
    while True:
        
        try:
            #print(str(n) + ' recv')
            prefix = await reader.readline()
            data = await reader.readexactly(int(prefix))
        except:
            #print('read op in handler ' + str(n) + ' no data, quitting')
            writer.close()
            await writer.wait_closed()
            current_connection_number -= 1
            return

        #print(str(n) + ' pickle loads')
        obs = pickle.loads(data)

        if not obs:
            writer.close()
            #print(str(n) + ' wait_closed')
            await writer.wait_closed()
            current_connection_number -= 1
            return

        gamestate_data[n] = obs
        obs_received_this_frame += 1

        this_frame_raw_actions = [0, 0, 0]

        obs_received_this_frame -= 1
        this_request_action = this_frame_raw_actions[n]
        #print(this_request_action)
        pickled = pickle.dumps(this_request_action,  pickle.HIGHEST_PROTOCOL)
        sz_bytes = b'%d\n' % len(pickled)
        writer.write(sz_bytes)
        writer.write(pickled)
        #print(str(n) + ' drain')
        await writer.drain()

async def listen(reader, writer):
    global current_connection_number

    if current_connection_number >= max_connection_number:
        print('error max conn achieved')
        return
    task = asyncio.create_task(lockstep_compute_n(reader, writer, current_connection_number))
    #task = asyncio.create_task(dummy_compute_n(reader, writer, current_connection_number))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    current_connection_number += 1


async def main():
    global trained_agent

    trained_agent_generator = get_trained_agent_entrypoint_multienv(get_config(trained_checkpoint_path), args.num_envs)
    trained_agent = trained_agent_generator(
        obs_space_v1.get_observation_space_v1(2),
        act_space_v1.get_action_space_v1(2))


    server = await asyncio.start_server(
        listen, '127.0.0.1', 5555)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())