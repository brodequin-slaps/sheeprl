from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import melee_env
import gymnasium as gym
import melee_env.agents.basic
import melee_env.env_v2
import numpy as np

import melee_env.agents
import melee

from melee_env.agents.basic import agent_type
from melee_env.gamestate_to_obs import gamestate_to_obs_v1, gamestate_to_obs_v3
from melee_env.raw_to_logical_inputs import raw_to_logical_inputs_v1, raw_to_logical_inputs_v3
from melee_env.logical_to_libmelee_inputs import logical_to_libmelee_inputs_v1, logical_to_libmelee_inputs_v3
from melee_env.act_space import act_space_v1, act_space_v3
from melee_env.obs_space import obs_space_v1, obs_space_v3

from omegaconf import DictConfig
from sheeprl.cli import get_trained_agent_entrypoint, sam_build_config, get_trained_agent_entrypoint_multienv
import torch
import threading
import multiprocessing.shared_memory as shm

#from eval_client_lib import client
import pickle
import socket
import time
import random


def get_formatted_obs(obs):
    formatted_obs = {}
    for obs_key in obs[0].keys():
        formatted_obs[obs_key] = np.array([obs[env_idx][obs_key] for env_idx in obs.keys()])
    
    return formatted_obs

def current_nanos_time():
    return round(time.time() * 1000000000)

# clean this up (old)
trained_against_NOOP_bots_checkpoint_path_v0 = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-07_22-18-40_dreamer_v3_melee_5/version_0/checkpoint/ckpt_530000_0.ckpt"
v1_trained_against_v0 = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-08_20-46-16_dreamer_v3_melee_7/version_0/checkpoint/ckpt_160000_0.ckpt"

# current
test_path = "/home/sam/dev/sheeprl/trained_agent_checkpoint/2024-04-19_18-42-53_dreamer_v3_melee_7/version_0/checkpoint/ckpt_1550000_0.ckpt"
trained_checkpoint_path = "/home/sam/dev/sheeprl/trained_agent_checkpoint/2024-05-10_23-38-21_dreamer_v3_melee_7/version_0/checkpoint/ckpt_5365000_0.ckpt"


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


class MeleeWrapper(gym.Wrapper):
    shared_memory = shm.SharedMemory(name='my_shared_memory', create=False, size=1)

    def __init__(self, id:str, seed:int | None = None) -> None:
        #if isinstance(screen_size, int):
            #screen_size = (screen_size,) * 2
        
        iso_path = "/home/sam/dev/melee.iso"
        slippi_game_path = "/home/sam/dev/Ishiiruka/Slippi_Online-x86_64.AppImage"
        
        #with MeleeWrapper.env_counter_lock:
        env_num = MeleeWrapper.shared_memory.buf[0]
        MeleeWrapper.shared_memory.buf[0] = env_num+1
        print('init env: ' + str(env_num))
        slippi_port = str(51441 + env_num)
        #players = [melee_env.env_v2.sam_ai(melee.enums.Character.JIGGLYPUFF), melee_env.agents.basic.Rest()]
        
        
        # load trained agent stuff
        #trained_agent_generator = get_trained_agent_entrypoint(get_config(test_path))
        #trained_agent = trained_agent_generator(
        #    melee_env.env_v2.MeleeEnv_v2.get_observation_space_v1(2),
        #    melee_env.env_v2.MeleeEnv_v2.get_action_space_v1(2))
        #
        #def trained_agent_fn(obs):
        #    return trained_agent.act(obs)

        
        #players = [
        #    melee_env.agents.basic.step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #      raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.trained_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #        trained_agent_fn,
        #        raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #        logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
        #        agent_type.enemy_controlled_AI, 12)]
        #        # todo: read these configs (function versions, act_every, etc. )from config.yaml

        players = [
            melee_env.agents.basic.step_controlled_ai(
              raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
              logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
              agent_type.step_controlled_AI), 
            melee_env.agents.basic.NOOP(melee_env.agents.basic.enums.Character.BOWSER)]

        env = melee_env.env_v2.MeleeEnv_v2(
            iso_path, 
            slippi_game_path, 
            players,
            act_space_v3.get_action_space_v3(2),
            obs_space_v1.get_observation_space_v1(2),
            gamestate_to_obs_v1.gamestate_to_obs_v1,
            "melee_shm_frame",
            64,
            fast_forward=True, 
            shuffle_controllers_after_each_game=True, 
            randomize_character=True, 
            randomize_stage=True, 
            num_players=2, 
            action_repeat=12, 
            max_match_steps=60*60*8,
            env_num=str(env_num),
            slippi_port=slippi_port,
            seed=101)
        super().__init__(env)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)        

        # render
        self._render_mode = None
        # metadata
        #self._metadata = {"render_fps": 30}

        @property
        def render_mode(self) -> str | None:
            return self._render_mode
    
        def step(self, action:Any):
            return self.env.step(action)
        
        def reset(self, seed=None, options=None):
            return self.env.reset(seed, options)
        
        def render(self):
            return self.env.render()
        
        def close(self):
            return self.env.close()
        



class MeleeWrapper_multienv_using_gpu_asyncio_server(gym.Wrapper):
    shared_memory = shm.SharedMemory(name='my_shared_memory', create=False, size=1)

    def __init__(self, id:str, seed:int | None = None, render_mode="rgb_array", screen_size=64) -> None:
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2
        
        #self._render_mode = render_mode
        self._render_mode = render_mode

        iso_path = "/home/sam/dev/melee.iso"
        slippi_game_path = "/home/sam/dev/Ishiiruka/Slippi_Online-x86_64.AppImage"
        
        #with MeleeWrapper.env_counter_lock:
        self.env_num = MeleeWrapper_multienv.shared_memory.buf[0]
        MeleeWrapper_multienv.shared_memory.buf[0] = self.env_num+1
        print('init env: ' + str(self.env_num))
        slippi_port = str(51441 + self.env_num)
        #players = [melee_env.env_v2.sam_ai(melee.enums.Character.JIGGLYPUFF), melee_env.agents.basic.Rest()]
        
        #self.clientclass = clientclass('localhost', 9999)
        
        def trained_agent_fn(obs):
            #print('env num trained agent: ' + str(self.env_num))
            #obs['env_idx'] = self.env_num - 1 # because they create a dummy env so we start at 1...
            #obs['env_idx'] = 0
            return client('localhost', 9999, pickle.dumps(obs, pickle.HIGHEST_PROTOCOL))
        
        class trained_agent_class:
            def __init__(self) -> None:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(('localhost', 5555))
                print('trained agent connected')
            
            def __call__(self, obs):
                pickled = pickle.dumps(obs, pickle.HIGHEST_PROTOCOL)
                self.sock.sendall(b'%d\n' % len(pickled))
                self.sock.sendall(pickled)

                sz_bytes = b''
                current = self.sock.recv(1)
                while current.decode('utf-8') != '\n':
                    sz_bytes += current
                    current = self.sock.recv(1)
                
                #print('sz bytes = ' + str(sz_bytes))
                sz = int.from_bytes(sz_bytes, byteorder='big')
                data = self.sock.recv(sz)
                unpickled = pickle.loads(data)
                return unpickled

        
        #players = [
        #    melee_env.agents.basic.step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #      raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.trained_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #        trained_agent_fn,
        #        raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #        logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
        #        agent_type.enemy_controlled_AI, 12)]
        #        # todo: read these configs (function versions, act_every, etc. )from config.yaml

        #players = [
        #    melee_env.agents.basic.step_controlled_ai(
        #      raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.NOOP(melee_env.agents.basic.enums.Character.BOWSER)]
        

        players = [
            melee_env.agents.basic.step_controlled_ai(
              raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
              logical_to_libmelee_inputs_v3.logical_to_libmelee_inputs_v3, 
              agent_type.step_controlled_AI,
              character=melee_env.agents.basic.enums.Character.JIGGLYPUFF), 
            melee_env.agents.basic.trained_ai(act_space_v1.get_action_space_v1(2),
                obs_space_v1.get_observation_space_v1(2),
                gamestate_to_obs_v1.gamestate_to_obs_v1,
                #trained_agent_fn,
                trained_agent_class(),
                raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
                logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1_1,
                agent_type.enemy_controlled_AI, 12,
                character=melee_env.agents.basic.enums.Character.JIGGLYPUFF)]

        env = melee_env.env_v2.MeleeEnv_v2(
            iso_path, 
            slippi_game_path, 
            players, 
            act_space_v3.get_action_space_v3(2),
            obs_space_v3.get_observation_space_v3(2),
            gamestate_to_obs_v3.gamestate_to_obs_v3,
            64,
            fast_forward=True, 
            shuffle_controllers_after_each_game=True, 
            randomize_character=False, 
            randomize_stage=True, 
            num_players=2, 
            action_repeat=6, 
            max_match_steps=60*60*8,
            env_num=str(self.env_num),
            slippi_port=slippi_port,
            seed=seed)
        super().__init__(env)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)        

        # metadata
        #self._metadata = {"render_fps": 30}

        @property
        def render_mode(self) -> str | None:
            return self._render_mode
    
        def step(self, action:Any):
            return self.env.step(action)
        
        def reset(self, seed=None, options=None):
            return self.env.reset(seed, options)
        
        def render(self):
            return self.env.render()
        
        def close(self):
            return self.env.close()



class MeleeWrapper_multienv(gym.Wrapper):
    shared_memory = shm.SharedMemory(name='my_shared_memory', create=False, size=1)

    def __init__(self, id:str, seed:int | None = None, render_mode="rgb_array", screen_size=64) -> None:
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2
        
        #self._render_mode = render_mode
        self._render_mode = render_mode

        iso_path = "/home/sam/dev/melee.iso"
        slippi_game_path = "/home/sam/dev/Ishiiruka/Slippi_Online-x86_64.AppImage"
        
        #with MeleeWrapper.env_counter_lock:
        self.env_num = MeleeWrapper_multienv.shared_memory.buf[0]
        MeleeWrapper_multienv.shared_memory.buf[0] = self.env_num+1
        print('init env: ' + str(self.env_num))
        slippi_port = str(51441 + self.env_num)

        
        #trained_agent_generator = get_trained_agent_entrypoint_multienv(get_config(trained_checkpoint_path), 1)
        #trained_agent = trained_agent_generator(
        #    obs_space_v1.get_observation_space_v1(2),
        #    act_space_v1.get_action_space_v1(2))

        
        def fsmash_agent_fn(obs):
            actions = [12, 12, 12, 12, 8, 8, 8, 8]
            fsmash_agent_fn.i += 1

            if fsmash_agent_fn.i % fsmash_agent_fn.next == 0:
                fsmash_agent_fn.next = random.randint(15, 25)
                fsmash_agent_fn.i = 0
                fsmash_agent_fn.action = (fsmash_agent_fn.action + 1) % len(actions)
                return actions[fsmash_agent_fn.action]
            else:
                return 0
        
        fsmash_agent_fn.next = 10
        fsmash_agent_fn.action = 0
        fsmash_agent_fn.i = 0

        #def trained_agent_fn(obs):
        #    gamestate_data = {}
        #    gamestate_data[0] = obs
        #    return trained_agent.act_v2(get_formatted_obs(gamestate_data))[0]
        
        class trained_agent_class:
            def __init__(self) -> None:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(('localhost', 5555))
                print('trained agent connected')
            
            def __call__(self, obs):
                pickled = pickle.dumps(obs, pickle.HIGHEST_PROTOCOL)
                self.sock.sendall(b'%d\n' % len(pickled))
                self.sock.sendall(pickled)

                sz_bytes = b''
                current = self.sock.recv(1)
                while current.decode('utf-8') != '\n':
                    sz_bytes += current
                    current = self.sock.recv(1)
                
                #print('sz bytes = ' + str(sz_bytes))
                sz = int.from_bytes(sz_bytes, byteorder='big')
                data = self.sock.recv(sz)
                unpickled = pickle.loads(data)
                return unpickled

        
        #players = [
        #    melee_env.agents.basic.step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #      raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.trained_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #        trained_agent_fn,
        #        raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #        logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
        #        agent_type.enemy_controlled_AI, 12)]
        #        # todo: read these configs (function versions, act_every, etc. )from config.yaml

        #players = [
        #    melee_env.agents.basic.step_controlled_ai(
        #      raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.NOOP(melee_env.agents.basic.enums.Character.BOWSER)]
        

        players = [
            melee_env.agents.basic.step_controlled_ai(
              raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
              logical_to_libmelee_inputs_v3.logical_to_libmelee_inputs_v3, 
              agent_type.step_controlled_AI,
              character=melee_env.agents.basic.enums.Character.JIGGLYPUFF), 
            melee_env.agents.basic.trained_ai(act_space_v3.get_action_space_v3(2),
                obs_space_v3.get_observation_space_v3(2),
                gamestate_to_obs_v3.gamestate_to_obs_v3,
                fsmash_agent_fn,
                #trained_agent_fn,
                #trained_agent_class(),
                raw_to_logical_inputs_v3.raw_to_logical_inputs_v3, 
                logical_to_libmelee_inputs_v3.logical_to_libmelee_inputs_v3,
                agent_type.enemy_controlled_AI, 6,
                character=melee_env.agents.basic.enums.Character.JIGGLYPUFF)]

        env = melee_env.env_v2.MeleeEnv_v2(
            iso_path, 
            slippi_game_path, 
            players, 
            act_space_v3.get_action_space_v3(2),
            obs_space_v3.get_observation_space_v3(2),
            gamestate_to_obs_v3.gamestate_to_obs_v3,
            64,
            fast_forward=True, 
            shuffle_controllers_after_each_game=True, 
            randomize_character=False, 
            randomize_stage=True, 
            num_players=2, 
            action_repeat=6, 
            max_match_steps=60*60*8,
            env_num=str(self.env_num),
            slippi_port=slippi_port,
            seed=current_nanos_time())
        super().__init__(env)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)        

        # metadata
        #self._metadata = {"render_fps": 30}

        @property
        def render_mode(self) -> str | None:
            return self._render_mode
    
        def step(self, action:Any):
            return self.env.step(action)
        
        def reset(self, seed=None, options=None):
            return self.env.reset(seed, options)
        
        def render(self):
            return self.env.render()
        
        def close(self):
            return self.env.close()
