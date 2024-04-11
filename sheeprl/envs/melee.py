from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import melee_env
import gymnasium as gym
import melee_env.agents.basic
import melee_env.env_v2
import numpy as np

import melee_env.agents
import melee

from melee_env.agents.basic import agent_type
from melee_env.gamestate_to_obs import gamestate_to_obs_v1
from melee_env.raw_to_logical_inputs import raw_to_logical_inputs_v1
from melee_env.logical_to_libmelee_inputs import logical_to_libmelee_inputs_v1

from omegaconf import DictConfig
from sheeprl.cli import get_trained_agent_entrypoint, sam_build_config
import torch
import threading
import multiprocessing.shared_memory as shm

# clean this up (old)
trained_against_NOOP_bots_checkpoint_path_v0 = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-07_22-18-40_dreamer_v3_melee_5/version_0/checkpoint/ckpt_530000_0.ckpt"
v1_trained_against_v0 = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-08_20-46-16_dreamer_v3_melee_7/version_0/checkpoint/ckpt_160000_0.ckpt"

# current
v2_trained_against_v1 = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-09_07-26-56_dreamer_v3_melee_7/version_0/checkpoint/ckpt_420000_0.ckpt"

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
        #trained_agent_generator = get_trained_agent_entrypoint(get_config(v2_trained_against_v1))
        #trained_agent = trained_agent_generator(
        #    melee_env.env_v2.MeleeEnv_v2.get_observation_space_v1(2),
        #    melee_env.env_v2.MeleeEnv_v2.get_action_space_v1(2))
        
        #def trained_agent_fn(obs):
        #    return trained_agent.act(obs)
        
        def random_agent_fn(obs):
            return self.action_space.sample()

        
        players = [
            melee_env.agents.basic.step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
              raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
              logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
              agent_type.step_controlled_AI), 
            melee_env.agents.basic.trained_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
                #trained_agent_fn,
                random_agent_fn,
                raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
                logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
                agent_type.enemy_controlled_AI, 12)]
                # todo: read these configs (function versions, act_every, etc. )from config.yaml

        #players = [
        #    melee_env.agents.basic.step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
        #      raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
        #      logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1, 
        #      agent_type.step_controlled_AI), 
        #    melee_env.agents.basic.NOOP(melee_env.agents.basic.enums.Character.BOWSER)]

        env = melee_env.env_v2.MeleeEnv_v2(
            iso_path, 
            slippi_game_path, 
            players, 
            fast_forward=True, 
            shuffle_controllers_after_each_game=True, 
            randomize_character=True, 
            randomize_stage=False, 
            num_players=2, 
            action_repeat=12, 
            max_match_steps=60*60*8,
            env_num=str(env_num),
            slippi_port=slippi_port)
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
