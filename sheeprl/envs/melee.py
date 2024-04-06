from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import melee_env
import gymnasium as gym
import melee_env.agents.basic
import melee_env.env_v2
import numpy as np

import melee_env.agents
import melee

class MeleeWrapper(gym.Wrapper):
    def __init__(self, id:str, seed:int | None = None) -> None:
        #if isinstance(screen_size, int):
            #screen_size = (screen_size,) * 2
        
        iso_path = "/home/sam/dev/melee.iso"
        slippi_game_path = "/home/sam/dev/Ishiiruka/Slippi_Online-x86_64.AppImage"
        env_num = "0"
        slippi_port = "51441"
        #players = [melee_env.env_v2.sam_ai(melee.enums.Character.JIGGLYPUFF), melee_env.agents.basic.Rest()]
        players = [melee_env.env_v2.sam_ai(melee.enums.Character.JIGGLYPUFF), melee_env.agents.basic.NOOP(melee.enums.Character.JIGGLYPUFF)]


        env = melee_env.env_v2.MeleeEnv_v2(
            iso_path, 
            slippi_game_path, 
            players, 
            fast_forward=True, 
            shuffle_controllers_after_each_game=True, 
            randomize_character=True, 
            randomize_stage=True, 
            num_players=2, 
            action_repeat=12, 
            max_match_steps=60*60*8,
            env_num=env_num,
            slippi_port=slippi_port)
        super().__init__(env)

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


        # trained agents stuff for the enemy
        self.trained_agent_puff_v1_path = "/home/sam/dev/sheeprl/logs/runs/dreamer_v3/melee/2024-04-05_13-48-00_dreamer_v3_melee_5/version_0/checkpoint/ckpt_40000_0.ckpt"
        


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
