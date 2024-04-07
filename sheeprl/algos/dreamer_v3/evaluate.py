from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v3.agent import PlayerDV3, build_agent
from sheeprl.algos.dreamer_v3.utils import test
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation

import torch

def get_trained_agent(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    class trained_agent():
        def __init__(self, obs_space, action_space):
            self.observation_space = obs_space
            self.action_space = action_space

            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {self.observation_space}")

            fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

            is_continuous = isinstance(self.action_space, gym.spaces.Box)
            is_multidiscrete = isinstance(self.action_space, gym.spaces.MultiDiscrete)
            actions_dim = tuple(
                self.action_space.shape if is_continuous else (self.action_space.nvec.tolist() if is_multidiscrete else [self.action_space.n])
            )

            # Create the actor and critic models
            world_model, actor, _, _ = build_agent(
                fabric,
                actions_dim,
                is_continuous,
                cfg,
                self.observation_space,
                state["world_model"],
                state["actor"],
            )
            self.player = PlayerDV3(
                world_model.encoder.module,
                world_model.rssm,
                actor.module,
                actions_dim,
                cfg.env.num_envs,
                cfg.algo.world_model.stochastic_size,
                cfg.algo.world_model.recurrent_model.recurrent_state_size,
                fabric.device,
                discrete_size=cfg.algo.world_model.discrete_size,
                decoupled_rssm=cfg.algo.decoupled_rssm,
            )

            self.cfg = cfg
            self.player.num_envs = 1
            self.player.init_states()
            self.device = fabric.device


        @torch.no_grad()
        def act(self, obs):
            for k in obs.keys():
                obs[k] = torch.from_numpy(obs[k]).view(1, *obs[k].shape).float()
            
            preprocessed_obs = {}
            for k, v in obs.items():
                if k in self.cfg.algo.cnn_keys.encoder:
                    preprocessed_obs[k] = v[None, ...].to(self.device) / 255 - 0.5
                elif k in self.cfg.algo.mlp_keys.encoder:
                    preprocessed_obs[k] = v[None, ...].to(self.device)
            
            real_actions = self.player.get_greedy_action(
                    preprocessed_obs, False, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
            )

            if self.player.actor.is_continuous:
                real_actions = torch.cat(real_actions, -1).cpu().numpy()
            else:
                real_actions = torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

            reshaped_actions = real_actions.reshape(self.action_space.shape)
            return reshaped_actions
    
    def trained_agent_generator(obs_space, action_space):
        return trained_agent(obs_space, action_space)

    return trained_agent_generator


@register_evaluation(algorithms="dreamer_v3")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env = make_env(
        cfg,
        cfg.seed,
        0,
        log_dir,
        "test",
        vector_env_idx=0,
    )()
    observation_space = env.observation_space
    action_space = env.action_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    # Create the actor and critic models
    world_model, actor, _, _ = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        state["actor"],
    )
    player = PlayerDV3(
        world_model.encoder.module,
        world_model.rssm,
        actor.module,
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
        decoupled_rssm=cfg.algo.decoupled_rssm,
    )

    test(player, fabric, cfg, log_dir, sample_actions=True)
