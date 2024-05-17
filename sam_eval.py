from sheeprl.cli import get_trained_agent_entrypoint, sam_build_config
from omegaconf import DictConfig, OmegaConf, open_dict

import gymnasium as gym 
from sheeprl.utils.env import make_env
from sheeprl.utils.timer import timer
from torchmetrics import MaxMetric, MeanMetric, CatMetric

test_path = "logs/runs/dreamer_v3/MsPacmanNoFrameskip-v4/2024-04-18_08-13-05_dreamer_v3_MsPacmanNoFrameskip-v4_5/version_0/checkpoint/ckpt_1152_0.ckpt"

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

def get_trained_agent(checkpoint_path):
    trained_agent_generator, cfg = get_trained_agent_entrypoint(get_config(checkpoint_path, fabric_accelerator="gpu"))
    return trained_agent_generator(env.observation_space, env.action_space)

if __name__ == "__main__":
    cfg =  sam_build_config(get_config(test_path))
    trained_agent_generator = get_trained_agent_entrypoint(get_config(test_path))
    env = make_env(cfg, 5, 0)()

    trained_agent = trained_agent_generator(env.observation_space, env.action_space)

    for _ in range(1000):
        with timer("max_inference_time", MaxMetric), timer("mean_inference_time", MeanMetric):#, timer("cat_inference_time", CatMetric):
            trained_agent.act(env.observation_space.sample())
    
    metrics = timer.compute()
    print('max: ' + str(metrics["max_inference_time"]) + ' | mean: ' + str(metrics["mean_inference_time"]))
    print('mean:' + str(metrics["mean_inference_time"]))
    #print(metrics["cat_inference_time"])
