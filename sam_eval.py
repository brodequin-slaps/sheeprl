from sheeprl.cli import get_trained_agent_entrypoint
from omegaconf import DictConfig, OmegaConf, open_dict

import gymnasium as gym 
from sheeprl.utils.env import make_env
from sheeprl.utils.timer import timer
from torchmetrics import MaxMetric, MeanMetric, CatMetric



def get_config():
    cfg = DictConfig({
        "checkpoint_path": "logs/runs/dreamer_v3/MsPacmanNoFrameskip-v4/2024-03-14_20-31-19_dreamer_v3_MsPacmanNoFrameskip-v4_5/version_0/checkpoint/ckpt_600000_0.ckpt",
        "disable_grads": True,
        "env": {
            "capture_video": False,
            "input_buffer": 0
        },
        "fabric": {
            "accelerator": "gpu"
        },
        "float32_matmul_precsion": "high",
        "num_threads": 1,
        "seed": None
    })
    return cfg

if __name__ == "__main__":
    cfg = get_config()
    trained_agent_generator, cfg = get_trained_agent_entrypoint(cfg)
    env = make_env(cfg, 5, 0)()

    trained_agent = trained_agent_generator(env.observation_space, env.action_space)

    for _ in range(1000):
        with timer("max_inference_time", MaxMetric), timer("mean_inference_time", MeanMetric):#, timer("cat_inference_time", CatMetric):
            trained_agent.act(env.observation_space.sample())
    
    metrics = timer.compute()
    print('max: ' + str(metrics["max_inference_time"]) + ' | mean: ' + str(metrics["mean_inference_time"]))
    print('mean:' + str(metrics["mean_inference_time"]))
    #print(metrics["cat_inference_time"])
