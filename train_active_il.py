import os
import numpy as np
from algo import ALGOS
from utils.config import parse_args, load_yml_config
from utils.exp import eval, preprare_training
from train_expert import train


def get_expert(configs):
    """Train expert if necessary"""
    expert_configs = load_yml_config(configs["expert_config"])
    if configs.get("expert_model") and os.path.exists(configs["expert_model"]):
        expert_configs["state_dim"], expert_configs["action_dim"], expert_configs["action_high"] = (
            configs["state_dim"],
            configs["action_dim"],
            configs["action_high"],
        )
        expert = ALGOS[configs["expert_name"]](expert_configs)
        expert.load_model(configs["expert_model"])
    else:
        expert = train(*preprare_training(expert_configs))
    return expert, configs


def train_imitator(configs, agent, env, logger, writer, seed, model_path):
    # get expert
    expert = get_expert(configs)

    # evaluate before update, to get baseline
    writer.add_scalar(
        f"evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, logger),
        global_step=0,
    )

    # train agent
    best_avg_reward = -np.inf
    expert_traj_num = 0
    for i in range(configs["max_iters"]):
        # 1. ask expert
        if i % configs["rollout_freq"] == 0:
            agent.rollout(expert)  # \beta = 1
            expert_traj_num += 1
            logger.info(
                f"Asking for new expert data, currently have {expert_traj_num} expert trajectories"
            )
        # 2. update
        snapshot = agent.learn()
        if snapshot != None:
            for key, value in snapshot.items():
                writer.add_scalar(key, value, global_step=i + 1)
        # 3. evaluate
        if (i + 1) % configs["eval_freq"] == 0:
            avg_reward = eval(agent, configs["env_name"], seed, logger)
            writer.add_scalar(
                f"evaluation_averaged_return",
                avg_reward,
                global_step=i + 1,
            )  # evaluate before update, to get baseline
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(model_path)


if __name__ == "__main__":
    # read configs
    args = parse_args()
    assert args.config in ['dagger.yml'], "Currently only dagger is supperted for active imitation learning"
    configs = load_yml_config(args.config)
    # train imitator
    train_imitator(*preprare_training(configs))
