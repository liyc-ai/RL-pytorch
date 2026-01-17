import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

from src.utils.logger import tb2dict, window_smooth
from src.utils.ospy import filter_from_list

# Hyper-param
WORK_DIR = os.getcwd()
LOG_DIRs = [
    "runs/2026-01-17-15-12-21__comment@benchmark__seed@3407__agent.algo@ppo__env.id@Hopper-v5",
    "runs/2026-01-17-16-59-43__comment@benchmark__seed@3407__agent.algo@ppo__env.id@Hopper-v5",
]
KEYs = ["return/eval", "return/train"]
RULE = "events.out.tfevents*"
SMOOTH_WINDOW_SIZE = 10

# Drawing
sbn.set_style("darkgrid")

## 1. Convert tensorboard files to a list of data points
datas = {key: list() for key in KEYs}
env_id, algo = None, None
for log_dir in LOG_DIRs:
    # check
    _log_dir = log_dir.split("__")
    tmp_env_id = filter_from_list(_log_dir, "env.id@*")[0].split("@")[-1]
    tmp_algo = filter_from_list(_log_dir, "agent.algo@*")[0].split("@")[-1]
    if env_id is None:
        env_id = tmp_env_id
    else:
        assert (
            env_id == tmp_env_id
        ), "The data used to plot must from the same environment!"
    if algo is None:
        algo = tmp_algo
    else:
        assert algo == tmp_algo, "The data used to plot must from the same algorithm"
    # get data
    dir_path = osp.join(WORK_DIR, log_dir)
    tb_file = filter_from_list(os.listdir(dir_path), RULE)[0]
    data = tb2dict(osp.join(dir_path, tb_file), KEYs)
    for key in KEYs:
        datas[key].append(data[key])
merged_datas = {key: {"steps": list(), "values": list()} for key in KEYs}
for key in KEYs:  # smooth
    for i in range(len(datas[key])):
        merged_datas[key]["steps"] += datas[key][i]["steps"]
        merged_datas[key]["values"] += window_smooth(
            datas[key][i]["values"], SMOOTH_WINDOW_SIZE
        )

## 2. Drawing multiple lines in different subplots
fig, axes = plt.subplots(len(KEYs), 1, figsize=(10, 5))
for i, key in enumerate(KEYs):
    sbn.lineplot(
        data=pd.DataFrame(merged_datas[key]),
        x="steps",
        y="values",
        label=key,
        ax=axes[i],
    )
    axes[i].set_title(f"Learning Curves of {algo} on {env_id}", size=14)
    axes[i].set_xlabel("Steps", size=14)
    axes[i].set_ylabel("Return", size=14)
    axes[i].tick_params(axis="both", which="major", labelsize=14)
    axes[i].legend(loc="lower right", fontsize=14)
fig.tight_layout()
fig.savefig("result.pdf")
