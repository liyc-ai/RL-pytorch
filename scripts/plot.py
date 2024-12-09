import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from drlplugs.logger import tb2dict, window_smooth
from drlplugs.ospy import filter_from_list

# Hyper-param
WORK_DIR = osp.expanduser("~/workspace/RL-pytorch/runs")
LOG_DIRs = [
    "2024-01-27__18-19-31~seed=3407~agent.algo=ppo~env.id=Hopper-v4",
    "2024-01-27__19-14-55~seed=1290~agent.algo=ppo~env.id=Hopper-v4",
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
    _log_dir = log_dir.split("~")
    tmp_env_id = filter_from_list(_log_dir, "env.id=*")[0].split("=")[-1]
    tmp_algo = filter_from_list(_log_dir, "agent.algo=*")[0].split("=")[-1]
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

## 2. Drawing multiple lines in a single picture
for key in KEYs:
    sbn.lineplot(data=pd.DataFrame(merged_datas[key]), x="steps", y="values", label=key)
plt.title(f"Learning Curves of {algo} on {env_id}")
plt.xlabel("Steps", size=14)
plt.ylabel("Return", size=14)
plt.yticks(size=14)
plt.legend(loc="lower right", fontsize=14)
plt.savefig("result.pdf")
