# ILAlgo
Baseline Algorithms of Imitation Learning and Reinforcment Learning , written in Pytorch. All code are tested on Mujoco.

**Note**: Code are under development. Issues,  bug reports and PRs are welcome.

## Install Dependency

```bash
pip install -r requirements
```

## Run experiment

```bash
# train expert
python train_expert.py --config sac.yml --env-name HalfCheetah-v3 --generate-data
# train imitator
python train_expert.py --config bc.yml
```

**Watch Result**

```bash
tensorboard --log_dir out/[your exp name]
```

## Currently Implemented Algorithms

**Reinforcement Learning**

1. [TRPO](https://arxiv.org/abs/1502.05477)
2. [PPO](https://arxiv.org/abs/1707.06347)
3. [SAC](https://arxiv.org/abs/1812.05905)
4. [TD3](https://arxiv.org/abs/1802.09477)
5. [DDPG](https://arxiv.org/abs/1509.02971)

**Imitation Learning**

1. [BC](https://proceedings.neurips.cc/paper/1990/hash/248e844336797ec98478f85e7626de4a-Abstract.html)
2. [DAgger](https://arxiv.org/abs/1011.0686)
3. [GAIL](https://arxiv.org/abs/1606.03476)
4. [AIRL](https://arxiv.org/abs/1710.11248)

## Acknowledgement
During my implementation of IL and RL algorithms, a lot of classic open-source materials on the Internet served as good references. And I highly appreciate their author's effort. Below is a detailed list. By the way, thanks to my friends from [LAMDA-RL](https://github.com/LAMDA-RL) for their helpful discussions.

**RL Library**

+ [tianshou](https://github.com/thu-ml/tianshou)
+ [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
+ [stable-baselines-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
+ [spinningup](https://github.com/openai/spinningup)
+ [RL-Adventure2](https://github.com/higgsfield/RL-Adventure-2)
+ [d4rl_evaluations](https://github.com/rail-berkeley/d4rl_evaluations)

**Open Source Code Repo**

+ [TD3](https://github.com/sfujim/TD3)
+ [gail-airl-ppo.pytorch](https://github.com/ku2482/gail-airl-ppo.pytorch)
+ [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
+ [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
+ [imitation](https://github.com/HumanCompatibleAI/imitation)
+ [imitation-learning](https://github.com/Kaixhin/imitation-learning)
+ [ILSwiss](https://github.com/Ericonaldo/ILSwiss)
+ [unstable_baselines](https://github.com/x35f/unstable_baselines)
+ [Code-for-Error-Bounds-of-Imitating-Policies-and-Environments](https://github.com/tianxusky/Code-for-Error-Bounds-of-Imitating-Policies-and-Environments)

**Blog**

+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr.iro.umontreal.ca/679b37e0-caab-4710-921b-b59a688075df_1642188062/blog/)

**Tutorial**

+ [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/index.html)
