# ILAlgo
Imitation Learning Algorithms Codebase with Baseline of Reinforcement Learning Algorithms

## Install Dependency

```bash
pip install -r requirements
```

## Run experiment

```bash
python run_rl.py trpo  # which will run trpo with config file in config/trpo.yml
```

## Currently Implemented Algorithms:

1. [TRPO](https://arxiv.org/abs/1502.05477)
2. [PPO](https://arxiv.org/abs/1707.06347)
3. [SAC](https://arxiv.org/abs/1812.05905)
4. [TD3](https://arxiv.org/abs/1802.09477)
5. [DDPG](https://arxiv.org/abs/1509.02971)

## Acknowledgement
During my implementation of IL and RL algorithms, a lot of classic open-source materials on the Internet served as good references. And I highly appreciate their author's effort. Below is a detailed list.

**RL Library**

+ [tianshou](https://github.com/thu-ml/tianshou)
+ [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
+ [stable-baselines-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
+ [spinningup](https://github.com/openai/spinningup)
+ [RL-Adventure2](https://github.com/higgsfield/RL-Adventure-2)
+ [imitation](https://github.com/HumanCompatibleAI/imitation)
+ [d4rl_evaluations](https://github.com/rail-berkeley/d4rl_evaluations)
+ [imitation-learning](https://github.com/Kaixhin/imitation-learning)
+ [ILSwiss](https://github.com/Ericonaldo/ILSwiss)

**Open Source Code Repo**

+ [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
+ [TD3](https://github.com/sfujim/TD3)
+ [gail-airl-ppo.pytorch](https://github.com/ku2482/gail-airl-ppo.pytorch)

**Blog**

+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr.iro.umontreal.ca/679b37e0-caab-4710-921b-b59a688075df_1642188062/blog/)

**Tutorials**

+ [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/index.html)
