# RL-pytorch
Re-implementations of Deep Reinforcement Learning (DRL) algorithms, written in PyTorch.

## Installation

```bash
git clone https://github.com/liyc-ai/RL-pytorch.git
cd RL-pytorch
pip install .

# update installation if you make modifications
pip install --upgrade .
```

## Implemented Algorithms

- [x] Deep Q Networks (DQN) [[paper](https://www.nature.com/articles/nature14236.pdf)] [[official code](https://github.com/deepmind/dqn)]
- [x] Deep Double Q Networks (DDQN) [[paper](https://arxiv.org/pdf/1509.06461.pdf)]
- [x] Dueling Network Architectures for Deep Reinforcement Learning (DuelDQN) [[paper](https://arxiv.org/pdf/1511.06581.pdf)]
- [x] Continuous control with deep reinforcement learning (DDPG) [[paper](https://arxiv.org/pdf/1509.02971.pdf)]
- [x] Addressing Function Approximation Error in Actor-Critic Methods (TD3) [[paper](https://arxiv.org/pdf/1802.09477.pdf)] [[official code](https://github.com/sfujim/TD3)]
- [x] Soft Actor-Critic Algorithms and Applications (SAC) [[paper](https://arxiv.org/pdf/1812.05905.pdf)] [[official code](https://github.com/rail-berkeley/softlearning/)]
- [x] Trust Region Policy Optimization (TRPO) [[paper](https://arxiv.org/pdf/1502.05477.pdf)] [[official code](https://github.com/joschu/modular_rl)]
- [x] Proximal Policy Optimization (PPO) [[paper](https://arxiv.org/pdf/1707.06347.pdf)] [[official code](https://github.com/openai/baselines)]

## Run Experiments

```bash
python scripts/train_agent.py agent=ppo env.id=Hopper-v4
```

By default, the results are stored at the `runs` dir.

## Acknowledgement
With the progress of this project, I found many open-source materials on the Internet to be excellent references. I am deeply grateful for the efforts of their authors. Below is a detailed list. Additionally, I would like to extend my thanks to my friends from [LAMDA-RL](https://github.com/LAMDA-RL) for our helpful discussions.

**Codebase**

+ [tianshou](https://github.com/thu-ml/tianshou)
+ [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
+ [stable-baselines-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
+ [stable-baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
+ [spinningup](https://github.com/openai/spinningup)
+ [RL-Adventure2](https://github.com/higgsfield/RL-Adventure-2)
+ [unstable_baselines](https://github.com/x35f/unstable_baselines)
+ [d4rl_evaluations](https://github.com/rail-berkeley/d4rl_evaluations)
+ [TD3](https://github.com/sfujim/TD3)
+ [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)

**Blog**

+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr.iro.umontreal.ca/679b37e0-caab-4710-921b-b59a688075df_1642188062/blog/)

**Tutorial**

+ [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/index.html)
