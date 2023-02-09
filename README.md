# ilkit
Imitation Learning (IL) and Deep Reinforcement Learning (RL) with PyTorch.

**Status**: Under development, and there may be some bugs. Welcome to issues, bug reports and PRs!

## Installation

```bash
git clone https://github.com/BepfCp/ilkit
cd ilkit
pip install -e .
```

### Optional

#### D4RL

We support to load offline (expert) data from the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark.
```bash
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```

#### nni
We support to use [nni](https://github.com/microsoft/nni) to auto-tune hyperparameters.

```bash
pip install nni
```

#### wandb

```
pip install wandb
```

#### RLAssistant

We use [RLAssistant](https://github.com/polixir/RLAssistant) to manage experiments.

```bash
git clone https://github.com/polixir/RLAssistant.git
cd RLAssistant
python install -e .
```

#### SmartLogger

We support to view the experiment result in front page via [SmartLogger](https://github.com/FanmingL/SmartLogger)

```bash
git clone https://github.com/FanmingL/SmartLogger.git
cd SmartLogger
pip install -e .
```

## Implemented Algorithms

Welcome to make PRs on new algorithm implementations :) .

### RL

- [x] Deep Q Networks (DQN) [[paper](https://www.nature.com/articles/nature14236.pdf)] [[official code](https://github.com/deepmind/dqn)]
- [x] Deep Double Q Networks (DDQN) [[paper](https://arxiv.org/pdf/1509.06461.pdf)]
- [x] Dueling Network Architectures for Deep Reinforcement Learning (DuelDQN) [[paper](https://arxiv.org/pdf/1511.06581.pdf)]
- [x] Continuous control with deep reinforcement learning (DDPG) [[paper](https://arxiv.org/pdf/1509.02971.pdf)]
- [x] Addressing Function Approximation Error in Actor-Critic Methods (TD3) [[paper](https://arxiv.org/pdf/1802.09477.pdf)] [[official code](https://github.com/sfujim/TD3)]
- [x] Soft Actor-Critic Algorithms and Applications (SAC) [[paper](https://arxiv.org/pdf/1812.05905.pdf)] [[official code](https://github.com/rail-berkeley/softlearning/)]
- [x] Trust Region Policy Optimization (TRPO) [[paper](https://arxiv.org/pdf/1502.05477.pdf)] [[official code](https://github.com/joschu/modular_rl)]
- [x] Proximal Policy Optimization (PPO) [[paper](https://arxiv.org/pdf/1707.06347.pdf)] [[official code](https://github.com/openai/baselines)]

### IL

- [x] Behavioral Cloning (BC, for both continuous and discrete action space) [[paper](https://proceedings.neurips.cc/paper/1990/file/248e844336797ec98478f85e7626de4a-Paper.pdf)]
- [x] A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger, for both continuous and discrete action space) [[paper](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf)] 
- [x] Generative Adversarial Imitation Learning (GAIL) [[paper](https://arxiv.org/pdf/1606.03476.pdf)] [[official code](https://github.com/openai/imitation)]
- [x] Learning Robust Rewards with Adversarial Inverse Reinforcement Learning (AIRL) [[paper](https://arxiv.org/pdf/1710.11248.pdf)]
- [ ] IQ-Learn: Inverse soft-Q Learning for Imitation (IQ-Learn, for both continuous and discrete action space) [[paper](https://arxiv.org/pdf/2106.12142.pdf)] [[official code](https://github.com/Div99/IQ-Learn)]
- [ ] InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations (InfoGAIL) [[paper](https://arxiv.org/pdf/1703.08840.pdf)] [[official code](https://github.com/YunzhuLi/InfoGAIL)]
- [ ] Imitation Learning via Off-Policy Distribution Matching (ValueDICE) [[paper](https://arxiv.org/pdf/1912.05032.pdf)] [[official code](https://github.com/google-research/google-research/tree/master/value_dice)]
- [ ] Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning (DAC) [[paper](https://arxiv.org/pdf/1809.02925.pdf)] [[official code](https://github.com/google-research/google-research/tree/master/dac)]

## Run Experiments

### Train RL or IL agent

```bash
# RL
python example/train_agent.py agent=rl/ppo env.id=Hopper-v4

# IL
python example/train_agent.py agent=il/gail env.id=Hopper-v4 expert_dataset.d4rl_env_id=hopper-expert-v2
```

By default, the results are stored at the `logs` dir.

### Collect demonstrations

```bash
python example/collect_demo.py agent=rl/sac env.id=Hopper-v4 model_path=data/sac_hopper.pt
```

### Hyper-parameter Fine-Tuning

Remember to specify your `nni_optim_fn` function in `example/hyper_param_tuning.py`.

```bash
pip install nni
# start to fine tune hyper parameters
nnictl create -c conf/_hyper_param_tuning.yaml -p 8080

# stop to fine tune hyper parameters
nnictl stop --all
nnictl stop [experiment ID]
nnictl stop [port]

# view experiment
nnictl view [--port PORT] [--experiment_dir EXPERIMENT_DIR] id
```

Then, watch the results at `http://localhost:8080`; if you want to stop the experiment, please use command `nnictl stop [Experiment ID]`. For more commands, please see [the official documentation](https://nni.readthedocs.io/en/stable/reference/nnictl.html).

## TODO

- [x] Support nni
- [ ] Support the `__target__` usage of Hydra
- [ ] Support parallel sampling
- [ ] Support Ray for distributional traning

## Acknowledgement
During implementing the IL and RL algorithms, a lot of classic open-source materials on the Internet served as good references. And I highly appreciate their author's effort. Below is a detailed list. By the way, thanks to my friends from [LAMDA-RL](https://github.com/LAMDA-RL) for their helpful discussions.

**RL Library**

+ [tianshou](https://github.com/thu-ml/tianshou)
+ [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
+ [stable-baselines-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
+ [stable-baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
+ [spinningup](https://github.com/openai/spinningup)
+ [RL-Adventure2](https://github.com/higgsfield/RL-Adventure-2)
+ [unstable_baselines](https://github.com/x35f/unstable_baselines)
+ [d4rl_evaluations](https://github.com/rail-berkeley/d4rl_evaluations)

**IL Library**
+ [imitation](https://github.com/HumanCompatibleAI/imitation)
+ [imitation-learning](https://github.com/Kaixhin/imitation-learning)
+ [ILSwiss](https://github.com/Ericonaldo/ILSwiss)
+ [gail-airl-ppo.pytorch](https://github.com/ku2482/gail-airl-ppo.pytorch)
+ [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

**Open Source Code Repo**

+ [TD3](https://github.com/sfujim/TD3)
+ [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
+ [Code-for-Error-Bounds-of-Imitating-Policies-and-Environments](https://github.com/tianxusky/Code-for-Error-Bounds-of-Imitating-Policies-and-Environments)
+ [IQ-Learn](https://github.com/Div99/IQ-Learn)

**Blog**

+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr.iro.umontreal.ca/679b37e0-caab-4710-921b-b59a688075df_1642188062/blog/)

**Tutorial**

+ [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/index.html)