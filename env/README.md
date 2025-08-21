[**ENGLISH**](README.md) | [**中文**](assets/README_CN.md)


# TODO: More complex fire cmd, choose which missile to shoot (DONE)
# TODO: Rule based agent and its test (Still Testing), add more commands and same(?) rule based agent for enemy
# TODO: Missile types nasil olmali hepsi meteor mi olsa? Mavi takımda TFX içn AIM SL ve Meteor olabilir eşit sayılarda, kırmızı takımda Meteor ve Mica olabilir
# TODO: General cleaning of code files etc.
# TODO: Converting current harfang gym to stable_baselines compatible, action and state spaces must be correct.
# TODO: Clean harfang system commit to repo
# TODO: start RL testing


# TODO: not Agents, ActionHelper

# class Agents
# self.state = state
# self.team = "ally" or "oppo"
# def behave(self, state)
# def update(self, state)
#  self.state = state
# Agents -> Ally, Oppo
# Ally behave() very simple tactics
#   if self.state[4] == 0:
# Oppo behave() bit more complex tactics
# ally_action = ally.behave(ally.state) , "fire"
# oppo_action = oppo.behave(oppo.state), "track"



# obs, reward, done, oppo_state, success = env.step(action, oppo_action)
# env.step(action)
# def play_oppo(self)
# def step(self, action):
#  if oppo_is_enabled:
#       self.play_oppo()

# obs, reward, done, info = env.step(action)
# obs = [int float, no dict]
# reward = single number, int or float
# done True or False boolean
# info can be a dict
# dict = {"ally_state": {},  "oppo_state": {}}
# oppo_state = info["oppo_state"]
# oppo.update(oppo_state)

# STATE REFACTOR: {}
# {"speed": 0, "heading": 44, "missile_threat": 1, RELATIVEBEARING: 34.67, "missile_wreck"..........}
# what was the index of relative bearing? state[24] or state[25] ?
# placeholders.py RELATIVEBEARING = "relative_bearing_ally"

# vectorizer(dict)
# state_list.append(dict["heading"])
# state_list.append(dict["speed"])
# return obs


## Useful commands
```bash
 python env/hirl/train_rule_yaw_agent.py --env simple_enemy --agent agents --port 50888 --episodes 10 --render  
 python env/hirl/train_rule_yaw_agent.py --env simple_enemy --agent agents --port 50888 --episodes 1 --render  --command fire
```

<h1 align='center'> Highly Imitative Reinforcement Learning for UCAV </h1>

This is the implementation of [*An Imitative Reinforcement Learning Framework for Autonomous Dogfight*](https://arxiv.org/abs/2406.11562).
The expert dataset, the trained models, and recorded videos of the learned policies are available at [Google Drive](https://drive.google.com/drive/folders/1lAllxmsy0MhW714ZmT8fb0MkdJktUxzJ?usp=sharing).

## Installation

### Install Environment Dependencies

Please follow [the official instructions](https://github.com/harfang3d/dogfight-sandbox-hg2) to install Dogfight Sandbox.

<!-- Alternatively, installation can also be done directly from one of the following links: [Link 1](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/download/v1.3.1/dogfight-sandbox-hg2.zip) or [Link 2](https://drive.google.com/file/d/1FihtrwnwGt0FXaVlGS4881yN3oYpbdlw/view?usp=drive_link). -->

### Install HIRL

```bash
conda create -n harfang_env python=3.8
conda activate harfang_env
git clone https://github.com/zrc0622/HIRL4UCAV.git
cd HIRL4UCAV
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
```

## Usage

### Prerequisites
Download the `expert_data` and `bc_actor` folders from [Google Drive](https://drive.google.com/drive/folders/1lAllxmsy0MhW714ZmT8fb0MkdJktUxzJ?usp=sharing) and place them in the repository. 
Update the IP address in the `local_config.yaml` file.

### Run Experiments
Once the prerequisites are set, you can run experiments using the following command:

```bash
python harfang_env/train_all.py --port=<ENV_PORT> --env=<ENV_TYPE> --random --agent=HIRL --type=<HIRL_TYPE> --model_name=<MODEL_NAME>
```

Replace placeholders with your specific setup.

<!-- - `<ENV_PORT>`: The port number for the training environment (e.g., 12345).
- `<ENV_TYPE>`: The type of training environment (e.g., "straight_line", "serpentine", "circular").
- `<HIRL_TYPE>`: The variant of the HIRL algorithm (e.g., "soft", "linear", "fixed").
- `<MODEL_NAME>`: The name of the trained model to be saved (e.g., "HIRL_soft"). -->

## Performance

### Comparative Results

<div align="center">
  <img src="./assets/fig1.png" width="100%"/>
</div>

### Policy Trajectories

<div align="center">
  <img src="./assets/fig2.png" width="100%"/>
</div>

## Citation

```bibtex
@misc{li2024imitative,
    title={An Imitative Reinforcement Learning Framework for Autonomous Dogfight}, 
    author={Siyuan Li and Rongchang Zuo and Peng Liu and Yingnan Zhao},
    year={2024},
    eprint={2406.11562},
    archivePrefix={arXiv}
}
```