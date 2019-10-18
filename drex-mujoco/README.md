# D-REX mujoco

## Requirements

This code are tested with `python 3.6` and `tensorflow-gpu==v1.14.0`. For other library dependencies, please refer `requirements.txt` or check `env.yml`.


## Run Experiment

1. Behavior Cloning (BC)

```
python bc_mujoco.py --env_id HalfCheetah-v2 --log_path ./log/drex/halfcheetah/bc --demo_trajs demos/suboptimal_demos/halfcheetah/dataset.pkl
python bc_mujoco.py --env_id Hopper-v2 --log_path ./log/drex/hopper/bc --demo_trajs demos/suboptimal_demos/hopper/dataset.pkl
```

2. Generate Noise Injected Trajectories

```
python bc_noise_dataset.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --bc_agent ./log/drex/halfcheetah/bc/model.ckpt --demo_trajs ./demos/suboptimal_demos/halfcheetah/dataset.pkl
python bc_noise_dataset.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --bc_agent ./log/drex/hopper/bc/model.ckpt --demo_trajs ./demos/suboptimal_demos/hopper/dataset.pkl
```

3. Run T-REX

```
python drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --bc_trajs ./demos/suboptimal_demos/halfcheetah/dataset.pkl --unseen_trajs ./demos/full_demos/halfcheetah/trajs.pkl --noise_injected_trajs ./log/drex/halfcheetah/prebuilt.pkl
python drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --bc_trajs ./demos/suboptimal_demos/hopper/dataset.pkl --unseen_trajs ./demos/full_demos/hopper/trajs.pkl --noise_injected_trajs ./log/drex/hopper/prebuilt.pkl
```

You can download pregenerated unseen trajectories from [here](https://github.com/dsbrown1331/CoRL2019-DREX/releases). Instead, you can just erase the `--unseeen_trajs` option. It just used for generating the plot shown in the paper.

4. Run PPO

```
python drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --mode train_rl --ctrl_coeff 0.1
python drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --mode train_rl --ctrl_coeff 0.001
```
