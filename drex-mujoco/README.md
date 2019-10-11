# D-REX mujoco

## Requirements

This code are tested with `python 3.6` and `tensorflow-gpu==v1.14.0`. For other library dependencies, please refer `requirements.txt` or check `env.yml`.


## Run Experiment

1. Behavior Cloning (BC)

```
python bc_mujoco.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --demo_chkpt 'range(70,71,1)' --stochastic --log_path ./log/hopper/ --mode bc --num_trajs 0 --min_length 1000
python bc_mujoco.py --env_id HalfCheetah-v2 --env_type mujoco --learners_path ./learner/demo_models/halfcheetah/checkpoints --demo_chkpt 'range(60,61,1)' --stochastic --log_path ./log/halfcheetah/ --mode bc --num_trajs 0 --min_length 1000
```

2. Run experiments

```
python bc_noise_trex.py --log_dir ./log/hopper --env_id Hopper-v2 --ctrl_coeff 0.001
python bc_noise_trex.py --log_dir ./log/halfcheetah --env_id HalfCheetah-v2 --ctrl_coeff 0.1
```
