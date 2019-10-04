# Generative Adversarial Imitation Learning (GAIL)

## Mujoco

- Train

```
mpirun -np 4 python -m baselines.gail.run_mujoco --env_id Hopper-v2 --expert_path ./path/to/data/*.npz --log_dir ./log/dir/you/want
```

- Eval

```
python -m baselines.gail.run_mujoco --task evaluate --load_model_path ./path/to/model --stochastic_policy
```

- Generate Dataset

  - Check the parent projects `gail_dst_gen.py`.

    ```
    python gail_dst_gen.py --env_id 'Hopper-v2' --learners_path ./learner/demo_models/hopper/checkpoints/ --train_chkpt '120' --num_trajs 1 --stochastic
    ```

## Atari

- Generate Dataset

    ```
    cd baselines/gail/dataset
    python atari_gen.py --env_id PongNoFrameskip-v4 --out ../../../data/pong --learners_path ../../../../demo_models/pong/checkpoints --chkpts 3600 --num_traj 3 --stochastic
    ```

- Train

```
mpirun -np 4 python -m baselines.gail.run_atari --env_id PongNoFrameskip-v4 --expert_path ./data/breakout --log_dir ./log
```

- Eval

```
python -m baselines.gail.run_atari --task evaluate --env_id PongNoFrameskip-v4 --load_model_path ./log_atari/with_best_demo/pong/test2/chckpts/model-500 --stochastic_policy
```


# Original README


- Original paper: https://arxiv.org/abs/1606.03476

For results benchmarking on MuJoCo, please navigate to [here](result/gail-result.md)

## If you want to train an imitation learning agent

### Step 1: Download expert data

Download the expert data into `./data`, [download link](https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing)

### Step 2: Run GAIL

Run with single thread:

```bash
python -m baselines.gail.run_mujoco
```

Run with multiple threads:

```bash
mpirun -np 16 python -m baselines.gail.run_mujoco
```

See help (`-h`) for more options.

#### In case you want to run Behavior Cloning (BC)

```bash
python -m baselines.gail.behavior_clone
```

See help (`-h`) for more options.


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/openai/baselines/pulls.

## Maintainers

- Yuan-Hong Liao, andrewliao11_at_gmail_dot_com
- Ryan Julian, ryanjulian_at_gmail_dot_com

## Others

Thanks to the open source:

- @openai/imitation
- @carpedm20/deep-rl-tensorflow
