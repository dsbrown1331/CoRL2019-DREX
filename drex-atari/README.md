## behavioral_cloning_atari

#Depenencies
See environment.yaml for conda dependencies

# Code structure
./baselines/ contains code for the OpenAIBaselines that we used for PPO reinforcement learning.

You will need to install baselines using following commands:

cd baselines
pip install -e .


./bc_degredation_data/ contains the data, plots, and code for creating the behavioral cloning degredation plots that vary epsilon greedy noise from 0.01 to 1.0. 

./checkpoints/ [Removed due to size constraints on supplement] contains checkpointed behavioral cloning policies for the different games

./figs/ contains the reward extrapolation plots and the reward attention mask plots

./learned_models/ contains the learned reward models learned by D-REX and used for the attention and extrapolation plots, as well as used for RL using PPO

./models/ [Removed due to size constraints] contains the partially trained PPO checkpoints that we used for algorithmic demonstrations.




# To generate reward degredation data

python bc_degredation_data_generator.py --env_name beamrider --checkpoint_path ./models/beamrider_25/00700

python bc_degredation_data_generator.py --env_name breakout --checkpoint_path ./models/breakout_25/01200

python bc_degredation_data_generator.py --env_name enduro --checkpoint_path ./models/enduro_25/003600

python bc_degredation_data_generator.py --env_name pong --checkpoint_path ./models/pong_25/00750

python bc_degredation_data_generator.py --env_name qbert --checkpoint_path ./models/qbert_25/00500

python bc_degredation_data_generator.py --env_name seaquest --checkpoint_path ./models/seaquest_5/00070

python bc_degredation_data_generator.py --env_name spaceinvaders --checkpoint_path ./models/spaceinvaders_25/00750




# To run reward learning:

python LearnAtariSyntheticRankingsBinning.py --env_name spaceinvaders --reward_model_path ./learned_models/spaceinvaders.params --checkpoint_path ./models/spaceinvaders_25/00500



#To run RL:

First install baselines (see above for instructions)

Then run

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/tflogs/spaceinvaders_0 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path ./learned_models/spaceinvaders.params --seed 0 --num_timesteps=5e7  --save_interval=1000 --num_env 9

The code that takes the learned reward and feeds it to the PPO agent is found in 
./baselines/baselines/common/custon_reward_wrapper.py

The code for masking out game scores and ship counts, etc, is in:
./baselines/baselines/common/trex_utils.py
