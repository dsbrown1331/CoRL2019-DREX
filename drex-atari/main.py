import torch
import argparse
import numpy as np
from train import train
from pdb import set_trace
import dataset
import tensorflow as tf
from run_test import *


def normalize_state(obs):
    obs_highs = env.observation_space.high
    obs_lows = env.observation_space.low
    #print(obs_highs)
    #print(obs_lows)
    #return  2.0 * (obs - obs_lows) / (obs_highs - obs_lows) - 1.0
    return obs / 255.0


def mask_score(obs):
    #takes a stack of four observations and blacks out (sets to zero) top n rows
    n = 10
    #no_score_obs = copy.deepcopy(obs)
    obs[:,:n,:,:] = 0
    return obs

def generate_novice_demos(env, env_name, agent):
    checkpoint_min = 50
    checkpoint_max = 200
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000')
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = "../learning-rewards-of-learners/learner/models/" + env_name + "_25/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                traj.append((mask_score(normalize_state(ob)),action))
                ob, r, done, _ = env.step(action)
                #print(ob.shape)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)
    print(np.mean(learning_returns), np.max(learning_returns))
    return demonstrations, learning_returns, learning_rewards



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ##################################################
    # ##             Algorithm parameters             ##
    # ##################################################
    #parser.add_argument("--dataset-size", type=int, default=75000)
    #parser.add_argument("--updates", type=int, default=10000)#200000)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hist-len", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--env_name", type=str, help="Atari environment name in lowercase, i.e. 'beamrider'")

    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--min-squared-gradient", type=float, default=0.01)
    parser.add_argument("--l2-penalty", type=float, default=0.0001)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--num_eval_episodes", type=int, default = 30)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--use_best', default=1.0, type=float, help="fraction of best demos to use for BC")

    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    stochastic = True


    #load novice demonstrations
    #pkl_file = open("../learning-rewards-of-learners/learner/novice_demos/" + args.env_name + "12_50.pkl", "rb")
    #novice_data = pickle.load(pkl_file)
    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    demonstrations, learning_returns, _ = generate_novice_demos(env, env_name, agent)
    print("choosing best {} percent of demos".format(args.use_best * 100))
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    start_index = len(demonstrations) - int(len(demonstrations) * args.use_best)
    demonstrations = demonstrations[start_index:]
    print(len(demonstrations))
    for d in demonstrations:
        print(len(d))



    dataset_size = sum([len(d) for d in demonstrations])
    print("Data set size = ", dataset_size)

    data = dataset.Dataset(dataset_size, args.hist_len)
    episode_index_counter = 0
    num_data = 0
    action_set = set()
    action_cnt_dict = {}
    for episode in demonstrations:
        for sa in episode:
            state, action = sa
            action = action[0]
            action_set.add(action)
            if action in action_cnt_dict:
                action_cnt_dict[action] += 1
            else:
                action_cnt_dict[action] = 0
            #transpose into 4x84x84 format
            state = np.transpose(np.squeeze(state), (2, 0, 1))*255.
            data.add_item(state, action)
            num_data += 1
            if num_data == dataset_size:
                print("data set full")
                break
        if num_data == dataset_size:
            print("data set full")
            break
    print("available actions", action_set)
    print(action_cnt_dict)

    train(args.env_name,
        action_set,
        args.learning_rate,
        args.alpha,
        args.l2_penalty,
        args.minibatch_size,
        args.hist_len,
        args.discount,
        args.checkpoint_dir,
        dataset_size,
        data,
        [],
        args.num_eval_episodes,
        0.01,
        "standard_bc"
        )
