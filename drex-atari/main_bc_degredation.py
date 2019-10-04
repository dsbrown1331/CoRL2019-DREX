import torch
import argparse
import numpy as np
from train import train
from pdb import set_trace
import dataset
import tensorflow as tf
from run_test import *
from baselines.common.trex_utils import preprocess

def generate_novice_demos(env, env_name, agent, model_dir):
    checkpoint_min = 550
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    #if env_name == "pong":
    #    checkpoints = ['00025','00050','00175','00200','00250','00350','00450','00500','00550','00600','00700','00700']
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

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
                ob_processed = preprocess(ob, env_name)
                #ob_processed = ob_processed[0] #get rid of spurious first dimension ob.shape = (1,84,84,4)
                traj.append((ob_processed,action))
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


def generate_expert_demos(env, env_name, agent, epsilon_greedy):


    demonstrations = []
    learning_returns = []
    learning_rewards = []
    model_path = "path_to_model"

    agent.load(model_path)
    episode_count = 25
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
            if np.random.rand() < epsilon_greedy:
                action = [env.action_space.sample()]
            else:
                action = agent.act(ob, r, done)
            ob_processed = preprocess(ob, env_name)
            traj.append((ob_processed, action))
            ob, r, done, _ = env.step(action)
            #print(ob.shape)

            gt_rewards.append(r[0])
            steps += 1
            acc_reward += r[0]
            if done or steps > 4000:
                print("steps: {}, return: {}".format(steps,acc_reward))
                break
        if acc_reward > 300:
            print("traj length", len(traj))
            demonstrations.append(traj)
            print("demo length", len(demonstrations))
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
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--env_name", type=str, help="Atari environment name in lowercase, i.e. 'beamrider'")
    parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")

    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--l2-penalty", type=float, default=0.00)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--num_eval_episodes", type=int, default = 30)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--use_best', default=1.0, type=float, help="fraction of best demos to use for BC")
    parser.add_argument('--epsilon_greedy', default = 0.01, type=float, help="fraction of actions chosen at random for rollouts")

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

    extra_checkpoint_info = "novice_demos"

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

    #demonstrations, learning_returns, _ = generate_demos(env, env_name, agent, args.epsilon_greedy)
    demonstrations, learning_returns, _ = generate_novice_demos(env, env_name, agent, args.models_dir)

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
        dataset_size*7,
        data, args.num_eval_episodes,
        args.epsilon_greedy,
        extra_checkpoint_info)
