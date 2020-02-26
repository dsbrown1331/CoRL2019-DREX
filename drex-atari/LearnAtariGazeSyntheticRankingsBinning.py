import sys
sys.path.insert(0, './baselines')

import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.trex_utils import preprocess
from bc import Clone
from synthesize_rankings_bc import DemoGenerator
from train import train
from pdb import set_trace
from main_bc_degredation import generate_novice_demos
from run_test import PPO2Agent
import dataset

from gaze_cnn import Net
import atari_head_dataset as ahd 
import gaze_utils
from tensorboardX import SummaryWriter

def create_gaze_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length, gaze_coords, use_gaze):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    training_gaze = []
    num_demos = len(demonstrations)

    
    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        
        #pick two random demonstrations
        ti = np.random.randint(num_demos)
        tj = np.random.randint(num_demos)

        #create random snippets, doesn't matter where they are since we don't have rankings
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length_ti = len(demonstrations[ti])
        min_length_tj = len(demonstrations[tj])
        #print(min_length)
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        ti_start = np.random.randint(min_length_ti - rand_length + 1)
        tj_start = np.random.randint(min_length_tj - rand_length + 1)
        
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip every other framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        
        gaze_i = gaze_coords[ti][ti_start:ti_start+rand_length:2]
        gaze_j = gaze_coords[tj][tj_start:tj_start+rand_length:2]

        training_obs.append((traj_i, traj_j))
        training_gaze.append((gaze_i, gaze_j))

    return training_obs, training_gaze

#Takes as input a list of lists of demonstrations where first list is lowest ranked and last list is highest ranked
def create_training_data_from_bins(_ranked_demos, num_snippets, min_snippet_length, max_snippet_length):


    step = 4
    #n_train = 3000 #number of pairs of trajectories to create
    #snippet_length = 50
    training_obs = []
    training_labels = []
    num_ranked_bins = len(ranked_demos)
    #pick progress based snippets
    for n in range(num_snippets):
        #pick two distinct bins at random
        bi = 0
        bj = 0
        #only add trajectories that are different returns
        while(bi == bj):
            #pick two random demonstrations
            bi = np.random.randint(num_ranked_bins)
            bj = np.random.randint(num_ranked_bins)
        #given these two bins, we now pick a random trajectory from each bin
        ti = random.choice(ranked_demos[bi])
        tj = random.choice(ranked_demos[bj])

        #Given these trajectories create a random snippet
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(ti), len(tj))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        #print("batch size", rand_length)
        if bi < bj: #bin_j is better so pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(tj) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(ti) - rand_length + 1)
        #print("start", ti_start, tj_start)
        snip_i = ti[ti_start:ti_start+rand_length:step] #use step to skip everyother framestack to reduce size
        snip_j = tj[tj_start:tj_start+rand_length:step]
            #print('traj', traj_i, traj_j)
            #return_i = sum(learning_rewards[ti][ti_start:ti_start+snippet_length])
            #return_j = sum(learning_rewards[tj][tj_start:tj_start+snippet_length])
            #print("returns", return_i, return_j)

        #if return_i > return_j:
        #    label = 0
        #else:
        #    label = 1
        if bi > bj:
            label = 0
        else:
            label = 1
        training_obs.append((snip_i, snip_j))
        training_labels.append(label)


    return training_obs, training_labels






def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir, use_gaze, gaze_obs, gaze_coords, gaze_loss_type, gaze_reg, gaze_conv_layer):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    if use_gaze: #only loss supported
        gaze_training_data = list(zip(gaze_obs, gaze_coords))

    #partition into training and validation sets with 90/10 split
    np.random.shuffle(training_data)
    training_data_size = int(len(training_data) * 0.8)
    training_dataset = training_data[:training_data_size]
    validation_dataset = training_data[training_data_size:]
    print("training size = {}, validation size = {}".format(len(training_dataset), len(validation_dataset)))

    best_v_accuracy = -np.float('inf')
    early_stop = False
    for epoch in range(num_iter):
        np.random.shuffle(training_dataset)
        training_obs, training_labels = zip(*training_dataset)
        if use_gaze:
            np.random.shuffle(gaze_training_data)
            gaze_training_obs, gaze_training_coords = zip(*gaze_training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i) / 255.
            traj_j = np.array(traj_j) / 255.
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards, _, _ = reward_network.forward(traj_i, traj_j, gaze_conv_layer)
            #print(outputs[0], outputs[1])
            #print(labels.item())
            outputs = outputs.unsqueeze(0)
            #print("outputs", outputs)
            #print("labels", labels)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            # if labels == 0:
            #     #print("label 0")
            #     loss = torch.log(1 + torch.exp(outputs[1] - outputs[0]))
            # else:
            #     #print("label 1")
            #     loss = torch.log(1 + torch.exp(outputs[0] - outputs[1]))
            
            
            
            
            if gaze_loss_type == 'KL':
                #get gaze conv maps
                obs_i, obs_j = gaze_training_obs[i]
                obs_i = np.array(obs_i) / 255.
                obs_j = np.array(obs_j) / 255.
                obs_i = torch.from_numpy(obs_i).float().to(device)
                obs_j = torch.from_numpy(obs_j).float().to(device)
                _, _, conv_map_i, conv_map_j = reward_network.forward(obs_i, obs_j, gaze_conv_layer)

                # ground truth human gaze maps (7x7)
                gaze_i, gaze_j = gaze_training_coords[i]

                gaze_i = torch.squeeze(torch.tensor(gaze_i, device=device)) # list of torch tensors
                gaze_j = torch.squeeze(torch.tensor(gaze_j, device=device))

                if gaze_loss_type == 'KL':
                    gaze_loss_i = get_gaze_KL_loss(gaze_i, torch.squeeze(conv_map_i))
                    gaze_loss_j = get_gaze_KL_loss(gaze_j, torch.squeeze(conv_map_j))

                    gaze_loss_total = (gaze_loss_i + gaze_loss_j)
                    writer.add_scalar('KL_loss', gaze_loss_total.item(), epoch*len(training_labels)+i) 

                loss += gaze_reg * gaze_loss_total
            
            
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 1000 == 999  :
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                #evaluate validation_dataset error
                validation_obs, validation_labels = zip(*validation_dataset)
                v_accuracy = calc_accuracy(reward_net, validation_obs, validation_labels, gaze_conv_layer)
                print("Validation accuracy = {}".format(v_accuracy))
                if v_accuracy > best_v_accuracy:
                    print("check pointing")
                    torch.save(reward_net.state_dict(), checkpoint_dir)
                    count = 0
                    best_v_accuracy = v_accuracy
                else:
                    count += 1
                    if count > 5:
                        print("Stopping to prevent overfitting after {} ".format(count))
                        early_stop = True
                        break
                print(abs_rewards)
                cum_loss = 0.0
        if early_stop:
            print("early stop!!!!!!!")
            break
    print("finished training")






def calc_accuracy(reward_network, training_inputs, training_outputs, gaze_conv_layer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            #print(inputs)
            #print(labels)
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i) / 255.
            traj_j = np.array(traj_j) / 255.
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return, _, _ = reward_network.forward(traj_i, traj_j, gaze_conv_layer)
            #print(outputs)
            _, pred_label = torch.max(outputs,0)
            #print(pred_label)
            #print(label)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s]) / 255.).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


def get_gaze_KL_loss(true_gaze, conv_gaze): # order of 60s
    import torch.nn.functional as F
    loss = 0
    batch_size = true_gaze.shape[0]

    #print("true_gaze", true_gaze.shape)
    #print("conv_gaze_shape", conv_gaze.shape)
    # assert batch size for both conv and true gaze is the same
    assert(batch_size==conv_gaze.shape[0])

    epsilon = 1e-10 # introduce epsilon to avoid log and division by zero error
    true_gaze2 = torch.clamp(true_gaze, epsilon, 1)
    conv_gaze = torch.clamp(conv_gaze, epsilon, 1)
    return torch.sum(torch.mul(torch.mul(true_gaze2,true_gaze),torch.log(torch.div(true_gaze2 ,conv_gaze))))/batch_size



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument("--num_bc_eval_episodes", type=int, default = 5, help="number of epsilon greedy BC demos to generate")
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument("--num_epsilon_greedy_demos", type=int, default=20, help="number of times to generate rollouts from each noise level")
    parser.add_argument("--num_demos", help="number of demos to generate", default=10, type=int)
    parser.add_argument("--num_bc_steps", default = 75000, type=int, help='number of steps of BC to run')
   
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hist-len", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--l2-penalty", type=float, default=0.00)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument('--epsilon_greedy', default = 0.0, type=float, help="fraction of actions chosen at random for rollouts")


    parser.add_argument('--data_dir', help="where atari-head data is located")
    parser.add_argument('--gaze_loss', default=None, type=str, help="type of gaze loss function: sinkhorn, exact, coverage, KL, None")
    parser.add_argument('--gaze_reg', default=0.01, type=float, help="gaze loss multiplier")
    parser.add_argument('--gaze_conv_layer', default=4, type=int, help='the convlayer of the reward network to which gaze should be compared')
    parser.add_argument('--use_motion', action="store_true")


    parser.add_argument('--num_snippets', default = 40000, type=int)


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
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    print("Training reward for", env_id)
    #n_train = 200 #number of pairs of trajectories to create
    #snippet_length = 50 #length of trajectory for training comparison
    lr = 0.00001
    weight_decay = 0.0
    num_iter = 10 #num times through training data
    l1_reg=0.0
    stochastic = True
    bin_width = 0 #only bin things that have the same score
    num_snippets = args.num_snippets
    min_snippet_length = 50
    max_snippet_length = 200
    extra_checkpoint_info = "novice_demos"  #for finding checkpoint again
    epsilon_greedy_list = [1.0,0.75,0.5,0.25,0.01]#, 0.4, 0.2, 0.1]#[1.0, 0.5, 0.3, 0.1, 0.01]



    hist_length = 4


    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    
    action_meanings = env.unwrapped.envs[0].unwrapped.get_action_meanings()

    print("="*10)
    print(action_meanings)
    num_actions = len(action_meanings)
    print("="*10)


    env = VecFrameStack(env, 4)

    # gaze-related arguments
    use_gaze = args.gaze_loss
    gaze_loss_type = args.gaze_loss
    gaze_reg = args.gaze_reg
    gaze_conv_layer = args.gaze_conv_layer

    print("Usinging gaze set to {}!!".format(use_gaze))

    print("Downloading gaze data for BC with actions")
    data_dir = args.data_dir
    gaze_dataset = ahd.AtariHeadDataset(env_name, data_dir)
    demonstrations, learning_returns, learning_rewards, learning_gaze = gaze_utils.get_preprocessed_trajectories(env_name, gaze_dataset, data_dir, use_gaze, gaze_conv_layer)
    print([len(d) for d in demonstrations])
    print([r for r in learning_returns])
    print("demos downloaded")


    


    #Run BC on demos
    dataset_size = sum([len(d) for d in demonstrations])
    print("Data set size = ", dataset_size)


    episode_index_counter = 0
    num_data = 0
    action_set = set()
    action_cnt_dict = {}
    data = []
    cnt = 0
    for episode in demonstrations:
        print("adding demonstration", cnt)
        cnt += 1
        
        for sa in episode:
            state, action = sa
            #Need to translate from gaze actions to ale gym
            action = gaze_utils.translate_action(action, action_meanings)
            action_set.add(action)
            if action in action_cnt_dict:
                action_cnt_dict[action] += 1
            else:
                action_cnt_dict[action] = 0
          
            #transpose into 4x84x84 format
            state = np.transpose(np.squeeze(state), (2, 0, 1))
            data.append((state, action))
        #del demonstrations[0]
    #del demonstrations

    #take 10% as validation data
    np.random.shuffle(data)
    training_data_size = int(len(data) * 0.8)
    training_data = data[:training_data_size]
    validation_data = data[training_data_size:]
    print("training size = {}, validation size = {}".format(len(training_data), len(validation_data)))
    training_dataset = dataset.Dataset(training_data_size, hist_length)
    validation_dataset = dataset.Dataset(len(validation_data), hist_length)
    for state,action in training_data:
        training_dataset.add_item(state, action)
        num_data += 1
        if num_data == training_dataset.size:
            print("data set full")
            break

    for state, action in validation_data:
        validation_dataset.add_item(state, action)
        num_data += 1
        if num_data == validation_dataset.size:
            print("data set full")
            break
    del training_data
    del validation_data
    print("available actions", action_set)
    print(action_cnt_dict)


    agent = train(args.env_name,
        action_meanings,
        args.learning_rate,
        args.alpha,
        args.l2_penalty,
        args.minibatch_size,
        args.hist_len,
        args.discount,
        args.checkpoint_dir,
        args.num_bc_steps,
        training_dataset,
        validation_dataset,
        args.num_bc_eval_episodes,
        0.01,
        extra_checkpoint_info)
    del training_dataset
    del validation_dataset
    #minimal_action_set = acion_set

    #agent = Clone(list(minimal_action_set), hist_length, args.checkpoint_bc_policy)
    print("beginning evaluation")
    generator = DemoGenerator(agent, args.env_name, args.num_epsilon_greedy_demos, args.seed)
    ranked_demos = generator.get_pseudo_rankings(epsilon_greedy_list, add_noop=True)

    # ## Add the demonstrators demos as the highest ranked batch of trajectories, don't need actions
    demo_demos = []
    for d in demonstrations:
        traj = []
        for s,a in d:
            traj.append(np.expand_dims(s, axis=0))
        demo_demos.append(traj)
    ranked_demos.append(demo_demos)



    # input("let's check the demos")
    # print(len(ranked_demos))
    # for dset in ranked_demos:
    #     print("this many demos in first bin")
    #     print(len(dset))
    #     print(dset[0][0].shape)
    #     print(type(dset[0][0][0,0,0,0]))
    #     print(dset[0][0][0,20:40,20:40,:])
    # input()

    #remove the extra first dimension on the observations
    _ranked_demos = ranked_demos
    ranked_demos = []
    for _r in _ranked_demos:
        r = []
        for _d in _r:
            d = []
            for _ob in _d:
                ob = _ob[0]
                d.append(ob)
            r.append(d)
        ranked_demos.append(r)

    print("Learning from ", len(ranked_demos), "synthetically ranked batches of demos")

    if use_gaze:
        gaze_obs, gaze_coords = create_gaze_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length, learning_gaze, use_gaze)
    else:
        gaze_obs = []
        gaze_coords = []

    training_obs, training_labels = create_training_data_from_bins(ranked_demos, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(gaze_loss_type)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path, use_gaze, gaze_obs, gaze_coords, gaze_loss_type, gaze_reg, gaze_conv_layer)
    torch.cuda.empty_cache() 


    with torch.no_grad():
        for bin_num, demonstrations in enumerate(ranked_demos):
            pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
            if bin_num == 0:
                print("No-op demos")

            elif bin_num < len(ranked_demos) - 1:
                print("Epsilon = ", epsilon_greedy_list[bin_num-1], "bin results:")
                
            else:
                print("Demonstrator demos:")
            for i, p in enumerate(pred_returns):
                print(i,p)


    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels, gaze_conv_layer))


    #TODO:add checkpoints to training process
    torch.save(reward_net.state_dict(), args.reward_model_path)
