import argparse
#Take a cloned policy and plot the degredation


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


def generate_demos(env, env_name, agent, checkpoint_path, num_demos):
    print("generating demos from checkpoint:", checkpoint_path)

    demonstrations = []
    learning_returns = []

    model_path = checkpoint_path

    agent.load(model_path)
    episode_count = num_demos
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
                print("demo: {}, steps: {}, return: {}".format(i, steps,acc_reward))
                break
        print("traj length", len(traj))
        print("demo length", len(demonstrations))
        demonstrations.append(traj)
        learning_returns.append(acc_reward)
    print("Mean", np.mean(learning_returns), "Max", np.max(learning_returns))

    return demonstrations, learning_returns


#Takes as input a list of lists of demonstrations where first list is lowest ranked and last list is highest ranked
def create_training_data_from_bins(_ranked_demos, num_snippets, min_snippet_length, max_snippet_length):


    step = 2
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







class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.shape)
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        #x = x.view(-1, 1936)
        x = F.leaky_relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print("sum rewards", sum_rewards)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j




def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
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
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 1000 == 999:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")






def calc_accuracy(reward_network, training_inputs, training_outputs):
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
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
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
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
    parser.add_argument("--num_bc_eval_episodes", type=int, default = 0, help="number of epsilon greedy BC demos to generate")
    parser.add_argument("--num_epsilon_greedy_demos", type=int, default=20, help="number of times to generate rollouts from each noise level")
    parser.add_argument("--checkpoint_path", help="path to checkpoint to run agent for demos")
    parser.add_argument("--num_demos", help="number of demos to generate", default=10, type=int)
    parser.add_argument("--num_bc_steps", default = 50000, type=int, help='number of steps of BC to run')

    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hist-len", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--l2-penalty", type=float, default=0.00)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument('--epsilon_greedy', default = 0.0, type=float, help="fraction of actions chosen at random for rollouts")

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

    extra_checkpoint_info = "bc_degredation"  #for finding checkpoint again


    hist_length = 4


    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })

    stochastic = True
    env = VecFrameStack(env, 4)
    demonstrator = PPO2Agent(env, env_type, stochastic)

    ##generate demonstrations for use in BC
    demonstrations, learning_returns = generate_demos(env, env_name, demonstrator, args.checkpoint_path, args.num_demos)

    #Run BC on demos
    dataset_size = sum([len(d) for d in demonstrations])
    print("Data set size = ", dataset_size)


    episode_index_counter = 0
    num_data = 0
    action_set = set()
    action_cnt_dict = {}
    data = []
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
            data.append((state, action))

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

    print("available actions", action_set)
    print(action_cnt_dict)

    agent = train(args.env_name,
        action_set,
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

    #minimal_action_set = acion_set
    epsilon_greedy_list = [0.01]
    for i in range(20):
        epsilon_greedy_list.append(0.05 + i*0.05)
    print("epsilon_greedy_list", epsilon_greedy_list)

    #agent = Clone(list(minimal_action_set), hist_length, args.checkpoint_bc_policy)
    print("beginning evaluation")
    generator = DemoGenerator(agent, args.env_name, args.num_epsilon_greedy_demos, args.seed)
    batch_rewards = generator.get_pseudo_ranking_returns(epsilon_greedy_list)

    #write the batch rewards out to a file
    writer = open("./bc_degredation_data/" + env_name + "_batch_rewards.csv",'w')
    #first write the returns of the demonstrations
    writer.write("demos, ")
    for i in range(len(learning_returns)-1):
        writer.write("{}, ".format(learning_returns[i]))
    writer.write("{}\n".format(learning_returns[-1]))
    for i,batch in enumerate(batch_rewards):
        writer.write("{}, ".format(epsilon_greedy_list[i]))
        for i in range(len(batch)-1):
            writer.write("{}, ".format(batch[i][0]))
        writer.write("{}\n".format(batch[-1][0]))
