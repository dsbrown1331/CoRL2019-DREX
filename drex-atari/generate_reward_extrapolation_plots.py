
# coding: utf-8

# In[1]:


#jupyter notebook for generating reward extrapolation plots


# In[2]:

import sys
import argparse
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
from LearnAtariSyntheticRankingsBinning import *


# In[3]:


env_name = sys.argv[1]
model_dir = './models/'
if env_name == 'breakout':
    checkpoint_path = model_dir + 'models/breakout_25/01200'
elif env_name == "beamrider":
    checkpoint_path = model_dir + 'models/beamrider_25/00700'
elif env_name == "enduro":
    checkpoint_path = model_dir +'models/enduro_25/03600'
elif env_name == "pong":
    checkpoint_path = model_dir + "models/pong_25/00750"
elif env_name == "qbert":
    checkpoint_path = model_dir + "models/qbert_25/00500"
elif env_name == "seaquest":
    checkpoint_path = model_dir + "models/seaquest_5/00070"
elif env_name == "spaceinvaders":
    checkpoint_path = model_dir + "models/spaceinvaders_25/00750"

seed = 0
num_demos = 10
num_eval_episodes = 20

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
seed = int(seed)
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
num_snippets = 40000
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


env = VecFrameStack(env, 4)
demonstrator = PPO2Agent(env, env_type, stochastic)


# In[4]:


##generate demonstrations for use in BC
demonstrations, learning_returns = generate_demos(env, env_name, demonstrator, checkpoint_path, num_demos)

#Run BC on demos
dataset_size = sum([len(d) for d in demonstrations])
print("Data set size = ", dataset_size)


# In[5]:


#get minimal action set
episode_index_counter = 0
num_data = 0
action_set = set()

for episode in demonstrations:
    for sa in episode:
        state, action = sa
        action = action[0]
        action_set.add(action)
print(action_set)


# In[6]:


print(demonstrations[0][0][0].shape)


# In[7]:


#get ride of action labels and spurious first dimension from gym to make it easier to predict returns of demonstrations
_demonstrations = demonstrations
demonstrations = []
for _t in _demonstrations:
    t = []
    for _ob,a in _t:
        ob = _ob[0]
        t.append(ob)
    demonstrations.append(t)
print(len(demonstrations))


# In[8]:


checkpoint_policy = 'checkpoints/' + env_name +'_novice_demos_network.pth.tar'

epsilon_greedy_list = [1.0,0.75,0.5,0.25,0.01]#, 0.4, 0.2, 0.1]#[1.0, 0.5, 0.3, 0.1, 0.01]

agent = Clone(list(action_set), hist_length, checkpoint_policy)
print("beginning evaluation")
generator = DemoGenerator(agent, env_name, num_eval_episodes, seed)
ranked_demos, demo_returns = generator.get_pseudo_rankings(epsilon_greedy_list, returns = True)
print(len(ranked_demos))
print(demo_returns)

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


# In[9]:


#generate some extrapolation trajectories
#generate some trajectories for inspecting learned reward

if env_name == "breakout":
    checkpoint_min = 1250
    checkpoint_max = 1450
    checkpoint_step = 25
    episode_count = 2
elif env_name == "beamrider":
    checkpoint_min = 700
    checkpoint_max = 1450
    checkpoint_step = 25
    episode_count = 1
elif env_name == "enduro":
    checkpoint_min = 3625
    checkpoint_max = 4425
    checkpoint_step = 50
    episode_count = 1
elif env_name == "pong":
    checkpoint_min = 800
    checkpoint_max = 1400
    checkpoint_step = 25
    episode_count = 1
elif env_name == "seaquest":
    checkpoint_min = 100
    checkpoint_max = 1400
    checkpoint_step = 100
    episode_count = 1
elif env_name == "qbert":
    checkpoint_min = 550
    checkpoint_max = 1450
    checkpoint_step = 50
    episode_count = 1
elif env_name == "spaceinvaders":
    checkpoint_min = 750
    checkpoint_max = 1450
    checkpoint_step = 25
    episode_count = 1
checkpoints_extrapolate = []
for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints_extrapolate.append('0000' + str(i))
        elif i < 100:
            checkpoints_extrapolate.append('000' + str(i))
        elif i < 1000:
            checkpoints_extrapolate.append('00' + str(i))
        elif i < 10000:
            checkpoints_extrapolate.append('0' + str(i))
print(checkpoints_extrapolate)


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 7, stride=3)
        self.conv2 = nn.Conv2d(32, 16, 5, stride=2)
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
        r = torch.sigmoid(self.fc2(x)) #clip reward?
        #r = F.celu(self.fc2(x))
        #r = self.fc2(x)
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



# In[11]:


#import the reward network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
reward_net_path = "./learned_models/" + env_name +"_five_bins_noop_earlystop.params"
reward_net = Net()
reward_net.load_state_dict(torch.load(reward_net_path))
reward_net.to(device)


# In[12]:


learning_returns_extrapolate = []
pred_returns_extrapolate = []


for checkpoint in checkpoints_extrapolate:

    model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint

    demonstrator.load(model_path)

    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = demonstrator.act(ob, r, done)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            ob_processed = ob_processed[0] #get rid of spurious first dimension ob.shape = (1,84,84,4)
            traj.append(ob_processed)
            steps += 1
            acc_reward += r[0]
            if done:
                print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                break
        print("traj length", len(traj))
        #print("demo length", len(demonstrations))
        #demonstrations.append(traj)
        learning_returns_extrapolate.append(acc_reward)
        pred_returns_extrapolate.append(reward_net.cum_return(torch.from_numpy(np.array(traj)).float().to(device))[0].item())
        print("pred return", pred_returns_extrapolate[-1])


# In[13]:


batch_pred_returns = []
with torch.no_grad():
    for bin_num, r_demonstrations in enumerate(ranked_demos):
        pred_returns = [predict_traj_return(reward_net, traj) for traj in r_demonstrations]
        batch_pred_returns.append(pred_returns)
        print("Epsilon = ", epsilon_greedy_list[bin_num], "bin results:")
            #print("Demonstrator demos:")
        for i, p in enumerate(pred_returns):
            print(i,p, demo_returns[bin_num][i])


# In[14]:


print(len(learning_returns))
print(len(demonstrations))
print(demonstrations[0][0].shape)
with torch.no_grad():
    pred_demo_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_demo_returns):
        print(i,p, learning_returns[i])


# In[15]:


def convert_range(x,minimum, maximum,a,b):
    return (x - minimum)/(maximum - minimum) * (b - a) + a


# In[16]:



import matplotlib.pylab as plt
learning_returns_noise = []
for rbin in demo_returns:
    for r in rbin:
        learning_returns_noise.append(r)

pred_returns_noise = []
for rbin in batch_pred_returns:
    for r in rbin:
        pred_returns_noise.append(r)

learning_returns_all = np.array(learning_returns_noise + learning_returns + learning_returns_extrapolate)
pred_returns_all = np.array(pred_returns_noise + pred_demo_returns + pred_returns_extrapolate)
learning_returns_noise = np.array(learning_returns_noise)
pred_returns_noise = np.array(pred_returns_noise)
learning_returns = np.array(learning_returns)
pred_returns = np.array(pred_returns)


# In[17]:


learning_returns_extrapolate = np.array(learning_returns_extrapolate)
pred_returns_extrapolate = np.array(pred_returns_extrapolate)


# In[18]:


# buffer = 10
# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'xx-large',
#           'figure.figsize': (6, 5),
#          'axes.labelsize': 'xx-large',
#          'axes.titlesize':'xx-large',
#          'xtick.labelsize':'xx-large',
#          'ytick.labelsize':'xx-large'}
# pylab.rcParams.update(params)
# print(pred_returns_all)
# print(learning_returns_all)
# #plt.plot(learning_returns_all, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_all],'ro')
# plt.plot(learning_returns_noise/max(learning_returns_all), pred_returns_noise/max(pred_returns_all), 'bo')
# plt.plot(learning_returns/max(learning_returns_all), pred_demo_returns/max(pred_returns_all), 'ro')
# plt.plot(learning_returns_extrapolate/max(learning_returns_all), pred_returns_extrapolate/max(pred_returns_all), 'go')
# #plt.plot(learning_returns_extrapolate, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_extrapolate],'bo')
# plt.plot([0,1],[0,1],'k--')
# #plt.plot([0,max(learning_returns_demos)],[0,max(learning_returns_demos)],'k-', linewidth=2)
# #plt.axis([0,max(learning_returns_all) + buffer,0,max(learning_returns_all)+buffer])
# plt.xlabel("Ground Truth Returns")
# plt.ylabel("Predicted Returns (normalized)")
# plt.tight_layout()
buffer = 20
if env_name == "pong":
    buffer = 2
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         # 'figure.figsize': (6, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

print(pred_returns_all)
print(learning_returns_all)
plt.plot(learning_returns_noise, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_noise],'bo')
plt.plot(learning_returns, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_demo_returns],'ro')
plt.plot(learning_returns_extrapolate, [convert_range(p,max(pred_returns_all), min(pred_returns_all),max(learning_returns_all), min(learning_returns_all)) for p in pred_returns_extrapolate],'go')
plt.plot([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],[min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer],'k--')
plt.axis([min(0, min(learning_returns_all)-2),max(learning_returns_all) + buffer,min(0, min(learning_returns_all)-2),max(learning_returns_all)+buffer])
plt.xlabel("Ground Truth Returns")
plt.ylabel("Predicted Returns (normalized)")
plt.tight_layout()
plt.savefig("./figs/" + env_name + "_gt_vs_pred_rewards_progress_sigmoid.png")

#plt.axis('square')
