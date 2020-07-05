import argparse
import os
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

import gym

import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
        'figure.figsize': (5, 4),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
from matplotlib import pyplot as plt

from bc_mujoco import Policy
from utils import RandomAgent, gen_traj
################################

class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=0.033, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

################################

class NoiseInjectedPolicy(object):
    def __init__(self,env,policy,action_noise_type,noise_level):
        self.action_space = env.action_space
        self.policy = policy
        self.action_noise_type = action_noise_type

        if action_noise_type == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mu=mu,sigma=std)
        elif action_noise_type == 'ou':
            mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=mu,sigma=std)
        elif action_noise_type == 'epsilon':
            self.epsilon = noise_level
            self.scale = self.action_space.high[0]
            assert np.all(self.scale == self.action_space.high) and \
                np.all(self.scale == -1.*self.action_space.low)
        else:
            assert False, "no such action noise type: %s"%(action_noise_type)

    def act(self, obs, reward, done):
        if self.action_noise_type == 'epsilon':
            if np.random.random() < self.epsilon:
                return np.random.uniform(-self.scale,self.scale,self.action_space.shape)
            else:
                act = self.policy.act(obs,reward,done)
        else:
            act = self.policy.act(obs,reward,done)
            act += self.action_noise()

        return np.clip(act,self.action_space.low,self.action_space.high)

    def reset(self):
        self.action_noise.reset()

################################
class BCNoisePreferenceDataset(object):
    def __init__(self,env,max_steps=None,min_margin=None):
        self.env = env

        self.max_steps = max_steps
        self.min_margin = min_margin

    def prebuild(self,agent,noise_range,num_trajs,min_length,logdir):
        trajs = []
        for noise_level in tqdm(noise_range):
            noisy_policy = NoiseInjectedPolicy(self.env,agent,'epsilon',noise_level)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs,actions,rewards = gen_traj(self.env,noisy_policy,-1)
                agent_trajs.append((obs,actions,rewards))

            trajs.append((noise_level,agent_trajs))

        self.trajs = trajs

        with open(os.path.join(logdir,'prebuilt.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

    def load_prebuilt(self,fname):
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                self.trajs = pickle.load(f)
            return True
        else:
            return False

    def draw_fig(self,log_dir,demo_trajs):
        demo_returns = [np.sum(rewards) for _,_,rewards in demo_trajs]
        demo_ave, demo_std = np.mean(demo_returns), np.std(demo_returns)

        noise_levels = [noise for noise,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs in self.trajs])

        random_agent = RandomAgent(self.env.action_space)
        random_returns = [np.sum(gen_traj(self.env,random_agent,-1)[2]) for _ in range(20)]
        random_ave, random_std = np.mean(random_returns), np.std(random_returns)

        from_to = [np.min(noise_levels), np.max(noise_levels)]

        plt.figure()
        plt.fill_between(from_to,
                         [demo_ave - demo_std, demo_ave - demo_std], [demo_ave + demo_std, demo_ave + demo_std], alpha = 0.3)
        plt.plot(from_to,[demo_ave, demo_ave], label='demos')

        plt.fill_between(noise_levels,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(noise_levels, np.mean(returns, axis = 1),'-.', label="bc")

        #plot the average of pure noise in dashed line for baseline
        plt.fill_between(from_to,
                         [random_ave - random_std, random_ave - random_std], [random_ave + random_std, random_ave + random_std], alpha = 0.3)
        plt.plot(from_to,[random_ave, random_ave], '--', label='random')

        plt.legend(loc="best")
        plt.xlabel("Epsilon")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,"degredation_plot.pdf"))
        plt.close()

    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1) # if noise level is small, then it is better traj.
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                        )

        return D

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True, help='log dir')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--bc_agent',required=True)
    parser.add_argument('--demo_trajs',required=True, help='suboptimal demo trajectories used for bc (used to generate a figure)')
    # Noise Injection Hyperparams
    parser.add_argument('--noise_range', default='np.arange(0.,1.0,0.05)', help='decide upto what learner stage you want to give')
    parser.add_argument('--num_trajs', default=5,type=int, help='number of trajectory generated by each agent')
    parser.add_argument('--min_length', default=0,type=int, help='minimum length of trajectory generated by each agent')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    # Generate a Noise Injected Trajectories
    env = gym.make(args.env_id)
    env.seed(args.seed)

    dataset = BCNoisePreferenceDataset(env)
    agent = Policy(env)
    agent.load(args.bc_agent)

    dataset.prebuild(agent,eval(args.noise_range),args.num_trajs,args.min_length,args.log_dir)

    with open(args.demo_trajs,'rb') as f:
        demo_trajs = pickle.load(f)

    dataset.draw_fig(args.log_dir,demo_trajs)
