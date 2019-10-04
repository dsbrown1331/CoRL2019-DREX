#from ale_wrapper import ALEInterfaceWrapper
from preprocess import Preprocessor
from state import *
import numpy as np
import utils
import gym
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.trex_utils import preprocess

def normalize_state(obs):
    #obs_highs = env.observation_space.high
    #obs_lows = env.observation_space.low
    #print(obs_highs)
    #print(obs_lows)
    #return  2.0 * (obs - obs_lows) / (obs_highs - obs_lows) - 1.0
    return obs / 255.0



class Evaluator:

    def __init__(self, env_name, num_eval_episodes, checkpoint_dir, epsilon_greedy):
        self.env_name = env_name
        self.num_eval_episodes = num_eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.epsilon_greedy = epsilon_greedy

    def evaluate(self, agent):
        ale = self.setup_eval_env(self.env_name)
        self.eval(ale, self.env_name, agent)

    def setup_eval_env(self, env_name):
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
        #env id, env type, num envs, and seed
        env = make_vec_env(env_id, env_type, 1, 0, wrapper_kwargs={'clip_rewards':False,'episode_life':False,})
        if env_type == 'atari':
            env = VecFrameStack(env, 4)
        return env


    def eval(self, env, env_name, agent):
        rewards = []
        # 100 episodes
        episode_count = self.num_eval_episodes
        reward = 0
        done = False
        rewards = []
        writer = open(self.checkpoint_dir + "/" +self.env_name + "_bc_results.txt", 'w')
        for i in range(int(episode_count)):
            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                #preprocess the state
                state = preprocess(ob, env_name)
                state = np.transpose(state, (0, 3, 1, 2))
                if np.random.rand() < self.epsilon_greedy:
                    #print('eps greedy action')
                    action = env.action_space.sample()
                else:
                    #print('policy action')
                    action = agent.get_action(state)
                ob, reward, done, _ = env.step(action)

                steps += 1
                acc_reward += reward
                if done:
                    print("Episode: {}, Steps: {}, Reward: {}".format(i,steps,acc_reward))
                    writer.write("{}\n".format(acc_reward[0]))
                    rewards.append(acc_reward)
                    break

        print("Mean reward is: " + str(np.mean(rewards)))
