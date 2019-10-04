import os
import sys
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
import argparse
import gym
from tqdm import tqdm

sys.path.append(os.path.abspath('../../../../baselines/'))

class PPO2Agent(object):
    def __init__(self, env, env_type, path, stochastic=False, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model

        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0})

        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env,'cnn')
                else:
                    assert False,' not supported env_type'

                make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                nsteps=1, ent_coef=0., vf_coef=0.,
                                max_grad_norm=0.)
                self.model = make_model()

                self.model_path = path
                self.model.load(path)

        self.stochastic = stochastic

    def act(self, obs, reward, done):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.stochastic:
                    a,v,state,neglogp = self.model.step(obs[None])
                else:
                    a = self.model.act_model.act(obs[None])
        return a[0]

def gen_traj(env,policy,max_len):
    ob, r, done = env.reset(), 0., False

    #ob has a shape of [H,W,#frame_stack=4]. Note that the frames between obs overlap.
    frames, actions, rewards = [ob], [], []

    while not done:
        a = policy(ob,r,done)
        ob,r,done,_ = env.step(a)

        frames.append(ob)
        actions.append(a)
        rewards.append(r)

        if max_len and len(frames) > max_len:
            break

    return np.array(frames), np.array(actions), np.array(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate trajectories and save it in tfrecords format")
    parser.add_argument('--out', required=True)
    parser.add_argument('--env_id', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--learners_path', help='learner path', required=True)
    parser.add_argument('--chkpts', default=600, help='range or number / checkpoints to use')
    parser.add_argument('--stochastic',action='store_true')
    #parser.add_argument('--ret_thresh', help='Return threshhold', type=int, default=4000) #skip poor trajectory.
    parser.add_argument('--num_traj', help='Number of trajectories to collect for each given checkpoints', type=int, default=15)
    parser.add_argument('--max_len', help='Maximum trajecroy length', type=int, default=None) #it was step_thresh before.
    args = parser.parse_args()

    from baselines.common.atari_wrappers import make_atari, wrap_deepmind
    env = wrap_deepmind(make_atari(args.env_id),episode_life=False,clip_rewards=False,frame_stack=True,scale=False)

    # Load Agents
    chkpts = eval(args.chkpts)
    if type(chkpts) == int:
        chkpts = [chkpts]
    else:
        chkpts = list(chkpts)
    agents = []

    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) in chkpts])
    for path in models:
        agent = PPO2Agent(env,'atari',str(path),stochastic=args.stochastic)
        agents.append(agent)

    Path(args.out).mkdir(parents=True,exist_ok=True)

    trajs = {
        'obs': [],
        'acs': [],
        'rews': [],
        'ep_rets': [],
    }
    for agent in tqdm(agents):
        for i in tqdm(range(args.num_traj)):
            frames, actions, rewards = gen_traj(env,agent.act,args.max_len)

            trajs['obs'].append(frames)
            trajs['acs'].append(actions)
            trajs['rews'].append(rewards)
            trajs['ep_rets'].append(np.sum(rewards))

    with open(str(Path(args.out)/'demo_trajs.pkl'), 'wb') as f:
        pickle.dump(trajs, f)
