import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import gym

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

from tf_commons.ops import Linear
from utils import PPO2Agent

class Policy(object):
    def __init__(self,ob_dim,ac_dim,num_layers,embed_size):
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            self.inp = tf.placeholder(tf.float32,[None,ob_dim])
            self.l = tf.placeholder(tf.float32,[None,ac_dim])
            self.l2_reg = tf.placeholder(tf.float32,[])

            with tf.variable_scope('weights') as param_scope:
                self.param_scope = param_scope

                fcs = []
                last_dims = ob_dim
                for l in range(num_layers):
                    fcs.append(Linear('fc%d'%(l+1),last_dims,embed_size)) #(l+1) is gross, but for backward compatibility
                    last_dims = embed_size
                fcs.append(Linear('fc%d'%(num_layers+1),last_dims,ac_dim))

            # build graph
            def _build(x):
                for fc in fcs[:-1]:
                    x = tf.nn.relu(fc(x))
                pred_a = fcs[-1](x)
                return pred_a

            self.ac = _build(self.inp)

            loss = tf.reduce_sum((self.ac-self.l)**2,axis=1)
            self.loss = tf.reduce_mean(loss,axis=0)

            weight_decay = 0.
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.w**2)

            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

            self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self,train=False):
        with self.graph.as_default():
            if train:
                return tf.trainable_variables(self.param_scope.name)
            else:
                return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size,iter,l2_reg,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs)-1)
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            batch = []
            for i in idxes:
                batch.append((obs[i],acs[i]))
            b_s,b_a = [np.array(e) for e in zip(*batch)]

            return b_s,b_a

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_s,b_a = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,_ = sess.run([self.loss,self.l2_loss,self.update_op],feed_dict={
                    self.inp:b_s,
                    self.l:b_a,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_s,b_a= _batch(valid_idxes)
                    valid_loss = sess.run(self.loss,feed_dict={
                        self.inp:b_s,
                        self.l:b_a,
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f'%(loss,l2_loss,valid_loss)))

    def act(self, observation, reward, done):
        sess = self.sess

        with self.graph.as_default():
            ac = sess.run(self.ac,feed_dict={self.inp:observation[None]})[0]

        return ac

    def save(self,path):
        with self.graph.as_default():
            self.saver.save(self.sess,path,write_meta_graph=False)

    def load(self,path):
        with self.graph.as_default():
            self.saver.restore(self.sess,path)

class Dataset(object):
    def __init__(self,env, env_type):
        self.env = env
        self.env_type = env_type

        if hasattr(self.env.unwrapped,'num_envs'):
            self.venv = True
        else:
            self.venv = False

    def gen_traj(self,agent,min_length):
        if self.venv:
            obs, actions, rewards = [self.env.reset()[0]], [], []
        else:
            obs, actions, rewards = [self.env.reset()], [], []

        # For debug purpose
        last_episode_idx = 0
        acc_rewards = []
        last_x_poses = []

        while True:
            action = agent.act(obs[-1], None, None)
            if self.venv:
                ob, reward, done, _ = self.env.step(action[None])
                ob, reward, done = ob[0], reward[0], done[0]

                if not done: #virtual env reset by itself...
                    last_x_pos = self.env.venv.envs[0].unwrapped.sim.data.qpos.copy()[0]
            else:
                ob, reward, done, _ = self.env.step(action)

                last_x_pos = self.env.unwrapped.sim.data.qpos.copy()[0]

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                if self.env_type == 'mujoco':
                    last_x_poses.append(last_x_pos)

                if len(obs) < min_length:
                    obs.pop()

                    if self.venv:
                        obs.append(self.env.reset()[0])
                    else:
                        obs.append(self.env.reset())

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                else:
                    obs.pop()

                    acc_rewards.append(np.sum(rewards[last_episode_idx:]))
                    last_episode_idx = len(rewards)
                    break

        return np.stack(obs,axis=0), np.stack(actions,axis=0), np.array(rewards), np.mean(acc_rewards), np.std(acc_rewards), last_x_poses

    def prebuilt(self,agents,num_trajs,min_length):
        assert len(agents)>0, 'no agent given'
        trajs = []
        last_x_poses = []

        for agent in tqdm(agents):
            agent_trajs = []
            agent_last_x_poses = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_ in agent_trajs])  < min_length) or \
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                (*traj), avg_acc_reward, std_acc_reward, last_x_pos = self.gen_traj(agent,-1)

                agent_trajs.append(traj)
                agent_last_x_poses += last_x_pos

            returns = [np.sum(rewards) for _,_,rewards in agent_trajs]

            trajs += agent_trajs
            last_x_poses += agent_last_x_poses

            tqdm.write('model: %s avg reward: %f std: %f'%(agent.model_path,np.mean(returns),np.std(returns)))

        self.trajs = trajs
        self.last_x_poses = last_x_poses

    def getD(self):
        obs,actions,rewards = zip(*self.trajs)
        obs,actions,rewards = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))
        print(obs.shape,actions.shape,rewards.shape)
        return obs,actions,rewards

    def infer_action(self,inv_model):
        obs, _, rewards = self.trajs
        acs = inv_model.infer_action(obs)
        self.trajs = (obs,acs,rewards)

    def save(self,path,filename='dataset.pkl'):
        with open(os.path.join(path,filename),'wb') as f:
            pickle.dump(self.trajs,f)

        if self.env_type == 'mujoco':
            with open(os.path.join(path,'last_x_pos.txt'),'w') as f:
                for x_pos in self.last_x_poses:
                    print(x_pos,file=f)

    def save_for_gail(self,filename):
        """
        npz file:
            obs: N-length list (N is # trajectories used for training) of numpy array having shape of [traj_len,ob_dim]
            acs : N-length list (N is # trajectories used for training) of numpy array having shape of [traj_len,ac_dim]
            reward: N-length list (N is # trajectories used for training) of numpy array having shape of [traj_len,]
            ep_rets : N-length list of float
        """
        obs_list, acs_list, rews_list, ep_rets = [], [], [], []
        for obs,acs,rewards in self.trajs:
            obs_list.append(obs)
            acs_list.append(acs)
            rews_list.append(rewards)
            ep_rets.append(np.sum(rewards))

        np.savez(filename,obs=obs_list,acs=acs_list,rews=rews_list,ep_rets=np.array(ep_rets))

    def load(self,path,filename='dataset.pkl'):
        with open(os.path.join(path,filename),'rb') as f:
            self.trajs = pickle.load(f)

def setup_logdir(log_path,args):
    logdir = Path(log_path)
    if logdir.exists() :
        c = input('%s is already exist. continue [Y/etc]? '%(log_path))
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
    logdir.mkdir(parents=True)
    with open(str(logdir/'args.txt'),'w') as f:
        f.write( str(args) )
    return str(logdir)

def build_dataset(args,env):
    dataset = Dataset(env,args.env_type)

    demo_agents = []
    demo_chkpt = eval(args.demo_chkpt)
    if type(demo_chkpt) == int:
        demo_chkpt = list(range(demo_chkpt+1))
    else:
        demo_chkpt = list(demo_chkpt)

    models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) in demo_chkpt])
    for path in models:
        agent = PPO2Agent(env,args.env_type,str(path),stochastic=args.stochastic)
        demo_agents.append(agent)

    dataset.prebuilt(demo_agents,args.num_trajs,args.min_length)
    return dataset

def bc(args):
    logdir = setup_logdir(os.path.join(args.log_path,'bc'),args)

    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
    else:
        assert False

    dataset = build_dataset(args,env)
    dataset.save(logdir)

    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)

    D = dataset.getD()
    try:
        policy.train(D,args.batch_size,args.num_iter,l2_reg=args.l2_reg,debug=True)
    except KeyboardInterrupt:
        pass
    policy.save(os.path.join(logdir,'model.ckpt'))

def eval_(args):
    env = gym.make(args.env_id)

    logdir = str(Path(args.log_path))
    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],args.num_layers,args.embed_size)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    policy.load(os.path.join(logdir,'model.ckpt'))

    from performance_checker import gen_traj
    from gym.wrappers import Monitor
    perfs = []
    for j in tqdm(range(args.num_tries)):
        if j == 0 and args.video_record:
            wrapped = Monitor(env, './video/',force=True)
        else:
            wrapped = env

        perfs.append(gen_traj(wrapped,policy,args.render,99999))
    print(logdir, ',', np.mean(perfs), ',', np.std(perfs))

def gen_gail(args):
    env = gym.make(args.env_id)

    bc_dataset = Dataset(env,args.env_type)
    bc_dataset.load(os.path.join(args.log_path,'bc'))

    bc_dataset.save_for_gail(os.path.join(args.log_path,'bc','gail_data.npz'))

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    # Train Setting
    parser.add_argument('--mode', required=True, choices=['bc','eval','gen_gail'])
    parser.add_argument('--log_path', required=True, help='path to log base (mode & env_id will be concatenated at the end)')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--env_type', default='mujoco', help='mujoco, atari, or robosuite', choices=['mujoco','robosuite'])
    # Dataset
    parser.add_argument('--learners_path', required=True, help='path of learning agents')
    parser.add_argument('--demo_chkpt', default='240', help='decide upto what learner stage you want to give')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--num_trajs', default=0,type=int, help='')
    parser.add_argument('--min_length', default=1000,type=int, help='minimum length of trajectory generated by each agent')
    # Network
    parser.add_argument('--num_layers', default=4,type=int)
    parser.add_argument('--embed_size', default=256,type=int)
    # Training
    parser.add_argument('--l2_reg', default=0.001, type=float, help='l2 regularization size')
    parser.add_argument('--inv_model', default='') # For BCO only
    parser.add_argument('--num_iter',default=50000,type=int)
    parser.add_argument('--batch_size',default=128,type=int)
    # For Eval
    parser.add_argument('--num_tries', default=10, type=int, help='path of learning agents')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args()

    if args.mode == 'bc':
        bc(args)
    elif args.mode == 'eval':
        eval_(args)
    elif args.mode == 'gen_gail':
        gen_gail(args)
    else:
        assert False
