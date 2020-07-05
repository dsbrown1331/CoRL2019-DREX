import sys
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import gym

from tf_commons.ops import Linear

# Import my own libraries
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/learner/baselines/')

####################

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample() #[None]

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
                if isinstance(env.observation_space,gym.spaces.Dict):
                    ob_space = env.observation_space.spaces['ob_flattened']
                else:
                    ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env,'cnn')
                elif env_type in ['mujoco','robosuite']:
                    policy = build_policy(env,'mlp')
                else:
                    assert False,' not supported env_type'

                make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                nsteps=1, ent_coef=0., vf_coef=0.,
                                max_grad_norm=0.)
                self.model = make_model()

                self.model_path = path
                self.model.load(path)

        if env_type in ['mujoco','robosuite']:
            with open(path+'.env_stat.pkl', 'rb') as f :
                import pickle
                s = pickle.load(f)
            self.ob_rms = s['ob_rms']
            #self.ret_rms = s['ret_rms']
            self.clipob = 10.
            self.epsilon = 1e-8
        else:
            self.ob_rms = None

        self.stochastic = stochastic

    def act(self, obs, reward, done):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)

        with self.graph.as_default():
            with self.sess.as_default():
                if self.stochastic:
                    a,v,state,neglogp = self.model.step(obs[None])
                else:
                    a = self.model.act_model.act(obs[None])
        return np.clip(a[0],-1.,1.)

def gen_traj(env,agent,min_length):
    obs, actions, rewards = [env.reset()], [], []
    while True:
        action = agent.act(obs[-1], None, None)
        ob, reward, done, _ = env.step(action)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)

        if done:
            if len(obs) < min_length:
                obs.pop()
                obs.append(env.reset())
            else:
                obs.pop()
                break

    return np.stack(obs,axis=0), np.array(actions), np.array(rewards)

####################

class RewardNet():
    def __init__(self,include_action,ob_dim,ac_dim,num_layers=2,embedding_dims=256):
        in_dims = ob_dim+ac_dim if include_action else ob_dim

        with tf.variable_scope('weights') as param_scope:
            fcs = []
            last_dims = in_dims
            for l in range(num_layers):
                fcs.append(Linear('fc%d'%(l+1),last_dims,embedding_dims)) #(l+1) is gross, but for backward compatibility
                last_dims = embedding_dims
            fcs.append(Linear('fc%d'%(num_layers+1),last_dims,1))

        self.fcs = fcs
        self.in_dims = in_dims
        self.include_action = include_action
        self.param_scope = param_scope

    def input_preprocess(self,obs,acs):
        assert len(obs) == len(acs) or len(obs) == len(acs)+1

        return \
            np.concatenate((obs[:len(acs)],acs),axis=1) if self.include_action \
            else obs

    def build_input_placeholder(self,name):
        return tf.placeholder(tf.float32,[None,self.in_dims],name=name)

    def build_reward(self,x):
        for fc in self.fcs[:-1]:
            x = tf.nn.relu(fc(x))
        r = tf.squeeze(self.fcs[-1](x),axis=1)
        return x, r

    def build_weight_decay(self):
        weight_decay = 0.
        for fc in self.fcs:
            weight_decay += tf.reduce_sum(fc.w**2)
        return weight_decay

class Model(object):
    def __init__(self,net:RewardNet,batch_size=64):
        self.B = batch_size
        self.net = net

        self.x = net.build_input_placeholder('x') # tensor shape of [B*steps] + input_dims
        self.x_split = tf.placeholder(tf.int32,[self.B]) # B-lengthed vector indicating the size of each steps
        self.y = net.build_input_placeholder('y') # tensor shape of [B*steps] + input_dims
        self.y_split = tf.placeholder(tf.int32,[self.B]) # B-lengthed vector indicating the size of each steps
        self.l = tf.placeholder(tf.int32,[self.B]) # [0 when x is better 1 when y is better]

        self.l2_reg = tf.placeholder(tf.float32,[]) # [0 when x is better 1 when y is better]

        # Graph Ops for Inference
        self.fv, self.r = net.build_reward(self.x)

        # Graph ops for training
        _, rs_xs = net.build_reward(self.x)
        self.v_x = tf.stack([tf.reduce_sum(rs_x) for rs_x in tf.split(rs_xs,self.x_split,axis=0)],axis=0)

        _, rs_ys = net.build_reward(self.y)
        self.v_y = tf.stack([tf.reduce_sum(rs_y) for rs_y in tf.split(rs_ys,self.y_split,axis=0)],axis=0)

        logits = tf.stack([self.v_x,self.v_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        # Regualarizer Ops
        weight_decay = net.build_weight_decay()
        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.v_y,self.v_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.net.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.net.param_scope.name)

    def train(self,D,iter=10000,l2_reg=0.01,noise_level=0.1,debug=False,early_term=False):
        """
        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
                while
                    sigma^{1,2}: shape of [steps,in_dims]
                    mu : 0 or 1
            l2_reg
            noise_level: input label noise to prevent overfitting
            debug: print training statistics
            early_term:  training will be early-terminate when validation accuracy is larger than 95%
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D)*0.8)]
        valid_idxes = idxes[int(len(D)*0.8):]

        def _load(idxes,add_noise=True):
            batch = []

            for i in idxes:
                batch.append(D[i])

            b_x,b_y,b_l = zip(*batch)
            x_split = np.array([len(x) for x in b_x])
            y_split = np.array([len(y) for y in b_y])
            b_x,b_y,b_l = np.concatenate(b_x,axis=0),np.concatenate(b_y,axis=0),np.array(b_l)

            if add_noise:
                b_l = (b_l + np.random.binomial(1,noise_level,self.B)) % 2 #Flip it with probability 0.1

            return b_x.astype(np.float32),b_y.astype(np.float32),x_split,y_split,b_l


        def _batch(idx_list,add_noise):
            if len(idx_list) > self.B:
                idxes = np.random.choice(idx_list,self.B,replace=False)
            else:
                idxes = idx_list

            return _load(idxes,add_noise)

        def load(idxes):
            b_x,b_y,x_split,y_split,b_l =\
                tf.py_func(_load, [idxes], [tf.float32,tf.float32,tf.int64,tf.int64,tf.int64], stateful=False)
            b_x = tf.reshape(b_x,[-1,84,84,4])
            b_y = tf.reshape(b_y,[-1,84,84,4])
            x_split = tf.reshape(x_split,[self.B])
            y_split = tf.reshape(y_split,[self.B])
            b_l = tf.reshape(b_l,[self.B])

            return b_x,b_y,x_split,y_split,b_l

        ds = tf.data.Dataset.range(len(D))
        ds = ds.repeat(-1)
        ds = ds.shuffle(len(D))
        ds = ds.batch(64)
        ds = ds.map(load, num_parallel_calls=8)
        ds = ds.prefetch(10)

        batch_op = ds.make_one_shot_iterator().get_next()

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_x,b_y,x_split,y_split,b_l = _batch(train_idxes,add_noise=True)
            #b_x,b_y,x_split,y_split,b_l = sess.run(batch_op)

            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.x_split:x_split,
                self.y_split:y_split,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_x,b_y,x_split,y_split,b_l = _batch(valid_idxes,add_noise=False)
                    valid_acc = sess.run(self.acc,feed_dict={
                        self.x:b_x,
                        self.y:b_y,
                        self.x_split:x_split,
                        self.y_split:y_split,
                        self.l:b_l
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))

            if early_term and valid_acc >= 0.95:
                print('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc))
                print('early termination@%08d'%it)
                break

    def get_reward(self,obs,acs,batch_size=1024):
        sess = tf.get_default_session()

        inp = self.net.input_preprocess(obs,acs)

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.x:inp[i:i+batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)

