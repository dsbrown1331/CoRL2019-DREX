'''
from baselines/ppo1/cnn_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    recurrent = False

    #def __init__(self, name, ob_space, ac_space, kind='large'):
    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = ob / 255.0
        def _build(name,x):
            if kind == 'small': # from A3C paper
                x = tf.nn.relu(U.conv2d(x, 16, "%s_l1"%name, [8, 8], [4, 4], pad="VALID"))
                x = tf.nn.relu(U.conv2d(x, 32, "%s_l2"%name, [4, 4], [2, 2], pad="VALID"))
                x = U.flattenallbut0(x)
                x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            elif kind == 'large': # Nature DQN
                x = tf.nn.relu(U.conv2d(x, 32, "%s_l1"%name, [8, 8], [4, 4], pad="VALID"))
                x = tf.nn.relu(U.conv2d(x, 64, "%s_l2"%name, [4, 4], [2, 2], pad="VALID"))
                x = tf.nn.relu(U.conv2d(x, 64, "%s_l3"%name, [3, 3], [1, 1], pad="VALID"))
                x = U.flattenallbut0(x)
                x = tf.nn.relu(tf.layers.dense(x, 512, name='%s_lin'%name, kernel_initializer=U.normc_initializer(1.0)))
            else:
                raise NotImplementedError
            return x

        last_out = _build('vf',x)
        self.vpred = tf.layers.dense(last_out, 1, name='vf_final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        last_out = _build('pol',x)
        logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='polfinal', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

