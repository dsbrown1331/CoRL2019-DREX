import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, env, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = (env.action_space.n,)
        self.input_shape = tuple([84,84,4+env.action_space.n])
        self.num_actions = env.action_space.n
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.int32, (None,) , name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.int32, (None,), name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        """
        obs_ph: tf tensor shape of [None,84,84,4]
        acs_ph: tf tensor shape of [None,ac_dim] #one hot encoding

        """
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            one_hot_ac = tf.one_hot(acs_ph,self.num_actions,dtype=tf.float32)
            x = tf.concat([obs_ph/255.0, tf.tile(one_hot_ac[:,None,None,:],[1,84,84,1])],
                          axis=3) #[None,84,84,4+ac_dim]
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.tanh(tf.layers.dense(x, 512, name='lin1', kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(x, 1, name='lin2', kernel_initializer=U.normc_initializer(1.0))
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        obs = np.array(obs)

        sess = tf.get_default_session()
        if len(obs.shape) == 3:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 0:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
