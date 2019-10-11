import os
import pickle
from pathlib import Path
from argparse import Namespace
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import gym

from utils import Model, MujocoNet, PPO2Agent
from bc_noise import BCNoisePreferenceDataset

PREGEN_TRAJS = {
    'Ant-v2':'./learner/demo_models/ant/Ant-v2_trajs.pkl',
    'Hopper-v2':'./learner/demo_models/hopper/Hopper-v2_trajs.pkl',
    'HalfCheetah-v2':'./learner/demo_models/halfcheetah/HalfCheetah-v2_trajs.pkl',
    'SawyerReach':'./learner/demo_models/sawyerreacher_v3/SawyerReach_trajs.pkl',
}

def _generate_trajs(
    env_id,
    env_type,
    demo_paths,
    num_trajs_per_agent,
    log_dir,
):
    if env_type == 'mujoco':
        env = gym.make(env_id)
    else:
        assert False

    agents = sorted([p for p in Path(demo_paths).glob('?????')])

    def _run_episode(agent):
        obs, actions, rewards, done = [env.reset()], [], [], False
        while not done:
            action = agent.act(obs[-1], None, None)
            ob, reward, done, _ = env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

        return (np.stack(obs,axis=0), np.array(actions), np.array(rewards))

    trajs = []
    for path in tqdm(agents):
        agent = PPO2Agent(env,env_type,str(path),stochastic=True)

        for _ in range(num_trajs_per_agent):
            trajs.append(_run_episode(agent))

    with open(os.path.join(log_dir,'%s_trajs.pkl'%env_id),'wb') as f:
        pickle.dump(trajs,f)

def anal(model_dir,unseen_trajs=None,rl_trajs=None,save_path=None):
    graph = tf.Graph()
    config = tf.ConfigProto() # Run on CPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph,config=config)

    # Load Seen Trajectory for T-REX
    with graph.as_default():
        with sess.as_default():
            print(os.path.realpath(model_dir))

            with open(str(Path(model_dir)/'args.txt')) as f:
                args = eval(f.read())

            if args.env_type == 'mujoco':
                env = gym.make(args.env_id)
            else:
                False

            dataset = BCNoisePreferenceDataset(env,args.env_type,None,None)
            if args.prebuilt_noise_dataset:
                loaded = dataset.load_prebuilt(args.prebuilt_noise_dataset)
            else:
                loaded = dataset.load_prebuilt(model_dir)
            assert loaded, 'prebuilt.pkl does not exist'

            models = []
            for i in range(args.num_models):
                with tf.variable_scope('model_%d'%i):
                    net = MujocoNet(args.include_action,env.observation_space.shape[-1],env.action_space.shape[-1],num_layers=args.num_layers,embedding_dims=args.embedding_dims)

                    model = Model(net,batch_size=1)
                    model.saver.restore(sess,os.path.join(model_dir,'model_%d.ckpt'%i))

                    models.append(model)

            seen_trajs = [
                (obs,actions,rewards) for _,trajs in dataset.trajs for obs,actions,rewards,_ in trajs
            ]

    # Load Unseen Trajectories
    if unseen_trajs is None:
        traj_file = PREGEN_TRAJS.get(args.env_id,None)
        if traj_file is not None and os.path.exists(traj_file):
            with open(traj_file,'rb') as f:
                unseen_trajs = pickle.load(f)
        else:
            unseen_trajs = []

    # Load Demo Trajectories used for BC
    from bc_mujoco import Dataset as BCDataset
    bc_dataset = BCDataset(env,args.env_type)

    bc_agent_path = args.bc_agent_path if args.bc_agent_path else os.path.join(args.log_dir,'bc')
    bc_dataset.load(bc_agent_path)

    bc_trajs = bc_dataset.trajs

    # Calculate Predicted Returns
    def _get_return(obs,acs):
        with sess.as_default():
            return np.sum([model.get_reward(obs,acs) for model in models]) / len(models)

    seen = [1] * len(seen_trajs) + [0] * len(unseen_trajs) + [2] * len(bc_trajs)
    gt_returns, pred_returns = [], []

    for obs,actions,rewards in seen_trajs+unseen_trajs+bc_trajs:
        gt_returns.append(np.sum(rewards))
        pred_returns.append(_get_return(obs,actions))

    # Show RL-progres (optional)
    if rl_trajs:
        rl_returns = []
        for trajs in rl_trajs:
            for step,(obs,actions,rewards) in enumerate(trajs):
                x = np.sum(rewards)
                y = _get_return(obs,actions)
                c = step

                rl_returns.append((x,y,c))
    else:
        rl_returns = None

    save_path = save_path if save_path else os.path.join(args.log_dir,'gt_vs_pred_rewards.pdf')
    draw(np.array(gt_returns),np.array(pred_returns),np.array(seen),rl_returns,save_path)

    sess.close()

def draw(gt_returns,pred_returns,seen,rl_returns,figname=False):
    """
    gt_returns: [N] length
    pred_returns: [N] length
    seen: [N] length
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pylab
    from matplotlib import pyplot as plt
    from imgcat import imgcat

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('ggplot')
    params = {
        'text.color':'black',
        'axes.labelcolor':'black',
        'xtick.color':'black',
        'ytick.color':'black',
        'legend.fontsize': 'xx-large',
        #'figure.figsize': (6, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'xx-large'}
    matplotlib.pylab.rcParams.update(params)

    def _convert_range(x,minimum, maximum,a,b):
        return (x - minimum)/(maximum - minimum) * (b - a) + a

    def _no_convert_range(x,minimum, maximum,a,b):
        return x

    convert_range = _convert_range
    #convert_range = _no_convert_range

    gt_max,gt_min = max(gt_returns),min(gt_returns)
    pred_max,pred_min = max(pred_returns),min(pred_returns)
    max_observed = np.max(gt_returns[np.where(seen!=1)])

    # Draw P
    fig,ax = plt.subplots()

    ax.plot(gt_returns[np.where(seen==0)],
            [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==0)]], 'go') # unseen trajs
    ax.plot(gt_returns[np.where(seen==1)],
            [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==1)]], 'bo') # seen trajs for T-REX
    ax.plot(gt_returns[np.where(seen==2)],
            [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==2)]], 'ro') # seen trajs for BC

    if rl_returns:
        x,y,c = zip(*rl_returns)
        x,y,c = np.array(x), np.array(y), np.array(c)

        y = convert_range(y,pred_min,pred_max,gt_min,gt_max)
        sc = ax.scatter(x,y,c=c)
        cbar = fig.colorbar(sc)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Training steps (relative)', rotation=270)

    ax.plot([gt_min-5,gt_max+5],[gt_min-5,gt_max+5],'k--')
    #ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)
    #ax.set_xlim([gt_min-5,gt_max+5])
    #ax.set_ylim([gt_min-5,gt_max+5])
    ax.set_xlabel("Ground Truth Returns")
    ax.set_ylabel("Predicted Returns (normalized)")
    fig.tight_layout()

    if figname:
        plt.savefig(figname)
        plt.close()
    else:
        imgcat(fig)
        plt.close()
