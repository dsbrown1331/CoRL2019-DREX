import os
import argparse
import pickle
from functools import partial
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm

from bc_mujoco import Policy, setup_logdir
from bc_noise import BCNoisePreferenceDataset
from utils import MujocoNet, Model

def train_reward(args):
    # set random seed
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    trex_dir = args.trex_dir if args.trex_dir else 'infer_reward'
    log_dir = os.path.join(args.log_dir,trex_dir)
    setup_logdir(log_dir,args)

    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
    else:
        assert False

    ob_shape = env.observation_space.shape
    ac_dims = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[-1]

    dataset = BCNoisePreferenceDataset(env,args.env_type,args.max_steps,args.min_noise_margin)

    if args.prebuilt_noise_dataset:
        loaded = dataset.load_prebuilt(args.prebuilt_noise_dataset)
        assert loaded
    else:
        bc_agent_path = args.bc_agent_path if args.bc_agent_path else os.path.join(args.log_dir,'bc')

        if args.init_rl_with_bc_model:
            from utils import PPO2Agent
            policy = PPO2Agent(env,args.env_type,os.path.join(bc_agent_path,'bc_model_policy_only'),stochastic=args.stochastic)
        else:
            policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],num_layers=4,embed_size=256)
            policy.load(os.path.join(bc_agent_path,'model.ckpt'))

        dataset.prebuild(policy,eval(args.noise_range),args.num_trajs,args.min_length,log_dir)

        from bc_mujoco import Dataset as BCDataset
        bc_dataset = BCDataset(env,args.env_type)
        bc_dataset.load(bc_agent_path)

        dataset.draw_fig(args.log_dir,bc_dataset.trajs)

        if args.add_bc_dataset:
            dataset.trajs.append((-99999, bc_dataset.trajs))

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            net = MujocoNet(args.include_action,ob_shape[-1],ac_dims,num_layers=args.num_layers,embedding_dims=args.embedding_dims)
            model = Model(net,batch_size=64)
            models.append(model)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        D = dataset.sample(args.D,include_action=args.include_action)

        model.train(D,iter=args.iter,l2_reg=args.l2_reg,noise_level=args.noise,debug=True)

        model.saver.save(sess,os.path.join(log_dir,'model_%d.ckpt'%(i)),write_meta_graph=False)

    sess.close()

def eval_reward(args):
    from model_analysis import anal

    trex_dir = args.trex_dir if args.trex_dir else 'infer_reward'
    model_dir = os.path.join(args.log_dir,trex_dir)

    anal(model_dir)

def train_rl(args):
    # Train an agent
    import pynvml as N
    import subprocess, multiprocessing
    ncpu = multiprocessing.cpu_count()
    N.nvmlInit()
    ngpu = N.nvmlDeviceGetCount()

    rl_dir = args.rl_dir if args.rl_dir else 'ppo'
    log_dir = os.path.abspath(os.path.join(args.log_dir,rl_dir))
    setup_logdir(log_dir,args)

    trex_dir = args.trex_dir if args.trex_dir else 'infer_reward'
    model_dir = os.path.join(args.log_dir,trex_dir)

    kwargs = {
        "model_dir":os.path.abspath(model_dir),
        "ctrl_coeff":args.ctrl_coeff,
        "alive_bonus":args.alive_bonus
    }

    procs = []
    for i in range(args.rl_runs):
        # Prepare Command
        template = 'python -m baselines.run --alg=ppo2 --env={env} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma} --seed {seed}'

        cmd = template.format(
            env=args.env_id,
            nenv=1, #ncpu//ngpu,
            num_timesteps=args.num_timesteps,
            save_interval=args.save_interval,
            custom_reward=args.custom_reward,
            gamma=args.gamma,
            seed=i,
            kwargs=str(kwargs)
        )

        if args.init_rl_with_bc_model:
            bc_agent_path = args.bc_agent_path if args.bc_agent_path else os.path.join(args.log_dir,'bc')
            p = Path(bc_agent_path)/'bc_model_policy_only'

            cmd += f" --load_path {str(p.absolute())}"

        # Prepare Log settings through env variables
        env = os.environ.copy()
        env["OPENAI_LOGDIR"] = os.path.join(log_dir,'run_%d'%i)
        if i == 0:
            env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
        else:
            env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)

        # run process
        procs.append(p)

    for line in procs[0].stdout:
        print(line.decode(),end='')

    for p in procs[1:]:
        p.wait()

def eval_rl(args):
    from utils import PPO2Agent
    if args.env_type == 'mujoco':
        env = gym.make(args.env_id)
        from utils import get_perf
    else:
        assert False

    with open(os.path.join(args.log_dir,'rl_results.txt'),'w') as f:
        def _get_perf(agent, num_eval=20):
            return [ get_perf(env,agent) for _ in range(num_eval)]

        # Load BC agent; to compare with this data
        bc_agent_path = args.bc_agent_path if args.bc_agent_path else os.path.join(args.log_dir,'bc')

        if args.init_rl_with_bc_model:
            agent = PPO2Agent(env,args.env_type,os.path.join(bc_agent_path,'bc_model_policy_only'),stochastic=args.stochastic)
        else:
            agent = Policy(env.observation_space.shape[0],env.action_space.shape[0],num_layers=4,embed_size=256)
            agent.load(os.path.join(bc_agent_path,'model.ckpt'))

        bc_perfs = _get_perf(agent)
        print('[%s] %f %f'%('bc',np.mean(bc_perfs),np.std(bc_perfs)))
        print('[%s] %f %f'%('bc',np.mean(bc_perfs),np.std(bc_perfs)),file=f)

        # Load T-REX learned agent
        rl_dir = args.rl_dir if args.rl_dir else 'ppo'
        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir,rl_dir)))
        print(str(agents_dir))

        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir/('run_%d'%i)/'checkpoints'/step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env,args.env_type,str(path),stochastic=True)
                agent_perfs = _get_perf(agent)
                print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs[-5:]),np.std(agent_perfs[-5:])))
                print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs[-5:]),np.std(agent_perfs[-5:])),file=f)

                perfs += agent_perfs
            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)))
            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)),file=f)

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--mode', default='all',choices=['all','train_reward','eval_reward','train_rl','eval_rl'])
    # Env Setting
    parser.add_argument('--env_id', default='Hopper-v2', help='Select the environment to run')
    parser.add_argument('--env_type', default='mujoco',choices=['mujoco'])
    # Demonstrator Setting
    parser.add_argument('--prebuilt_noise_dataset', default=None, help='if not specified, it will generate a new dataset')
    parser.add_argument('--bc_agent_path', default=None, help='default will be log_dir/bc')
    parser.add_argument('--trex_dir', default='infer_reward')
    parser.add_argument('--noise_range', default='np.arange(0.,1.00,0.05)', help='decide upto what learner stage you want to give')
    parser.add_argument('--num_trajs', default=5,type=int, help='number of trajectory generated by each agent')
    parser.add_argument('--min_length', default=0,type=int, help='minimum length of trajectory generated by each agent')
    parser.add_argument('--init_rl_with_bc_model', action='store_true')
    parser.add_argument('--stochastic', action='store_true', help='underlying policy for noisy dataset is stochastic or not')
    # Dataset setting
    parser.add_argument('--D', default=5000, type=int, help='|D| in the preference paper')
    parser.add_argument('--max_steps', default=50, type=int, help='maximum length of subsampled trajecotry')
    parser.add_argument('--add_bc_dataset', action='store_true')
    parser.add_argument('--min_noise_margin', default=0.3, type=float, help='')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    # Network setting
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.0, type=float, help='noise level to add on training label (another regularization)')
    parser.add_argument('--iter', default=1000, type=int, help='# trainig iters')
    # Args for PPO
    parser.add_argument('--rl_runs', default=5, type=int)
    parser.add_argument('--rl_dir', default=None)
    parser.add_argument('--custom_reward', default='preference_normalized_v2', choices=['preference','preference_normalized','preference_normalized_v2'])
    parser.add_argument('--num_timesteps', default=int(1e6), type=int)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()

    if args.mode == 'all':
        tf.reset_default_graph()
        train_reward(args)
        tf.reset_default_graph()
        eval_reward(args)
        tf.reset_default_graph()
        train_rl(args)
        tf.reset_default_graph()
        eval_rl(args)
    elif args.mode == 'train_reward':
        train_reward(args)
        tf.reset_default_graph()
        eval_reward(args)
    elif args.mode == 'eval_reward':
        eval_reward(args)
    elif args.mode =='train_rl':
        train_rl(args)
        tf.reset_default_graph()
        eval_rl(args)
    elif args.mode == 'eval_rl':
        eval_rl(args)
