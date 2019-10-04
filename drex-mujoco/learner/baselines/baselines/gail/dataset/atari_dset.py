import os
import sys
sys.path.append(os.path.abspath('../../../../baselines/'))

from baselines.gail.dataset.mujoco_dset import Dset
import numpy as np
import pickle
from pathlib import Path
from baselines import logger

class Atari_Dset(object):
    frame_stack = 4

    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        with open(str(Path(expert_path)/'demo_trajs.pkl'),'rb') as f:
            traj_data = pickle.load(f)

        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])

        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        self.obs = np.concatenate([ob[:-1] for ob in obs],axis=0)
        self.acs = np.concatenate(acs,axis=0)

        print(self.obs.shape,self.acs.shape)

        # Dataset statistics
        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))

        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction)],
                              self.acs[:int(self.num_transition*train_fraction)],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):],
                            self.acs[int(self.num_transition*train_fraction):],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Best returns: %f" % np.max(self.rets))
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Atari_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, required=True)
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)

