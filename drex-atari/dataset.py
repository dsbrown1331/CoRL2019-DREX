import numpy as np
import gc
import os
import random
from collections import namedtuple

Example = namedtuple('Example', 'state action')

def normalize_state(obs):

    #print(obs_highs)
    #print(obs_lows)
    #return  2.0 * (obs - obs_lows) / (obs_highs - obs_lows) - 1.0
    return obs / 255.0


class Dataset:

    def __init__(self, size, hist_len):
        self.size = size
        self.hist_len = hist_len
        self.states = np.empty((size, hist_len, 84, 84), dtype=np.float32)
        self.actions = np.empty(size)
        self.index = 0
        self.sample_indices = list(range(size))
        self.shuffle_indices()
        self.minibatch_index = 0

    def shuffle_indices(self):
        random.shuffle(self.sample_indices)

    def clear(self):
        self.states = np.empty((self.size, 84, 84), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.uint8)

    def store_experiences(self, storage_dir):
        np.save(os.path.join(storage_dir, "states"), self.states)
        np.save(os.path.join(storage_dir, "actions"), self.actions)

    def load_experiences(self, storage_dir):
        self.states = np.load(os.path.join(storage_dir, "states" + ".npy"))
        self.actions = np.load(os.path.join(storage_dir, "actions" + ".npy"))

    def get_dataset(self):
        dataset = []
        for i in range(self.size):
            dataset.append(Example(state=normalize_state(self.states[i]),
                            action=self.actions[i]))
        return dataset



    def add_item(self, state, action):
        if self.index == self.size:
            raise ValueError("Dataset is full. Clear dataset before adding anything.")
        # input a_t, r_t, f_t+1, episode done at t+1
        self.states[self.index, ...] = state
        self.actions[self.index] = action
        self.index += 1



    def sample_minibatch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            index = self.sample_indices[self.minibatch_index]
            batch.append(Example(state=normalize_state(self.states[index]),
                                action=self.actions[index]))
            self.minibatch_index = self.minibatch_index + 1
            if self.minibatch_index >= self.size:
                self.minibatch_index = 0
                self.shuffle_indices()
        return batch
