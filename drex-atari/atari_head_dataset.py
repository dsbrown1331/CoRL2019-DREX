from scipy import stats as st
import math
import csv
import os
from os import path, listdir
import numpy as np


class AtariHeadDataset():

    def __init__(self, env_name, data_path):
        '''
            Loads the dataset trajectories into memory.
            data_path is the root of the dataset (the folder, which contains
            the game folders, each of which contains the trajectory folders.
        '''

        self.trajs_path = data_path
        self.env_name = env_name

        # check that the we have the trajs where expected
        print(self.trajs_path)
        assert path.exists(self.trajs_path)
        self.trajectories = self.load_trajectories()

        # compute the stats after loading
        self.stats = {}
        for g in self.trajectories.keys():
            self.stats[g] = {}
            nb_games = self.trajectories[g].keys()

            total_frames = sum([len(self.trajectories[g][traj])
                                for traj in self.trajectories[g]])
            final_scores = [self.trajectories[g][traj][-1]['score']
                            for traj in self.trajectories[g]]

            self.stats[g]['total_replays'] = len(nb_games)
            self.stats[g]['total_frames'] = total_frames
            self.stats[g]['max_score'] = np.max(final_scores)
            self.stats[g]['min_score'] = np.min(final_scores)
            self.stats[g]['avg_score'] = np.mean(final_scores)
            self.stats[g]['stddev'] = np.std(final_scores)
            self.stats[g]['sem'] = st.sem(final_scores)

    def load_trajectories(self):
        print('env name: ', self.env_name)
        # read the meta data csv file
        trial_nums = []
        loaded_from = {}
        final_trial_episode = {}
        with open(self.trajs_path+'meta_data.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    game_name = row[0].lower()
                    if game_name == self.env_name:
                        trial_num = int(row[1])
                        trial_nums.append(trial_num)
                        if int(row[3]) != 0:
                            loaded_from[trial_num] = int(row[3])
                            final_trial_episode[trial_num] = 0
                    line_count += 1

        d = path.join(self.trajs_path, self.env_name)
        trials = [o for o in listdir(d)
                  if path.isdir(path.join(d, o))]

        # discard trial numbers <180 (episode # not recorded)
        trial_nums = [t for t in trial_nums if t >= 180]

        # trajectory folder names for the chosen env
        valid_trials = [t for t in trials if int(
            t.split('_')[0]) in trial_nums]
        valid_trial_nums = [int(t.split('_')[0])
                            for t in trials if int(t.split('_')[0]) in trial_nums]
        print('valid trials:', valid_trials)
        print('valid trial nums:', valid_trial_nums)

        trajectories = {}
        game = self.env_name
        trajectories[game] = {}
        last_traj = {}

        game_dir = d

        # sort the trajectories by trial nums
        traj_list = listdir(game_dir)
        traj_list.sort(key=lambda x: x.split('_')[0])

        for traj in traj_list:
            curr_trial = int(traj.split('_')[0])
            if(traj.endswith('.txt') and curr_trial in valid_trial_nums):
                if curr_trial in loaded_from.keys():
                    curr_traj = last_traj[loaded_from[curr_trial]]
                    last_episode = final_trial_episode[loaded_from[curr_trial]]
                else:
                    curr_traj = []
                    # TODO: only if starting a new episode, update for loading from previous game
                    last_episode = 0
                with open(path.join(game_dir, traj)) as f:
                    for i, line in enumerate(f):
                        # first line is the metadata, second is the header
                        if i > 1:

                            #frame,reward,score,terminal, action
                            curr_data = line.rstrip('\n').split(',')

                            # skipping the frames with a NULL score
                            if curr_data[2] != 'null':
                                curr_trans = {}
                                curr_trans['frame'] = (curr_data[0])
                                curr_trans['episode'] = int(
                                    curr_data[1]) if curr_data[1] != 'null' else float('nan')
                                curr_trans['score'] = int(curr_data[2])
                                curr_trans['duration'] = int(
                                    curr_data[3]) if curr_data[3] != 'null' else float('nan')
                                curr_trans['reward'] = int(
                                    curr_data[4]) if curr_data[4] != 'null' else float('nan')
                                curr_trans['action'] = int(
                                    curr_data[5]) if curr_data[5] != 'null' else float('nan')
                                curr_trans['gaze_positions'] = (
                                    [float(gp) if gp != 'null' else float('nan') for gp in curr_data[6:]])
                                curr_trans['img_dir'] = traj.strip('.txt')

                                # start a new current trajectory if next epiosde begins
                                # save traj number beginning from 0 for these initial episodes
                                if(curr_trans['episode'] != last_episode and not math.isnan(curr_trans['episode'])):
                                    extra_episode_num = curr_trial
                                    while(extra_episode_num in valid_trial_nums):
                                        print(
                                            'randomly sampling a trial number for extra episodes in a trajectory')
                                        extra_episode_num = np.random.randint(
                                            0, 1000)
                                    print(extra_episode_num, traj,
                                          curr_trans['episode'], last_episode)
                                    trajectories[game][extra_episode_num] = curr_traj
                                    curr_traj = []
                                else:
                                    curr_traj.append(curr_trans)
                                last_episode = curr_trans['episode']
                                final_trial_episode[curr_trial] = last_episode
                                last_traj[curr_trial] = curr_traj
                trajectories[game][int(traj.split('_')[0])] = curr_traj
        return trajectories
